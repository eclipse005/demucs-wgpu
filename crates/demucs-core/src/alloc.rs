//! A caching global allocator.
//!
//! The host forward allocates and frees multi-megabyte intermediates hundreds of
//! times per chunk, and on Windows a fresh large allocation costs page faults on
//! first touch: the attention's 231 MB score matrix runs at 162 GFLOP/s when its
//! pages are faulted in every call and 724 GFLOP/s when the same buffer is
//! reused (measured, `demucs kernels` with `DEMUCS_GEMM_SCALING=1`). PyTorch's
//! allocator exists for the same reason and is a large part of why torch+CPU is
//! hard to catch.
//!
//! Design: round every request up to a size class, hand back a cached block for
//! that class when one is free, and keep a bounded free list otherwise. Every
//! block is allocated with 64-byte alignment, which covers the alignments the
//! model's tensors ask for (the largest is a 32-byte SIMD type), so the block a
//! cached pointer refers to is valid for any request of its class.
//!
//! This is only installed by the `demucs` binary (`#[global_allocator]`), so a
//! library consumer keeps the system allocator unless it opts in.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

/// Alignment every cached block is allocated with, and the largest the cache
/// will serve. A request needing more falls through to the system allocator.
const CACHE_ALIGN: usize = 64;

/// Bytes the free lists may hold before blocks start going back to the system.
/// The model's working set peaks around 1 GB, so this keeps a chunk's worth of
/// buffers resident without holding on to everything.
const MAX_CACHED_BYTES: usize = 2 << 30;

/// Requests at or above this size are passed straight through: a single block
/// that large is not worth pooling behind a mutex.
const MAX_CACHED_BLOCK: usize = 512 << 20;

/// Rounds a request up: powers of two to 1 MiB, then whole MiB steps. Sizes
/// repeat across iterations of the same op, so this gives a small number of
/// classes to search.
fn size_class(size: usize) -> usize {
    const SMALL: usize = 1 << 20;
    if size <= SMALL {
        size.next_power_of_two().max(CACHE_ALIGN)
    } else {
        (size + SMALL - 1) / SMALL * SMALL
    }
}

/// Blocks cached per small size class. Small classes are indexed directly (see
/// `class_index`), so a lookup touches at most this many slots.
const SLOTS_PER_CLASS: usize = 8;

/// Distinct large classes kept, each with up to `SLOTS_PER_CLASS` blocks.
const LARGE_CLASSES: usize = 32;

/// Smallest class that goes to the "large" table instead of the indexed one.
const SMALL_LIMIT: usize = 1 << 20;

/// Slot index for a size class, or `None` for the large table.
///
/// The first version of this allocator scanned a flat free list from the end,
/// which is O(entries): once a workload pushed thousands of small blocks (the
/// direct convolution's per-task accumulators), *every* later allocation in the
/// model paid a scan over the whole list, and the slowdown showed up in stages
/// that never touch the allocator's newest entries.
fn class_index(class: usize) -> Option<usize> {
    if class > SMALL_LIMIT {
        return None;
    }
    Some(class.trailing_zeros() as usize)
}

/// `(class, pointers)` for the classes above `SMALL_LIMIT`.
struct FreeList {
    small: [[*mut u8; SLOTS_PER_CLASS]; 21],
    small_len: [u8; 21],
    large_class: [usize; LARGE_CLASSES],
    large: [[*mut u8; SLOTS_PER_CLASS]; LARGE_CLASSES],
    large_len: [u8; LARGE_CLASSES],
    large_used: usize,
}

// SAFETY: the arrays hold pointers `System` returned and that nobody owns; all
// access is behind the mutex.
unsafe impl Send for FreeList {}
unsafe impl Sync for FreeList {}

pub struct CachingAllocator {
    free: Mutex<FreeList>,
    cached_bytes: AtomicUsize,
    hits: AtomicUsize,
    misses: AtomicUsize,
}

impl CachingAllocator {
    pub const fn new() -> Self {
        Self {
            free: Mutex::new(FreeList {
                small: [[std::ptr::null_mut(); SLOTS_PER_CLASS]; 21],
                small_len: [0; 21],
                large_class: [0; LARGE_CLASSES],
                large: [[std::ptr::null_mut(); SLOTS_PER_CLASS]; LARGE_CLASSES],
                large_len: [0; LARGE_CLASSES],
                large_used: 0,
            }),
            cached_bytes: AtomicUsize::new(0),
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
        }
    }

    /// `(hits, misses)` so far, for the benchmark output.
    pub fn stats(&self) -> (usize, usize) {
        (
            self.hits.load(Ordering::Relaxed),
            self.misses.load(Ordering::Relaxed),
        )
    }
}

impl Default for CachingAllocator {
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl GlobalAlloc for CachingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.align() > CACHE_ALIGN || layout.size() > MAX_CACHED_BLOCK {
            return System.alloc(layout);
        }
        let class = size_class(layout.size());
        {
            let mut free = self.free.lock().unwrap_or_else(|e| e.into_inner());
            let taken = match class_index(class) {
                Some(index) => {
                    let len = free.small_len[index] as usize;
                    if len > 0 {
                        free.small_len[index] = (len - 1) as u8;
                        Some(free.small[index][len - 1])
                    } else {
                        None
                    }
                }
                None => {
                    let mut found = None;
                    for slot in 0..free.large_used {
                        if free.large_class[slot] == class && free.large_len[slot] > 0 {
                            let len = free.large_len[slot] as usize;
                            free.large_len[slot] = (len - 1) as u8;
                            found = Some(free.large[slot][len - 1]);
                            break;
                        }
                    }
                    found
                }
            };
            if let Some(pointer) = taken {
                self.cached_bytes.fetch_sub(class, Ordering::Relaxed);
                self.hits.fetch_add(1, Ordering::Relaxed);
                return pointer;
            }
        }
        self.misses.fetch_add(1, Ordering::Relaxed);
        let classed = Layout::from_size_align_unchecked(class, CACHE_ALIGN);
        System.alloc(classed)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if layout.align() > CACHE_ALIGN || layout.size() > MAX_CACHED_BLOCK {
            System.dealloc(ptr, layout);
            return;
        }
        let class = size_class(layout.size());
        let room = self.cached_bytes.load(Ordering::Relaxed) + class <= MAX_CACHED_BYTES;
        let mut free = self.free.lock().unwrap_or_else(|e| e.into_inner());
        let stored = if room {
            match class_index(class) {
                Some(index) => {
                    let len = free.small_len[index] as usize;
                    if len < SLOTS_PER_CLASS {
                        free.small[index][len] = ptr;
                        free.small_len[index] = (len + 1) as u8;
                        true
                    } else {
                        false
                    }
                }
                None => {
                    let mut slot = None;
                    for candidate in 0..free.large_used {
                        if free.large_class[candidate] == class {
                            slot = Some(candidate);
                            break;
                        }
                    }
                    let slot = match slot {
                        Some(slot) => slot,
                        None if free.large_used < LARGE_CLASSES => {
                            let slot = free.large_used;
                            free.large_class[slot] = class;
                            free.large_used += 1;
                            slot
                        }
                        None => usize::MAX,
                    };
                    if slot != usize::MAX && (free.large_len[slot] as usize) < SLOTS_PER_CLASS {
                        let len = free.large_len[slot] as usize;
                        free.large[slot][len] = ptr;
                        free.large_len[slot] = (len + 1) as u8;
                        true
                    } else {
                        false
                    }
                }
            }
        } else {
            false
        };
        drop(free);
        if !stored {
            let classed = Layout::from_size_align_unchecked(class, CACHE_ALIGN);
            System.dealloc(ptr, classed);
            return;
        }
        self.cached_bytes.fetch_add(class, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_classes_are_stable_and_cover_the_request() {
        for size in [1usize, 4, 63, 64, 65, 1000, 1 << 20, (1 << 20) + 1, 5 << 20] {
            let class = size_class(size);
            assert!(class >= size, "{size} -> {class}");
            assert_eq!(class, size_class(size), "not idempotent");
            assert_eq!(class % CACHE_ALIGN, 0);
        }
        assert_eq!(size_class(0), CACHE_ALIGN);
    }

    #[test]
    fn a_reused_block_keeps_its_class() {
        let allocator = CachingAllocator::new();
        unsafe {
            let layout = Layout::from_size_align(4096, CACHE_ALIGN).unwrap();
            let first = allocator.alloc(layout);
            assert!(!first.is_null());
            allocator.dealloc(first, layout);
            let second = allocator.alloc(layout);
            assert_eq!(first, second, "the block should come back from the cache");
            assert_eq!(allocator.stats(), (1, 1));
            allocator.dealloc(second, layout);
        }
    }

    #[test]
    fn oversized_alignments_bypass_the_cache() {
        let allocator = CachingAllocator::new();
        unsafe {
            let layout = Layout::from_size_align(4096, 256).unwrap();
            let pointer = allocator.alloc(layout);
            assert!(!pointer.is_null());
            allocator.dealloc(pointer, layout);
            assert_eq!(allocator.stats(), (0, 0), "256-byte alignment is not cached");
        }
    }
}
