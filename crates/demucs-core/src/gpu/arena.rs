//! Device-resident execution: tensors, a recycling arena, and command recording.
//!
//! The reason this exists is a measurement: host↔device transfer on this machine
//! runs at ~520 MB/s, so a single 1.18 GB activation round trip costs 2.2 s —
//! more than the entire model forward on GPU. Any architecture that hands
//! intermediates back to the host cannot be fast no matter how good the kernels
//! are, and no amount of kernel tuning fixes it. So the rules are:
//!
//! * every intermediate lives in device memory for its whole lifetime,
//! * all dispatches of a forward pass go into **one** command encoder and are
//!   submitted once,
//! * the host sees only the audio at the front and the audio at the back.
//!
//! Buffers come from [`Arena`], which allocates one per tensor and **recycles**
//! them: a tensor references a [`Slot`], and when the last reference to a slot
//! dies the buffer moves to the arena's free lists instead of being destroyed.
//! Recycling is what keeps a chunk's footprint near its live set — without it a
//! 7.8 s segment held 7.0 GiB / 4903 buffers resident, because every
//! intermediate survived to the end of the chunk. The single queue is what makes
//! recycling sound: a recycled buffer is handed out only after every dispatch
//! that read the old contents was already recorded, and dispatches execute in
//! the order they were recorded.
//!
//! Two kinds of buffer are never recycled. A buffer the **host** wrote into
//! (`Arena::upload`, `Arena::clear`) is out, because `Queue::write_buffer` is
//! ordered against *submission*, not against individual dispatches, so a later
//! write in the same pass would clobber data an already-queued dispatch still
//! has to read. And an allocation below [`POOL_MIN_BYTES`] is out, because the
//! per-dispatch params blocks are 16 bytes and a pile of them costs less than
//! the bookkeeping would.

use crate::error::{Error, Result};
use crate::gpu::Gpu;
use std::cell::{Cell, RefCell};
use std::ops::Deref;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};

/// A contiguous device buffer holding `shape` elements of `f32`, row-major.
#[derive(Clone, Debug)]
pub struct DevTensor {
    /// Refcounted, so the buffer returns to its pool when the last *view* of it
    /// — a `with_shape` or `slice` counts as a view — goes away, not when the
    /// allocation that created it does.
    pub buffer: Arc<Slot>,
    pub offset: u64,
    pub shape: Vec<usize>,
}

impl DevTensor {
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn rows(&self) -> usize {
        if self.shape.len() <= 1 {
            1
        } else {
            self.shape[0]
        }
    }

    /// Trailing dimension, i.e. the one a row-wise kernel walks.
    pub fn cols(&self) -> usize {
        self.shape.last().copied().unwrap_or(1)
    }

    pub fn with_shape(&self, shape: Vec<usize>) -> Result<Self> {
        let len: usize = shape.iter().product();
        if len != self.len() {
            return Err(Error::Shape(format!(
                "reshape from {:?} to {shape:?} is not length preserving",
                self.shape
            )));
        }
        Ok(Self {
            buffer: self.buffer.clone(),
            offset: self.offset,
            shape,
        })
    }

    /// A view of a byte range of this tensor, reusing the same buffer.
    pub fn slice(&self, offset_elements: usize, shape: Vec<usize>) -> Result<Self> {
        let len: usize = shape.iter().product();
        if offset_elements + len > self.len() {
            return Err(Error::Shape(format!(
                "slice [{offset_elements}, {}) runs past a tensor of {}",
                offset_elements + len,
                self.len()
            )));
        }
        let offset = self.offset + (offset_elements * 4) as u64;
        // The device asks for a 32-byte storage binding offset, so a slab whose
        // base sits mid-alignment cannot be bound: better to say which tensor here
        // than to let the driver reject 900 bind groups later.
        if offset % BINDING_OFFSET_ALIGNMENT != 0 {
            return Err(Error::Shape(format!(
                "slice at element {offset_elements} of a tensor at byte {} lands on byte \
                 {offset}, which is not a multiple of the {BINDING_OFFSET_ALIGNMENT}-byte \
                 binding offset alignment (shape {shape:?} of parent {:?} with {} elements)",
                self.offset, self.shape, self.len()
            )));
        }
        Ok(Self {
            buffer: self.buffer.clone(),
            offset,
            shape,
        })
    }

    /// Declares that the host wrote into this tensor's buffer.
    ///
    /// `Queue::write_buffer` is ordered against the *next submission*, not
    /// against individual dispatches, so once a host write is in flight for a
    /// buffer, that buffer must not be handed to another tensor before the
    /// submission that consumes the written data has been submitted — the
    /// second write would overwrite the first one's data. Marked slots are
    /// therefore dropped rather than recycled.
    pub fn mark_host_written(&self) {
        self.buffer
            .host_written
            .store(true, Ordering::Release);
    }
}

/// A pooled device buffer: the allocation, its byte size, and whether the host
/// wrote into it.
///
/// `Drop` is the whole point of the type — it returns the buffer to the free
/// lists of the [`Pool`] that produced it, unless the host wrote to it or it
/// is too small to be worth a list, in which case `wgpu` reclaims the buffer
/// together with the `Arc`.
pub struct Slot {
    /// `Option` so `Drop` can hand the buffer to the pool without moving out
    /// of a borrowed field; it is only ever taken by `Drop` itself.
    buffer: Option<wgpu::Buffer>,
    /// The buffer's byte size, which is also its size class in the pool.
    size: u64,
    host_written: AtomicBool,
    /// The pool this buffer was allocated from. Always set for arena
    /// allocations — small ones borrow it for the live-set accounting and
    /// never re-enter the free lists — and `None` only for the detached
    /// description-only slots the geometry tests build.
    pool: Option<Weak<Pool>>,
}

impl Slot {
    /// A buffer allocated by an [`Arena`], whether or not it will be recycled.
    fn pooled(buffer: wgpu::Buffer, size: u64, pool: &Arc<Pool>) -> Arc<Self> {
        Arc::new(Self {
            buffer: Some(buffer),
            size,
            host_written: AtomicBool::new(false),
            pool: Some(Arc::downgrade(pool)),
        })
    }

    /// A buffer that belongs to no pool: dropped, never recycled, and never
    /// accounted. Only for building a `DevTensor` to describe in tests.
    pub fn detached(buffer: wgpu::Buffer, size: u64) -> Arc<Self> {
        Arc::new(Self {
            buffer: Some(buffer),
            size,
            host_written: AtomicBool::new(false),
            pool: None,
        })
    }
}

impl Deref for Slot {
    type Target = wgpu::Buffer;

    fn deref(&self) -> &Self::Target {
        self.buffer
            .as_ref()
            .expect("a Slot's buffer is only taken by its own Drop")
    }
}

impl std::fmt::Debug for Slot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Slot")
            .field("size", &self.size)
            .field("host_written", &self.host_written.load(Ordering::Relaxed))
            .finish()
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        let Some(pool) = self.pool.as_ref().and_then(Weak::upgrade) else {
            return;
        };
        // The live set shrinks whichever way the buffer goes afterwards.
        pool.note_release(self.size);
        // A host-written buffer must not be recycled: `write_buffer` is
        // ordered against submission, so the next write could clobber data an
        // already-queued dispatch still has to read. Neither is one below the
        // pooling minimum worth a free list.
        if self.host_written.load(Ordering::Acquire) || self.size < POOL_MIN_BYTES {
            return;
        }
        if let Some(buffer) = self.buffer.take() {
            pool.give_back(self.size, buffer);
        }
    }
}

/// The arena's free lists and its accounting.
///
/// It lives behind an `Arc` so a `Slot`'s `Drop` can return a buffer long after
/// the borrow of the arena that allocated it is gone.
struct Pool {
    /// One free list per size class: index is `class_of(size)` and each entry
    /// remembers the buffer's exact size. Bucketed, not a linear scan, because
    /// the host-side allocator learned that the hard way: a linear free list
    /// turned a 231 MB scores allocation from 10.2 ms into 45.7 ms.
    free: RefCell<Vec<Vec<(u64, wgpu::Buffer)>>>,
    /// Bytes currently checked out by live tensors, their count, and the
    /// high-water mark since the last reset. This is the resident set a chunk
    /// actually costs — what the old cumulative counter could not show.
    live_bytes: Cell<u64>,
    live_buffers: Cell<usize>,
    peak_live_bytes: Cell<u64>,
    /// Bytes idle in the free lists. Bounded by `cap` so a long session with
    /// drifting shapes cannot accumulate without limit.
    pooled_bytes: Cell<u64>,
    cap: u64,
    /// Per-chunk counters, cleared by `Arena::reset`: how many buffers the
    /// chunk asked the driver for, how many bytes those cost, and how many
    /// allocations the free lists served instead.
    created: Cell<usize>,
    created_bytes: Cell<u64>,
    reused: Cell<usize>,
}

/// Allocations below this size are not pooled: the per-dispatch params blocks
/// are 16 bytes and there are thousands of them per chunk, so the bookkeeping
/// would cost more than the buffers.
const POOL_MIN_BYTES: u64 = 256 << 10;

/// Size-class grain, 1 MiB. The model's shapes repeat exactly from chunk to
/// chunk, so the common case is an exact-class hit; the grain only has to
/// absorb the `pad_ceil` rounding that makes two versions of the same tensor
/// differ by a few hundred bytes.
const CLASS_GRAIN_BYTES: u64 = 1 << 20;

/// A free buffer may serve a request at most this many times smaller than
/// itself. The old 2-class window was only 2 MiB of slack, so a 297 MiB im2col
/// scratch sitting idle during the transformer could not cover a 220 MiB
/// attention-score matrix — Task Manager then showed both, and they fought the
/// pool cap. 4× covers that pair (and the smaller cross-attention scores)
/// without letting a few-megabyte tensor strand the scratch.
const REUSE_SIZE_RATIO: u64 = 4;

fn class_of(size: u64) -> usize {
    size.div_ceil(CLASS_GRAIN_BYTES) as usize
}

const MIB: u64 = 1024 * 1024;

impl Pool {
    fn new(cap: u64) -> Self {
        Self {
            free: RefCell::new(Vec::new()),
            live_bytes: Cell::new(0),
            live_buffers: Cell::new(0),
            peak_live_bytes: Cell::new(0),
            pooled_bytes: Cell::new(0),
            cap,
            created: Cell::new(0),
            created_bytes: Cell::new(0),
            reused: Cell::new(0),
        }
    }

    /// Takes a buffer of at least `size` bytes out of the free lists, or
    /// `None` if nothing within [`REUSE_SIZE_RATIO`] of the request is free.
    /// Returns the buffer with its real size, which may exceed the request.
    ///
    /// Best-fit among buffers in `[size, size * 4]`: the class is only a
    /// coarse filter, not a guarantee. A 512 KiB+1 buffer and a 1 MiB one land
    /// in the same class, so a hit still has to be checked against the exact
    /// size on record. Skipping the check would hand out a buffer too small
    /// for its bindings, which wgpu rejects at submit time as an invalid bind
    /// group — with no hint of which allocation was at fault.
    fn take(&self, size: u64) -> Option<(u64, wgpu::Buffer)> {
        let start = class_of(size);
        let max_size = size.saturating_mul(REUSE_SIZE_RATIO);
        let mut free = self.free.borrow_mut();
        let mut best: Option<(usize, usize)> = None;
        let mut best_size = u64::MAX;
        for class in start..free.len() {
            for (index, (buffer_size, _)) in free[class].iter().enumerate() {
                if *buffer_size >= size && *buffer_size <= max_size && *buffer_size < best_size {
                    best = Some((class, index));
                    best_size = *buffer_size;
                }
            }
        }
        let (class, index) = best?;
        let (buffer_size, buffer) = free[class].remove(index);
        self.pooled_bytes.set(self.pooled_bytes.get() - buffer_size);
        self.reused.set(self.reused.get() + 1);
        Some((buffer_size, buffer))
    }

    /// Returns a buffer to its size class. If that would exceed the cap,
    /// smaller idle buffers are dropped first so a large, reused plane (the
    /// im2col scratch, attention scores) is not evicted by a pile of leftover
    /// unique sizes. If the buffer still does not fit — it is larger than the
    /// cap, or the pool is already full of same-or-larger buffers — it is
    /// dropped instead of retained.
    fn give_back(&self, size: u64, buffer: wgpu::Buffer) {
        if self.pooled_bytes.get() + size > self.cap {
            self.evict_smaller_to_fit(size);
        }
        if self.pooled_bytes.get() + size > self.cap {
            return;
        }
        let class = class_of(size);
        let mut free = self.free.borrow_mut();
        if free.len() <= class {
            free.resize(class + 1, Vec::new());
        }
        self.pooled_bytes.set(self.pooled_bytes.get() + size);
        free[class].push((size, buffer));
    }

    /// Drops the smallest idle buffers until `needed` would fit under the cap,
    /// but never a buffer as large as `needed` itself — swapping a 300 MiB
    /// scratch for another 300 MiB one is a wash, and dropping it for a
    /// smaller newcomer is how the cap used to churn 1 GiB every layer.
    fn evict_smaller_to_fit(&self, needed: u64) {
        loop {
            if self.pooled_bytes.get() + needed <= self.cap {
                return;
            }
            let mut free = self.free.borrow_mut();
            let mut smallest: Option<(usize, usize, u64)> = None;
            for (class, bucket) in free.iter().enumerate() {
                for (index, (buffer_size, _)) in bucket.iter().enumerate() {
                    let take = match smallest {
                        None => true,
                        Some((_, _, size)) => *buffer_size < size,
                    };
                    if take {
                        smallest = Some((class, index, *buffer_size));
                    }
                }
            }
            match smallest {
                Some((class, index, buffer_size)) if buffer_size < needed => {
                    drop(free[class].remove(index));
                    self.pooled_bytes
                        .set(self.pooled_bytes.get() - buffer_size);
                }
                _ => return,
            }
        }
    }

    /// Records an allocation entering the live set.
    fn note_alloc(&self, bytes: u64) {
        self.live_bytes.set(self.live_bytes.get() + bytes);
        self.live_buffers.set(self.live_buffers.get() + 1);
        self.peak_live_bytes
            .set(self.peak_live_bytes.get().max(self.live_bytes.get()));
    }

    /// Records a buffer leaving the live set — for the pool or for `wgpu`, the
    /// live set does not care which.
    fn note_release(&self, bytes: u64) {
        self.live_bytes.set(self.live_bytes.get() - bytes);
        self.live_buffers.set(self.live_buffers.get() - 1);
    }
}

/// Hands out [`DevTensor`] buffers, recycling the ones whose last view died.
///
/// Each tensor gets its own buffer rather than a slice of a shared one. That
/// is a correctness requirement, not a preference: wgpu tracks buffer usage
/// per dispatch, and a buffer bound simultaneously as a read-only storage
/// binding and a read-write one is a validation error (`STORAGE_READ_WRITE` is
/// exclusive within a usage scope). With a shared bump arena an op's output
/// routinely lands in the same buffer as one of its inputs, which breaks
/// exactly that rule.
///
/// `budget` is the cap on bytes retained in the free lists, not a bound on
/// concurrent allocation: what bounds a chunk is the live set of its own
/// intermediates, and recycling is what holds that down. A buffer whose use is
/// over — a layer's scores, a permute scratch — is gone the moment its
/// `DevTensor` goes out of scope, which happens between layers, long before
/// the chunk ends. Sound because the queue is single and in order: the buffer
/// can only be handed out again after every dispatch that read the old
/// contents was already recorded.
pub struct Arena {
    pool: Arc<Pool>,
}

/// `[mem]` lines on stderr: how much a chunk really needs, which buffers the
/// pool served, and where the driver said no.
fn mem_trace() -> bool {
    std::env::var("DEMUCS_MEM").map(|v| !v.is_empty()).unwrap_or(false)
}

/// `Limits::default()` asks for this `min_storage_buffer_offset_alignment`, which
/// is what a slice's byte offset has to satisfy to be bindable.
const BINDING_OFFSET_ALIGNMENT: u64 = 32;

impl Arena {
    /// `budget` caps the free lists; see [`Arena`].
    pub fn new(_gpu: &Gpu, budget: u64) -> Self {
        Self {
            pool: Arc::new(Pool::new(budget)),
        }
    }

    /// High-water mark of live bytes since the last reset: the memory a chunk
    /// actually needed at its worst moment.
    pub fn peak(&self) -> u64 {
        self.pool.peak_live_bytes.get()
    }

    /// Bytes currently checked out. Zero between chunks.
    pub fn bytes_resident(&self) -> u64 {
        self.pool.live_bytes.get()
    }

    /// Buffers currently checked out.
    pub fn allocations(&self) -> usize {
        self.pool.live_buffers.get()
    }

    /// Allocates a dedicated buffer, 256-byte aligned so it can back a uniform
    /// binding as well as a storage one.
    pub fn alloc(&mut self, gpu: &Gpu, bytes: u64, label: &str) -> Result<DevTensor> {
        let started = std::time::Instant::now();
        let tensor = self.alloc_inner(gpu, bytes, label);
        host_timing_add("alloc", started.elapsed());
        tensor
    }

    /// Allocates straight from the driver, never from the pool.
    ///
    /// For a buffer the host writes *after* dispatches have been recorded on a
    /// recorder that has not submitted yet: a pooled buffer may still be read by
    /// those pending dispatches (its last Rust reference died, but the queue has
    /// not run it), and a `write_buffer` lands ahead of the whole pending
    /// submission — so recycling it would silently corrupt them. A fresh buffer
    /// has no such reader.
    pub fn alloc_dedicated(&mut self, gpu: &Gpu, bytes: u64, label: &str) -> Result<DevTensor> {
        let started = std::time::Instant::now();
        let aligned = (bytes.max(4) + 255) & !255;
        let tensor = if aligned > gpu.info.max_buffer_bytes {
            Err(Error::Gpu(format!(
                "{label} needs {} MiB, more than the {} MiB a single buffer may hold",
                aligned / MIB,
                gpu.info.max_buffer_bytes / MIB
            )))
        } else {
            let buffer = gpu.scratch(label, aligned).map_err(Error::Gpu)?;
            self.pool.created.set(self.pool.created.get() + 1);
            self.pool
                .created_bytes
                .set(self.pool.created_bytes.get() + aligned);
            let slot = Slot::pooled(buffer, aligned, &self.pool);
            self.pool.note_alloc(slot.size);
            Ok(DevTensor {
                buffer: slot,
                offset: 0,
                shape: vec![(aligned / 4) as usize],
            })
        };
        host_timing_add("alloc", started.elapsed());
        tensor
    }

    /// Uploads host data into a fresh, never-pooled allocation — the form to use
    /// when dispatches are already pending on the chunk's recorder (see
    /// [`Arena::alloc_dedicated`] for why).
    pub fn upload_dedicated(
        &mut self,
        gpu: &Gpu,
        shape: &[usize],
        data: &[f32],
        label: &str,
    ) -> Result<DevTensor> {
        let mut tensor = self.alloc_dedicated(gpu, (data.len() * 4) as u64, label)?;
        tensor.shape = shape.to_vec();
        tensor.mark_host_written();
        gpu.queue
            .write_buffer(&tensor.buffer, tensor.offset, bytemuck::cast_slice(data));
        Ok(tensor)
    }

    fn alloc_inner(&mut self, gpu: &Gpu, bytes: u64, label: &str) -> Result<DevTensor> {
        let aligned = (bytes.max(4) + 255) & !255;
        if aligned > gpu.info.max_buffer_bytes {
            return Err(Error::Gpu(format!(
                "{label} needs {} MiB, more than the {} MiB a single buffer may hold",
                aligned / MIB,
                gpu.info.max_buffer_bytes / MIB
            )));
        }

        // A pooled buffer may be larger than the request; the slot records the
        // buffer's real size so it goes back to the class it came from.
        let (buffer, buffer_bytes, from_pool) = {
            let want_pool = aligned >= POOL_MIN_BYTES;
            if want_pool {
                match self.pool.take(aligned) {
                    Some((size, buffer)) => (buffer, size, true),
                    None => {
                        let buffer = gpu.scratch(label, aligned).map_err(Error::Gpu)?;
                        self.pool.created.set(self.pool.created.get() + 1);
                        self.pool
                            .created_bytes
                            .set(self.pool.created_bytes.get() + aligned);
                        (buffer, aligned, false)
                    }
                }
            } else {
                let buffer = gpu.scratch(label, aligned).map_err(Error::Gpu)?;
                self.pool.created.set(self.pool.created.get() + 1);
                self.pool
                    .created_bytes
                    .set(self.pool.created_bytes.get() + aligned);
                (buffer, aligned, false)
            }
        };

        // Even a sub-minimum allocation carries the pool reference: it never
        // re-enters the free lists, but the live-set accounting still has to
        // see it go.
        let slot = Slot::pooled(buffer, buffer_bytes, &self.pool);
        self.pool.note_alloc(slot.size);
        if mem_trace() && aligned >= 8 << 20 {
            eprintln!(
                "[mem] {label} <-> {:>6} MiB [{}] (live {:>6} MiB, {} buffers)",
                aligned / MIB,
                if from_pool { "pool" } else { "new " },
                self.pool.live_bytes.get() / MIB,
                self.pool.live_buffers.get()
            );
        }
        Ok(DevTensor {
            buffer: slot,
            offset: 0,
            shape: vec![(bytes / 4) as usize],
        })
    }

    /// Allocates and shapes a tensor in one step.
    pub fn tensor(&mut self, gpu: &Gpu, shape: &[usize], label: &str) -> Result<DevTensor> {
        let elements: usize = shape.iter().product();
        let mut tensor = self.alloc(gpu, (elements * 4) as u64, label)?;
        tensor.shape = shape.to_vec();
        Ok(tensor)
    }

    /// Uploads host data into a fresh allocation.
    pub fn upload(&mut self, gpu: &Gpu, shape: &[usize], data: &[f32], label: &str) -> Result<DevTensor> {
        self.upload_bytes(gpu, shape, bytemuck::cast_slice(data), label)
    }

    /// Uploads raw bytes into a fresh allocation of `shape` elements.
    ///
    /// The result is [marked host-written](DevTensor::mark_host_written):
    /// `write_buffer` is ordered against the next submission, so the buffer
    /// must stay put until the submission that reads it has gone out.
    pub fn upload_bytes(
        &mut self,
        gpu: &Gpu,
        shape: &[usize],
        data: &[u8],
        label: &str,
    ) -> Result<DevTensor> {
        let tensor = self.tensor(gpu, shape, label)?;
        if data.len() > tensor.len() * 4 {
            return Err(Error::Shape(format!(
                "{label}: {} bytes do not fit {shape:?}",
                data.len()
            )));
        }
        tensor.mark_host_written();
        let started = std::time::Instant::now();
        gpu.queue.write_buffer(&tensor.buffer, tensor.offset, data);
        host_timing_add("stage_write", started.elapsed());
        Ok(tensor)
    }

    /// Zeroes a tensor.
    ///
    /// `write_buffer` is ordered against everything submitted afterwards, so
    /// the zeros land before any dispatch recorded after this call — no kernel
    /// needed. The tensor is [marked
    /// host-written](DevTensor::mark_host_written) for the ordering reason
    /// above: a recycled buffer must not be host-written mid-pass.
    pub fn clear(&self, gpu: &Gpu, tensor: &DevTensor) {
        let zeros = vec![0u8; (tensor.len() * 4) as usize];
        tensor.mark_host_written();
        gpu.queue
            .write_buffer(&tensor.buffer, tensor.offset, &zeros);
    }

    /// Starts a new round: reports what the last chunk cost and re-arms the
    /// per-chunk counters. The free lists survive — chunk two's big allocations
    /// hit the pool — but a tensor still checked out does not, and if one is
    /// alive here something held it across the chunk boundary, which the
    /// report says out loud.
    pub fn reset(&mut self) {
        if mem_trace() {
            eprintln!(
                "[mem] ---- chunk: created {} buffers ({} MiB), reused {}, peak live {} MiB, {} buffers still live ({} MiB)",
                self.pool.created.get(),
                self.pool.created_bytes.get() / MIB,
                self.pool.reused.get(),
                self.pool.peak_live_bytes.get() / MIB,
                self.pool.live_buffers.get(),
                self.pool.live_bytes.get() / MIB,
            );
            eprintln!(
                "[mem] ---- pool: {} buffers / {} MiB retained, cap {} MiB",
                self.pool.free.borrow().iter().map(Vec::len).sum::<usize>(),
                self.pool.pooled_bytes.get() / MIB,
                self.pool.cap / MIB,
            );
            if self.pool.live_buffers.get() > 0 {
                eprintln!(
                    "[mem] ---- WARNING: {} buffers still live at reset; something outlived the chunk",
                    self.pool.live_buffers.get()
                );
            }
        }
        self.pool.peak_live_bytes.set(0);
        self.pool.created.set(0);
        self.pool.created_bytes.set(0);
        self.pool.reused.set(0);
    }
}

/// Timestamp capacity. A full model forward is ~2600 dispatches per chunk; one
/// query per dispatch (pass begin, differenced against the next dispatch's
/// begin) covers 4095 of them. 4096 is also the adapter-side cap on a single
/// QuerySet.
const TIMING_CAPACITY: u32 = 4096;

/// One timed dispatch: what ran and where its query sits.
///
/// The interval between two consecutive dispatches' begin timestamps is the
/// first one's kernel time *plus* the pass-boundary gap in front of the next —
/// which is exactly the dispatch overhead this diagnostic is meant to expose.
struct TimedDispatch {
    label: String,
    grid: (u32, u32, u32),
    /// Query index of the pass-begin timestamp.
    index: u32,
}

/// Optional per-dispatch GPU timing state behind [`Recorder`].
///
/// Each dispatch is already its own compute pass, so the pass-level
/// `timestamp_writes` give exactly a before-and-after pair per dispatch — no
/// resubmitting per stage, which is the measurement method this replaces (that
/// one serialises the queue and distorts what it measures).
struct Timing {
    query_set: wgpu::QuerySet,
    /// Resolve target, 8 bytes per query, `QUERY_RESOLVE | COPY_SRC`.
    resolve: wgpu::Buffer,
    /// Map-readable copy of `resolve`.
    readback: wgpu::Buffer,
    dispatches: Vec<TimedDispatch>,
    /// Next free query index, two per dispatch.
    next: u32,
}

impl Timing {
    fn new(gpu: &Gpu) -> Self {
        let buffer = |label: &str, usage: wgpu::BufferUsages| {
            gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (TIMING_CAPACITY as u64) * 8,
                usage,
                mapped_at_creation: false,
            })
        };
        Self {
            query_set: gpu.device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("dispatch-timings"),
                ty: wgpu::QueryType::Timestamp,
                count: TIMING_CAPACITY,
            }),
            resolve: buffer(
                "dispatch-timings.resolve",
                wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            ),
            readback: buffer(
                "dispatch-timings.readback",
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            ),
            dispatches: Vec::new(),
            next: 0,
        }
    }

    /// Resolves, reads back, and prints the aggregate GPU time per kernel plus
    /// the slowest individual dispatches.
    fn report(self, gpu: &Gpu, total_dispatches: usize) -> Result<()> {
        let used = self.next as u64;
        if used == 0 || self.dispatches.is_empty() {
            return Ok(());
        }
        // The readback buffer is `MAP_READ | COPY_DST` and the resolve buffer
        // landed the values there, so this maps directly — no staging hop.
        let bytes = (used * 8) as usize;
        let slice = self.readback.slice(0..bytes as u64);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(format!("poll for dispatch timings: {e}")))?;
        rx.recv()
            .map_err(|_| Error::Gpu("map callback dropped".into()))?
            .map_err(|e| Error::Gpu(format!("map dispatch timings: {e}")))?;
        let data: Vec<u8> = slice.get_mapped_range().map_err(|e| Error::Gpu(format!("mapped range: {e}")))?.to_vec();
        self.readback.unmap();
        let ticks: &[u64] = bytemuck::cast_slice(&data);
        // Device-dependent; on Vulkan this is typically 1 ns per tick.
        let period = gpu.queue.get_timestamp_period() as f64;
        // Interval between consecutive begin timestamps: kernel time plus the
        // pass-boundary gap before the next dispatch. The final dispatch has no
        // successor to difference against and is left out of the per-kernel
        // table; the total below is therefore the sum of full intervals.
        let mut rows: Vec<(String, (u32, u32, u32), f64)> = self
            .dispatches
            .iter()
            .zip(self.dispatches.iter().skip(1))
            .map(|(d, next)| {
                let begin = ticks[d.index as usize];
                let end = ticks[next.index as usize];
                let ms = end.saturating_sub(begin) as f64 * period / 1.0e6;
                (d.label.clone(), d.grid, ms)
            })
            .collect();

        // Aggregate by kernel label, in first-seen order so the output is stable.
        let mut order: Vec<String> = Vec::new();
        let mut agg: std::collections::HashMap<String, (usize, f64, f64)> =
            std::collections::HashMap::new();
        for (label, _, ms) in &rows {
            let entry = agg.entry(label.clone()).or_insert_with(|| {
                order.push(label.clone());
                (0, 0.0, 0.0)
            });
            entry.0 += 1;
            entry.1 += ms;
            entry.2 = entry.2.max(*ms);
        }
        let total: f64 = rows.iter().map(|(_, _, ms)| *ms).sum();
        println!(
            "=== GPU dispatch timings: {} intervals timed ({} dispatches total, last interval ignored), {total:.1} ms on GPU ===",
            rows.len(),
            total_dispatches
        );
        for label in &order {
            let (count, sum, max) = &agg[label];
            println!(
                "  {label:<24} x{count:<5} total {sum:>9.1} ms  avg {:>7.2} ms  max {max:>8.2} ms",
                sum / *count as f64
            );
        }
        rows.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        println!("  slowest dispatches:");
        for (label, grid, ms) in rows.iter().take(10) {
            println!(
                "    {label:<24} grid [{:>6},{:>4},{:>4}] {ms:>8.2} ms",
                grid.0, grid.1, grid.2
            );
        }
        // Full per-dispatch rows for offline analysis of shape-dependent cost.
        if let Some(path) = std::env::var_os("DEMUCS_GPU_DISPATCH_CSV") {
            let mut csv = String::from("label,grid_x,grid_y,grid_z,ms\n");
            for (label, grid, ms) in &rows {
                csv.push_str(&format!(
                    "{label},{},{},{},{ms:.4}\n",
                    grid.0, grid.1, grid.2
                ));
            }
            match std::fs::write(&path, csv) {
                Ok(()) => println!("  wrote per-dispatch rows to {}", path.to_string_lossy()),
                Err(e) => println!("  could not write CSV to {path:?}: {e}"),
            }
        }
        Ok(())
    }
}

/// Records dispatches into a single command encoder.
///
/// One submission per forward pass: per-op submits would serialize on the queue
/// and throw away the overlap between kernels.
pub struct Recorder {
    encoder: wgpu::CommandEncoder,
    dispatches: usize,
    timing: Option<Timing>,
}

impl Recorder {
    pub fn new(gpu: &Gpu) -> Self {
        // Per-dispatch GPU timings on demand: `DEMUCS_GPU_TIMINGS=1` makes every
        // stage report its per-kernel aggregate (and `DEMUCS_GPU_DISPATCH_CSV`
        // writes the full per-dispatch rows). The timestamps cost a query pair
        // per dispatch, so it is off by default. The feature has to be on the
        // device, which `Gpu::with_selector_async` already requested when the
        // adapter offers it.
        let timing = if gpu
            .device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY)
            && std::env::var("DEMUCS_GPU_TIMINGS")
                .map(|v| !v.is_empty())
                .unwrap_or(false)
        {
            Some(Timing::new(gpu))
        } else {
            None
        };
        Self {
            encoder: gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("forward"),
                }),
            dispatches: 0,
            timing,
        }
    }

    /// Records GPU-side timestamps around every dispatch. Read the results with
    /// [`Recorder::submit_timed`]; requires the `TIMESTAMP_QUERY` feature.
    pub fn timed(gpu: &Gpu) -> Self {
        Self {
            encoder: gpu
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("forward-timed"),
                }),
            dispatches: 0,
            timing: Some(Timing::new(gpu)),
        }
    }

    pub fn dispatches(&self) -> usize {
        self.dispatches
    }

    /// Binds a pipeline and issues `grid` workgroups. Does not wait.
    ///
    /// `label` names the op for the per-dispatch timing report; it costs a
    /// borrow per dispatch and a `String` only when timing.
    pub fn dispatch(
        &mut self,
        label: &str,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        grid: (u32, u32, u32),
    ) {
        // Claim one query before building the pass, so the descriptor and
        // the recorded label agree on the index. Past capacity the dispatch
        // still runs, it is just not timed.
        let begin = self.timing.as_mut().and_then(|t| {
            if t.next >= TIMING_CAPACITY {
                return None;
            }
            let begin = t.next;
            t.next += 1;
            t.dispatches.push(TimedDispatch {
                label: label.to_string(),
                grid,
                index: begin,
            });
            Some(begin)
        });
        let timestamp_writes = begin.map(|index| {
            let timing = self.timing.as_ref().expect("claimed a slot, so timing exists");
            wgpu::ComputePassTimestampWrites {
                query_set: &timing.query_set,
                beginning_of_pass_write_index: Some(index),
                end_of_pass_write_index: None,
            }
        });
        let recorded = std::time::Instant::now();
        let mut pass = self.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(grid.0.max(1), grid.1.max(1), grid.2.max(1));
        drop(pass);
        HOST_TIMING_NS[1].fetch_add(
            recorded.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        HOST_TIMING_OPS[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.dispatches += 1;
    }

    /// Submits everything recorded so far and waits for it.
    ///
    /// The submit runs inside a validation error scope, so a binding mistake
    /// surfaces as an error here instead of as a buffer full of zeros: wgpu
    /// reports validation failures to the uncaptured-error handler and lets the
    /// dispatch do nothing, which is indistinguishable from a correct kernel
    /// writing the wrong answer.
    pub fn submit(self, gpu: &Gpu) -> Result<()> {
        // A timed recorder reports its per-kernel GPU time and then behaves
        // exactly like a plain submit (the resolve rides the same encoder).
        if self.timing.is_some() {
            return self.submit_timed(gpu);
        }
        gpu.flush_uniforms();
        let guard = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let started = std::time::Instant::now();
        gpu.queue.submit([self.encoder.finish()]);
        let poll = gpu.flush();
        let validation = pollster::block_on(guard.pop());
        HOST_TIMING_NS[2].fetch_add(
            started.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        HOST_TIMING_OPS[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        gpu.reset_uniforms();
        if let Some(error) = validation {
            return Err(Error::Gpu(format!("compute pass failed validation: {error}")));
        }
        poll
    }

    /// Submits like [`Recorder::submit`], then resolves the timing queries and
    /// prints the per-kernel GPU-time aggregate. Only meaningful on a recorder
    /// built with [`Recorder::timed`]; the resolve rides the same encoder, so
    /// the measured dispatches are exactly what a normal pass would run.
    pub fn submit_timed(self, gpu: &Gpu) -> Result<()> {
        let Self {
            mut encoder,
            dispatches,
            timing,
        } = self;
        let used = timing.as_ref().map(|t| t.next).unwrap_or(0);
        if used > 0 {
            let t = timing.as_ref().expect("used > 0 implies timing exists");
            encoder.resolve_query_set(&t.query_set, 0..used, &t.resolve, 0);
            encoder.copy_buffer_to_buffer(&t.resolve, 0, &t.readback, 0, (used as u64) * 8);
        }
        gpu.flush_uniforms();
        let guard = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
        gpu.queue.submit([encoder.finish()]);
        let poll = gpu.flush();
        let validation = pollster::block_on(guard.pop());
        gpu.reset_uniforms();
        if let Some(error) = validation {
            return Err(Error::Gpu(format!("compute pass failed validation: {error}")));
        }
        poll?;
        if used > 0 {
            timing
                .expect("used > 0 implies timing exists")
                .report(gpu, dispatches)?;
        }
        Ok(())
    }

    /// Submits without waiting.
    pub fn submit_async(self, gpu: &Gpu) {
        gpu.flush_uniforms();
        gpu.queue.submit([self.encoder.finish()]);
        gpu.reset_uniforms();
    }

    /// Records a copy out of a device tensor into a host-visible staging buffer.
    ///
    /// Riding the same encoder as the dispatches is what makes the copy free to
    /// order: it lands after everything the pass wrote and before anything the
    /// next pass writes, with no second submission and no fence to reason about.
    pub fn copy_to_staging(&mut self, source: &DevTensor, staging: &wgpu::Buffer, bytes: u64) {
        let size = bytes.div_ceil(4) * 4;
        self.encoder
            .copy_buffer_to_buffer(&source.buffer, source.offset, staging, 0, size.max(4));
    }

    /// Submits and hands back the submission index, so a caller can wait for
    /// *this* submission rather than for the queue to drain: with two chunks in
    /// flight that is the difference between overlapping the host's tail work
    /// with the device's next chunk and serialising them.
    ///
    /// Deliberately no flush and no error scope: waiting here would give up the
    /// overlap this exists for, and anything the pass gets wrong surfaces at the
    /// next device poll — which is where the caller waits for this index.
    pub fn submit_indexed(self, gpu: &Gpu) -> wgpu::SubmissionIndex {
        gpu.flush_uniforms();
        let index = gpu.queue.submit([self.encoder.finish()]);
        gpu.reset_uniforms();
        index
    }
}

/// A binding table for one dispatch, so callers do not repeat the builder dance.
/// Host-side cost attribution for the per-chunk gap the profiler cannot see:
/// `bind_group` creation, dispatch recording, queue submit/flush, host→device
/// staging writes and arena allocation. Enabled by `DEMUCS_STAGE_TIMING=1` (read
/// at [`host_timing_report`] time); the counters are plain atomics, so the
/// instrumented path costs one relaxed add per op.
pub static HOST_TIMING_NS: [std::sync::atomic::AtomicU64; 5] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];
pub static HOST_TIMING_OPS: [std::sync::atomic::AtomicU64; 5] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// Names in the same order as the [`HOST_TIMING_NS`] slots.
pub const HOST_TIMING_LABELS: [&str; 5] = ["bind_group", "dispatch", "submit", "stage_write", "alloc"];

/// Resets and returns the host-side counters: `(labels, ms, ops)` per slot.
pub fn host_timing_take() -> ([&'static str; 5], [f64; 5], [u64; 5]) {
    let mut ms = [0.0f64; 5];
    let mut ops = [0u64; 5];
    for slot in 0..5 {
        let ns = HOST_TIMING_NS[slot].swap(0, std::sync::atomic::Ordering::Relaxed);
        let n = HOST_TIMING_OPS[slot].swap(0, std::sync::atomic::Ordering::Relaxed);
        ms[slot] = ns as f64 / 1e6;
        ops[slot] = n;
    }
    (HOST_TIMING_LABELS, ms, ops)
}

/// Times one op against a host-side slot by name; a no-op for unknown names.
pub fn host_timing_add(slot: &str, elapsed: std::time::Duration) {
    let index = HOST_TIMING_LABELS
        .iter()
        .position(|label| *label == slot)
        .unwrap_or(usize::MAX);
    if index == usize::MAX {
        return;
    }
    HOST_TIMING_NS[index].fetch_add(elapsed.as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
    HOST_TIMING_OPS[index].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

pub fn bind_group(
    gpu: &Gpu,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    entries: &[(&wgpu::Buffer, u64, u64)],
) -> wgpu::BindGroup {
    let entries: Vec<wgpu::BindGroupEntry> = entries
        .iter()
        .enumerate()
        .map(|(index, (buffer, offset, len))| wgpu::BindGroupEntry {
            binding: index as u32,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer,
                offset: *offset,
                size: std::num::NonZeroU64::new(*len),
            }),
        })
        .collect();
    let started = std::time::Instant::now();
    let group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &entries,
    });
    HOST_TIMING_NS[0].fetch_add(
        started.elapsed().as_nanos() as u64,
        std::sync::atomic::Ordering::Relaxed,
    );
    HOST_TIMING_OPS[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    group
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensor_geometry_is_derived_from_the_last_dimension() {
        let tensor = DevTensor {
            buffer: dummy_slot(),
            offset: 0,
            shape: vec![4, 61, 60, 384],
        };
        assert_eq!(tensor.len(), 4 * 61 * 60 * 384);
        assert_eq!(tensor.rows(), 4);
        assert_eq!(tensor.cols(), 384);
    }

    #[test]
    fn reshape_has_to_preserve_length() {
        let tensor = DevTensor {
            buffer: dummy_slot(),
            offset: 0,
            shape: vec![2, 6],
        };
        assert!(tensor.with_shape(vec![3, 4]).is_ok());
        assert!(tensor.with_shape(vec![5, 5]).is_err());
    }

    #[test]
    fn slices_stay_inside_the_parent() {
        let tensor = DevTensor {
            buffer: dummy_slot(),
            offset: 0,
            shape: vec![4, 8],
        };
        // 32 elements total: a 16-element window starting at 16 exactly fits.
        assert!(tensor.slice(0, vec![2, 8]).is_ok());
        assert!(tensor.slice(16, vec![2, 8]).is_ok());
        assert!(tensor.slice(17, vec![2, 8]).is_err());
        assert!(tensor.slice(0, vec![5, 8]).is_err());
    }

    #[test]
    fn slices_are_byte_offset() {
        let tensor = DevTensor {
            buffer: dummy_slot(),
            offset: 1024,
            shape: vec![4, 8],
        };
        let sliced = tensor.slice(8, vec![8]).unwrap();
        assert_eq!(sliced.offset, 1024 + 32);
    }

    /// A binding offset has to be 32-byte aligned, and only aligned tensors can
    /// ever be bound — so a slice that lands mid-alignment is rejected here.
    #[test]
    fn a_slice_must_keep_the_binding_offset_aligned() {
        let tensor = DevTensor {
            buffer: dummy_slot(),
            offset: 1024,
            shape: vec![4, 8],
        };
        let err = tensor.slice(3, vec![8]).unwrap_err().to_string();
        assert!(err.contains("alignment"), "{err}");
    }

    /// The recycling contract: when the last view of a tensor dies, the very
    /// next same-sized allocation is served from the free list instead of the
    /// driver. This is what keeps a chunk's resident set near its live set
    /// instead of its history.
    #[test]
    fn a_dropped_pooled_tensor_is_handed_out_again() {
        let Ok(gpu) = Gpu::new() else { return };
        let mut arena = Arena::new(&gpu, 64 << 20);
        let first = arena.tensor(&gpu, &[1 << 18], "first").unwrap();
        drop(first);
        // The drop put the buffer in a free list rather than the shredder.
        assert_eq!(arena.pool.pooled_bytes.get(), 1 << 20);
        assert_eq!(arena.pool.live_buffers.get(), 0);
        let second = arena.tensor(&gpu, &[1 << 18], "second").unwrap();
        assert_eq!(arena.pool.reused.get(), 1);
        assert_eq!(arena.pool.created.get(), 1);
        assert_eq!(arena.allocations(), 1);
    }

    /// A slice outlives the tensor it came from and still keeps the buffer
    /// alive: the pool must not hand it out while a view is alive, or the view
    /// would watch another tensor scribble over its data.
    #[test]
    fn a_live_view_keeps_the_buffer_out_of_the_pool() {
        let Ok(gpu) = Gpu::new() else { return };
        let mut arena = Arena::new(&gpu, 64 << 20);
        let parent = arena.tensor(&gpu, &[1 << 18], "parent").unwrap();
        let half = parent.slice(0, vec![1 << 17]).unwrap();
        drop(parent);
        let other = arena.tensor(&gpu, &[1 << 18], "other").unwrap();
        assert!(!Arc::ptr_eq(&half.buffer, &other.buffer));
        assert_eq!(arena.pool.reused.get(), 0);
        assert_eq!(arena.pool.live_buffers.get(), 2);
    }

    /// Host-written buffers are the one thing the pool must never hand out:
    /// `write_buffer` is ordered against submission, so a recycled host-written
    /// buffer would have its later write clobber data an already queued
    /// dispatch still needs.
    #[test]
    fn a_host_written_tensor_is_never_recycled() {
        let Ok(gpu) = Gpu::new() else { return };
        let mut arena = Arena::new(&gpu, 64 << 20);
        let first = arena
            .upload(&gpu, &[1 << 18], &vec![0.0; 1 << 18], "first")
            .unwrap();
        let slot = first.buffer.clone();
        drop(first);
        let second = arena.tensor(&gpu, &[1 << 18], "second").unwrap();
        assert!(!Arc::ptr_eq(&slot, &second.buffer));
        assert_eq!(arena.pool.reused.get(), 0);
    }

    /// The size class is a coarse filter, not a size guarantee: `1 MiB + 4`
    /// bytes and `2 MiB` share a class, and handing the small one out for the
    /// big request produces bindings larger than the buffer — which wgpu
    /// rejects at submit time as an invalid bind group, naming the bind group
    /// and not the allocation. The `conv2d` test hit exactly this.
    #[test]
    fn a_pooled_buffer_that_is_too_small_is_not_handed_out() {
        let Ok(gpu) = Gpu::new() else { return };
        let mut arena = Arena::new(&gpu, 64 << 20);
        let small = arena.tensor(&gpu, &[(1 << 20) + 1], "small").unwrap();
        assert!(small.buffer.size() >= ((1 << 20) + 1) * 4);
        drop(small);
        let big = arena.tensor(&gpu, &[2 << 20], "big").unwrap();
        assert_eq!(
            arena.pool.reused.get(), 0,
            "the small buffer must not serve the big request"
        );
        assert_eq!(arena.pool.created.get(), 2);
        assert!(big.buffer.size() >= (2 << 20) * 4);
    }

    /// Params blocks are 16 bytes apiece and there are thousands of them per
    /// chunk; the bookkeeping must not cost more than the buffers.
    #[test]
    fn an_allocation_below_the_minimum_is_never_pooled() {
        let Ok(gpu) = Gpu::new() else { return };
        let mut arena = Arena::new(&gpu, 64 << 20);
        let first = arena.tensor(&gpu, &[4], "params").unwrap();
        let slot = first.buffer.clone();
        drop(first);
        let second = arena.tensor(&gpu, &[4], "params").unwrap();
        assert!(!Arc::ptr_eq(&slot, &second.buffer));
        assert_eq!(arena.pool.reused.get(), 0);
        assert_eq!(arena.pool.created.get(), 2);
    }

    /// The cap bounds what the free lists retain, so drifting shapes cannot
    /// accumulate without limit.
    #[test]
    fn the_pool_does_not_retain_past_its_cap() {
        let Ok(gpu) = Gpu::new() else { return };
        let mut arena = Arena::new(&gpu, 2 << 20);
        let first = arena.tensor(&gpu, &[1 << 20], "first").unwrap();
        drop(first);
        // 4 MiB would exceed the 2 MiB cap, so the drop shreds it instead.
        assert_eq!(arena.pool.pooled_bytes.get(), 0);
        let second = arena.tensor(&gpu, &[1 << 20], "second").unwrap();
        assert_eq!(arena.pool.created.get(), 2);
        assert_eq!(arena.pool.reused.get(), 0);
        assert_eq!(arena.allocations(), 1);
    }

    /// The accounting the memory report is built on: resident and count track
    /// the live set, not the allocation history.
    #[test]
    fn the_live_set_falls_as_tensors_die() {
        let Ok(gpu) = Gpu::new() else { return };
        let mut arena = Arena::new(&gpu, 64 << 20);
        {
            let _a = arena.tensor(&gpu, &[1 << 18], "a").unwrap();
            let _b = arena.tensor(&gpu, &[1 << 18], "b").unwrap();
            assert_eq!(arena.bytes_resident(), 2 << 20);
            assert_eq!(arena.allocations(), 2);
        }
        assert_eq!(arena.bytes_resident(), 0);
        assert_eq!(arena.allocations(), 0);
        assert_eq!(arena.peak(), 2 << 20);
    }

    /// A plain (never-recycled) slot, for tests that only need a real
    /// `wgpu::Buffer` to describe.
    fn dummy_slot() -> Arc<Slot> {
        Slot::detached(dummy_buffer(), 4096)
    }

    fn dummy_buffer() -> wgpu::Buffer {
        // `DevTensor` is only a description; these tests never touch the device.
        // A real buffer is needed to construct one, so build a 1-byte one.
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()))
            .into_iter()
            .next()
            .expect("test machine has an adapter");
        let (device, _queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("tensor-test"),
            ..Default::default()
        }))
        .expect("device");
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dummy"),
            size: 4096,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    }
}
