//! Where does the GEMM plateau come from?
//!
//! The square-tile GEMM sustains ~3.3 TFLOP/s on this card where the
//! register-only FMA probe reaches 5.8, uniformly across shapes, and no tile
//! arithmetic explains it. [`Kernels::plateau_probe`] times the same kernel
//! with the global staging loads replaced by constants and with the barriers
//! removed, so the gap has a number attached to each part instead of a theory.
//!
//! `#[ignore]`d: it is a measurement, not a gate.
//!
//! ```text
//! cargo test --release -p demucs-core --test gpu_gemm_plateau -- --ignored --nocapture
//! ```

use demucs_core::gpu::kernels::Kernels;
use demucs_core::gpu::Gpu;

fn gpu_or_skip() -> Option<Gpu> {
    match Gpu::new() {
        Ok(gpu) => {
            println!("adapter: {}", gpu.info.describe());
            Some(gpu)
        }
        Err(e) => {
            eprintln!("skipping: no usable wgpu adapter ({e})");
            None
        }
    }
}

#[test]
#[ignore = "a measurement, not a gate; run with --ignored --nocapture"]
fn the_linear_gemm_plateau_is_split_between_its_parts() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let kernels = Kernels::new(&gpu).unwrap();

    // The three Linear shapes the cross-transformer calls: the projections
    // (25 calls, K=512) and the feed-forward's first projection (5 calls).
    for (m, n, k) in [(2688, 512, 512), (1408, 512, 512), (2688, 2048, 512)] {
        kernels.plateau_probe(&gpu, 20, m, n, k).unwrap();
    }
}

/// The other half of the plateau: how much of it the register tile explains.
///
/// The FMA probe reaches 5.8 TFLOP/s while the GEMM sits at 3.3, and the
/// mutants rule out global traffic and barriers; what is left is latency, and
/// the tile sets how many warps are available to hide it.
#[test]
#[ignore = "a measurement, not a gate; run with --ignored --nocapture"]
fn the_tile_sets_how_much_of_the_plateau_is_occupancy() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let kernels = Kernels::new(&gpu).unwrap();

    for (m, n, k) in [(2688, 512, 512), (2688, 2048, 512), (2688, 512, 2048)] {
        kernels.tile_probe(&gpu, 20, m, n, k).unwrap();
    }
}
