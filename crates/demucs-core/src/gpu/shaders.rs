//! WGSL kernels.
//!
//! Shaders are built as Rust strings with the tuning constants interpolated, so
//! the host's padding and dispatch arithmetic cannot drift from what the shader
//! assumes —the single most common way this kind of kernel silently produces
//! wrong results.
//!
//! Two conventions in the GEMM body are load-bearing rather than stylistic:
//!
//! * **Fully unrolled scalar names.** An accumulator or operand vector indexed by
//!   a loop variable is not promoted to registers and lands in per-thread local
//!   memory instead.
//! * **`vec2` shared tiles.** The shared tile is `array<vec2<f32>, ..>` and the
//!   reduction steps by two, so each thread issues 16 `LDS.64` per K-step
//!   instead of 64 `LDS.32`. The row pitch is `BK + 2` (even, so every row is
//!   8-byte aligned) and stays coprime enough with 32 banks that the strided
//!   per-thread ownership does not collide.
//!
//! Numbers in these comments are quoted only where the measurement behind them
//! was trustworthy. An earlier version of the GEMM benchmark timed the dispatch
//! *and* the operand upload together, which made host-to-device bandwidth look
//! like kernel throughput and invalidated several rounds of tuning; see
//! `gpu::gemm::Prepared`.

/// GEMM tile geometry. Also exported so the host pads to exactly these multiples.
///
/// `BK = 32` was measured and rejected: the shared tiles grow to 34 KB, which
/// leaves room for only one workgroup per SM, and throughput drops from 3.27 to
/// 2.78 TFLOP/s. Occupancy beats fewer barriers here.
///
/// The tile is a parameter rather than four constants because the tall variant
/// below is worth a different trade on the model's shapes.
pub const BM: usize = 128;
pub const BN: usize = 128;
pub const BK: usize = 16;

/// A GEMM tile and the thread block that fills it.
///
/// Two shapes are compiled: the square 128x128 one every variant has always
/// used, and a 256x128 one whose purpose is traffic. Measured on this card, the
/// kernel's cost tracks the bytes each K-step stages rather than the FMAs it
/// issues — two shapes with identical FLOPs but 2x the workgroups differ by 5%,
/// while each K-step costs ~6 us per workgroup whatever the tile — so a taller
/// M tile is the lever: B panels are re-read `M / BM` times, and doubling BM
/// halves that. On the feed-forward's shape that is 2.08 GB of traffic down to
/// 1.63 GB.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tile {
    pub bm: usize,
    pub bn: usize,
    pub tx: usize,
    pub ty: usize,
}

/// Row count of the 64x128 convolution tile, exported so the dispatcher's grid
/// and the compiled shader agree.
pub const BM64: usize = 64;
/// Row count of the 96×128 convolution tile.
pub const BM96: usize = 96;

/// Whether the convolutions fold their im2col gather into the GEMM.
/// `DEMUCS_CONV_DIRECT=0` restores the materialised patch matrix, which is how the
/// two were compared.
pub fn conv_direct_for_convs() -> bool {
    // Measured off, on the DTTNet 20 s clip: 15.08 s against 12.25 s for the
    // materialised patch matrix (same output). The gather's per-element arithmetic
    // and bounds checks cost more than the traffic they save — the staging loop is
    // execute-bound, not bandwidth-bound, which is the same conclusion the
    // coalesced-walk experiment reaches from the other side. Kept in the tree
    // because it is the right shape for a machine whose staging is bandwidth-bound,
    // and because the next person will otherwise write it again.
    std::env::var("DEMUCS_CONV_DIRECT").map(|v| v == "1").unwrap_or(false)
}

/// Whether the convolution's coalesced `B` staging walk is used.
/// `DEMUCS_GEMM_B_WALK=scattered` restores the default walk, which is how the two
/// were compared.
pub fn coalesced_b_for_convs() -> bool {
    // Neutral on the convolution shapes (14.87 s against 14.90 s on the DTTNet
    // clip, same output): the 32 lanes' scattered reads all land in the same
    // workgroup's tile, so the L2 line is consumed either way. The default stays
    // with the walk every other GEMM uses.
    std::env::var("DEMUCS_GEMM_B_WALK").map(|v| v == "coalesced").unwrap_or(false)
}

/// Whether a single-batch `transb` GEMM uses the shader with batch bases compiled
/// out. HTDemucs Linear projections are `batches == 1` but historically shared
/// the batched `gemm_transb` (the QK^T path); carrying dead `abase`/`bbase`/
/// `cbase` through the K loop cost 15% on this driver for the plain GEMM
/// (3.27 TFLOP/s without vs 2.76 with). `DEMUCS_GEMM_TRANSB_PLAIN=0` restores
/// the batched shader for A/B. Neutral on HTDemucs Linear shapes.
pub fn transb_plain_for_single_batch() -> bool {
    std::env::var("DEMUCS_GEMM_TRANSB_PLAIN")
        .map(|v| v != "0")
        .unwrap_or(true)
}

/// Whether the split `group_norm` reduces per-slice stats once, then applies.
/// The original apply had every thread of every segment workgroup walk all
/// `segments` partials — O(segments²) on the waveform DConv's 2016-segment
/// slices, ~8 ms × 4. `DEMUCS_GN_COMBINE=0` restores that loop for A/B.
pub fn group_norm_combine_stats() -> bool {
    std::env::var("DEMUCS_GN_COMBINE")
        .map(|v| v != "0")
        .unwrap_or(true)
}

/// Whether the DConv's GLU and the LayerScale that follows it are one dispatch.
///
/// `DEMUCS_FUSE_GLU_SCALE=0` restores the two-step form, which is how the two
/// were compared; a trace that wants the GLU's own output also takes it.
pub fn fuse_glu_scale() -> bool {
    std::env::var("DEMUCS_FUSE_GLU_SCALE")
        .map(|v| v != "0")
        .unwrap_or(true)
}

/// Segment count below which the apply replays the partials itself instead of
/// taking a `group_norm_combine` dispatch first.
///
/// The combine exists because the replay is O(segments²) per pair; most calls
/// in the model have only a handful of segments, where the dispatch costs more
/// than the replay (`group_norm_combine` measured ~50 µs each, 62 of them per
/// chunk). Replay and combine sum the partials left to right in the same
/// order, so the two paths are bit-identical. `DEMUCS_GN_COMBINE_WORK=0`
/// always combines, a huge value never does.
pub fn group_norm_combine_for(pairs: usize, segments: usize) -> bool {
    if !group_norm_combine_stats() {
        return false;
    }
    let limit: usize = std::env::var("DEMUCS_GN_COMBINE_WORK")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4096);
    // The replay costs `pairs * segments` workgroups × `segments` loads each.
    pairs.saturating_mul(segments).saturating_mul(segments) > limit
}

/// Whether the convolution's narrow-M tile is used. `DEMUCS_CONV_TILE=128` restores
/// the square tile, which is how the two were compared.
pub fn bm64_for_convs() -> bool {
    match std::env::var("DEMUCS_CONV_TILE").ok().as_deref() {
        Some("128") => false,
        Some("64") => true,
        Some("96") => true,
        // Measured on the DTTNet 20 s clip: 16.87 s with the square tile against
        // 15.14 s with the narrow one, and the outputs identical to the last
        // digit of the SNR test's 129.65 dB. See the README's conv section.
        _ => true,
    }
}

/// 96-row convolution tile for `64 < oc <= 192` (HTDemucs 96 and 192).
/// `DEMUCS_CONV_TILE=128` or `64` turns it off.
pub fn bm96_for_convs() -> bool {
    match std::env::var("DEMUCS_CONV_TILE").ok().as_deref() {
        Some("128") | Some("64") => false,
        _ => true,
    }
}

impl Default for Tile {
    fn default() -> Self {
        Self {
            bm: BM,
            bn: BN,
            tx: THREADS_X,
            ty: THREADS_Y,
        }
    }
}

impl Tile {
    /// The 64-wide N tile: `n = dim_head` for the AV product, and the frequency
    /// axis' 60 bands for `Q K^T`. Square in M, since the M tile is shared with
    /// the caller's grid.
    pub const fn bn64() -> Self {
        Self {
            bn: 64,
            ..Tile::square()
        }
    }

    /// Both sides 64: for the frequency axis, whose whole attention problem is
    /// `60 x 60` per (frame, head). A 128-row tile covers those 60 rows with
    /// 128 and computes zeros for the rest; at 64 the waste is 7% instead of
    /// 113%.
    pub const fn bm64_bn64() -> Self {
        Self {
            bm: 64,
            bn: 64,
            tx: 16,
            ty: 16,
        }
    }

    /// Half-height: 64 rows, 128 columns, for the convolutions' narrow output
    /// channel counts.
    pub const fn bm64_bn128() -> Self {
        Self {
            bm: 64,
            bn: BN,
            tx: THREADS_X,
            ty: THREADS_Y,
        }
    }

    /// 96×128: HTDemucs `oc = 96` is one M tile (a 128-row tile wastes 25%),
    /// and `oc = 192` is two exact tiles rather than two 128-row tiles padded
    /// to 256.
    pub const fn bm96_bn128() -> Self {
        Self {
            bm: 96,
            bn: BN,
            tx: THREADS_X,
            ty: THREADS_Y,
        }
    }

    /// The tile every variant has always used.
    pub const fn square() -> Self {
        Self {
            bm: BM,
            bn: BN,
            tx: THREADS_X,
            ty: THREADS_Y,
        }
    }
}

// A 256x128 tile with 512 threads was built and measured here: it halves the
// B-panel re-reads (2.08 GB of traffic down to 1.63 GB on the feed-forward's
// shape) and is **9% slower** (17.0 ms against 15.6 ms, same binary, same
// shape). The kernel is not bandwidth-bound at these sizes, and putting twice
// the warps behind one barrier costs more than the traffic it saves. Keep the
// 128x128 tile with 256 threads; the two workgroups per SM it allows hide
// latency better than one workgroup twice the size.
/// Shared-memory row pitch in `f32`, kept even so rows are 8-byte aligned for
/// `vec2` access. The extra slots also skew column reads across banks: a stride
/// coprime with 32 in `vec2` units gives every one of the 16 threads reading a
/// column its own bank. A stride of `BK` alone would not — 16 rows on an
/// 8-word stride land 4 rows apart on the same bank pair.
const PAD: usize = if BK >= 16 { BK + 2 } else { BK + 6 };
/// Row pitch in `vec2` units.
const PW: usize = PAD / 2;

pub const THREADS_X: usize = 16;
pub const THREADS_Y: usize = 16;
/// `vec2` slots per tile row that actually hold data, i.e. one K block's payload.
const ROW_SLOTS: usize = BK / 2;

/// `C[m,n] = A[m,k] @ B[k,n]`, all row-major f32.
///
/// A and B must already be zero-padded to multiples of [`BM`]/[`BK`] and
/// [`BK`]/[`BN`] respectively, which is what lets the inner loop drop every
/// bounds check: padding reads land on zeros and contribute nothing. Only `C` is
/// guarded, in the epilogue.
///
/// `wid.y` drives the M tile, `wid.x` the N tile.
///
/// `batched` selects whether the shader derives a per-workgroup base offset from
/// `wid.z`. Both forms are emitted from this one generator and compute the same
/// thing when `inner_count == 1`; the flag exists because carrying three extra
/// live offsets through the K loop costs measured throughput —3.27 TFLOP/s
/// without them against 2.76 with —and only the attention GEMMs need them.
pub fn gemm_f32(batched: bool) -> String {
    gemm_impl(batched, false)
}

/// The same GEMM with B read as `(n, k)` instead of `(k, n)`.
///
/// Attention needs it: the score matrix is `Q @ K^T`, and in the fused QKV
/// tensor K is stored `[frame][dim_head]` row-major, so its transpose is what a
/// `(k, n)`-strided read cannot express. Only the global addressing of the B tile
/// changes —the shared layout and the reduction are identical, so the two forms
/// are the same arithmetic.
pub fn gemm_f32_transb(batched: bool) -> String {
    gemm_impl(batched, true)
}

/// Batched GEMM with a 64-wide N tile. The attention AV product has
/// `n = dim_head = 64`, half of a 128-wide tile: the wasted half was measured
/// at ~86 ms per forward. The AV product's B operand is a strided view of the
/// fused QKV tensor, so nothing about the weight padding changes.
pub fn gemm_batched_bn64() -> String {
    gemm_impl_with(Tile::bn64(), true, false, GemmEpilogue::None)
}

/// The attention's `P·V` with the softmax folded into the `A` staging.
///
/// `A` is the *raw* score matrix (`Q Kᵀ`, already scaled by `1/sqrt(d_head)` by
/// the `exp`'s own factor), and the staging applies
/// `exp(a * scale - max_row) * inv_row` while it copies, reading the two
/// per-row statistics [`softmax_stats_warp`] wrote. The probabilities are then
/// never materialised: the softmax does one read pass instead of a read pass
/// plus a read-modify-write of the full `(heads·tokens, tokens)` matrix, which
/// on the 7.8 s segment is 231 MB per call.
///
/// Bit-identical to the two-kernel form: the same `exp`, the same subtraction,
/// and the product is rounded through the same `f32` shared tile.
pub fn gemm_batched_bn64_row_exp() -> String {
    gemm_impl_full(Tile::bn64(), true, false, GemmEpilogue::None, false, false, true)
}

/// The transposed form with a 64-wide N tile, for `Q K^T` when the sequence is
/// short.
///
/// The frequency axis runs `n = bands = 60`, so a 128-wide tile computes zeros
/// for more than half of every workgroup and a 128-wide *M* tile -- which this
/// shares with the caller's grid -- covers 60 rows with 128. Dropping N to 64
/// takes the waste from 4.5x to 1.9x; the M half needs the grid to agree and is
/// left for a variant that carries its own tile size.
pub fn gemm_transb_bn64() -> String {
    gemm_impl_with(Tile::bn64(), true, true, GemmEpilogue::None)
}

/// The transposed and plain batched forms at 64x64: the frequency axis'
/// attention, where both operands are 60 wide.
pub fn gemm_transb_bm64() -> String {
    gemm_impl_with(Tile::bm64_bn64(), true, true, GemmEpilogue::None)
}

pub fn gemm_batched_bm64() -> String {
    gemm_impl_with(Tile::bm64_bn64(), true, false, GemmEpilogue::None)
}

/// A 64-row, 128-column tile, unbatched: the shape a convolution wants.
///
/// A conv's `M` is its output channel count — 32, 64 or 96 in DTTNet — so a
/// 128-row tile computes zeros for most of its rows: 4x the FMAs at 32 channels,
/// 2x at 64. The `BN = 128` half is deliberate: the tile's other cost is the `B`
/// panel re-read once per M tile, and at these shapes a 64-row tile still leaves
/// *one* M tile for m <= 64, so narrowing M costs no traffic at all. That is what
/// distinguishes this from the 64x64 and 256x128 experiments recorded in the
/// negative-results table, both of which changed the N side of a much larger
/// problem.
pub fn gemm_bm64() -> String {
    gemm_impl_with(Tile::bm64_bn128(), false, false, GemmEpilogue::None)
}

/// A convolution whose patch matrix is never materialised.
///
/// `conv2d_into` builds the im2col matrix and runs the GEMM over it, which costs
/// two full passes over `k / in_channels` times the input — nine times, for the
/// 3x3 stacks that dominate DTTNet. Measured, the convolution pair runs at
/// ~80 GB/s against this card's ~190, and that traffic is the cost: the patch
/// matrix is 604 MB for the first block alone, written once and read once per
/// layer.
///
/// This is the same GEMM with the gather folded into its `B` staging, which is
/// what cuDNN's implicit GEMM does and why it is 3x faster here. The trick is
/// that a tile's `B` row for one tap is a *contiguous run of the input*: for a
/// fixed `(ic, ky, kx)`, consecutive output columns read consecutive input
/// columns, so the staging is a coalesced read with a per-row offset — no gather,
/// no patch matrix, and the input is read once per `(k-block, tap)` instead of
/// being written out and read back.
///
/// The one structural requirement is that a tile's positions lie in a single
/// output row — `out_w` a multiple of `BN`, not merely `>= BN`: a tile starting
/// at `n0 = 128` with `out_w = 200` spans row 0's tail and row 1's head, and
/// every position past the boundary gathers from the wrong row's input.
/// `conv2d_into` falls back to the im2col path when a layer's width straddles.
pub fn gemm_conv_direct() -> String {
    // Coalesced: the gather's reads are contiguous runs of the input, so the walk
    // that puts consecutive output columns in consecutive lanes turns a warp's
    // request into one 128-byte line.
    gemm_impl_full(Tile::bm64_bn128(), false, false, GemmEpilogue::None, true, true, false)
}

/// `gemm_conv_direct` with the batch base terms emitted, for a forward that
/// carries several segments through one pass.
///
/// The gather then starts at `bbase`, each batch's own block of the input, so a
/// batched conv is as free of a patch matrix as the single-segment one. The
/// terms measurably cost throughput on this driver, which is why the plain form
/// stays the batch-1 path rather than this being the only variant.
pub fn gemm_conv_direct_batched() -> String {
    gemm_impl_full(Tile::bm64_bn128(), true, false, GemmEpilogue::None, true, true, false)
}

/// The convolution tile with a **coalesced `B` staging walk**.
///
/// The GEMM's two staging walks are not symmetric, and which one wins depends on
/// the operand's row pitch. The default walk gives each lane a slot of the tile
/// and walks `k` fastest within it, so a warp's 32 lanes read 32 *different rows*
/// of `B`; that is faster on the roformer's operands, where `ldb` is 1.5 KB and
/// the panel is a handful of pages wide (measured: 20.3 ms against 15.6 on
/// `ff_in`, which is why the default is what it is).
///
/// A convolution's `B` is the patch matrix, whose rows are `positions` apart —
/// 128 KB at the bottleneck and **2 MB** in the first block at 256 frames. There,
/// 32 lanes reading 32 rows means 32 cache lines per warp instruction and a ~4x
/// sector amplification on the largest operand in the model. This variant walks
/// `n` across the lanes instead, so one warp instruction is one 128-byte run, and
/// only `conv2d_into` uses it.
pub fn gemm_bm64_coalesced() -> String {
    gemm_impl_full(Tile::bm64_bn128(), false, false, GemmEpilogue::None, true, false, false)
}

/// The batched forms of the convolution tile, 64x128 with the same B-walk
/// choices as `gemm_bm64` / `gemm_bm64_coalesced`.
///
/// A convolution's per-batch blocks have to be bound whole for the *output*
/// offset to stay 32-byte aligned: the stride `out_channels * positions` is
/// 6 * 33 = 198 at a real segment's frame count, and 198 is not a multiple of
/// 8, so a per-batch bind lands mid-alignment and the driver rejects the pass.
/// The batch stride therefore rides the job (`bsb_outer` / `csc_outer`) and the
/// batch base is computed in the shader, which is what these two are.
///
/// Distinct from `gemm_batched_bm64`, which is the attention's 64x64 tile.
pub fn gemm_batched_bm64_bn128() -> String {
    gemm_impl_full(Tile::bm64_bn128(), true, false, GemmEpilogue::None, false, false, false)
}

/// Batched 96×128 convolution tile. See [`Tile::bm96_bn128`].
pub fn gemm_batched_bm96_bn128() -> String {
    gemm_impl_full(Tile::bm96_bn128(), true, false, GemmEpilogue::None, false, false, false)
}

/// The coalesced-walk form of [`gemm_batched_bm64_bn128`], for
/// `DEMUCS_GEMM_B_WALK=coalesced`.
pub fn gemm_batched_bm64_bn128_coalesced() -> String {
    gemm_impl_full(Tile::bm64_bn128(), true, false, GemmEpilogue::None, true, false, false)
}

/// What the GEMM's epilogue does to each accumulator before it stores it.
///
/// Fusing the bias add and the feed-forward GELU into the store keeps two
/// full passes over the widest activation in the model from ever happening.
/// Both replay the unfused order exactly -- `acc + bias` first, GELU of that
/// sum second -- so the outputs are bit-identical to the separate kernels.
///
/// The residual forms go one step further and carry the transformer block's
/// skip connection: they read the destination element back (`C` is already a
/// `read_write` binding) and add it. That replaces a `normed = proj(...); x +=
/// normed` pair with a single dispatch that never writes the intermediate —
/// and it is bit-identical too, because `acc + bias` is computed and rounded
/// before the residual is added, exactly as the unfused pair rounds it.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GemmEpilogue {
    None,
    /// `C = acc + Bias[col]`, the `nn.Linear` bias.
    Bias,
    /// `C = gelu(acc + Bias[col])`, the feed-forward's first layer.
    GeluBias,
    /// `C = acc + C`, the block's skip connection.
    Residual,
    /// `C = acc + Bias[col] + C`.
    BiasResidual,
    /// `C = acc + Bias[row]`: the convolution's bias, which runs along the
    /// output channels — this GEMM's M axis, not the Linear's N.
    RowBias,
}

impl GemmEpilogue {
    fn has_bias(self) -> bool {
        matches!(
            self,
            GemmEpilogue::Bias
                | GemmEpilogue::GeluBias
                | GemmEpilogue::BiasResidual
                | GemmEpilogue::RowBias
        )
    }

    fn has_residual(self) -> bool {
        matches!(self, GemmEpilogue::Residual | GemmEpilogue::BiasResidual)
    }
}

pub fn gemm_bias() -> String {
    gemm_impl_with(Tile::default(), true, false, GemmEpilogue::Bias)
}

/// The batched form with the row bias: what a convolution's GEMM can fold in
/// instead of running `add_row_bias_in_place` over its own output.
pub fn gemm_batched_row_bias(tile: Tile, coalesced_b: bool) -> String {
    gemm_impl_full(
        tile,
        true,
        false,
        GemmEpilogue::RowBias,
        coalesced_b,
        false,
        false,
    )
}

/// Whether a convolution's bias rides its GEMM's epilogue. `DEMUCS_CONV_BIAS_FUSE=0`
/// takes the extra pass instead.
pub fn conv_bias_fuse() -> bool {
    std::env::var("DEMUCS_CONV_BIAS_FUSE")
        .map(|v| v != "0")
        .unwrap_or(true)
}

/// Single-batch `transb` with the Linear bias folded into the store.
/// HTDemucs projections are `transb` + a separate affine; fusing the bias
/// drops that extra pass. `DEMUCS_LINEAR_BIAS_FUSE=0` restores the split.
pub fn gemm_transb_plain_bias() -> String {
    gemm_impl_with(Tile::square(), false, true, GemmEpilogue::Bias)
}

pub fn linear_bias_fuse() -> bool {
    std::env::var("DEMUCS_LINEAR_BIAS_FUSE")
        .map(|v| v != "0")
        .unwrap_or(true)
}

pub fn gemm_gelu_bias() -> String {
    gemm_impl_with(Tile::default(), true, false, GemmEpilogue::GeluBias)
}

pub fn gemm_residual() -> String {
    gemm_impl_with(Tile::default(), true, false, GemmEpilogue::Residual)
}

pub fn gemm_bias_residual() -> String {
    gemm_impl_with(Tile::square(), true, false, GemmEpilogue::BiasResidual)
}

fn gemm_impl(batched: bool, transb: bool) -> String {
    gemm_impl_with(Tile::default(), batched, transb, GemmEpilogue::None)
}

fn gemm_impl_with(tile: Tile, batched: bool, transb: bool, ep: GemmEpilogue) -> String {
    gemm_impl_full(tile, batched, transb, ep, false, false, false)
}

/// One GEMM in an arbitrary (tile, transb, epilogue, slot width) form, for the
/// probes.
///
/// The tree compiles a fixed set of these; the plateau work needs to time tiles
/// that no path uses yet without wiring a pipeline for each into `Kernels`.
pub fn gemm_variant(tile: Tile, transb: bool, ep: GemmEpilogue, wide: bool) -> String {
    gemm_impl_width(tile, false, transb, ep, false, false, false, wide)
}

fn gemm_impl_full(
    tile: Tile,
    batched: bool,
    transb: bool,
    ep: GemmEpilogue,
    coalesced_b: bool,
    conv_direct: bool,
    a_row_exp: bool,
) -> String {
    gemm_impl_width(tile, batched, transb, ep, coalesced_b, conv_direct, a_row_exp, false)
}

/// The same generator with the shared-memory slot width chosen: `wide` makes
/// every staged slot four K elements (`vec4<f32>`) instead of two.
///
/// Why that matters, measured on the Linear shapes (`plateau_probe`'s mutants):
/// replacing the *global* staging loads with cache hits buys 8-20%, removing the
/// barriers 5-15%, and making the shared->register reads loop-invariant buys
/// **45-60%** — so the shared reads, not the global traffic, are what the kernel
/// spends its time and issue slots on. `vec2` fragments cost one LDS per 8 FMAs;
/// reading `vec4` costs one per 16, which is the only way to move that number
/// without shrinking the register tile (and the tile probe shows shrinking it
/// under `vec2` loses 10%: the wider slot is what pays for a smaller tile).
///
/// The staging stores widen with it, so the staging instruction count halves
/// too, and the row pitch keeps the same "coprime with 32" trick in the wider
/// unit (`PW` of 9 `vec2` becomes 5 `vec4`).
#[allow(clippy::too_many_arguments)]
fn gemm_impl_width(
    tile: Tile,
    batched: bool,
    transb: bool,
    ep: GemmEpilogue,
    coalesced_b: bool,
    conv_direct: bool,
    a_row_exp: bool,
    wide: bool,
) -> String {
    // Elements per shared slot and the WGSL type that holds them.
    let fw = if wide { 4 } else { 2 };
    let fvt = if wide { "vec4<f32>" } else { "vec2<f32>" };
    let zero = if wide {
        "vec4<f32>(0.0, 0.0, 0.0, 0.0)"
    } else {
        "vec2<f32>(0.0, 0.0)"
    };
    // Row pitch in `f32`, and in slots. Both are kept even in their own unit so
    // rows stay aligned for the slot type, and coprime with 32 slots' worth of
    // banks so the 16 lanes reading a column get their own bank each.
    let pad = if wide { BK + 4 } else { PAD };
    let pw = pad / fw;
    let row_slots = BK / fw;
    assert!(
        !(coalesced_b && transb),
        "the coalesced walk is written for the plain `(k, n)` operand only"
    );
    assert!(
        !(conv_direct && transb),
        "the conv-direct walk is the plain operand's gather"
    );
    assert!(
        !(a_row_exp && (transb || conv_direct || coalesced_b || !batched)),
        "the row-exp A path is the batched plain operand's staging"
    );
    let Tile { bm, bn, tx, ty } = tile;
    let tn = bn / tx;
    let tm = bm / ty;
    // The base terms are emitted only when they can be non-zero. Declaring
    // `let abase = 0u;` and adding it to every address costs 15% of the kernel's
    // throughput on this driver —the compiler does not fold it away —and the
    // plain path is the one every `nn.Linear` in the model goes through.
    let (batch_base, a_pre, b_pre, c_pre) = if batched {
        (
            "    let outer = wid.z / gd.inner_count;\n    \
             let inner = wid.z % gd.inner_count;\n    \
             let abase = outer * gd.asa_outer + inner * gd.asa_inner;\n    \
             let bbase = outer * gd.bsb_outer + inner * gd.bsb_inner;\n    \
             let cbase = outer * gd.csc_outer + inner * gd.csc_inner;\n",
            "abase + ",
            "bbase + ",
            "cbase + ",
        )
    } else {
        ("", "", "", "")
    };

    let mut accumulators = String::new();
    for i in 0..tm {
        for j in 0..tn {
            accumulators.push_str(&format!("    var c{i}_{j}: f32 = 0.0;\n"));
        }
    }

    // When the tile is batched the conv-direct gather starts at `bbase`, the
    // batch's own block of the input, so a batched forward still needs no patch
    // matrix. The base reaches the gather as a parameter because its helper is
    // emitted outside the entry point, where `bbase` lives.
    let conv_base = if batched { "bbase" } else { "0u" };

    // Staging helpers. Each thread copies its share of the A tile and of the B
    // tile; both walks decompose the flat slot index the same way —row on the
    // quotient, k pair on the remainder —and `by`/`bx` are A's names reused for
    // B's (row = the tile's outer axis, column = its k pair).
    //
    // A "coalesced" walk for B was built and measured: consecutive lanes taking
    // consecutive n of one k, which turns each instruction's 32 scattered sectors
    // into one contiguous 128-byte request. It is 30% *slower* on this card
    // (20.3 ms against 15.6 ms on the model's `ff_in` shape, same binary, same
    // shared layout). The scattered form is not a bug to fix: with the kernel
    // latency-bound rather than bandwidth-bound, 16 outstanding sectors per
    // instruction buy more memory-level parallelism than 4 do. Do not "fix" this
    // again without measuring.
    let threads = tx * ty;
    let slots_a = bm * (BK / fw) / threads;
    // Sized to the B tile's own row count rather than A's: a 64-wide N tile needs
    // half the staging, and covering 128 rows of a 64-row tile was pure waste.
    let slots_b = bn * (BK / fw) / threads;
    assert!(slots_a > 0 && slots_b > 0, "a tile must stage at least one slot");

    // One slot constructor from its element expressions, so the wide and narrow
    // forms differ only in how many there are.
    let ctor = |parts: &[String]| -> String {
        format!("{fvt}(\n               {})", parts.join(",\n               "))
    };
    // The `k`-th component of a slot as a WGSL swizzle.
    let comp = |slot: String, c: usize| -> String {
        let swizzle = ["x", "y", "z", "w"][c];
        format!("{slot}.{swizzle}")
    };
    // Each slot element carries its own K guard: the tail block reads past
    // `gd.k` and must stage zeros there.
    let guarded = |parts: &[String], base: &str| -> Vec<String> {
        parts
            .iter()
            .enumerate()
            .map(|(c, part)| {
                // The first element's guard stays bare, so the two-wide source
                // is textually what it always was.
                if c == 0 {
                    format!("select(0.0, {part}, {base} < gd.k)")
                } else {
                    format!("select(0.0, {part}, {base} + {c}u < gd.k)")
                }
            })
            .collect()
    };

    // Global-load double buffering: stage `kb + 1` is fetched into registers
    // while `kb` is being consumed from shared, so the ~600-cycle global latency
    // overlaps the reduction instead of preceding it.
    let mut prefetch_decl = String::new();
    let mut prefetch = String::new();
    let mut store_prefetch = String::new();
    for e in 0..slots_a {
        prefetch_decl.push_str(&format!("    var pfa{e}: {fvt} = {zero};\n"));
    }
    for e in 0..slots_b {
        prefetch_decl.push_str(&format!("    var pfb{e}: {fvt} = {zero};\n"));
    }
    for e in 0..slots_a {
        let flat = format!("(ty * {tx}u + tx + {e}u * {threads}u)");
        // A is row-major in k, so a whole slot is a contiguous run: the wide form
        // is one 16-byte global load per slot instead of two 8-byte ones.
        let reads: Vec<String> = (0..fw)
            .map(|c| {
                format!(
                    "A[@A@(m0 + ar{e}) * gd.lda + k1 + ac{e} * {fw}u + {c}u]"
                )
            })
            .collect();
        prefetch.push_str(&format!(
            "            let ai{e} = {flat};\n\
             \x20           let ar{e} = ai{e} / {row}u;\n\
             \x20           let ac{e} = ai{e} % {row}u;\n\
             \x20           pfa{e} = {ctor};\n",
            row = row_slots,
            ctor = ctor(&reads),
        ));
        // Re-derived rather than carried over: the prefetch above lives inside
        // the same `if` block, but the store runs after the reduction's barrier,
        // outside it. Each element keeps its own guard, so a K tail that runs
        // past the block still stages zeros.
        let parts: Vec<String> = (0..fw)
            .map(|c| {
                let value = comp(format!("pfa{e}"), c);
                if a_row_exp {
                    format!("a_row_scale({value}, ast{e})")
                } else {
                    value
                }
            })
            .collect();
        store_prefetch.push_str(&format!(
            "        let si{e} = {flat};\n\
             \x20       let sr{e} = si{e} / {row}u;\n\
             \x20       let sc{e} = si{e} % {row}u;\n\
             \x20       As[sr{e} * PW + sc{e}] = {ctor};\n",
            row = row_slots,
            ctor = ctor(&guarded(&parts, &format!("k1 + sc{e} * {fw}u"))),
        ));
    }
    for e in 0..slots_b {
        let flat = format!("(ty * {tx}u + tx + {e}u * {threads}u)");
        // `by` is the shared tile's outer axis and `bx` its k pair. In the
        // transposed form (`B` stored `(n, k)`) the outer axis is n; in the plain
        // form the tile is transposed on the way in, so it is n there too —only
        // the global address differs. In the transposed form k is the *column*
        // index (stride 1), so the stage offset has to be added explicitly; in the
        // plain form k is the row index and `(k1 + ...) * ldb` already carries it.
        let row = format!("bi{e} / {row}u", row = row_slots);
        let col = format!("bi{e} % {row}u", row = row_slots);
        // Only the transposed operand keeps consecutive k in consecutive
        // addresses, so only there does a wide slot become one global load.
        let reads: Vec<String> = (0..fw)
            .map(|c| {
                let k = format!("k1 + bx{e} * {fw}u + {c}u");
                if conv_direct {
                    format!("conv_b({k}, oy, ox0 + by{e}, {conv_base})")
                } else if transb {
                    format!("B[@B@(n0 + by{e}) * gd.ldb + {k}]")
                } else {
                    // The parentheses matter: `k` is a sum, and it is the whole
                    // of it that strides by `ldb`.
                    format!("B[@B@({k}) * gd.ldb + n0 + by{e}]")
                }
            })
            .collect();
        prefetch.push_str(&format!(
            "            let bi{e} = {flat};\n\
             \x20           let by{e} = {row};\n\
             \x20           let bx{e} = {col};\n\
             \x20           pfb{e} = {ctor};\n",
            ctor = ctor(&reads),
        ));
        // Re-derived for the same reason as A's: the store is outside the `if`.
        let parts: Vec<String> = (0..fw).map(|c| comp(format!("pfb{e}"), c)).collect();
        store_prefetch.push_str(&format!(
            "        let ui{e} = {flat};\n\
             \x20       let ur{e} = ui{e} / {row}u;\n\
             \x20       let uc{e} = ui{e} % {row}u;\n\
             \x20       Bs[ur{e} * PW + uc{e}] = {ctor};\n",
            row = row_slots,
            ctor = ctor(&guarded(&parts, &format!("k1 + uc{e} * {fw}u"))),
        ));
    }

    // The conv-direct gather: one `B` element, read straight out of the input.
    //
    // The reason this is affordable per element is that a tile's positions lie in
    // a single output row (`BN <= out_w`, enforced by the caller), so `(oy, ox0)`
    // come from the workgroup id once and an element's input column is
    // `ox0 + by` — a contiguous run, which is what keeps the staging reads
    // coalesced. The expensive part of a general gather, `p / out_w`, never
    // happens.
    let conv_helper = if conv_direct {
        format!(
            "
@group(0) @binding(4) var<uniform> cd: vec4<u32>;  // out_w, out_h, w, h
         @group(0) @binding(5) var<uniform> ce: vec4<u32>;  // (stride_w,stride_h) (pad_w,pad_h) (kh,kw) in_channels
         
         fn conv_b(k: u32, oy: u32, col: u32, base: u32) -> f32 {{
             let kh = ce.z >> 16u;
             let kw = ce.z & 0xffffu;
             let taps = kh * kw;
             if (k >= ce.w * taps) {{ return 0.0; }}
             let ic = k / taps;
             let tap = k % taps;
             let ky = tap / kw;
             let kx = tap % kw;
             let stride_w = ce.x >> 16u;
             let stride_h = ce.x & 0xffffu;
             let pad_w = ce.y >> 16u;
             let pad_h = ce.y & 0xffffu;
             let iy = i32(oy * stride_h + ky) - i32(pad_h);
             let ix = i32(col * stride_w + kx) - i32(pad_w);
             if (iy < 0 || iy >= i32(cd.w) || ix < 0 || ix >= i32(cd.z)) {{ return 0.0; }}
             return X[base + (ic * cd.w + u32(iy)) * cd.z + u32(ix)];
         }}
"
        )
    } else {
        String::new()
    };
    // The attention's folded softmax: `A` is the raw score matrix and the A
    // staging rewrites every element as `exp(a*scale - max_row) * inv_row`
    // from the per-row statistics the softmax kernel wrote. Only
    // `gemm_batched_bn64_row_exp` turns it on, so the two bindings and the
    // helper are emitted for that variant alone — an unused binding would not
    // be in the automatic layout anyway.
    let row_exp_helper = if a_row_exp {
        "
@group(0) @binding(4) var<storage, read> AStats: array<vec2<f32>>;
@group(0) @binding(5) var<uniform> ae: vec4<u32>;  // scale bits, rows per batch, _, _

fn a_row_scale(v: f32, s: vec2<f32>) -> f32 {
    return exp(v * bitcast<f32>(ae.x) - s.x) * s.y;
}
"
            .to_string()
    } else {
        String::new()
    };
    // The A rows a staging thread owns are the same for every K block, so the
    // rows' `(max, inv)` pairs are loaded once and reused by both stores.
    let mut stats_prologue = String::new();
    if a_row_exp {
        for e in 0..slots_a {
            stats_prologue.push_str(&format!(
                "    let asr{e} = (ty * {tx}u + tx + {e}u * {threads}u) / {row}u;\n\
                 \x20   let ast{e} = AStats[wid.z * ae.y + m0 + asr{e}];\n",
                row = row_slots,
            ));
        }
    }
    let mut store_stage0 = String::new();
    for e in 0..slots_a {
        let flat = format!("(ty * {tx}u + tx + {e}u * {threads}u)");
        // With the folded softmax the staged value is rewritten on the way in;
        // otherwise it goes to shared exactly as loaded.
        let loaded = |c: usize| {
            let addr = format!("A[@A@(m0 + wr{e}) * gd.lda + wc{e} * {fw}u + {c}u]");
            if a_row_exp {
                format!("a_row_scale({addr}, ast{e})")
            } else {
                addr
            }
        };
        let parts: Vec<String> = (0..fw).map(loaded).collect();
        store_stage0.push_str(&format!(
            "        let wi{e} = {flat};\n\
             \x20       let wr{e} = wi{e} / {row}u;\n\
             \x20       let wc{e} = wi{e} % {row}u;\n\
             \x20       As[wr{e} * PW + wc{e}] = {ctor};\n",
            row = row_slots,
            ctor = ctor(&guarded(&parts, &format!("wc{e} * {fw}u"))),
        ));
    }
    for e in 0..slots_b {
        let flat = format!("(ty * {tx}u + tx + {e}u * {threads}u)");
        // The prologue stages the first K block, so the column offset is zero
        // here and only the prefetch above needs `k1`.
        let (row, col) = if coalesced_b {
            (format!("vi{e} % {bn}u"), format!("vi{e} / {bn}u"))
        } else {
            (
                format!("vi{e} / {row}u", row = row_slots),
                format!("vi{e} % {row}u", row = row_slots),
            )
        };
        // Only the transposed operand keeps consecutive k in consecutive
        // addresses, so only there does a wide slot become one global load.
        let parts: Vec<String> = (0..fw)
            .map(|c| {
                let k = format!("vc{e} * {fw}u + {c}u");
                if conv_direct {
                    format!("conv_b({k}, oy, ox0 + vr{e}, {conv_base})")
                } else if transb {
                    format!("B[@B@(n0 + vr{e}) * gd.ldb + {k}]")
                } else {
                    format!("B[@B@({k}) * gd.ldb + n0 + vr{e}]")
                }
            })
            .collect();
        store_stage0.push_str(&format!(
            "        let vi{e} = {flat};\n\
             \x20       let vr{e} = {row};\n\
             \x20       let vc{e} = {col};\n\
             \x20       Bs[vr{e} * PW + vc{e}] = {ctor};\n",
            ctor = ctor(&guarded(&parts, &format!("vc{e} * {fw}u"))),
        ));
    }

    // Reduction in steps of one shared slot: `fw` K elements per load, so a
    // `vec2` slot is an 8-byte load per 2 FMAs and a `vec4` one a 16-byte load
    // per 4.
    let mut compute = String::new();
    // The shared row bases do not move across the K loop, so they are computed
    // once and every access below is a constant offset from one. Leaving the
    // k-step as a loop variable instead costs an address calculation per load —
    // 128 of them against 512 FMAs per iteration — and the kernel measured 42%
    // of this card's peak with that, which is almost exactly the instruction
    // ratio it implies.
    for i in 0..tm {
        compute.push_str(&format!(
            "        let ar{i} = (ty + {i}u * TY) * PW;\n"
        ));
    }
    for j in 0..tn {
        compute.push_str(&format!(
            "        let br{j} = (tx + {j}u * TX) * PW;\n"
        ));
    }
    for q2 in 0..BK / fw {
        compute.push_str("        {\n");
        for i in 0..tm {
            compute.push_str(&format!(
                "            let a{i} = As[ar{i} + {q2}u];\n"
            ));
        }
        for j in 0..tn {
            compute.push_str(&format!(
                "            let b{j} = Bs[br{j} + {q2}u];\n"
            ));
        }
        for i in 0..tm {
            for j in 0..tn {
                // One FMA per element of the slot: `fw` shared loads feed
                // `fw` times the FMAs, which is what the width buys.
                for c in 0..fw {
                    let sw = ["x", "y", "z", "w"][c];
                    compute.push_str(&format!(
                        "            c{i}_{j} = fma(a{i}.{sw}, b{j}.{sw}, c{i}_{j});\n"
                    ));
                }
            }
        }
        compute.push_str("        }\n");
    }

    // The epilogue replays the unfused sequence exactly: the bias add of the
    // separate `add_in_place`, then the erf GELU of the separate
    // `gelu_in_place`, both elementwise on the same accumulator value. The
    // residual forms read the destination back — the block's skip connection —
    // and add it last, which is where the unfused `x += normed` added it too.
    let bias_binding = if ep.has_bias() {
        "@group(0) @binding(4) var<storage, read> Bias: array<f32>;\n".to_string()
    } else {
        String::new()
    };
    let mut epilogue = String::new();
    // Codegen for the store: a column bias is the same for every row of the
    // tile, so it is loaded once per column here rather than once per store —
    // eight loads a thread instead of sixty-four. On the row-bias twin the same
    // hoist was worth more (26.1 -> 22.2 ms on a chunk's worth of conv GEMMs)
    // than the fold that introduced the load in the first place. The index is
    // clamped so the load stays inside the tensor; the guard still decides
    // whether the store happens.
    let column_bias = ep.has_bias() && ep != GemmEpilogue::RowBias;
    if column_bias {
        for j in 0..tn {
            epilogue.push_str(&format!(
                "    let gn{j} = n0 + tx + {j}u * TX;\n\
                 \x20   let cb{j} = Bias[min(gn{j}, gd.n - 1u)];\n"
            ));
        }
    }
    for i in 0..tm {
        epilogue.push_str(&format!("    let gm{i} = m0 + ty + {i}u * TY;\n"));
        epilogue.push_str(&format!("    if (gm{i} < gd.m) {{\n"));
        if ep == GemmEpilogue::RowBias {
            // One load per row for the whole tile rather than one per store.
            epilogue.push_str(&format!("        let row_bias{i} = Bias[gm{i}];\n"));
        }
        for j in 0..tn {
            let mut write = format!("c{i}_{j}");
            if ep == GemmEpilogue::RowBias {
                // The convolution's bias runs along the output channels, which
                // the GEMM lays out along M: `add_row_bias_in_place` adds
                // `Bias[row % bias_rows]` with `row` the channel, and `gm{i}` is
                // that same index within the tile.
                write = format!("{write} + row_bias{i}");
            } else if column_bias {
                write = format!("{write} + cb{j}");
            }
            if ep.has_residual() {
                write = format!("{write} + C[@C@gm{i} * gd.ldc + gn{j}]");
            }
            if ep == GemmEpilogue::GeluBias {
                write = format!("gelu_erp({write})");
            }
            if column_bias {
                epilogue.push_str(&format!(
                    "        if (gn{j} < gd.n) {{ C[@C@gm{i} * gd.ldc + gn{j}] = {write}; }}\n"
                ));
            } else {
                epilogue.push_str(&format!(
                    "        let gn{j} = n0 + tx + {j}u * TX;\n\
                     \x20       if (gn{j} < gd.n) {{ C[@C@gm{i} * gd.ldc + gn{j}] = {write}; }}\n"
                ));
            }
        }
        epilogue.push_str("    }\n");
    }

    // Which grid axis carries the M tiles. `GemmJob::grid` decides the same thing
    // and the two must agree.
    let m_axis = "x";
    let n_axis = "y";

    format!(
        r#"
{bias_binding}struct Dims {{
    m: u32,
    n: u32,
    k: u32,
    lda: u32,
    ldb: u32,
    ldc: u32,
    // Batches are addressed as two levels, `wid.z / inner_count` and
    // `wid.z % inner_count`, each with its own stride. Attention needs exactly
    // this: its operands are indexed by (band, head) and the two levels are not
    // a single linear stride apart in the fused QKV layout.
    inner_count: u32,
    asa_outer: u32,
    asa_inner: u32,
    bsb_outer: u32,
    bsb_inner: u32,
    csc_outer: u32,
    csc_inner: u32,
}}

@group(0) @binding(0) var<storage, read> A: array<f32>;
{b_operand}
@group(0) @binding(2) var<storage, read_write> C: array<f32>;
@group(0) @binding(3) var<uniform> gd: Dims;
{conv_helper}{row_exp_helper}
const BM: u32 = {bm}u;
const BN: u32 = {bn}u;
const BK: u32 = {bk}u;
const PAD: u32 = {pad}u;
const PW: u32 = {pw}u;
const TM: u32 = {tm}u;
const TN: u32 = {tn}u;
const TX: u32 = {tx}u;
const TY: u32 = {ty}u;

var<workgroup> As: array<{fvt}, {as_len}>;
var<workgroup> Bs: array<{fvt}, {bs_len}>;

{gelu_fn}
@compute @workgroup_size({tx}, {ty})
fn gemm(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{
    let tx = lid.x;
    let ty = lid.y;
    let m0 = wid.{m_axis} * BM;
    let n0 = wid.{n_axis} * BN;
{conv_prologue}{batch_base}{stats_prologue}
{accumulators}
{prefetch_decl}
    let nk = (gd.k + BK - 1u) / BK;

    // Prologue: stage K block 0 straight into shared.
    {{
{store_stage0}    }}
    workgroupBarrier();

    for (var kb = 0u; kb < nk; kb = kb + 1u) {{
        let k1 = kb * BK + BK;

        // Start the next stage's global loads before touching shared, so their
        // latency overlaps the reduction below rather than preceding it.
        if (k1 < gd.k) {{
{prefetch}        }}

{compute}
        // The last K block has nothing to prefetch-store; skipping the pair of
        // barriers here is bit-identical (epilogue reads registers, not shared).
        // Uniform over the workgroup: k1 and gd.k do not vary by lane.
        if (k1 < gd.k) {{
            workgroupBarrier();
{store_prefetch}            workgroupBarrier();
        }}
    }}

{epilogue}
}}
"#,
        bm = bm,
        bn = bn,
        bk = BK,
        pad = pad,
        pw = pw,
        fvt = fvt,
        tm = tm,
        tn = tn,
        tx = tx,
        ty = ty,
        as_len = bm * pw,
        // Sized to the B tile's own rows: the staging walk now covers exactly the
        // rows the compute loop reads, so a 64-wide N tile halves this.
        bs_len = bn * pw,
        conv_prologue = if conv_direct {
            "    // One output row per tile: only `BN <= out_w` takes this path.
                 let oy = n0 / cd.x;
                 let ox0 = n0 % cd.x;
"
                .to_string()
        } else {
            String::new()
        },
        batch_base = batch_base,
        stats_prologue = stats_prologue,
        accumulators = accumulators,
        prefetch_decl = prefetch_decl,
        store_stage0 = store_stage0,
        prefetch = prefetch,
        compute = compute,
        store_prefetch = store_prefetch,
        epilogue = epilogue,
        bias_binding = bias_binding,
        b_operand = if conv_direct {
            "@group(0) @binding(1) var<storage, read> X: array<f32>;
".to_string()
        } else {
            "@group(0) @binding(1) var<storage, read> B: array<f32>;
".to_string()
        },
        conv_helper = conv_helper,
        row_exp_helper = row_exp_helper,
        m_axis = m_axis,
        n_axis = n_axis,
        gelu_fn = match ep {
            GemmEpilogue::GeluBias => {
                // Same erf formulation as the scalar gelu kernel, wrapped the same
                // way: `0.5 * x * (1 + erf(x / sqrt(2)))`. Returning the erf alone
                // would be a plausible-looking activation — it agrees with gelu to
                // within a few percent on `|x| < 1` and diverges to a factor of
                // `x` beyond — and that is exactly the shape of the silent error
                // this file's other epilogues are bit-exact to avoid.
                "fn gelu_erp(x: f32) -> f32 {\n    let scaled = x * 0.70710678118654752440;\n    let z = abs(scaled);\n    let t = 1.0 / (1.0 + 0.3275911 * z);\n    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t\n        - 0.284496736) * t + 0.254829592) * t) * exp(-z * z);\n    let e = select(-y, y, scaled >= 0.0);\n    return x * 0.5 * (1.0 + e);\n}\n"
                    .to_string()
            }
            _ => String::new(),
        },
    )
    // The nested generators cannot see the batch flags, so the base terms are
    // marked and substituted once here.
    .replace("@A@", a_pre)
    .replace("@B@", b_pre)
    .replace("@C@", c_pre)
}

/// A pure register-FMA loop, used to find what this driver/GPU combination can
/// actually sustain.
///
/// Without this number there is no way to tell a badly written GEMM from a
/// ceiling imposed by the stack: a kernel can only be judged against the
/// achievable rate, not against the datasheet.
///
/// The accumulators must be **independent chains**, and the body must be
/// unrolled: a version with two chains measured 3.1 TFLOP/s and one with eight
/// chains but only 8 FMAs per loop iteration also measured 3.1, while eight
/// chains with a 4x unroll reaches 5.8. The loop's own increment/compare/branch
/// are what cost the difference.
pub fn fma_probe(iters: usize) -> String {
    const CHAINS: usize = 8;
    const UNROLL: usize = 4;
    let mut init = String::new();
    let mut body = String::new();
    for c in 0..CHAINS {
        init.push_str(&format!(
            "    var acc{c} = f32(gid.x) * 1e-9 + {};\n",
            0.5 + c as f32 * 0.01
        ));
        for _ in 0..UNROLL {
            body.push_str(&format!("        acc{c} = fma(acc{c}, mul{c}, 1e-9);\n"));
        }
    }
    let sum = (0..CHAINS)
        .map(|c| format!("acc{c}"))
        .collect::<Vec<_>>()
        .join(" + ");
    format!(
        r#"
@group(0) @binding(0) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn fma_probe(@builtin(global_invocation_id) gid: vec3<u32>) {{
{init}
    let mul0 = 1.0000001;
    let mul1 = 0.9999999;
    let mul2 = 1.0000002;
    let mul3 = 0.9999998;
    let mul4 = 1.0000003;
    let mul5 = 0.9999997;
    let mul6 = 1.0000004;
    let mul7 = 0.9999996;
    for (var i = 0u; i < {iters}u; i = i + 1u) {{
{body}    }}
    out[(gid.x + gid.y * 1024u) & 0xFFFFu] = {sum};
}}
"#,
        init = init,
        body = body,
        sum = sum,
        iters = iters,
    )
}

/// FMAs the probe above issues per outer loop iteration.
pub const FMA_PROBE_PER_ITER: usize = 8 * 4;

/// Threads per workgroup for the row-wise elementwise kernels.
pub const ROW_THREADS: usize = 256;

/// Rows one dispatch may cover along the x grid axis.
///
/// WebGPU caps every dispatch dimension at 65535, and attention over the
/// frequency axis has 6408 * 60 = 384 480 rows, so a row-wise kernel cannot put
/// its row index in `wid.x` alone. `gd.w` carries the x extent and the row
/// becomes `wid.x + wid.y * gd.w`.
pub const ROW_GRID_X: usize = 65535;

/// `F.normalize(x) * sqrt(dim) * gamma`, one warp per row.
///
/// The same reduction as [`rms_norm`], folded with shuffles instead of a tree:
/// one `subgroupAdd` against eight barrier steps, and no shared memory at all.
/// The row is read once into registers and written back from them, so the
/// twice-read/half-idle shape of the workgroup version goes away too. The
/// summation order changes with the reduction shape — the same ~1e-7 relative
/// the attention tests already accept, and the same order the corpus of
/// alignment tests measures directly.
///
/// `SLOTS` must cover `ceil(dim / LANES)`; the caller keeps the workgroup
/// version for wider rows and for adapters without a 32-lane subgroup.
pub fn rms_norm_warp() -> String {
    const LANES: u32 = 32;
    const WARPS: u32 = 8;
    const THREADS: u32 = LANES * WARPS;
    const SLOTS: u32 = RMS_WARP_SLOTS;

    let mut load = String::new();
    let mut store = String::new();
    for k in 0..SLOTS {
        load.push_str(&format!(
            "    let x{k} = select(0.0, X[base + lane + {k}u * LANES], lane + {k}u * LANES < dim);\n\
             \x20   sum = fma(x{k}, x{k}, sum);\n"
        ));
        store.push_str(&format!(
            "    let i{k} = lane + {k}u * LANES;\n\
             \x20   if (i{k} < pitch_out) {{\n\
             \x20       Out[obase + i{k}] = select(0.0, x{k} * factor * Gamma[i{k}], i{k} < dim);\n\
             \x20   }}\n"
        ));
    }

    format!(
        r#"
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Gamma: array<f32>;
@group(0) @binding(2) var<storage, read_write> Out: array<f32>;
@group(0) @binding(3) var<uniform> gd: vec4<u32>;  // rows, dim, pitch_in, pitch_out

const LANES: u32 = {lanes}u;
const WARPS: u32 = {warps}u;
const ROW_GRID_X: u32 = {grid_x}u;

@compute @workgroup_size({threads})
fn rms_norm(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{
    let dim = gd.y;
    let pitch_in = gd.z;
    let pitch_out = gd.w;
    let warp = lid.x / LANES;
    let lane = lid.x % LANES;
    let row = (wid.x + wid.y * ROW_GRID_X) * WARPS + warp;
    let safe_row = min(row, gd.x - 1u);
    let base = safe_row * pitch_in;
    let obase = safe_row * pitch_out;

    var sum = 0.0;
{load}
    let squares = subgroupAdd(sum);
    let norm = max(sqrt(squares), 1e-12);
    let factor = sqrt(f32(dim)) / norm;

    if (row < gd.x) {{
{store}    }}
}}
"#,
        lanes = LANES,
        warps = WARPS,
        threads = THREADS,
        grid_x = ROW_GRID_X,
        load = load,
        store = store,
    )
}

/// Columns one lane of the warp rms_norm covers, i.e. `RMS_WARP_SLOTS * 32`.
pub const RMS_WARP_SLOTS: u32 = 12;
/// The widest row the warp rms_norm takes.
pub const RMS_WARP_COLS: usize = 32 * RMS_WARP_SLOTS as usize;
/// Rows one warp-per-row rms_norm workgroup covers.
pub const RMS_WARP_ROWS_PER_WG: usize = 8;

/// `F.normalize(x, dim=-1) * sqrt(dim) * gamma`.
///
/// Not the usual RMSNorm: this normalises by the **L2 norm of the feature
/// vector**, not by its root-mean-square, and the scale is `sqrt(dim)` rather
/// than a learned or fixed epsilon. `eps` is the clamp on the norm, matching
/// `F.normalize`'s `clamp_min(1e-12)`.
///
/// One workgroup per row, tree-reduced over `dim`; the reduction order is a
/// plain binary tree so it matches the CPU reference exactly enough for f32
/// agreement to be round-off rather than structure.
pub fn rms_norm() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Gamma: array<f32>;
@group(0) @binding(2) var<storage, read_write> Out: array<f32>;
@group(0) @binding(3) var<uniform> gd: vec4<u32>;  // rows, dim, pitch_in, pitch_out

const THREADS: u32 = {threads}u;
const ROW_GRID_X: u32 = {grid_x}u;
var<workgroup> partial: array<f32, {threads}>;

@compute @workgroup_size({threads})
fn rms_norm(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let row = wid.x + wid.y * ROW_GRID_X;
    if (row >= gd.x) {{ return; }}
    let dim = gd.y;
    let pitch_in = gd.z;
    let pitch_out = gd.w;
    let base = row * pitch_in;

    var sum = 0.0;
    for (var i = lid.x; i < dim; i = i + THREADS) {{
        let v = X[base + i];
        sum = fma(v, v, sum);
    }}
    partial[lid.x] = sum;
    workgroupBarrier();
    for (var step = THREADS / 2u; step > 0u; step = step / 2u) {{
        if (lid.x < step) {{
            partial[lid.x] = partial[lid.x] + partial[lid.x + step];
        }}
        workgroupBarrier();
    }}

    let norm = max(sqrt(partial[0]), 1e-12);
    let factor = sqrt(f32(dim)) / norm;
    let out_base = row * pitch_out;
    // Columns past `dim` are the output pitch's padding and are cleared: whatever
    // consumes this tensor next reads a full reduction step past the logical
    // feature width, and a width that is not a multiple of the step would
    // otherwise pick up the next row's values.
    for (var i = lid.x; i < pitch_out; i = i + THREADS) {{
        if (i < dim) {{
            Out[out_base + i] = X[base + i] * factor * Gamma[i];
        }} else {{
            Out[out_base + i] = 0.0;
        }}
    }}
}}
"#,
        threads = ROW_THREADS,
        grid_x = ROW_GRID_X
    )
}

/// Exact (erf-based) GELU, elementwise. `nn.GELU()`'s default, not the tanh
/// approximation —the two differ by ~2e-3 at |x| = 1.
///
/// The erf uses the A&S 7.1.26 rational form with `C_A = 0.3275911`. That
/// constant matters: a version of this in a sibling project carried 0.147 and
/// the resulting erf was off by ~5%.
pub fn gelu() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@group(0) @binding(2) var<uniform> gd: vec4<u32>;  // count, _, _, _

const GRID_X: u32 = 65535u;

fn erf(x: f32) -> f32 {{
    let z = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * z);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t
        - 0.284496736) * t + 0.254829592) * t) * exp(-z * z);
    return select(-y, y, x >= 0.0);
}}

@compute @workgroup_size({threads})
fn gelu(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let i = (wid.y * GRID_X + wid.x) * {threads}u + lid.x;
    if (i >= gd.x) {{ return; }}
    let v = X[i];
    Out[i] = v * 0.5 * (1.0 + erf(v * 0.70710678118654752440));
}}
"#,
        threads = ROW_THREADS
    )
}

/// Exact GELU applied in place.
///
/// The feed-forward's intermediate is the widest tensor in the model, so it is
/// worth not having a second one; a single read-write binding is also the only
/// way to write a buffer in place, since a dispatch may not bind the same buffer
/// as both read-only and read-write.
pub fn gelu_in_place() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<uniform> gd: vec4<u32>;  // count, _, _, _

const GRID_X: u32 = 65535u;

fn erf_in_place(x: f32) -> f32 {{
    let z = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * z);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t
        - 0.284496736) * t + 0.254829592) * t) * exp(-z * z);
    return select(-y, y, x >= 0.0);
}}

@compute @workgroup_size({threads})
fn gelu_in_place(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let i = (wid.y * GRID_X + wid.x) * {threads}u + lid.x;
    if (i >= gd.x) {{ return; }}
    let v = X[i];
    X[i] = v * 0.5 * (1.0 + erf_in_place(v * 0.70710678118654752440));
}}
"#,
        threads = ROW_THREADS
    )
}

/// `nn.GLU(dim=-1)`: `first_half * sigmoid(second_half)`.
///
/// One thread per output element; the two halves are `dim` apart, so a thread
/// reads from two places rather than one contiguous pair.
pub fn glu() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@group(0) @binding(2) var<uniform> gd: vec4<u32>;  // rows, half, _, _

const GRID_X: u32 = 65535u;

@compute @workgroup_size({threads})
fn glu(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let i = (wid.y * GRID_X + wid.x) * {threads}u + lid.x;
    let half = gd.y;
    let total = gd.x * half;
    if (i >= total) {{ return; }}
    let row = i / half;
    let col = i % half;
    let base = row * half * 2u;
    let gate = X[base + half + col];
    Out[i] = X[base + col] * (1.0 / (1.0 + exp(-gate)));
}}
"#,
        threads = ROW_THREADS
    )
}

/// `out = glu(x) * scale + shift` in one pass: the DConv's GLU is immediately
/// followed by its LayerScale, and the two between them read and write the whole
/// activation twice.
///
/// Bit-identical to the pair: the same `sig(X[base + half + col]) * X[base + col]`
/// per element, then the same `value * Scale[channel] + Shift[channel]` the
/// separate affine applies — that one is pure elementwise, so folding it into
/// the producer's store changes no rounding.
pub fn glu_channel_affine() -> String {
    glu_channel_affine_impl(false)
}

/// `out = base + (glu(x) * scale + shift)`, one pass.
///
/// The DConv layer ends with `out = current + gamma * glu(...)`, which as three
/// passes (the GLU, the LayerScale and the residual `current + ...`) is three
/// walks over the layer's output. The parentheses are load-bearing: scaling and
/// shifting first, then adding, is the association the three-pass form has, and
/// with a zero shift the two are the same arithmetic rather than merely close.
pub fn glu_channel_affine_add() -> String {
    glu_channel_affine_impl(true)
}

fn glu_channel_affine_impl(add: bool) -> String {
    let base_decl = if add {
        "@group(0) @binding(5) var<storage, read> Base: array<f32>;"
    } else {
        ""
    };
    let store = if add {
        "    Out[i] = Base[i] + (value * Scale[channel] + Shift[channel]);"
    } else {
        "    Out[i] = value * Scale[channel] + Shift[channel];"
    };
    let entry = if add {
        "glu_channel_affine_add"
    } else {
        "glu_channel_affine"
    };
    format!(
        r#"
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@group(0) @binding(2) var<storage, read> Scale: array<f32>;
@group(0) @binding(3) var<storage, read> Shift: array<f32>;
@group(0) @binding(4) var<uniform> gd: vec4<u32>;  // rows, half, channels, plane
{base_decl}

const GRID_X: u32 = 65535u;

@compute @workgroup_size({threads})
fn {entry}(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let i = (wid.y * GRID_X + wid.x) * {threads}u + lid.x;
    // `half` is the GLU's half, `channels * plane`; the LayerScale's channel
    // index is the one `channel_affine_act_in_place` uses, `(i / plane) % channels`.
    let half = gd.y;
    let channels = gd.z;
    let plane = gd.w;
    let total = gd.x * half;
    if (i >= total) {{ return; }}
    let row = i / half;
    let col = i % half;
    let base = row * half * 2u;
    let gate = X[base + half + col];
    let value = X[base + col] * (1.0 / (1.0 + exp(-gate)));
    let channel = (i / plane) % channels;
{store}
}}
"#,
        threads = ROW_THREADS,
    )
}

/// `out = a * sigmoid(b)` on a row, used for the attention gate.
///
/// The gate is per head while the value is per (head, dim_head), so the gate
/// index advances once every `dim_head` elements of the value row.
pub fn sigmoid_gate() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Gate: array<f32>;
@group(0) @binding(2) var<storage, read_write> Out: array<f32>;
@group(0) @binding(3) var<uniform> gd: vec4<u32>;  // count, heads, dim_head, _

const GRID_X: u32 = 65535u;

@compute @workgroup_size({threads})
fn sigmoid_gate(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let i = (wid.y * GRID_X + wid.x) * {threads}u + lid.x;
    if (i >= gd.x) {{ return; }}
    let heads = gd.y;
    let dim_head = gd.z;
    // Element `i` sits in head `h` of a row of `heads * dim_head`.
    let h = (i % (heads * dim_head)) / dim_head;
    let row = i / (heads * dim_head);
    Out[i] = X[i] * (1.0 / (1.0 + exp(-Gate[row * heads + h])));
}}
"#,
        threads = ROW_THREADS
    )
}

/// Interleaved rotary position embedding, matching `rotary_embedding_torch`.
///
/// `rotate_half` pairs *adjacent* elements (`(d r) -> d r` with `r = 2`), not
/// the two halves, and the angle is `position * freq` evaluated in f32. Applying
/// it to Q and K is what the reference does; the frequency table is shared across
/// all six layers.
///
/// This works directly on the fused QKV tensor, so no reordering pass is needed:
/// a row holds all `3 * heads * dim_head` columns of one frame. Q is heads
/// `0..heads` and K is `heads..2*heads`; V is left alone.
///
/// One thread owns one (row, frequency) pair across *all* rotating heads. The
/// angle depends only on the position and the frequency -- the same for every
/// head -- so the previous per-(row, head) form recomputed the same `sin`/`cos`
/// eight times over 384k half-idle workgroups. Each thread still reads and
/// writes exactly the same elements with exactly the same formula, so the
/// output is bit-identical.
///
/// Uniforms: `gd0 = (frames, row_stride, dim_head, head_begin)`,
/// `gd1 = (total_rows, head_count, 0, 0)`; the grid covers
/// `total_rows * (dim_head / 2)` threads.
pub fn rope() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read> Freqs: array<f32>;
@group(0) @binding(2) var<uniform> gd0: vec4<u32>;  // frames, row_stride, dim_head, head_begin
@group(0) @binding(3) var<uniform> gd1: vec4<u32>;  // total_rows, head_count, 0, 0

const THREADS: u32 = {threads}u;

@compute @workgroup_size({threads})
fn rope(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let frames = gd0.x;
    let row_stride = gd0.y;
    let dim_head = gd0.z;
    let head_begin = gd0.w;
    let total_rows = gd1.x;
    let head_count = gd1.y;
    let half = dim_head / 2u;

    let flat = wid.x * THREADS + lid.x;
    if (flat >= total_rows * half) {{ return; }}
    let row = flat / half;
    let k = flat % half;

    // The position is the frame index within the band, which is the row's low
    // bits because the layout is (band, frame, column).
    let pos = row % frames;
    let angle = f32(pos) * Freqs[k];
    let c = cos(angle);
    let s = sin(angle);
    let base = row * row_stride + head_begin * dim_head + 2u * k;

    for (var hh = 0u; hh < head_count; hh += 1u) {{
        let off = hh * dim_head;
        let a = X[base + off];
        let b = X[base + off + 1u];
        X[base + off] = a * c - b * s;
        X[base + off + 1u] = a * s + b * c;
    }}
}}
"#,
        threads = ROW_THREADS,
    )
}/// Side of the square tile the transpose moves through shared memory.
pub const TRANSPOSE_TILE: usize = 32;

/// `(batch, rows, cols, width) -> (batch, cols, rows, width)`.
///
/// The two axial transformers want the sequence on different axes: the time
/// transformer wants frames contiguous, the frequency one wants bands. Rather
/// than give every op a strided variant, the layout is flipped once per
/// transformer pair.
///
/// `width` is the innermost run that travels as a unit — the model's axial
/// transformers swap the band and frame axes while leaving the feature axis
/// untouched, which a plain matrix transpose cannot express. Because
/// consecutive threads walk consecutive `d` and `d` is contiguous on both
/// sides, the elementwise form needs no shared-memory tiling to stay
/// coalesced; a plain matrix transpose is the `width = 1` case.
///
/// `batch` keeps the independent instances (a batched forward's chunks) from
/// being reshaped into the swap: `(B*c, f, t)` must never be transposed as
/// `(f, B*c, t)`, which silently splits a batch's channels into the band axis.
///
/// `gd` is `(batch, rows, cols, width)`; the grid walks `batch*rows*cols*width`
/// elements in `ROW_THREADS`-wide groups.
pub fn transpose() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@group(0) @binding(2) var<uniform> gd: vec4<u32>;  // batch, rows, cols, width

const GRID_X: u32 = 65535u;

@compute @workgroup_size({threads})
fn transpose(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let i = (wid.y * GRID_X + wid.x) * {threads}u + lid.x;
    let plane = gd.y * gd.z * gd.w;
    if (i >= gd.x * plane) {{ return; }}
    let b = i / plane;
    let rest = i % plane;
    let width = gd.w;
    let row = rest / (gd.z * width);
    let tail = rest % (gd.z * width);
    let col = tail / width;
    let d = tail % width;
    let out_plane = gd.z * gd.y * gd.w;
    Out[b * out_plane + (col * gd.y + row) * width + d] = X[i];
}}
"#,
        threads = ROW_THREADS
    )
}

/// im2col gather for the conv family: `(batch, channels, h, w)` in,
/// `(batch, k, positions)` out, with `k = channels * kh * kw` in `(ic, ky, kx)`
/// order and `positions = out_h * out_w` in `(oy, ox)` order.
///
/// This is the operand order the GEMM wants for its B matrix, so the product
/// comes out channel-major (`(oc, positions)`, which is exactly one output
/// plane per channel) and no transpose pass is needed. It is also the same
/// layout `conv.rs` builds on the host, which is what makes the host patches a
/// usable reference for this kernel.
///
/// Zero padding is applied here rather than by the caller, so `X` is the
/// unpadded input.
///
/// Dimensions are plain 32-bit uniforms, one per slot across `gd`/`ge`/`gf`/`gg`:
/// the waveform branch feeds the full sample rate through here, so `w` and
/// `out_w` exceed what a packed `u16` pair could hold.
pub fn im2col() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@group(0) @binding(2) var<uniform> gd: vec4<u32>;  // ic, kh, kw, h
@group(0) @binding(3) var<uniform> ge: vec4<u32>;  // w, out_h, out_w, batch
@group(0) @binding(4) var<uniform> gf: vec4<u32>;  // stride_h, stride_w, pad_h, pad_w
@group(0) @binding(5) var<uniform> gg: vec4<u32>;  // pitch, rows, _, _

const THREADS: u32 = {threads}u;

@compute @workgroup_size({threads})
fn im2col(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let in_channels = gd.x;
    let kh = gd.y;
    let kw = gd.z;
    let h = gd.w;
    let w = ge.x;
    let out_h = ge.y;
    let out_w = ge.z;
    let batch = ge.w;
    let stride_h = gf.x;
    let stride_w = gf.y;
    let pad_h = gf.z;
    let pad_w = gf.w;
    // Row pitch of the destination: `positions` for a tight buffer, wider when
    // the GEMM that consumes this needs whole tiles (`BN`-padded). The margin
    // columns are the caller's zeros, never written here.
    let pitch = gg.x;
    // Rows per batch in the destination: `k` when the buffer is tight,
    // `pad_ceil(k, BK)` when the GEMM reads it (`BK`-padded rows). The batch
    // stride follows this, not the logical `k`, so the two agree.
    let rows = gg.y;

    // The dispatch carries the indices rather than the flat element number:
    // `wid.x` picks a block of output columns, `wid.y` a patch row of a batch, and
    // `wid.z` an output row. Deriving all three from one flat index cost six
    // integer divisions per element, which at these sizes made the gather
    // *division*-bound — 145 cycles per element against the card's ~10 for the
    // arithmetic — and this is the widest kernel in the model (39 dispatches and
    // 224 ms per chunk before the change, on 151M elements for the first block).
    let ox = wid.x * THREADS + lid.x;
    if (wid.z >= out_h) {{ return; }}
    let oy = wid.z;
    let flat = wid.y;
    let b = flat / rows;
    let krow = flat % rows;
    let k = in_channels * kh * kw;
    if (krow >= k) {{ return; }}   // the row margin of a `BK`-padded scratch

    let ic = krow / (kh * kw);
    let tap = krow % (kh * kw);
    let ky = tap / kw;
    let kx = tap % kw;

    let iy = i32(oy * stride_h + ky) - i32(pad_h);
    let src = (b * in_channels + ic) * h * w + u32(iy) * w;
    let dst = (b * rows + krow) * pitch + oy * out_w;
    let row_ok = iy >= 0 && iy < i32(h);
    // The decode above is uniform over the workgroup but costs every thread its
    // own copy of six integer divisions, which is what the previous version of
    // this kernel was bound on (145 cycles an element). Striding the columns
    // inside the thread pays it once for `cols` elements: `gg.z` is the x grid
    // in threads, so the loop's stride is the whole reduced x extent and the
    // lanes of a warp still cover consecutive columns on every trip.
    var col = ox;
    while (col < out_w) {{
        let ix = i32(col * stride_w + kx) - i32(pad_w);
        var value = 0.0;
        if (row_ok && ix >= 0 && ix < i32(w)) {{
            value = X[src + u32(ix)];
        }}
        Out[dst + col] = value;
        col += gg.z;
    }}
}}
"#,
        threads = ROW_THREADS
    )
}

/// Adds a bias that varies along the *row* axis, in place.
///
/// `add_in_place` broadcasts along the last axis, which is what a residual and
/// a transformer bias need; a convolution's bias belongs to its output channel,
/// and `conv2d_into` lays the output out as `(out_channels, positions)`, so the
/// bias index is the row index. `gd` is `(rows, cols, bias_rows, _)`: the bias
/// repeats every `bias_rows` rows, which is how one pass covers a whole
/// convolution's `(batch, out_channels, positions)` output when the batches are
/// stitched into one tensor rather than dispatched one at a time.
pub fn add_row_bias_in_place() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read> Bias: array<f32>;
@group(0) @binding(2) var<uniform> gd: vec4<u32>;  // rows, cols, bias_rows, _

const GRID_X: u32 = 65535u;

@compute @workgroup_size({threads})
fn add_row_bias_in_place(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let cols = gd.y;
    let i = (wid.y * GRID_X + wid.x) * {threads}u + lid.x;
    if (i >= gd.x * cols) {{ return; }}
    let row = i / cols;
    X[i] = X[i] + Bias[row % gd.z];
}}
"#,
        threads = ROW_THREADS
    )
}

/// The gather half of `nn.ConvTranspose2d`.
///
/// The GEMM produces the tap activations — `(out_channels * kh * kw,
/// positions_in)` from the `(in_channels, positions_in)` input — and this turns
/// them back into an image by summing, for every output pixel, the taps that
/// land on it. Written as a *gather* (one thread per output pixel, looping over
/// the taps) rather than a scatter, so no two threads ever touch the same
/// element: the result is deterministic and needs no atomics.
///
/// Dimensions are plain 32-bit uniforms, one per slot: the last decoder's
/// transposed conv spreads the full sample rate over `out_w`, which a packed
/// pair would cap at 65535.
pub fn col2im() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read> P: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@group(0) @binding(2) var<uniform> gd: vec4<u32>;  // oc, kh, kw, out_h
@group(0) @binding(3) var<uniform> ge: vec4<u32>;  // out_w, stride_h, stride_w, in_h
@group(0) @binding(4) var<uniform> gf: vec4<u32>;  // in_w, batch, pitch, rows

const GRID_X: u32 = 65535u;

@compute @workgroup_size({threads})
fn col2im(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let oc = gd.x;
    let kh = gd.y;
    let kw = gd.z;
    let out_h = gd.w;
    let out_w = ge.x;
    let stride_h = ge.y;
    let stride_w = ge.z;
    let in_h = ge.w;
    let in_w = gf.x;
    let batch = gf.y;
    // Rows of the tap matrix are padded to the GEMM's M tile, so the row pitch
    // is the caller's, not `positions_in`.
    let pitch = gf.z;
    // ... and the batch stride follows the *padded* row count, not `m`.
    let rows = gf.w;

    let positions_out = out_h * out_w;
    let m = oc * kh * kw;
    let i = (wid.y * GRID_X + wid.x) * {threads}u + lid.x;
    if (i >= batch * oc * positions_out) {{ return; }}

    let b = i / (oc * positions_out);
    let rest = i % (oc * positions_out);
    let channel = rest / positions_out;
    let p = rest % positions_out;
    let oh = p / out_w;
    let ow = p % out_w;

    var acc = 0.0;
    for (var ky = 0u; ky < kh; ky = ky + 1u) {{
        if (oh < ky) {{ continue; }}
        let num_h = oh - ky;
        if (num_h % stride_h != 0u) {{ continue; }}
        let iy = num_h / stride_h;
        if (iy >= in_h) {{ continue; }}
        for (var kx = 0u; kx < kw; kx = kx + 1u) {{
            if (ow < kx) {{ continue; }}
            let num_w = ow - kx;
            if (num_w % stride_w != 0u) {{ continue; }}
            let ix = num_w / stride_w;
            if (ix >= in_w) {{ continue; }}
            acc = acc + P[(b * rows + channel * kh * kw + ky * kw + kx) * pitch + iy * in_w + ix];
        }}
    }}
    Out[i] = acc;
}}
"#,
        threads = ROW_THREADS
    )
}

/// Per-channel affine followed by an activation, in place.
///
/// This is the convolution blocks' `conv -> norm -> act` tail: at load time a
/// BatchNorm's `running_mean`/`running_var` and `weight`/`bias` fold into one
/// `scale`/`shift` pair per channel, and the activation is fused here so the
/// activation does not pay a second read-modify-write of the widest activation
/// in the block. `gd` is `(channels, plane, activation, _)` where `plane` is the
/// per-channel element count (`h * w`), so the channel of element `i` is
/// `(i / plane) % channels`.
///
/// Activation mode: 0 identity, 1 ReLU, 2 exact (erf) GELU, 3 SiLU.
///
/// The GELU here is Abramowitz & Stegun 7.1.26 — good to ~1e-7 absolute, which
/// is two orders below the alignment tolerance the host models are held to, and
/// it is used only where a checkpoint asks for GELU (the roformer's GELU has its
/// own kernel).
pub fn channel_affine_act_in_place() -> String {
    channel_affine_act(false)
}

/// Whether a transformer block's `x + gamma * y` residual is one kernel or the
/// reference's three (scale in place, copy the base, add). `DEMUCS_RESIDUAL_FUSE=0`
/// takes the long way, which is also how the trace's intermediates stay visible.
pub fn residual_fuse() -> bool {
    std::env::var("DEMUCS_RESIDUAL_FUSE")
        .map(|v| v != "0")
        .unwrap_or(true)
}

/// `Out[i] = Base[i] + act(X[i] * Scale[c] + Shift[c])`, one pass.
///
/// The residual pair every transformer block ends with — an in-place LayerScale,
/// a copy of the block input and an add — is three passes over the same
/// `(tokens, dim)` tensor; this is the one pass they mean. `Base` is usually the
/// block input (`res = x + gamma * attn`).
pub fn channel_affine_act_add() -> String {
    channel_affine_act(true)
}

fn channel_affine_act(add: bool) -> String {
    let (x_decl, extra_decl, store) = if add {
        (
            "@group(0) @binding(0) var<storage, read> X: array<f32>;",
            "@group(0) @binding(4) var<storage, read> Base: array<f32>;\n\
             @group(0) @binding(5) var<storage, read_write> Out: array<f32>;",
            "    Out[i] = Base[i] + value;",
        )
    } else {
        (
            "@group(0) @binding(0) var<storage, read_write> X: array<f32>;",
            "",
            "    X[i] = value;",
        )
    };
    let entry = if add {
        "channel_affine_act_add"
    } else {
        "channel_affine_act_in_place"
    };
    format!(
        r#"
{x_decl}
@group(0) @binding(1) var<storage, read> Scale: array<f32>;
@group(0) @binding(2) var<storage, read> Shift: array<f32>;
@group(0) @binding(3) var<uniform> gd: vec4<u32>;  // channels, plane, activation, total
{extra_decl}

const GRID_X: u32 = 65535u;

fn erf_approx(z: f32) -> f32 {{
    let s = sign(z);
    let a = abs(z);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let poly = t * (0.254829592
        + t * (-0.284496736
        + t * (1.421413741
        + t * (-1.453152027 + t * 1.061405429))));
    return s * (1.0 - poly * exp(-a * a));
}}

@compute @workgroup_size({threads})
fn {entry}(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let i = (wid.y * GRID_X + wid.x) * {threads}u + lid.x;
    if (i >= gd.w) {{ return; }}
    let channel = (i / gd.y) % gd.x;
    var value = X[i] * Scale[channel] + Shift[channel];
    let mode = gd.z;
    if (mode == 1u) {{
        value = max(value, 0.0);
    }} else if (mode == 2u) {{
        value = 0.5 * value * (1.0 + erf_approx(value * 0.7071067811865476));
    }} else if (mode == 3u) {{
        value = value / (1.0 + exp(-value));
    }}
{store}
}}
"#,
        threads = ROW_THREADS
    )
}

/// Hyperbolic tangent, elementwise.
///
/// The mask estimator's MLP uses `nn.Tanh` between its hidden layers —the
/// transformer's feed-forward uses GELU, so the two cannot share a kernel.
pub fn tanh() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@group(0) @binding(2) var<uniform> gd: vec4<u32>;  // count, _, _, _

const GRID_X: u32 = 65535u;

@compute @workgroup_size({threads})
fn tanh_activation(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let i = (wid.y * GRID_X + wid.x) * {threads}u + lid.x;
    if (i >= gd.x) {{ return; }}
    // tanh via exp keeps the reference's own formulation rather than a rational
    // approximation, so the two agree to round-off.
    let e = exp(2.0 * X[i]);
    Out[i] = (e - 1.0) / (e + 1.0);
}}
"#,
        threads = ROW_THREADS
    )
}

/// `x = tanh(x + bias)`, in place, with a bias that varies per *block* of rows.
///
/// The mask estimator runs one 1536-wide MLP per band over the same 801 rows,
/// so its first layer is 60 GEMMs that are identical in shape and differ only
/// in their weights. Batching those into one dispatch needs the bias folded
/// into this pass instead of the GEMM epilogue: the batched form has no
/// epilogue, and the bias here is a function of the batch, not of the column
/// alone. Row `r` belongs to block `r / frames`, so the same `(band, column)`
/// bias applies to every row of a block -- which is what `frames` is for.
///
/// The arithmetic is the unfused order exactly: the sum is rounded in f32, then
/// tanh of the rounded value, so the result is bit-identical to `gemm_bias`
/// followed by `tanh_activation`.
pub fn tanh_bias_in_place() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read> Bias: array<f32>;
@group(0) @binding(2) var<uniform> gd: vec4<u32>;  // count, cols, frames, _

const GRID_X: u32 = 65535u;

@compute @workgroup_size({threads})
fn tanh_bias(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let i = (wid.y * GRID_X + wid.x) * {threads}u + lid.x;
    if (i >= gd.x) {{ return; }}
    let col = i % gd.y;
    let block = i / (gd.y * gd.z);
    let b = Bias[block * gd.y + col];
    let e = exp(2.0 * (X[i] + b));
    X[i] = (e - 1.0) / (e + 1.0);
}}
"#,
        threads = ROW_THREADS
    )
}

/// `out = in`, elementwise.
///
/// Needed because a kernel that reads one storage binding and writes another
/// cannot be pointed at the same buffer, so any in-place update that has to go
/// through a separate output ends with a copy.
pub fn copy() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@group(0) @binding(2) var<uniform> gd: vec4<u32>;  // count, _, _, _

const GRID_X: u32 = 65535u;

@compute @workgroup_size({threads})
fn copy(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let i = (wid.y * GRID_X + wid.x) * {threads}u + lid.x;
    if (i >= gd.x) {{ return; }}
    Out[i] = X[i];
}}
"#,
        threads = ROW_THREADS
    )
}

/// `out += source`, covering both the residual add (same shape) and the bias add
/// (source repeats along the last axis).
///
/// `gd.y` is the source length: `0` means "same shape as the output", which is
/// the residual case and costs one uniform branch rather than a modulo per
/// element. A non-zero value makes the source index wrap, so a `(n,)` bias adds
/// into every row of an `(rows, n)` tensor with the same kernel.
pub fn add_in_place() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read_write> Out: array<f32>;
@group(0) @binding(1) var<storage, read> Source: array<f32>;
@group(0) @binding(2) var<uniform> gd: vec4<u32>;  // count, src_len, _, _

const GRID_X: u32 = 65535u;

@compute @workgroup_size({threads})
fn add_in_place(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let i = (wid.y * GRID_X + wid.x) * {threads}u + lid.x;
    if (i >= gd.x) {{ return; }}
    let src_len = gd.y;
    var value = Source[i];
    if (src_len != 0u) {{
        value = Source[i % src_len];
    }}
    Out[i] = Out[i] + value;
}}
"#,
        threads = ROW_THREADS
    )
}

/// `out += source` in four-lane groups, covering both the residual add (same
/// shape) and the bias add (source repeats along the last axis). The scalar
/// form stays for counts that are not multiples of four.
pub fn add_in_place_vec4() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read_write> Out: array<f32>;
@group(0) @binding(1) var<storage, read> Source: array<f32>;
@group(0) @binding(2) var<uniform> gd: vec4<u32>;  // count, src_len, _, _

const GRID_X: u32 = 65535u;
const THREADS: u32 = {threads}u;

@compute @workgroup_size({threads})
fn add_in_place_vec4(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let g = (wid.y * GRID_X + wid.x) * THREADS + lid.x;
    let groups = gd.x / 4u;
    if (g >= groups) {{ return; }}
    let j = g * 4u;
    let src_len = gd.y;
    var v = vec4<f32>(Source[j], Source[j + 1u], Source[j + 2u], Source[j + 3u]);
    if (src_len != 0u) {{
        v = vec4<f32>(
            Source[j % src_len],
            Source[(j + 1u) % src_len],
            Source[(j + 2u) % src_len],
            Source[(j + 3u) % src_len]);
    }}
    Out[j] = Out[j] + v.x;
    Out[j + 1u] = Out[j + 1u] + v.y;
    Out[j + 2u] = Out[j + 2u] + v.z;
    Out[j + 3u] = Out[j + 3u] + v.w;
}}
"#,
        threads = ROW_THREADS
    )
}

/// GELU in place, four lanes per thread. The scalar form stays for counts that
/// are not multiples of four.
pub fn gelu_in_place_vec4() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<uniform> gd: vec4<u32>;  // count, _, _, _

const GRID_X: u32 = 65535u;
const THREADS: u32 = {threads}u;

fn erf4(x: f32) -> f32 {{
    let z = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * z);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t
        - 0.284496736) * t + 0.254829592) * t) * exp(-z * z);
    return select(-y, y, x >= 0.0);
}}

@compute @workgroup_size({threads})
fn gelu_in_place_vec4(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let g = (wid.y * GRID_X + wid.x) * THREADS + lid.x;
    let groups = gd.x / 4u;
    if (g >= groups) {{ return; }}
    let j = g * 4u;
    var v = vec4<f32>(X[j], X[j + 1u], X[j + 2u], X[j + 3u]);
    v = v * 0.5 * (1.0 + vec4<f32>(erf4(v.x * 0.70710678118654752440),
                                   erf4(v.y * 0.70710678118654752440),
                                   erf4(v.z * 0.70710678118654752440),
                                   erf4(v.w * 0.70710678118654752440)));
    X[j] = v.x;
    X[j + 1u] = v.y;
    X[j + 2u] = v.z;
    X[j + 3u] = v.w;
}}
"#,
        threads = ROW_THREADS
    )
}

/// Softmax over the last axis of a row, one workgroup per row.
///
/// Two passes over the row (max, then sum of exps) with the same tree shape as
/// [`rms_norm`]. Numerically this is the max-subtracted form, which is what
/// `torch.softmax` does. `gd.z` carries a scale applied before the max, which is
/// where attention folds in its `1/sqrt(dim_head)`.
pub fn softmax() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@group(0) @binding(2) var<uniform> gd: vec4<u32>;  // rows, cols, scale_bits, _

const THREADS: u32 = {threads}u;
const ROW_GRID_X: u32 = {grid_x}u;
var<workgroup> scratch: array<f32, {threads}>;

@compute @workgroup_size({threads})
fn softmax(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let row = wid.x + wid.y * ROW_GRID_X;
    if (row >= gd.x) {{ return; }}
    let cols = gd.y;
    let base = row * cols;
    let scale = bitcast<f32>(gd.z);

    // Pass 1: row max of the scaled scores.
    var local_max = -3.402823e38;
    for (var i = lid.x; i < cols; i = i + THREADS) {{
        local_max = max(local_max, X[base + i] * scale);
    }}
    scratch[lid.x] = local_max;
    workgroupBarrier();
    for (var step = THREADS / 2u; step > 0u; step = step / 2u) {{
        if (lid.x < step) {{
            scratch[lid.x] = max(scratch[lid.x], scratch[lid.x + step]);
        }}
        workgroupBarrier();
    }}
    let row_max = scratch[0];
    // See `softmax_in_place`: slot 0 is read by every thread and then overwritten
    // by the next pass, so the read has to be separated from the write.
    workgroupBarrier();

    // Pass 2: sum of exps.
    var local_sum = 0.0;
    for (var i = lid.x; i < cols; i = i + THREADS) {{
        local_sum = local_sum + exp(X[base + i] * scale - row_max);
    }}
    scratch[lid.x] = local_sum;
    workgroupBarrier();
    for (var step = THREADS / 2u; step > 0u; step = step / 2u) {{
        if (lid.x < step) {{
            scratch[lid.x] = scratch[lid.x] + scratch[lid.x + step];
        }}
        workgroupBarrier();
    }}
    let total = scratch[0];

    // Pass 3: normalise.
    let inv = select(0.0, 1.0 / total, total > 0.0);
    for (var i = lid.x; i < cols; i = i + THREADS) {{
        Out[base + i] = exp(X[base + i] * scale - row_max) * inv;
    }}
}}
"#,
        threads = ROW_THREADS,
        grid_x = ROW_GRID_X
    )
}

/// Rows wider than `THREADS * SOFTMAX_SLOTS` do not fit the register cache and
/// are rejected by the op layer rather than silently dropping their tails.
pub const SOFTMAX_SLOTS: u32 = 13;

/// Softmax over the last axis, in place, with a scale folded in.
///
/// A single `read_write` binding rather than a separate input and output: wgpu
/// tracks buffer usage per dispatch, and binding one buffer as both a read-only
/// and a read-write storage binding is a validation error. Attention wants the
/// softmax applied to the score buffer it already owns, so the shader declares
/// the one buffer it actually touches.
///
/// The scaled row is cached in registers on first read. The three passes used
/// to hit global memory three times each, and at 42 dispatches per chunk the
/// score tensor was being streamed ~9 GB per forward -- 2.9x the minimum. The
/// element assignment `i = lid.x + k*THREADS` is unchanged from the looped
/// form, so every partial sum folds in the same order and the output is
/// bit-identical; the exps are computed once and reused for the
/// normalisation, which is exact because `exp` is deterministic.
///
/// `SLOTS` must cover `ceil(max_cols / THREADS)`; the op layer rejects rows
/// wider than that rather than silently dropping their tails.
///
/// `ROWS_PER_WG` rows are reduced per workgroup, one 64-lane group each. It was
/// tried at 4 to amortise the barrier cost across the frequency attention's
/// narrow (64-column) rows and measured *neutral to worse*: those rows got 6%
/// slower (1.28 ms to 1.37 ms per dispatch) and the time axis' wide ones did not
/// move, because what limits the narrow case is barrier *latency*, which sharing
/// a barrier does not reduce. It is 1: one row per workgroup.
pub fn softmax_in_place() -> String {
    const SLOTS: u32 = SOFTMAX_SLOTS;
    /// Lanes per row, and rows per workgroup.
    const LANES: u32 = 64;
    const ROWS_PER_WG: u32 = 1;
    const THREADS_N: u32 = LANES * ROWS_PER_WG;

    let mut load = String::new();
    let mut max_fold = String::new();
    let mut exp_fold = String::new();
    let mut store = String::new();
    for k in 0..SLOTS {
        load.push_str(&format!(
            "    let s{k} = select(-3.402823e38, X[base + lane + {k}u * LANES] * scale,              lane + {k}u * LANES < cols);\n"
        ));
        max_fold.push_str(&format!("    local_max = max(local_max, s{k});\n"));
        exp_fold.push_str(&format!(
            "    let e{k} = select(0.0, exp(s{k} - row_max), lane + {k}u * LANES < cols);\n                 local_sum = local_sum + e{k};\n"
        ));
        store.push_str(&format!(
            "    if (lane + {k}u * LANES < pitch) {{\n                     X[base + lane + {k}u * LANES] =              select(0.0, exp(s{k} - row_max) * inv, lane + {k}u * LANES < cols);\n                 }}\n"
        ));
    }

    format!(
        r#"
@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<uniform> gd: vec4<u32>;  // rows, cols, scale_bits, pitch

const THREADS: u32 = {threads_n}u;
const LANES: u32 = {lanes}u;
const ROWS_PER_WG: u32 = {rows_per_wg}u;
const ROW_GRID_X: u32 = {grid_x}u;
var<workgroup> scratch: array<f32, {threads_n}>;

@compute @workgroup_size({threads_n})
fn softmax_in_place(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{
    let cols = gd.y;
    let pitch = gd.w;
    let scale = bitcast<f32>(gd.z);
    // Four rows per workgroup, so the barriers below are shared. The row is
    // clamped rather than skipped: an early return would take this invocation
    // out of those barriers, and only a whole workgroup may leave them.
    let group = lid.x / LANES;
    let lane = lid.x % LANES;
    let row = (wid.x + wid.y * ROW_GRID_X) * ROWS_PER_WG + group;
    let safe_row = min(row, gd.x - 1u);
    let base = safe_row * pitch;

    // Read the scaled row once into registers.
{load}
    // Pass 1: row max of the scaled scores.
    var local_max = -3.402823e38;
{max_fold}
    scratch[lid.x] = local_max;
    workgroupBarrier();
    for (var step = LANES / 2u; step > 0u; step = step / 2u) {{
        if (lane < step) {{
            scratch[lid.x] = max(scratch[lid.x], scratch[lid.x + step]);
        }}
        workgroupBarrier();
    }}
    let row_max = scratch[group * LANES];
    // See `softmax` above: the reduction result is read by every lane and the
    // next pass overwrites the slot, so the two are separated by a barrier.
    workgroupBarrier();

    // Pass 2: sum of exps, computed once and kept for the normalisation.
    var local_sum = 0.0;
{exp_fold}
    scratch[lid.x] = local_sum;
    workgroupBarrier();
    for (var step = LANES / 2u; step > 0u; step = step / 2u) {{
        if (lane < step) {{
            scratch[lid.x] = scratch[lid.x] + scratch[lid.x + step];
        }}
        workgroupBarrier();
    }}
    let total = scratch[group * LANES];
    let inv = select(0.0, 1.0 / total, total > 0.0);

    // Pass 3: normalise from the cached exps; the columns past `cols` are the
    // row pitch's zero padding, cleared so the AV product's K reads zeros.
    if (row < gd.x) {{
{store}    }}
}}
"#,
        threads_n = THREADS_N,
        lanes = LANES,
        rows_per_wg = ROWS_PER_WG,
        grid_x = ROW_GRID_X,
        load = load,
        max_fold = max_fold,
        exp_fold = exp_fold,
        store = store,
    )
}

/// Rows one softmax workgroup covers.
pub const SOFTMAX_ROWS_PER_WG: usize = 1;
/// Lanes per softmax row, i.e. the register cache's width.
pub const SOFTMAX_LANES: usize = 64;

/// One warp per row, reduced with subgroup shuffles instead of a tree.
///
/// The tree above costs six barriers per reduction and the frequency attention's
/// rows are 60 columns wide, so those rows spend almost all their time waiting
/// on barriers rather than doing arithmetic: 24 dispatches of 1.64 ms over
/// 384 480 rows is ~120 GB/s on a link that will do ~190. A warp's shuffle
/// reduction is five instructions and no barrier at all.
///
/// Only valid for rows of at most 64 columns (two slots per lane) and a pitch
/// the two slots cover; the caller picks it on `shuffle_reduction` and falls
/// back to the tree otherwise. The max is exact either way; the sum's
/// summation order changes with the reduction shape, which moves the
/// probabilities by ~1e-7 relative -- the same order the attention tests
/// already accept.
pub fn softmax_warp() -> String {
    /// Lanes per row, and rows per workgroup.
    const LANES: u32 = 32;
    const WARPS: u32 = 8;
    const THREADS: u32 = LANES * WARPS;

    format!(
        r#"
@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<uniform> gd: vec4<u32>;  // rows, cols, scale_bits, pitch

const LANES: u32 = {lanes}u;
const WARPS: u32 = {warps}u;
const ROW_GRID_X: u32 = {grid_x}u;

@compute @workgroup_size({threads})
fn softmax_in_place(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{
    let cols = gd.y;
    let pitch = gd.w;
    let scale = bitcast<f32>(gd.z);
    let warp = lid.x / LANES;
    let lane = lid.x % LANES;
    let row = (wid.x + wid.y * ROW_GRID_X) * WARPS + warp;
    // Clamped rather than skipped: the store is what is guarded, and a warp
    // that returned here would leave its row's padding stale.
    let safe_row = min(row, gd.x - 1u);
    let base = safe_row * pitch;

    let s0 = select(-3.402823e38, X[base + lane] * scale, lane < cols);
    let s1 = select(-3.402823e38, X[base + lane + LANES] * scale, lane + LANES < cols);

    var local_max = max(s0, s1);
    let row_max = subgroupMax(local_max);

    let e0 = select(0.0, exp(s0 - row_max), lane < cols);
    let e1 = select(0.0, exp(s1 - row_max), lane + LANES < cols);
    // `subgroupAdd` folds in the calling lane's own value, so adding it back
    // again would inflate the row sum by exactly one lane's share.
    let local_sum = e0 + e1;
    let total = subgroupAdd(local_sum);
    let inv = select(0.0, 1.0 / total, total > 0.0);

    if (row < gd.x) {{
        if (lane < pitch) {{
            X[base + lane] = select(0.0, e0 * inv, lane < cols);
        }}
        if (lane + LANES < pitch) {{
            X[base + lane + LANES] = select(0.0, e1 * inv, lane + LANES < cols);
        }}
    }}
}}
"#,
        lanes = LANES,
        warps = WARPS,
        threads = THREADS,
        grid_x = ROW_GRID_X,
    )
}

/// Out-of-place scaled softmax, one warp per row, any column count.
///
/// The in-place warp kernel only covers 64 columns (two loads per lane).
/// Attention's scores are thousands wide, so this walks the row in `LANES`
/// steps and uses the same shuffle reductions. No workgroup barrier: each
/// warp is an independent row.
pub fn softmax_scaled_warp() -> String {
    const LANES: u32 = 32;
    const WARPS: u32 = 8;
    const THREADS: u32 = LANES * WARPS;
    format!(
        r#"
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@group(0) @binding(2) var<uniform> gd: vec4<u32>;  // rows, cols, scale_bits, _

const LANES: u32 = {lanes}u;
const WARPS: u32 = {warps}u;
const ROW_GRID_X: u32 = {grid_x}u;

@compute @workgroup_size({threads})
fn softmax(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let cols = gd.y;
    let scale = bitcast<f32>(gd.z);
    let warp = lid.x / LANES;
    let lane = lid.x % LANES;
    let row = (wid.x + wid.y * ROW_GRID_X) * WARPS + warp;
    let safe_row = min(row, gd.x - 1u);
    let base = safe_row * cols;

    // Two passes, vec4 along the row: each lane owns 4 consecutive columns
    // and steps by 128. HTDemucs freq softmax is 2688-wide (exact 21 steps).
    var local_max = -3.402823e38;
    var local_sum = 0.0;
    let stride = LANES * 4u;
    var i = lane * 4u;
    loop {{
        if (i + 4u > cols) {{ break; }}
        let x0 = X[base + i] * scale;
        let x1 = X[base + i + 1u] * scale;
        let x2 = X[base + i + 2u] * scale;
        let x3 = X[base + i + 3u] * scale;
        let m2 = max(local_max, max(max(x0, x1), max(x2, x3)));
        let e = exp(local_max - m2);
        local_sum = local_sum * e + exp(x0 - m2) + exp(x1 - m2) + exp(x2 - m2) + exp(x3 - m2);
        local_max = m2;
        i = i + stride;
    }}
    loop {{
        if (i >= cols) {{ break; }}
        let x = X[base + i] * scale;
        let m2 = max(local_max, x);
        local_sum = local_sum * exp(local_max - m2) + exp(x - m2);
        local_max = m2;
        i = i + 1u;
    }}
    let row_max = subgroupMax(local_max);
    local_sum = local_sum * exp(local_max - row_max);
    let total = subgroupAdd(local_sum);
    let inv = select(0.0, 1.0 / total, total > 0.0);
    if (row < gd.x) {{
        var j = lane * 4u;
        loop {{
            if (j + 4u > cols) {{ break; }}
            Out[base + j] = exp(X[base + j] * scale - row_max) * inv;
            Out[base + j + 1u] = exp(X[base + j + 1u] * scale - row_max) * inv;
            Out[base + j + 2u] = exp(X[base + j + 2u] * scale - row_max) * inv;
            Out[base + j + 3u] = exp(X[base + j + 3u] * scale - row_max) * inv;
            j = j + stride;
        }}
        loop {{
            if (j >= cols) {{ break; }}
            Out[base + j] = exp(X[base + j] * scale - row_max) * inv;
            j = j + 1u;
        }}
    }}
}}
"#,
        lanes = LANES,
        warps = WARPS,
        threads = THREADS,
        grid_x = ROW_GRID_X,
    )
}

/// Rows one warp-per-row softmax workgroup covers.
pub const SOFTMAX_WARP_ROWS_PER_WG: usize = 8;

/// The same row scan as [`softmax_scaled_warp`], but it writes the two numbers
/// the probabilities need instead of the probabilities themselves: `(max, 1/Σexp)`
/// per row, as `vec2<f32>`.
///
/// The scaled-exponential form is what `gemm_batched_bn64_row_exp` applies while
/// it stages its `A` operand, so the `(heads·tokens, tokens)` matrix is never
/// read twice nor written at all — on the 7.8 s segment that is 462 MB per
/// attention call. Same max, same sum, same `exp`: the values the GEMM forms are
/// the ones the two-pass kernel would have written.
pub fn softmax_stats_warp() -> String {
    const LANES: u32 = 32;
    const WARPS: u32 = 8;
    const THREADS: u32 = LANES * WARPS;
    format!(
        r#"
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Stats: array<f32>;  // 2 per row: max, 1/sum
@group(0) @binding(2) var<uniform> gd: vec4<u32>;  // rows, cols, scale bits, _

const LANES: u32 = {lanes}u;
const WARPS: u32 = {warps}u;
const THREADS: u32 = {threads}u;
const ROW_GRID_X: u32 = {grid_x}u;

@compute @workgroup_size({threads})
fn softmax_stats(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let cols = gd.y;
    let scale = bitcast<f32>(gd.z);
    let warp = lid.x / LANES;
    let lane = lid.x % LANES;
    let row = (wid.x + wid.y * ROW_GRID_X) * WARPS + warp;
    let safe_row = min(row, gd.x - 1u);
    let base = safe_row * cols;

    // Two passes, vec4 along the row: each lane owns 4 consecutive columns
    // and steps by 128. HTDemucs freq softmax is 2688-wide (exact 21 steps).
    var local_max = -3.402823e38;
    var local_sum = 0.0;
    let stride = LANES * 4u;
    var i = lane * 4u;
    loop {{
        if (i + 4u > cols) {{ break; }}
        let x0 = X[base + i] * scale;
        let x1 = X[base + i + 1u] * scale;
        let x2 = X[base + i + 2u] * scale;
        let x3 = X[base + i + 3u] * scale;
        let m2 = max(local_max, max(max(x0, x1), max(x2, x3)));
        let e = exp(local_max - m2);
        local_sum = local_sum * e + exp(x0 - m2) + exp(x1 - m2) + exp(x2 - m2) + exp(x3 - m2);
        local_max = m2;
        i = i + stride;
    }}
    loop {{
        if (i >= cols) {{ break; }}
        let x = X[base + i] * scale;
        let m2 = max(local_max, x);
        local_sum = local_sum * exp(local_max - m2) + exp(x - m2);
        local_max = m2;
        i = i + 1u;
    }}
    let row_max = subgroupMax(local_max);
    local_sum = local_sum * exp(local_max - row_max);
    let total = subgroupAdd(local_sum);
    let inv = select(0.0, 1.0 / total, total > 0.0);
    // The two reductions leave both values warp-uniform, so one lane stores.
    if (row < gd.x && lane == 0u) {{
        Stats[safe_row * 2u] = row_max;
        Stats[safe_row * 2u + 1u] = inv;
    }}
}}
"#,
        lanes = LANES,
        warps = WARPS,
        threads = THREADS,
        grid_x = ROW_GRID_X,
    )
}
/// Columns one lane of the warp softmax covers, i.e. the widest row it takes.
pub const SOFTMAX_WARP_COLS: usize = 64;

/// Fused attention for `dim_head == 64`: `softmax(Q Kᵀ √d⁻¹ V` in one dispatch,
/// never materialising the score matrix.
///
/// One workgroup takes a 32-row block of Q for one (batch, head) pair —32
/// threads, one row each —and walks the whole sequence in K/V tiles. Two
/// passes over the K tiles keep the arithmetic as close as possible to the
/// unfused path: pass A finds each row's global max, pass B re-dots, folds the
/// running `Σ exp(s−m)` and the unnormalised `P·V` accumulation together, and
/// the division by the row sum happens only at the very end, where the unfused
/// path's softmax also normalises. What differs from the unfused path is the
/// summation *order* of the dots and of `P·V` —f32 rounding differences around
/// 1e-7 relative, which the alignment tests measure directly.
///
/// All three per-row arrays —the Q tile, the K tile, and the row's output
/// accumulator —live in shared memory, each thread owning one row. That last
/// one is load-bearing for a non-obvious reason: the accumulator started as 64
/// named scalars in registers, and the driver compiled them into per-thread
/// local memory instead. A full-chunk dispatch then ran 17× slower than the
/// FLOP count allows (0.9 s against ~50 ms) and its local-memory allocation
/// silently reset the device inside the model path, where VRAM is already
/// tight. 32-row tiles keep every array in shared memory with room to spare.
///
/// The tile rows are padded to 65 words: a 64-word stride puts every thread's
/// row on the same banks, a 65-word stride spreads them. V is read straight
/// through L2 —the workgroups of one (batch, head) sit adjacent in the grid,
/// so its K and V rows stay hot across the Q-block workgroups that reuse them.
pub fn flash_attention() -> String {
    const TILE: u32 = 32;
    const PAD: u32 = 65;
    const D_HEAD: u32 = 64;
    const THREADS: u32 = 32;

    // The accumulator lives in shared memory, one 64-wide row per thread.
    let o_accum = "
                o_tile[row * 64u + d] += p * qkv[v_row + d];";

    format!(
        r#"
struct FlashDims {{
    seq: u32,
    row_stride: u32,   // 3 * inner
    inner: u32,        // heads * dim_head
    heads: u32,
    scale_bits: u32,
}};

@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> fd: FlashDims;

const TILE: u32 = {tile}u;
const PAD: u32 = {pad}u;
const D_HEAD: u32 = {d_head}u;

var<workgroup> q_tile: array<f32, TILE * PAD>;
var<workgroup> k_tile: array<f32, TILE * PAD>;
var<workgroup> o_tile: array<f32, TILE * D_HEAD>;

@compute @workgroup_size({threads}u)
fn flash_attention(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{
    // One workgroup is one (batch, head) pair's 32-row Q block: wid.x selects
    // the pair, wid.y the block, and each thread owns one row of the tile.
    let bh = wid.x;
    let qb = wid.y;
    let b = bh / fd.heads;
    let h = bh % fd.heads;
    let q0 = qb * TILE;
    let q_rows = min(TILE, fd.seq - q0);
    let row = lid.x;
    let scale = bitcast<f32>(fd.scale_bits);

    let i = q0 + row;
    let q_valid = row < q_rows;
    // Clamp the row index so an invalid row's loads land inside the buffer;
    // its accumulator is never written out.
    let i_safe = min(i, fd.seq - 1u);
    let q_base = (b * fd.seq + i_safe) * fd.row_stride + h * D_HEAD;
    let k_head = b * fd.seq * fd.row_stride + fd.inner + h * D_HEAD;
    let v_head = b * fd.seq * fd.row_stride + 2u * fd.inner + h * D_HEAD;

    for (var d = 0u; d < D_HEAD; d += 1u) {{
        q_tile[row * PAD + d] = select(0.0, qkv[q_base + d], q_valid);
    }}
    workgroupBarrier();

    let kv_blocks = (fd.seq + TILE - 1u) / TILE;

    // Pass A: the global row max of the scaled scores.
    var m = -3.402823e38;
    for (var kb = 0u; kb < kv_blocks; kb += 1u) {{
        let j0 = kb * TILE;
        let kv_rows = min(TILE, fd.seq - j0);
        let j_safe = j0 + min(row, kv_rows - 1u);
        let j_valid = row < kv_rows;
        let k_row = k_head + j_safe * fd.row_stride;
        for (var d = 0u; d < D_HEAD; d += 1u) {{
            k_tile[row * PAD + d] = select(0.0, qkv[k_row + d], j_valid);
        }}
        workgroupBarrier();
        if (q_valid) {{
            for (var j2 = 0u; j2 < kv_rows; j2 += 1u) {{
                var s = 0.0;
                for (var d = 0u; d < D_HEAD; d += 1u) {{
                    s += q_tile[row * PAD + d] * k_tile[j2 * PAD + d];
                }}
                m = max(m, s * scale);
            }}
        }}
        workgroupBarrier();
    }}

    // Pass B: unnormalised P·V with the row max; the row sum of exp folds in
    // on the fly, and the single division lands at the end.
    var l = 0.0;
    for (var d = 0u; d < D_HEAD; d += 1u) {{
        o_tile[row * D_HEAD + d] = 0.0;
    }}
    for (var kb = 0u; kb < kv_blocks; kb += 1u) {{
        let j0 = kb * TILE;
        let kv_rows = min(TILE, fd.seq - j0);
        let j_safe = j0 + min(row, kv_rows - 1u);
        let j_valid = row < kv_rows;
        let k_row = k_head + j_safe * fd.row_stride;
        for (var d = 0u; d < D_HEAD; d += 1u) {{
            k_tile[row * PAD + d] = select(0.0, qkv[k_row + d], j_valid);
        }}
        workgroupBarrier();
        if (q_valid) {{
            for (var j2 = 0u; j2 < kv_rows; j2 += 1u) {{
                var s = 0.0;
                for (var d = 0u; d < D_HEAD; d += 1u) {{
                    s += q_tile[row * PAD + d] * k_tile[j2 * PAD + d];
                }}
                let p = exp(s * scale - m);
                l += p;
                let v_row = v_head + (j0 + j2) * fd.row_stride;
                for (var d = 0u; d < D_HEAD; d += 1u) {{
{o_accum}
                }}
            }}
        }}
        workgroupBarrier();
    }}

    if (q_valid) {{
        let inv_l = 1.0 / l;
        let o_row = (b * fd.seq + i) * fd.inner + h * D_HEAD;
        for (var d = 0u; d < D_HEAD; d += 1u) {{
            out[o_row + d] = o_tile[row * D_HEAD + d] * inv_l;
        }}
    }}
}}
"#,
        tile = TILE,
        pad = PAD,
        d_head = D_HEAD,
        threads = THREADS,
    )
}

/// `out *= source`, elementwise, both the same shape.
///
/// The decoding path of DTTNet multiplies each upsampled block by the encoder
/// feature of the matching resolution (`x = x * skip`), which is the one
/// elementwise combine `add_in_place` does not cover. Note this is a multiply,
/// not an add: the reference's decoder has no skip *connection* there, it gates
/// the upsampled tensor by the encoder's, and an `add` substituted for it would
/// change the model into one that does not exist.
pub fn mul_in_place() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read_write> Out: array<f32>;
@group(0) @binding(1) var<storage, read> Source: array<f32>;
@group(0) @binding(2) var<uniform> gd: vec4<u32>;  // count, _, _, _

const GRID_X: u32 = 65535u;

@compute @workgroup_size({threads})
fn mul_in_place(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let i = (wid.y * GRID_X + wid.x) * {threads}u + lid.x;
    if (i >= gd.x) {{ return; }}
    Out[i] = Out[i] * Source[i];
}}
"#,
        threads = ROW_THREADS
    )
}

/// Writes zeros over `gd.x` elements.
///
/// A device-side clear, because the host-side one is not free at these sizes:
/// `Arena::clear` goes through `Queue::write_buffer`, which is a host-to-device
/// copy at ~520 MB/s, so zeroing a 600 MB patch scratch that way costs over a
/// second — a thousand times what the same work costs on the device. The conv
/// workspace is sized in hundreds of megabytes, so every clear in the forward
/// pass has to be a dispatch.
pub fn fill_zero() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read_write> Out: array<f32>;
@group(0) @binding(1) var<uniform> gd: vec4<u32>;  // count, _, _, _

const GRID_X: u32 = 65535u;

@compute @workgroup_size({threads})
fn fill_zero(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let i = (wid.y * GRID_X + wid.x) * {threads}u + lid.x;
    if (i >= gd.x) {{ return; }}
    Out[i] = 0.0;
}}
"#,
        threads = ROW_THREADS
    )
}

/// Copies a `(batch, rows, cols)` block from one row pitch to another.
///
/// This exists because a GEMM operand cannot be a tight activation: the kernel
/// stages whole tiles, so `A`'s row count and `B`'s column count have to be
/// multiples of the tile and every read inside a tile has to land in a buffer.
/// The cheap way to satisfy that is to have the producer write the padded layout
/// in the first place — which is what the roformer does, since every one of its
/// operands is produced by another kernel. A `ConvTranspose2d`'s input is not:
/// it is either the uploaded activation or the output of a convolution whose
/// layout the *next* layer wants tight, so it needs a copy either way. One pass
/// over a few tens of megabytes at device bandwidth costs microseconds; asking
/// every producer to emit both layouts would cost every consumer a branch.
///
/// `gd` is `(batch, rows, cols, src_pitch)`, `ge` is `(dst_rows, dst_pitch, _, _)`.
/// Only `cols` elements of each row are written: the destination's margins stay
/// whatever the caller left there, and the caller is expected to have cleared
/// them (see [`Kernels::copy_pitched_into`](crate::gpu::kernels::Kernels::copy_pitched_into)
/// for why a conv-transpose does not have to).
/// `(batch, src_rows, cols) -> (batch, dst_rows, cols)`, keeping
/// `row_offset .. row_offset + dst_rows` of each batch's rows. The frequency
/// branch's transposed-conv crop: the kept rows sit `row_offset` in, and the
/// batch's own stride is independent of `dst_rows`.
pub fn crop_rows() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@group(0) @binding(2) var<uniform> gd: vec4<u32>;  // src_batch_stride, dst_rows, cols, row_offset
@group(0) @binding(3) var<uniform> ge: vec4<u32>;  // batch, _, _, _

const GRID_X: u32 = 65535u;

@compute @workgroup_size(256)
fn crop_rows(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let total = ge.x * gd.y * gd.z;
    let i = (wid.y * GRID_X + wid.x) * 256u + lid.x;
    if (i >= total) {{ return; }}

    let col = i % gd.z;
    let row = (i / gd.z) % gd.y;
    let batch = i / (gd.y * gd.z);

    Out[i] = X[batch * gd.x + (row + gd.w) * gd.z + col];
}}
"#
    )
}

pub fn copy_pitched() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@group(0) @binding(2) var<uniform> gd: vec4<u32>;  // batch, rows, cols, src_pitch
@group(0) @binding(3) var<uniform> ge: vec4<u32>;  // dst_rows, dst_pitch, _, _

const GRID_X: u32 = 65535u;

@compute @workgroup_size({threads})
fn copy_pitched(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let rows = gd.y;
    let cols = gd.z;
    let total = gd.x * rows * cols;
    let i = (wid.y * GRID_X + wid.x) * {threads}u + lid.x;
    if (i >= total) {{ return; }}

    let per_batch = rows * cols;
    let batch = i / per_batch;
    let rest = i % per_batch;
    let row = rest / cols;
    let col = rest % cols;

    Out[(batch * ge.x + row) * ge.y + col] = X[(batch * rows + row) * gd.w + col];
}}
"#,
        threads = ROW_THREADS
    )
}

/// Group normalisation over the channel axis of a row, one workgroup per
/// `(row, group)` pair.
///
/// Layout is described by strides rather than assumed: the DTTNet band sequence
/// normalises `(channels = n_features, len = sequence)` where the sequence axis
/// is *not* contiguous — its elements are `per_head` apart, because the channels
/// are the innermost run. Passing `len_stride` and `channel_stride` keeps that
/// out of the model code; a contiguous `(rows, channels, len)` tensor is just the
/// `channel_stride = 1, len_stride = len` case.
///
/// The statistics are the host's: `mean = sum / n` and
/// `var = max(sum_sq / n - mean^2, 0)`, then `(x - mean) / sqrt(var + eps) *
/// gamma + beta`. That is `nn.GroupNorm` in eval mode, whose `eps` is 1e-5 by
/// default — passed in rather than baked in, because a checkpoint that overrode
/// it would otherwise normalise differently by ~1e-4.
///
/// One workgroup per `(row, group)` also means the reduction is over a single
/// contiguous run per thread and no cross-workgroup combine is needed: the
/// slice's element count is at most a few thousand here, which is why this can
/// afford the shared-memory tree rather than the warp shuffle.
///
/// `gd` is `(groups, per_group, len, row_stride)`, `ge` is
/// `(rows, len_stride, channel_stride, eps_bits)`.
pub fn group_norm() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Gamma: array<f32>;
@group(0) @binding(2) var<storage, read> Beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> Out: array<f32>;
@group(0) @binding(4) var<uniform> gd: vec4<u32>;
@group(0) @binding(5) var<uniform> ge: vec4<u32>;

const THREADS: u32 = {threads}u;
const ROW_GRID_X: u32 = {grid_x}u;
var<workgroup> partial_sum: array<f32, {threads}>;
var<workgroup> partial_sq: array<f32, {threads}>;

fn group_norm_index(local: u32, len: u32, len_stride: u32, channel_stride: u32) -> u32 {{
    return (local / len) * channel_stride + (local % len) * len_stride;
}}

@compute @workgroup_size({threads})
fn group_norm(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let pair = wid.x + wid.y * ROW_GRID_X;
    let rows = ge.x;
    let groups = gd.x;
    if (pair >= rows * groups) {{ return; }}
    let row = pair / groups;
    let group = pair % groups;

    let per_group = gd.y;
    let len = gd.z;
    let row_stride = gd.w;
    let len_stride = ge.y;
    let channel_stride = ge.z;
    let eps = bitcast<f32>(ge.w);

    let group_size = per_group * len;
    // The group's first channel is `group * per_group`, and channels are
    // `channel_stride` apart inside a row.
    let base = row * row_stride + group * per_group * channel_stride;

    var sum = 0.0;
    var sum_sq = 0.0;
    for (var local = lid.x; local < group_size; local = local + THREADS) {{
        let v = X[base + group_norm_index(local, len, len_stride, channel_stride)];
        sum = sum + v;
        sum_sq = fma(v, v, sum_sq);
    }}
    partial_sum[lid.x] = sum;
    partial_sq[lid.x] = sum_sq;
    workgroupBarrier();
    for (var step = THREADS / 2u; step > 0u; step = step / 2u) {{
        if (lid.x < step) {{
            partial_sum[lid.x] = partial_sum[lid.x] + partial_sum[lid.x + step];
            partial_sq[lid.x] = partial_sq[lid.x] + partial_sq[lid.x + step];
        }}
        workgroupBarrier();
    }}

    let n = f32(group_size);
    let mean = partial_sum[0] / n;
    let variance = max(partial_sq[0] / n - mean * mean, 0.0);
    let scale = 1.0 / sqrt(variance + eps);

    for (var local = lid.x; local < group_size; local = local + THREADS) {{
        let offset = base + group_norm_index(local, len, len_stride, channel_stride);
        let channel = group * per_group + local / len;
        Out[offset] = (X[offset] - mean) * scale * Gamma[channel] + Beta[channel];
    }}
}}
"#,
        threads = ROW_THREADS,
        grid_x = ROW_GRID_X
    )
}

/// The split form of `group_norm`, for slices far too big for one workgroup.
///
/// The single-workgroup form is right for the DConv's few-hundred-element
/// slices, but the waveform branch's DConv normalises `(1, 96, 85995)` — 8.2
/// million elements — as one workgroup, which is 8 warps against the whole GPU
/// and measured 23.7 ms for one call (the host does the same statistics in
/// under a millisecond). So a slice larger than a few thousand elements is cut
/// into `segments` blocks, one workgroup each, in two dispatches: this one
/// reduces its block into a per-slice slot, `group_norm_apply` combines the
/// `segments` slots of its slice and normalises its block.
///
/// The slots are written, never accumulated into, so the partials buffer needs
/// no clearing, and the second dispatch's ordering against the first is the
/// submission order — same command encoder.
///
/// `gf` is `(segments, segment_len, _, _)`; the workgroup index is
/// `pair * segments + segment`.
pub fn group_norm_partial() -> String {
    format!(
        r#"
// Only the bindings this entry point actually reads or writes: wgpu's
// automatic layout is derived from the entry point's use, so a declared but
// unused binding is not in the layout and a bind group carrying it as well is
// rejected as invalid.
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<uniform> gd: vec4<u32>;  // groups, per_group, len, row_stride
@group(0) @binding(2) var<uniform> ge: vec4<u32>;  // rows, len_stride, channel_stride, eps_bits
@group(0) @binding(3) var<uniform> gf: vec4<u32>;  // segments, segment_len, _, _
@group(0) @binding(4) var<storage, read_write> Partials: array<f32>;  // 2 per (pair, segment)

const THREADS: u32 = {threads}u;
const ROW_GRID_X: u32 = {grid_x}u;
var<workgroup> partial_sum: array<f32, {threads}>;
var<workgroup> partial_sq: array<f32, {threads}>;

fn group_norm_index(local: u32, len: u32, len_stride: u32, channel_stride: u32) -> u32 {{
    return (local / len) * channel_stride + (local % len) * len_stride;
}}

@compute @workgroup_size({threads})
fn group_norm_partial(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let segments = gf.x;
    let segment_len = gf.y;
    let work = wid.x + wid.y * ROW_GRID_X;
    let pair = work / segments;
    let segment = work % segments;

    let rows = ge.x;
    let groups = gd.x;
    if (pair >= rows * groups) {{ return; }}
    let row = pair / groups;
    let group = pair % groups;

    let per_group = gd.y;
    let len = gd.z;
    let row_stride = gd.w;
    let len_stride = ge.y;
    let channel_stride = ge.z;

    let group_size = per_group * len;
    let base = row * row_stride + group * per_group * channel_stride;
    let start = segment * segment_len;
    let end = min(start + segment_len, group_size);

    var sum = 0.0;
    var sum_sq = 0.0;
    for (var local = start + lid.x; local < end; local = local + THREADS) {{
        let v = X[base + group_norm_index(local, len, len_stride, channel_stride)];
        sum = sum + v;
        sum_sq = fma(v, v, sum_sq);
    }}
    partial_sum[lid.x] = sum;
    partial_sq[lid.x] = sum_sq;
    workgroupBarrier();
    for (var step = THREADS / 2u; step > 0u; step = step / 2u) {{
        if (lid.x < step) {{
            partial_sum[lid.x] = partial_sum[lid.x] + partial_sum[lid.x + step];
            partial_sq[lid.x] = partial_sq[lid.x] + partial_sq[lid.x + step];
        }}
        workgroupBarrier();
    }}
    if (lid.x == 0u) {{
        let slot = work * 2u;
        Partials[slot] = partial_sum[0];
        Partials[slot + 1u] = partial_sq[0];
    }}
}}
"#,
        threads = ROW_THREADS,
        grid_x = ROW_GRID_X
    )
}

/// One thread per `(row, group)`: fold that pair's `segments` partials into
/// `(mean, scale)` with the same left-to-right sum the apply kernel used to
/// replay in every lane of every segment workgroup.
pub fn group_norm_combine() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read> Partials: array<f32>;
@group(0) @binding(1) var<storage, read_write> Stats: array<f32>;
@group(0) @binding(2) var<uniform> gd: vec4<u32>;
@group(0) @binding(3) var<uniform> ge: vec4<u32>;
@group(0) @binding(4) var<uniform> gf: vec4<u32>;  // segments, segment_len, _, _

const THREADS: u32 = {threads}u;
const ROW_GRID_X: u32 = {grid_x}u;

@compute @workgroup_size({threads})
fn group_norm_combine(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let segments = gf.x;
    let pair = (wid.x + wid.y * ROW_GRID_X) * THREADS + lid.x;
    let rows = ge.x;
    let groups = gd.x;
    if (pair >= rows * groups) {{ return; }}

    let per_group = gd.y;
    let len = gd.z;
    let eps = bitcast<f32>(ge.w);
    let group_size = per_group * len;

    var sum = 0.0;
    var sum_sq = 0.0;
    let first = pair * segments * 2u;
    for (var s = 0u; s < segments; s = s + 1u) {{
        sum = sum + Partials[first + s * 2u];
        sum_sq = sum_sq + Partials[first + s * 2u + 1u];
    }}
    let n = f32(group_size);
    let mean = sum / n;
    let variance = max(sum_sq / n - mean * mean, 0.0);
    Stats[pair * 2u] = mean;
    Stats[pair * 2u + 1u] = 1.0 / sqrt(variance + eps);
}}
"#,
        threads = ROW_THREADS,
        grid_x = ROW_GRID_X
    )
}

/// Apply using per-pair `(mean, scale)` from [`group_norm_combine`].
pub fn group_norm_apply() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Gamma: array<f32>;
@group(0) @binding(2) var<storage, read> Beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> Out: array<f32>;
@group(0) @binding(4) var<uniform> gd: vec4<u32>;
@group(0) @binding(5) var<uniform> ge: vec4<u32>;
@group(0) @binding(6) var<uniform> gf: vec4<u32>;
@group(0) @binding(7) var<storage, read> Stats: array<f32>;

const THREADS: u32 = {threads}u;
const ROW_GRID_X: u32 = {grid_x}u;

fn group_norm_index(local: u32, len: u32, len_stride: u32, channel_stride: u32) -> u32 {{
    return (local / len) * channel_stride + (local % len) * len_stride;
}}

@compute @workgroup_size({threads})
fn group_norm_apply(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let segments = gf.x;
    let segment_len = gf.y;
    let work = wid.x + wid.y * ROW_GRID_X;
    let pair = work / segments;
    let segment = work % segments;

    let rows = ge.x;
    let groups = gd.x;
    if (pair >= rows * groups) {{ return; }}
    let row = pair / groups;
    let group = pair % groups;

    let per_group = gd.y;
    let len = gd.z;
    let row_stride = gd.w;
    let len_stride = ge.y;
    let channel_stride = ge.z;

    let group_size = per_group * len;
    let base = row * row_stride + group * per_group * channel_stride;
    let mean = Stats[pair * 2u];
    let scale = Stats[pair * 2u + 1u];

    let start = segment * segment_len;
    let end = min(start + segment_len, group_size);
    // The transformer's position-wise norm stores `(token, channel)` with the
    // channels contiguous (`channel_stride == 1`) and the token axis exactly
    // `per_group` floats away — the generic walk below then puts consecutive
    // lanes `per_group` floats apart, so every lane pulls a 128-byte line to
    // use 4 bytes of it. When the two axes tile the group (channels fastest),
    // memory order *is* the element order: same elements, same arithmetic,
    // one line per lane. Measured on the 1.376M-element slices: 0.91 ms →
    // 0.06 ms per dispatch.
    if (channel_stride == 1u && len_stride == per_group) {{
        for (var local = start + lid.x; local < end; local = local + THREADS) {{
            let offset = base + local;
            let channel = group * per_group + local % per_group;
            Out[offset] = (X[offset] - mean) * scale * Gamma[channel] + Beta[channel];
        }}
    }} else {{
        for (var local = start + lid.x; local < end; local = local + THREADS) {{
            let offset = base + group_norm_index(local, len, len_stride, channel_stride);
            let channel = group * per_group + local / len;
            Out[offset] = (X[offset] - mean) * scale * Gamma[channel] + Beta[channel];
        }}
    }}
}}
"#,
        threads = ROW_THREADS,
        grid_x = ROW_GRID_X
    )
}

/// The original apply: every lane of every segment workgroup re-sums the
/// pair's partials. Kept so `DEMUCS_GN_COMBINE=0` can A/B the combine pass.
pub fn group_norm_apply_from_partials() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Gamma: array<f32>;
@group(0) @binding(2) var<storage, read> Beta: array<f32>;
@group(0) @binding(3) var<storage, read_write> Out: array<f32>;
@group(0) @binding(4) var<uniform> gd: vec4<u32>;
@group(0) @binding(5) var<uniform> ge: vec4<u32>;
@group(0) @binding(6) var<uniform> gf: vec4<u32>;
@group(0) @binding(7) var<storage, read> Partials: array<f32>;

const THREADS: u32 = {threads}u;
const ROW_GRID_X: u32 = {grid_x}u;

fn group_norm_index(local: u32, len: u32, len_stride: u32, channel_stride: u32) -> u32 {{
    return (local / len) * channel_stride + (local % len) * len_stride;
}}

@compute @workgroup_size({threads})
fn group_norm_apply(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let segments = gf.x;
    let segment_len = gf.y;
    let work = wid.x + wid.y * ROW_GRID_X;
    let pair = work / segments;
    let segment = work % segments;

    let rows = ge.x;
    let groups = gd.x;
    if (pair >= rows * groups) {{ return; }}
    let row = pair / groups;
    let group = pair % groups;

    let per_group = gd.y;
    let len = gd.z;
    let row_stride = gd.w;
    let len_stride = ge.y;
    let channel_stride = ge.z;
    let eps = bitcast<f32>(ge.w);

    let group_size = per_group * len;
    let base = row * row_stride + group * per_group * channel_stride;

    var sum = 0.0;
    var sum_sq = 0.0;
    let first = pair * segments * 2u;
    for (var s = 0u; s < segments; s = s + 1u) {{
        sum = sum + Partials[first + s * 2u];
        sum_sq = sum_sq + Partials[first + s * 2u + 1u];
    }}
    let n = f32(group_size);
    let mean = sum / n;
    let variance = max(sum_sq / n - mean * mean, 0.0);
    let scale = 1.0 / sqrt(variance + eps);

    let start = segment * segment_len;
    let end = min(start + segment_len, group_size);
    // See [`group_norm_apply`]: the same memory-order fast path, for the
    // small-segment case where the replay of the partials replaces the
    // combine dispatch.
    if (channel_stride == 1u && len_stride == per_group) {{
        for (var local = start + lid.x; local < end; local = local + THREADS) {{
            let offset = base + local;
            let channel = group * per_group + local % per_group;
            Out[offset] = (X[offset] - mean) * scale * Gamma[channel] + Beta[channel];
        }}
    }} else {{
        for (var local = start + lid.x; local < end; local = local + THREADS) {{
            let offset = base + group_norm_index(local, len, len_stride, channel_stride);
            let channel = group * per_group + local / len;
            Out[offset] = (X[offset] - mean) * scale * Gamma[channel] + Beta[channel];
        }}
    }}
}}
"#,
        threads = ROW_THREADS,
        grid_x = ROW_GRID_X
    )
}

/// One direction of a single-layer LSTM's recurrence.
///
/// The recurrence is sequential, so the only parallel work per row is the gate
/// projections — but each step's gates depend on the step before, so a
/// step-per-dispatch design pays a full dispatch (~0.15 ms on this card) for
/// ~40 us of arithmetic. DTTNet has 8 modules x 2 directions x up to 512 steps,
/// which is the ~4600 dispatches per chunk this exists to avoid. The `h`/`c`
/// state lives in workgroup shared memory and the whole time axis is walked
/// inside the shader: **one dispatch per (module, direction)**, not per step.
///
/// **`ROWS` rows per workgroup.** At DTTNet's `hidden = 96` the recurrent weight
/// matrix is 147 KB, larger than this card's 48 KB shared allotment, so it cannot
/// be staged once — it is re-read from global memory every step, and with one row
/// per workgroup that is 147 KB per row per step. Two rows per workgroup halve
/// it: the weight element a lane loads is used `ROWS` times before the next
/// load, exactly the reuse a GEMM tile gets from its accumulators. The
/// accumulators are unrolled in the generated source rather than indexed in a
/// loop, because a register array indexed by a loop variable lands in local
/// memory instead of in registers.
///
/// **`W_hh` is stored transposed, as `(hidden, 4 * hidden)`.** Every thread owns
/// one gate and walks `j`, so the natural layout (`(4 * hidden, hidden)`, one row
/// per gate) has consecutive lanes reading addresses `hidden` floats apart: 32
/// separate 32-byte sectors per warp instruction, 8x the bytes the values
/// occupy. Measured on DTTNet's bottleneck, that layout ran the recurrence at
/// 3.7 GFLOP/s — 75 ms per dispatch, 68% of the whole forward pass. With the
/// transpose, lane `g` and lane `g + 1` read adjacent floats and one warp
/// instruction is one 128-byte transaction.
///
/// The accumulation order matches the host (`acc = proj + b_hh`, then the
/// recurrent terms in `j` order, one gate per thread), so this stage differs from
/// the host only in the rounding of the fused multiply-add and in the
/// transcendental functions.
///
/// `gd` is `(hidden, time, reverse, out_pitch)`; `ge` is
/// `(out_offset, rows, grid_x, proj_pitch)`. Rows are covered `ROWS` at a time
/// and a partial last group is handled by the guards.
/// Rows one `lstm_recur` workgroup walks. Exported because the dispatch's grid is
/// `ceil(rows / ROWS)`, and a grid that disagrees with the shader's own grouping
/// would have workgroups compute rows nobody asked for.
pub const LSTM_ROWS: usize = 4;

pub fn lstm_recur() -> String {
    let rows_per_workgroup = LSTM_ROWS;
    // The gate loop and the update loop, both unrolled over `ROWS`.
    let mut zero = String::new();
    for r in 0..rows_per_workgroup {
        zero.push_str(&format!(
            "        h_state[{r}u * MAX_HIDDEN + i] = 0.0;
        c_state[{r}u * MAX_HIDDEN + i] = 0.0;
"
        ));
    }
    let mut acc_decl = String::new();
    let mut acc_load = String::new();
    let mut dispatch_gate = String::new();
    let mut gate_store = String::new();
    for r in 0..rows_per_workgroup {
        acc_decl.push_str(&format!("            var acc{r}: f32 = 0.0;
"));
        acc_load.push_str(&format!(
            "            if (row + {r}u < rows) {{ acc{r} = Proj[((row + {r}u) * time + t) * proj_pitch + gate] + B_hh[gate]; }}
"
        ));
        dispatch_gate.push_str(&format!(
            "            acc{r} = fma(w, h_state[{r}u * MAX_HIDDEN + j], acc{r});
"
        ));
        gate_store.push_str(&format!(
            "            gates[{r}u * gate_count + gate] = acc{r};
"
        ));
    }
    format!(
        r#"
@group(0) @binding(0) var<storage, read> Proj: array<f32>;
@group(0) @binding(1) var<storage, read> W_hh: array<f32>;
@group(0) @binding(2) var<storage, read> B_hh: array<f32>;
@group(0) @binding(3) var<storage, read_write> Out: array<f32>;
@group(0) @binding(4) var<uniform> gd: vec4<u32>;
@group(0) @binding(5) var<uniform> ge: vec4<u32>;

const THREADS: u32 = {threads}u;
const MAX_HIDDEN: u32 = {max_hidden}u;
const ROWS: u32 = {rows}u;

var<workgroup> h_state: array<f32, {state_slots}>;
var<workgroup> c_state: array<f32, {state_slots}>;
var<workgroup> gates: array<f32, {gate_slots}>;

fn sigmoid(x: f32) -> f32 {{
    return 1.0 / (1.0 + exp(-x));
}}

@compute @workgroup_size({threads})
fn lstm_recur(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let group = wid.x + wid.y * ge.z;
    let row = group * ROWS;
    let rows = ge.y;
    if (row >= rows) {{ return; }}

    let hidden = gd.x;
    let time = gd.y;
    let reverse = gd.z != 0u;
    let out_pitch = gd.w;
    let out_offset = ge.x;
    let proj_pitch = ge.w;
    let gate_count = 4u * hidden;

    for (var i = lid.x; i < ROWS * MAX_HIDDEN; i = i + THREADS) {{
{zap}    }}
    workgroupBarrier();

    for (var step = 0u; step < time; step = step + 1u) {{
        let t = select(step, time - 1u - step, reverse);

        var slot = 0u;
        loop {{
            let gate = lid.x + slot * THREADS;
            if (gate >= gate_count) {{ break; }}
{acc_decl}{acc_load}            for (var j = 0u; j < hidden; j = j + 1u) {{
                let w = W_hh[j * gate_count + gate];
{dispatch}            }}
{gate_store}            slot = slot + 1u;
        }}
        workgroupBarrier();

        // The state arrays are strided at `MAX_HIDDEN`, so this walks the same
        // layout the gate loop indexed.
        for (var i = lid.x; i < ROWS * MAX_HIDDEN; i = i + THREADS) {{
            let r = i / MAX_HIDDEN;
            let j = i % MAX_HIDDEN;
            if (j >= hidden || row + r >= rows) {{ continue; }}
            let base = r * gate_count + j;
            let input_gate = sigmoid(gates[base]);
            let forget_gate = sigmoid(gates[base + hidden]);
            let cell_gate = tanh(gates[base + 2u * hidden]);
            let output_gate = sigmoid(gates[base + 3u * hidden]);
            let c = forget_gate * c_state[i] + input_gate * cell_gate;
            c_state[i] = c;
            let h = output_gate * tanh(c);
            h_state[i] = h;
            Out[((row + r) * time + t) * out_pitch + out_offset + j] = h;
        }}
        workgroupBarrier();
    }}
}}
"#,
        threads = ROW_THREADS,
        max_hidden = LSTM_MAX_HIDDEN,
        rows = rows_per_workgroup,
        state_slots = LSTM_MAX_HIDDEN * rows_per_workgroup,
        gate_slots = LSTM_MAX_HIDDEN * 4 * rows_per_workgroup,
        zap = zero,
        acc_decl = acc_decl,
        acc_load = acc_load,
        dispatch = dispatch_gate,
        gate_store = gate_store,
    )
}

/// One direction of a single-layer LSTM's recurrence, **with the weights held in
/// registers**.
///
/// This is [`lstm_recur`] with the one thing that cost it removed. That kernel
/// re-reads `W_hh` from memory on every step — at `hidden = 96` the matrix is
/// 147 KB, larger than this card's 48 KB of shared memory, so it cannot be staged
/// — and that is 1152 warp-level loads per step per row-group. Measured, that is
/// ~14 ms per dispatch against a floor of ~1 ms for the same FMAs. The re-read is
/// *latency*-bound rather than bandwidth-bound (four rows per workgroup instead of
/// two halves the traffic and gains 8%), which is what says the fix is to remove
/// the loads rather than to overlap them better.
///
/// Each thread owns one gate's `hidden / 2` weights for the whole time axis and
/// pairs of threads split each gate's reduction. `THREADS = 8 * hidden` (two
/// halves of `4 * hidden` gates), the halves laid out contiguously so a warp stays
/// on one half and its `h_state` reads broadcast. The partials meet through shared
/// memory, the only cross-thread communication per step besides the barriers.
///
/// Two rows per workgroup, fixed: the partial sums travel as a `vec2`.
pub fn lstm_recur_regs(hidden: usize) -> String {
    assert!(
        hidden >= 4 && hidden % 2 == 0 && hidden <= LSTM_REG_MAX_HIDDEN,
        "the register LSTM is compiled for even hidden widths up to {LSTM_REG_MAX_HIDDEN},          got {hidden}"
    );
    let gates = 4 * hidden;
    let threads = 2 * gates;
    let wpt = hidden / 2;
    let mut load_weights = String::new();
    let mut dot = String::new();
    for i in 0..wpt {
        // The thread's `j` range starts at `j0`, so the weight it needs for step
        // `i` is row `j0 + i` of the transposed matrix. Everything is unrolled: an
        // array indexed by a loop variable lands in local memory rather than in
        // registers, the same trap the GEMM's accumulators document.
        load_weights.push_str(&format!(
            "    w[{i}u] = W_hh[(j0 + {i}u) * GATES + gate];
"
        ));
        dot.push_str(&format!(
            "        let h0_{i} = h_state[j0 + {i}u];
                     let h1_{i} = h_state[MAX_HIDDEN + j0 + {i}u];
                     acc0 = fma(w[{i}u], h0_{i}, acc0);
                     acc1 = fma(w[{i}u], h1_{i}, acc1);
"
        ));
    }
    format!(
        r##"
@group(0) @binding(0) var<storage, read> Proj: array<f32>;
@group(0) @binding(1) var<storage, read> W_hh: array<f32>;
@group(0) @binding(2) var<storage, read> B_hh: array<f32>;
@group(0) @binding(3) var<storage, read_write> Out: array<f32>;
@group(0) @binding(4) var<uniform> gd: vec4<u32>;
@group(0) @binding(5) var<uniform> ge: vec4<u32>;

const THREADS: u32 = {threads}u;
const MAX_HIDDEN: u32 = {max_hidden}u;
const GATES: u32 = {gate_count}u;
const WPT: u32 = {weights_per_thread}u;

var<workgroup> h_state: array<f32, {state_slots}>;
var<workgroup> c_state: array<f32, {state_slots}>;
var<workgroup> partials: array<vec2<f32>, {partial_slots}>;
var<workgroup> gates: array<f32, {gate_slots}>;

fn sigmoid(x: f32) -> f32 {{
    return 1.0 / (1.0 + exp(-x));
}}

@compute @workgroup_size({threads})
fn lstm_recur_regs(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let group = wid.x + wid.y * ge.z;
    let row = group * 2u;
    let rows = ge.y;
    if (row >= rows) {{ return; }}

    let hidden = gd.x;
    let time = gd.y;
    let reverse = gd.z != 0u;
    let out_pitch = gd.w;
    let out_offset = ge.x;
    let proj_pitch = ge.w;

    // Thread `t` owns gate `t % GATES` over the `j` range of half `t / GATES`.
    // Laying the halves out contiguously rather than interleaved keeps every lane
    // of a warp on the same half, so the `h_state` reads broadcast.
    let half = lid.x / GATES;
    let gate = lid.x % GATES;
    let j0 = half * WPT;

    // This thread's slice of the recurrent weights, resident for the whole time
    // axis. Loading it once is the entire point of this kernel.
    var w: array<f32, WPT>;
{load_weights}
    for (var i = lid.x; i < {state_slots}u; i = i + THREADS) {{
        h_state[i] = 0.0;
        c_state[i] = 0.0;
    }}
    workgroupBarrier();

    for (var step = 0u; step < time; step = step + 1u) {{
        let t = select(step, time - 1u - step, reverse);

        // Phase 1: each thread's partial dot product over its `WPT` weights, for
        // both rows at once.
        var acc0 = 0.0;
        var acc1 = 0.0;
{dot}
        partials[lid.x] = vec2<f32>(acc0, acc1);
        workgroupBarrier();

        // Phase 2: the lower half sums its gate's two partials and adds the input
        // projection and the recurrent bias — the host's order, where the project-
        // ed value and `b_hh` are what the accumulator starts from.
        if (half == 0u) {{
            let p = partials[gate] + partials[GATES + gate];
            var g0 = p.x;
            var g1 = p.y;
            if (row < rows) {{
                g0 = g0 + Proj[(row * time + t) * proj_pitch + gate] + B_hh[gate];
            }}
            if (row + 1u < rows) {{
                g1 = g1 + Proj[((row + 1u) * time + t) * proj_pitch + gate] + B_hh[gate];
            }}
            gates[gate] = g0;
            gates[GATES + gate] = g1;
        }}
        workgroupBarrier();

        // Phase 3: the cell update, one thread per (row, hidden unit).
        for (var i = lid.x; i < {state_slots}u; i = i + THREADS) {{
            let r = i / MAX_HIDDEN;
            let j = i % MAX_HIDDEN;
            if (j >= hidden || row + r >= rows) {{ continue; }}
            let base = r * GATES + j;
            let input_gate = sigmoid(gates[base]);
            let forget_gate = sigmoid(gates[base + hidden]);
            let cell_gate = tanh(gates[base + 2u * hidden]);
            let output_gate = sigmoid(gates[base + 3u * hidden]);
            let c = forget_gate * c_state[i] + input_gate * cell_gate;
            c_state[i] = c;
            let h = output_gate * tanh(c);
            h_state[i] = h;
            Out[((row + r) * time + t) * out_pitch + out_offset + j] = h;
        }}
        workgroupBarrier();
    }}
}}"##,
        threads = threads,
        max_hidden = LSTM_MAX_HIDDEN,
        gate_count = gates,
        weights_per_thread = wpt,
        state_slots = LSTM_MAX_HIDDEN * LSTM_REG_ROWS,
        partial_slots = threads,
        gate_slots = gates * LSTM_REG_ROWS,
        load_weights = load_weights,
        dot = dot,
    )
}

/// Rows a `lstm_recur_regs` workgroup walks; fixed at two because its partial sums
/// travel as a `vec2`.
pub const LSTM_REG_ROWS: usize = 2;

/// Largest hidden width the register LSTM holds. `hidden / 2` weights plus the
/// loop's own state has to fit the register file of a 768-thread workgroup, which
/// is ~85 registers per thread.
pub const LSTM_REG_MAX_HIDDEN: usize = 96;

/// Largest hidden width [`lstm_recur`] can run, set by its shared arrays: at
/// `LSTM_ROWS` rows the state and gate arrays come to
/// `(2 + 4) * LSTM_MAX_HIDDEN * LSTM_ROWS` floats, which has to leave room for
/// several workgroups per SM. 128 is what fits in 48 KB alongside 8 workgroups.
pub const LSTM_MAX_HIDDEN: usize = 128;

/// The DTTNet band-sequence permutation, in both directions.
///
/// `BandSequenceModelModule` reshapes `(b, c, t, f)` to `(b * heads, c / heads,
/// t, f)` and permutes to `(b * heads, f, t, c / heads)`, runs its modules, and
/// permutes back. The final `permute(0, 3, 2, 1)` inverts it, so the two index
/// maps are each other's inverses and one kernel with a `mode` flag serves both
/// — which is the point: a permutation and its inverse written separately are
/// two chances to get the axis order wrong, and a wrong permutation here would
/// produce a model that trains fine and separates badly.
///
/// `mode = 0` scatters `(b, heads * per_head, t, f)` into the module's layout;
/// `mode = 1` gathers it back. Both are pure gathers over `gd.w` elements.
///
/// `gd` is `(batches, heads, per_head, t)`, `ge` is `(f, count, mode, _)`.
pub fn heads_permute() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Out: array<f32>;
@group(0) @binding(2) var<uniform> gd: vec4<u32>;  // batches, heads, per_head, t
@group(0) @binding(3) var<uniform> ge: vec4<u32>;  // f, count, mode, _

const GRID_X: u32 = 65535u;

@compute @workgroup_size({threads})
fn heads_permute(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let i = (wid.y * GRID_X + wid.x) * {threads}u + lid.x;
    if (i >= ge.y) {{ return; }}

    let batches = gd.x;
    let heads = gd.y;
    let per_head = gd.z;
    let t = gd.w;
    let f = ge.x;
    let channels = heads * per_head;

    if (ge.z == 0u) {{
        // (b, c, t, f) -> (b * heads, f, t, per_head)
        let per_batch = channels * t * f;
        let b = i / per_batch;
        let rest = i % per_batch;
        let c = rest / (t * f);
        let after_c = rest % (t * f);
        let ti = after_c / f;
        let fi = after_c % f;
        let head = c / per_head;
        let j = c % per_head;
        Out[((b * heads + head) * f + fi) * (t * per_head) + ti * per_head + j] = X[i];
    }} else {{
        // (b * heads, f, t, per_head) -> (b, c, t, f)
        let per_batch = f * t * per_head;
        let bb = i / per_batch;
        let rest = i % per_batch;
        let fi = rest / (t * per_head);
        let after_f = rest % (t * per_head);
        let ti = after_f / per_head;
        let j = after_f % per_head;
        let b = bb / heads;
        let head = bb % heads;
        Out[((b * channels + head * per_head + j) * t + ti) * f + fi] = X[i];
    }}
}}
"#,
        threads = ROW_THREADS
    )
}

/// 4096-point unnormalised inverse FFT of a one-sided Hermitian spectrum.
///
/// One workgroup per frame. HTDemucs rejects `nfft != 4096`, so the size is
/// compiled in: 4096 `vec2` of shared memory is 32 KiB, which this card's
/// adapter limits already grant (`request_device` copies the adapter's limits).
///
/// `X` is CaC-packed `(batch, 4*sources, nfft/2, frames)`. The Nyquist bin is
/// the zero `pad_for_ispec` would have added. Inverse twiddles are `exp(+2πi…)`,
/// matching rustfft; the store applies `window / sqrt(nfft)`, which is the
/// host's `normalized=True` inverse scale times the Hann window.
pub fn istft_irfft4096() -> String {
    r#"
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read> Win: array<f32>;
@group(0) @binding(2) var<storage, read_write> Out: array<f32>;
@group(0) @binding(3) var<uniform> gd: vec4<u32>;  // frames, bins, packed, batch

const N: u32 = 4096u;
const BITS: u32 = 12u;
const THREADS: u32 = 256u;
const PER: u32 = 16u;
const PI2: f32 = 6.283185307179586;

var<workgroup> sm: array<vec2<f32>, 4096>;

fn bitrev12(x: u32) -> u32 {
    return reverseBits(x) >> (32u - BITS);
}

@compute @workgroup_size(256)
fn istft_irfft4096(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let frames = gd.x;
    let bins = gd.y;
    let packed = gd.z;
    let frame = wid.x;
    let row = wid.y;
    if (frame >= frames) { return; }

    let packed_out = packed / 2u;
    let b = row / packed_out;
    let co = row % packed_out;
    let cpack = co * 2u;
    let spec_base = (b * packed + cpack) * bins * frames + frame;
    let spec_im = spec_base + bins * frames;

    for (var p = 0u; p < PER; p++) {
        let k = lid.x + p * THREADS;
        var z = vec2<f32>(0.0, 0.0);
        if (k == 0u) {
            z = vec2(X[spec_base], X[spec_im]);
        } else if (k < bins) {
            let re = X[spec_base + k * frames];
            let im = X[spec_im + k * frames];
            z = vec2(re, im);
        } else if (k > bins) {
            let m = N - k;
            let re = X[spec_base + m * frames];
            let im = X[spec_im + m * frames];
            z = vec2(re, -im);
        }
        sm[bitrev12(k)] = z;
    }
    workgroupBarrier();

    for (var stage = 0u; stage < BITS; stage++) {
        let m = 1u << (stage + 1u);
        let mh = m >> 1u;
        workgroupBarrier();
        for (var p = 0u; p < 8u; p++) {
            let bfly = lid.x + p * THREADS;
            let j = bfly % mh;
            let grp = bfly / mh;
            let idx = grp * m + j;
            let pair = idx + mh;
            let a = sm[idx];
            let bval = sm[pair];
            let angle = PI2 * f32(j) / f32(m);
            let wr = cos(angle);
            let wi = sin(angle);
            let t = vec2(bval.x * wr - bval.y * wi, bval.x * wi + bval.y * wr);
            sm[idx] = a + t;
            sm[pair] = a - t;
        }
    }
    workgroupBarrier();

    let inv = 0.015625; // 1/sqrt(4096)
    let dst = ((row * frames + frame) * N);
    for (var p = 0u; p < PER; p++) {
        let i = lid.x + p * THREADS;
        Out[dst + i] = sm[i].x * Win[i] * inv;
    }
}
"#
    .to_string()
}

/// 4096-point real forward FFT of one STFT frame, written as CaC `(re, im)`
/// channels. Twiddles are `exp(-2πi…)`, matching rustfft; the store applies
/// `1/sqrt(nfft)` (`normalized=True`) and drops the Nyquist bin.
///
/// `Mix` is `_spec`'s reflect-padded waveform `(batch, channels, padded_len)`.
/// Output frame `f` is STFT frame `f+2` (the two padding frames `_spec` drops).
/// Center pad `nfft/2` is applied in-index, not by writing a second buffer.
pub fn stft_rfft4096() -> String {
    r#"
@group(0) @binding(0) var<storage, read> Mix: array<f32>;
@group(0) @binding(1) var<storage, read> Win: array<f32>;
@group(0) @binding(2) var<storage, read_write> Out: array<f32>;
@group(0) @binding(3) var<uniform> gd: vec4<u32>;  // le, bins, channels, hop
@group(0) @binding(4) var<uniform> ge: vec4<u32>;  // padded_len, center, batch, _

const N: u32 = 4096u;
const BITS: u32 = 12u;
const THREADS: u32 = 256u;
const PER: u32 = 16u;
const PI2: f32 = 6.283185307179586;

var<workgroup> sm: array<vec2<f32>, 4096>;

fn bitrev12(x: u32) -> u32 {
    return reverseBits(x) >> (32u - BITS);
}

fn reflect_at(idx: i32, len: i32) -> u32 {
    if (idx < 0) {
        return u32(-idx);
    }
    if (idx >= len) {
        return u32(2 * len - 2 - idx);
    }
    return u32(idx);
}

@compute @workgroup_size(256)
fn stft_rfft4096(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let le = gd.x;
    let bins = gd.y;
    let channels = gd.z;
    let hop = gd.w;
    let padded_len = ge.x;
    let center = ge.y;
    let frame = wid.x;
    let row = wid.y;
    if (frame >= le) { return; }

    let b = row / channels;
    let ch = row % channels;
    let mix_base = (b * channels + ch) * padded_len;
    let start = (frame + 2u) * hop;
    let plen = i32(padded_len);
    let cpad = i32(center);

    for (var p = 0u; p < PER; p++) {
        let k = lid.x + p * THREADS;
        let src = reflect_at(i32(start + k) - cpad, plen);
        let v = Mix[mix_base + src] * Win[k];
        sm[bitrev12(k)] = vec2(v, 0.0);
    }
    workgroupBarrier();

    for (var stage = 0u; stage < BITS; stage++) {
        let m = 1u << (stage + 1u);
        let mh = m >> 1u;
        workgroupBarrier();
        for (var p = 0u; p < 8u; p++) {
            let bfly = lid.x + p * THREADS;
            let j = bfly % mh;
            let grp = bfly / mh;
            let idx = grp * m + j;
            let pair = idx + mh;
            let a = sm[idx];
            let bval = sm[pair];
            let angle = PI2 * f32(j) / f32(m);
            let wr = cos(angle);
            let wi = -sin(angle);
            let t = vec2(bval.x * wr - bval.y * wi, bval.x * wi + bval.y * wr);
            sm[idx] = a + t;
            sm[pair] = a - t;
        }
    }
    workgroupBarrier();

    let inv = 0.015625;
    let packed = channels * 2u;
    let re_ch = ch * 2u;
    let im_ch = re_ch + 1u;
    let re_base = ((b * packed + re_ch) * bins) * le + frame;
    let im_base = ((b * packed + im_ch) * bins) * le + frame;
    for (var p = 0u; p < PER; p++) {
        let k = lid.x + p * THREADS;
        if (k < bins) {
            Out[re_base + k * le] = sm[k].x * inv;
            Out[im_base + k * le] = sm[k].y * inv;
        }
    }
}
"#
    .to_string()
}

/// Per-batch mean and unbiased std of a contiguous plane.
pub fn batch_moments() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Mean: array<f32>;
@group(0) @binding(2) var<storage, read_write> Std: array<f32>;
@group(0) @binding(3) var<uniform> gd: vec4<u32>;  // plane, batch, _, _

const THREADS: u32 = {threads}u;
var<workgroup> sum_s: array<f32, {threads}>;
var<workgroup> sum_q: array<f32, {threads}>;

@compute @workgroup_size({threads})
fn batch_moments(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let plane = gd.x;
    let b = wid.x;
    if (b >= gd.y) {{ return; }}
    let base = b * plane;
    var s = 0.0;
    var q = 0.0;
    for (var i = lid.x; i < plane; i = i + THREADS) {{
        let v = X[base + i];
        s = s + v;
        q = q + v * v;
    }}
    sum_s[lid.x] = s;
    sum_q[lid.x] = q;
    workgroupBarrier();
    for (var step = THREADS / 2u; step > 0u; step = step / 2u) {{
        if (lid.x < step) {{
            sum_s[lid.x] = sum_s[lid.x] + sum_s[lid.x + step];
            sum_q[lid.x] = sum_q[lid.x] + sum_q[lid.x + step];
        }}
        workgroupBarrier();
    }}
    if (lid.x == 0u) {{
        let n = f32(plane);
        let mean = sum_s[0] / n;
        let variance = (sum_q[0] - mean * mean * n) / (n - 1.0);
        Mean[b] = mean;
        Std[b] = sqrt(max(variance, 0.0));
    }}
}}
"#,
        threads = ROW_THREADS,
    )
}

/// Segment size of the split `batch_moments`: the plane is cut into
/// `ceil(plane / this)` pieces, one workgroup each. `DEMUCS_BATCH_MOMENTS_SEG=0`
/// keeps the single-workgroup scan, which is how the two were compared.
pub fn batch_moments_segment_elements() -> usize {
    std::env::var("DEMUCS_BATCH_MOMENTS_SEG")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4096)
}

/// First half of the split `batch_moments`: one workgroup per
/// `(batch item, segment)` writes that segment's `(sum, sum of squares)`.
///
/// The single-workgroup form has one workgroup against the whole GPU, so the
/// frequency branch's 2.75M-element plane measured 2.68 ms — the same
/// pathology the waveform DConv's group-norm slices had before they were cut
/// into segments.
pub fn batch_moments_partial() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> Partials: array<f32>;  // 2 per (batch, segment)
@group(0) @binding(2) var<uniform> gd: vec4<u32>;  // plane, segments, segment_len, batch

const THREADS: u32 = {threads}u;
const ROW_GRID_X: u32 = {grid_x}u;
var<workgroup> partial_sum: array<f32, {threads}>;
var<workgroup> partial_sq: array<f32, {threads}>;

@compute @workgroup_size({threads})
fn batch_moments_partial(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let plane = gd.x;
    let segments = gd.y;
    let segment_len = gd.z;
    let work = wid.x + wid.y * ROW_GRID_X;
    let b = work / segments;
    let segment = work % segments;
    if (b >= gd.w) {{ return; }}

    let base = b * plane;
    let start = segment * segment_len;
    let end = min(start + segment_len, plane);
    var s = 0.0;
    var q = 0.0;
    for (var i = start + lid.x; i < end; i = i + THREADS) {{
        let v = X[base + i];
        s = s + v;
        q = fma(v, v, q);
    }}
    partial_sum[lid.x] = s;
    partial_sq[lid.x] = q;
    workgroupBarrier();
    for (var step = THREADS / 2u; step > 0u; step = step / 2u) {{
        if (lid.x < step) {{
            partial_sum[lid.x] = partial_sum[lid.x] + partial_sum[lid.x + step];
            partial_sq[lid.x] = partial_sq[lid.x] + partial_sq[lid.x + step];
        }}
        workgroupBarrier();
    }}
    if (lid.x == 0u) {{
        Partials[work * 2u] = partial_sum[0];
        Partials[work * 2u + 1u] = partial_sq[0];
    }}
}}
"#,
        threads = ROW_THREADS,
        grid_x = ROW_GRID_X
    )
}

/// Second half of the split `batch_moments`: fold one batch item's segment
/// partials into `(mean, std)`, with the formula the single-workgroup kernel
/// uses (unbiased variance).
pub fn batch_moments_combine() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read> Partials: array<f32>;
@group(0) @binding(1) var<storage, read_write> Mean: array<f32>;
@group(0) @binding(2) var<storage, read_write> Std: array<f32>;
@group(0) @binding(3) var<uniform> gd: vec4<u32>;  // plane, batch, segments, _

const THREADS: u32 = {threads}u;

@compute @workgroup_size({threads})
fn batch_moments_combine(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let b = wid.x * THREADS + lid.x;
    if (b >= gd.y) {{ return; }}

    let plane = gd.x;
    let segments = gd.z;
    var s = 0.0;
    var q = 0.0;
    let first = b * segments * 2u;
    for (var i = 0u; i < segments; i = i + 1u) {{
        s = s + Partials[first + i * 2u];
        q = q + Partials[first + i * 2u + 1u];
    }}
    let n = f32(plane);
    let mean = s / n;
    let variance = (q - mean * mean * n) / (n - 1.0);
    Mean[b] = mean;
    Std[b] = sqrt(max(variance, 0.0));
}}
"#,
        threads = ROW_THREADS,
    )
}

/// `x = (x - mean) / (1e-5 + std)` per batch item, in place.
pub fn batch_normalize_in_place() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read_write> Out: array<f32>;
@group(0) @binding(1) var<storage, read> Mean: array<f32>;
@group(0) @binding(2) var<storage, read> Std: array<f32>;
@group(0) @binding(3) var<uniform> gd: vec4<u32>;  // count, plane, _, _

const THREADS: u32 = {threads}u;
const GRID_X: u32 = {grid_x}u;

@compute @workgroup_size({threads})
fn batch_normalize_in_place(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let i = (wid.y * GRID_X + wid.x) * THREADS + lid.x;
    if (i >= gd.x) {{ return; }}
    let b = i / gd.y;
    Out[i] = (Out[i] - Mean[b]) / (1e-5 + Std[b]);
}}
"#,
        threads = ROW_THREADS,
        grid_x = ROW_GRID_X,
    )
}

/// Overlap-add of windowed iFFT frames onto `length` samples, then divide by
/// the Hann envelope. Frame `f` lands at offset `(f+2)*hop` because `_ispec`
/// pads two zero frames on the left; those frames are skipped (they are zero).
pub fn istft_ola() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read> Frames: array<f32>;
@group(0) @binding(1) var<storage, read> Win: array<f32>;
@group(0) @binding(2) var<storage, read_write> Out: array<f32>;
@group(0) @binding(3) var<uniform> gd: vec4<u32>;  // frames, hop, nfft, pad
@group(0) @binding(4) var<uniform> ge: vec4<u32>;  // length, packed_out, batch, _

const THREADS: u32 = {threads}u;
const GRID_X: u32 = {grid_x}u;

@compute @workgroup_size({threads})
fn istft_ola(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let i = (wid.y * GRID_X + wid.x) * THREADS + lid.x;
    let frames = gd.x;
    let hop = gd.y;
    let nfft = gd.z;
    let pad = gd.w;
    let length = ge.x;
    let packed_out = ge.y;
    let total = ge.z * packed_out * length;
    if (i >= total) {{ return; }}

    let sample = i % length;
    let row = i / length;
    let ola_idx = sample + (nfft / 2u) + pad;
    let nfft_i = i32(nfft);
    let hop_i = i32(hop);
    let ola_i = i32(ola_idx);
    let pf_hi = ola_idx / hop;
    var pf_lo = 0u;
    if (ola_idx + hop > nfft) {{
        pf_lo = (ola_idx + hop - nfft) / hop;
    }}
    let padded_frames = frames + 4u;

    var acc = 0.0;
    var env = 0.0;
    for (var pf = pf_lo; pf <= pf_hi; pf++) {{
        if (pf >= padded_frames) {{ continue; }}
        let local_i = ola_i - i32(pf) * hop_i;
        if (local_i < 0 || local_i >= nfft_i) {{ continue; }}
        let local = u32(local_i);
        let w = Win[local];
        env += w * w;
        if (pf >= 2u && pf < 2u + frames) {{
            acc += Frames[(row * frames + (pf - 2u)) * nfft + local];
        }}
    }}
    Out[i] = select(0.0, acc / env, env > 0.0);
}}
"#,
        threads = ROW_THREADS,
        grid_x = ROW_GRID_X,
    )
}

/// `x[i] = x[i] * scale[i / plane] + shift[i / plane]`, in place.
///
/// The HTDemucs epilogue denormalises a whole branch with one mean/std per
/// batch item, which is not a channel affine.
pub fn batch_affine_in_place() -> String {
    format!(
        r#"
@group(0) @binding(0) var<storage, read_write> Out: array<f32>;
@group(0) @binding(1) var<storage, read> Scale: array<f32>;
@group(0) @binding(2) var<storage, read> Shift: array<f32>;
@group(0) @binding(3) var<uniform> gd: vec4<u32>;  // count, plane, _, _

const THREADS: u32 = {threads}u;
const GRID_X: u32 = {grid_x}u;

@compute @workgroup_size({threads})
fn batch_affine_in_place(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let i = (wid.y * GRID_X + wid.x) * THREADS + lid.x;
    if (i >= gd.x) {{ return; }}
    let plane = gd.y;
    let b = i / plane;
    Out[i] = Out[i] * Scale[b] + Shift[b];
}}
"#,
        threads = ROW_THREADS,
        grid_x = ROW_GRID_X,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemm_tiles_divide_evenly() {
        assert_eq!(BM % THREADS_Y, 0);
        assert_eq!(BN % THREADS_X, 0);
        assert_eq!(BM96 % THREADS_Y, 0);
        assert_eq!(BM96 * ROW_SLOTS % (THREADS_X * THREADS_Y), 0);
        assert_eq!(BM * BK % (THREADS_X * THREADS_Y), 0);
        assert_eq!(BN * BK % (THREADS_X * THREADS_Y), 0);
        let t = Tile::square();
        assert_eq!((t.bm / t.ty) * (t.bn / t.tx), 64, "8x8 accumulators per thread");
        // The 64-wide N tile has to stage whole rows too.
        let n64 = Tile::bn64();
        assert_eq!(n64.bn * ROW_SLOTS % (n64.tx * n64.ty), 0);
    }

    #[test]
    fn shared_tiles_stay_vec2_aligned() {
        // A stride of `PAD` f32 per row has to be even, or every `vec2` access
        // past the first row lands on an odd word and the driver has to split it.
        assert_eq!(PAD % 2, 0);
        assert_eq!(BM * BK % (THREADS_X * THREADS_Y * 2), 0);
        assert_eq!(BN * BK % (THREADS_X * THREADS_Y * 2), 0);
        // The staging walk has to cover every payload slot exactly; a pitch that
        // does not divide the thread count leaves rows holding whatever was in
        // shared memory before.
        assert_eq!(BM * ROW_SLOTS % (THREADS_X * THREADS_Y), 0);
        assert_eq!(BN * ROW_SLOTS % (THREADS_X * THREADS_Y), 0);
        assert!(PW >= ROW_SLOTS, "the pitch must leave room for the payload");
        let bytes = 2 * BM * PW * 8;
        assert!(bytes <= 32 * 1024, "shared tiles use {bytes} bytes");
    }

    #[test]
    fn shader_substitutes_every_constant() {
        let source = gemm_f32(false);
        assert!(source.contains("const BM: u32 = 128u;"));
        assert!(source.contains("const PAD: u32 = 18u;"));
        assert!(source.contains("@workgroup_size(16, 16)"));
        assert!(source.contains("array<vec2<f32>, 1152>"));
        for placeholder in [
            "{bm}", "{bn}", "{bk}", "{pad}", "{pw}", "{tm}", "{tn}", "{tx}", "{ty}",
            "{as_len}", "{bs_len}", "{accumulators}", "{prefetch_decl}", "{store_stage0}",
            "{prefetch}", "{compute}", "{store_prefetch}", "{epilogue}", "{batch_base}",
            "@A@", "@B@", "@C@",
        ] {
            assert!(
                !source.contains(placeholder),
                "placeholder {placeholder} was left unsubstituted"
            );
        }
    }

    #[test]
    fn shared_reads_are_vectorised() {
        let source = gemm_f32(false);
        // 8 a-slices + 8 b-slices per vector step, each one load. The shared
        // row bases are hoisted out of the K loop (`ar{i}`/`br{j}`), so every
        // step's access is a constant offset from one.
        assert!(source.contains("let a0 = As[ar0 + 0u];"));
        assert!(source.contains("let b0 = Bs[br0 + 0u];"));
        // Two FMAs per accumulator pair, one per vector lane.
        assert!(source.contains("c0_0 = fma(a0.x, b0.x, c0_0);"));
        assert!(source.contains("c0_0 = fma(a0.y, b0.y, c0_0);"));
        assert!(!source.contains("As[(ty * TM"), "scalar row ownership is back");
    }

    #[test]
    fn accumulator_ownership_is_strided_to_avoid_bank_conflicts() {
        // Striding by the thread count makes the address stride `PAD` words
        // (18, even but spreading across banks) instead of `TM * PAD`.
        let source = gemm_f32(false);
        assert!(source.contains("let ar1 = (ty + 1u * TY) * PW;"));
        assert!(source.contains("let br1 = (tx + 1u * TX) * PW;"));
    }

    #[test]
    fn dumps_the_generated_source_on_request() {
        if let Some(path) = std::env::var_os("DEMUCS_DUMP_SHADER") {
            let source = match std::env::var("DEMUCS_DUMP_SHADER_WHICH").as_deref() {
                Ok("residual") => gemm_residual(),
                Ok("bias_residual") => gemm_bias_residual(),
                Ok("gelu_bias") => gemm_gelu_bias(),
                Ok("bn64") => gemm_batched_bn64(),
                _ => gemm_f32(false),
            };
            std::fs::write(path, source).expect("dump");
        }
    }
}
