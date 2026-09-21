//! GPU kernels, wrapped as ops over device-resident tensors.
//!
//! Every op allocates its output from the [`Arena`], records one dispatch into
//! the [`Recorder`], and returns the tensor. Nothing reads back to the host, so
//! a whole forward pass costs one upload, one submission and one readback.
//!
//! Each dispatch gets its **own** uniform buffer, allocated from the arena.
//! Sharing one would be wrong rather than merely slow: `Queue::write_buffer` is
//! ordered against submission, not against individual dispatches, so writing the
//! same uniform twice before submitting makes both dispatches see the second
//! value.

use crate::error::{Error, Result};
use crate::gpu::arena::{bind_group, Arena, DevTensor, Recorder};
use crate::gpu::shaders;
use crate::gpu::Gpu;

/// Workgroup size for the row-wise kernels, mirroring `shaders::ROW_THREADS`.
const ROW_THREADS: usize = shaders::ROW_THREADS;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GemmDims {
    m: u32,
    n: u32,
    k: u32,
    lda: u32,
    ldb: u32,
    ldc: u32,
    inner_count: u32,
    asa_outer: u32,
    asa_inner: u32,
    bsb_outer: u32,
    bsb_inner: u32,
    csc_outer: u32,
    csc_inner: u32,
    _pad: u32,
}

/// Shape of an `nn.ConvTranspose2d` call, as the col2im gather needs it.
///
/// The convolution itself is one GEMM: `(out_channels * kh * kw, in_channels) @
/// (in_channels, positions_in)` — so the weights are uploaded in that
/// `(oc, ky, kx)`-major layout and the tap matrix it produces is what
/// [`Kernels::col2im_into`] folds back into an image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Col2ImShape {
    pub batch: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub in_h: usize,
    pub in_w: usize,
    pub kernel: (usize, usize),
    pub stride: (usize, usize),
}

impl Col2ImShape {
    pub fn out_hw(&self) -> (usize, usize) {
        (
            (self.in_h - 1) * self.stride.0 + self.kernel.0,
            (self.in_w - 1) * self.stride.1 + self.kernel.1,
        )
    }

    /// Rows of the tap matrix the GEMM produces.
    pub fn m(&self) -> usize {
        self.out_channels * self.kernel.0 * self.kernel.1
    }

    pub fn positions_in(&self) -> usize {
        self.in_h * self.in_w
    }

    pub fn positions_out(&self) -> usize {
        let (out_h, out_w) = self.out_hw();
        out_h * out_w
    }
}

/// The activation fused into [`Kernels::channel_affine_act_in_place`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    Identity,
    Relu,
    Gelu,
    Silu,
}

impl Activation {
    fn code(self) -> u32 {
        match self {
            Activation::Identity => 0,
            Activation::Relu => 1,
            Activation::Gelu => 2,
            Activation::Silu => 3,
        }
    }
}

/// Shape of a [`Kernels::group_norm_into`] call.
///
/// The layout is strides rather than a plain `(rows, channels, len)` because the
/// DTTNet band sequence normalises a row whose sequence axis is not contiguous:
/// its channels are the innermost run (`per_head`), so the sequence elements are
/// `len_stride = per_head` apart while the channels are 1 apart. A contiguous
/// tensor is the `channel_stride = 1, len_stride = len` case.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroupNormShape {
    pub rows: usize,
    pub groups: usize,
    /// Channels per group; `groups * per_group` is the channel count.
    pub per_group: usize,
    pub len: usize,
    /// Distance between consecutive rows' first elements.
    pub row_stride: usize,
    pub len_stride: usize,
    pub channel_stride: usize,
    /// `nn.GroupNorm`'s `eps`, 1e-5 unless the checkpoint overrode it.
    pub eps: f32,
}

impl GroupNormShape {
    pub fn channels(&self) -> usize {
        self.groups * self.per_group
    }

    /// Elements the operators must own: one past the last element either of them
    /// touches.
    pub fn needed(&self) -> usize {
        (self.rows - 1) * self.row_stride
            + (self.groups - 1) * self.per_group * self.channel_stride
            + (self.per_group - 1) * self.channel_stride
            + (self.len - 1) * self.len_stride
            + 1
    }
}

/// Shape of a [`Kernels::lstm_recur_into`] call: one direction of one layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LstmJob {
    pub rows: usize,
    pub time: usize,
    pub hidden: usize,
    /// Walk the time axis backwards; the reverse direction of a BiLSTM.
    pub reverse: bool,
    /// Row pitch of the projected gates, `4 * hidden` unless padded.
    pub proj_pitch: usize,
    /// Row pitch of the output, `hidden * directions` unless padded.
    pub out_pitch: usize,
    /// Where this direction's `hidden` values start in an output row.
    pub out_offset: usize,
}

/// Shape of an `nn.Conv2d` call, as the im2col gather needs to see it.
///
/// Deliberately the same description the host `conv.rs` uses, so the two can be
/// diffed directly: patches come out `(batch, k, positions)` with
/// `k = in_channels * kh * kw` in `(ic, ky, kx)` order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Im2ColShape {
    pub batch: usize,
    pub in_channels: usize,
    pub h: usize,
    pub w: usize,
    pub kernel: (usize, usize),
    pub stride: (usize, usize),
    pub pad: (usize, usize),
}

impl Im2ColShape {
    pub fn out_hw(&self) -> (usize, usize) {
        (
            (self.h + 2 * self.pad.0 - self.kernel.0) / self.stride.0 + 1,
            (self.w + 2 * self.pad.1 - self.kernel.1) / self.stride.1 + 1,
        )
    }

    /// Reduction length of the GEMM this feeds.
    pub fn k(&self) -> usize {
        self.in_channels * self.kernel.0 * self.kernel.1
    }

    pub fn positions(&self) -> usize {
        let (out_h, out_w) = self.out_hw();
        out_h * out_w
    }

    pub fn output_len(&self) -> usize {
        self.batch * self.k() * self.positions()
    }

    /// Elements the destination must hold when its rows are `pitch` wide and
    /// each batch gets `rows` of them.
    ///
    /// `pitch` is `positions()` for a tight buffer, or `pad_ceil(positions(),
    /// BN)` when the GEMM consumes it; `rows` is `k()` for a tight buffer, or
    /// `pad_ceil(k(), BK)` for the GEMM. The gather writes the batch stride from
    /// `rows`, so the padded form must be described the same way here.
    pub fn pitched_len(&self, pitch: usize) -> usize {
        self.pitched_len_with(pitch, self.k())
    }

    pub fn pitched_len_with(&self, pitch: usize, rows: usize) -> usize {
        self.batch * rows * pitch
    }
}

/// Two parameters per `u32`, for the kernels still sized to fit that packing.
fn pack(high: usize, low: usize) -> u32 {
    ((high as u32) << 16) | (low as u32)
}

/// One batched matrix multiply.
///
/// Batches are addressed as two levels with independent strides, which is what
/// the model's attention needs: its operands are indexed by `(band, head)` and
/// those two levels are not a single linear stride apart inside the fused QKV
/// tensor.
#[derive(Debug, Clone, Copy)]
pub struct GemmJob {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub lda: usize,
    pub ldb: usize,
    pub ldc: usize,
    pub batches: usize,
    pub inner_count: usize,
    pub a_outer: usize,
    pub a_inner: usize,
    pub b_outer: usize,
    pub b_inner: usize,
    pub c_outer: usize,
    pub c_inner: usize,
    /// Read B as `(n, k)`. Required for `Q @ K^T`, where K is stored
    /// `[frame][dim_head]` and its transpose is not a `(k, n)`-strided read.
    pub transb: bool,
}

impl GemmJob {
    /// `C[m, n] = A[m, k] @ B[k, n]` with both operands contiguous and padded.
    pub fn simple(m: usize, n: usize, k: usize) -> Self {
        let lda = pad_pitch(k, shaders::BK);
        let ldb = pad_pitch(n, shaders::BN);
        let a_outer = lda * pad_ceil(m, shaders::BM);
        Self {
            m,
            n,
            k,
            lda,
            ldb,
            ldc: n,
            batches: 1,
            inner_count: 1,
            a_outer,
            a_inner: 0,
            b_outer: ldb * lda,
            b_inner: 0,
            c_outer: m * n,
            c_inner: 0,
            transb: false,
        }
    }

    /// Repeats the same multiply over `batches` slices of A and C.
    pub fn batched(mut self, batches: usize) -> Self {
        self.batches = batches;
        self
    }

    /// Strides for an operand whose batch axis is contiguous, given how many
    /// elements one batch occupies and how many inner levels there are.
    ///
    /// The outer stride has to cover the *whole* inner run — `per_batch *
    /// inner_count`, not `per_batch`. Getting this wrong is silent: the base of
    /// batch `outer*inner_count + inner` collapses onto an earlier batch's, so
    /// two batches write to the same place and the last one wins.
    pub fn contiguous_batch(per_batch: usize, inner_count: usize) -> (usize, usize) {
        (per_batch * inner_count.max(1), per_batch)
    }

    pub fn grid(&self) -> (u32, u32, u32) {
        self.grid_with(shaders::BM, shaders::BN)
    }

    /// The grid for a specific tile: the shader derives `m0`/`n0` from the
    /// workgroup id and its own `BM`/`BN`, so the two have to be told the same
    /// numbers or the tiles land in the wrong place.
    pub fn grid_with(&self, bm: usize, bn: usize) -> (u32, u32, u32) {
        // Workgroups are dispatched x-fastest, so x is the axis the *other*
        // operand's reuse turns on. M on x means a wave of consecutive
        // workgroups shares one B panel — a few hundred KB, which stays in L2 —
        // while A streams past once. The other order re-reads the whole B matrix
        // per M tile, and at 2.4 MB the weight matrix does not fit the 2 MB L2 to
        // absorb it. Measured equal on the model's `ff_in` shape (15.6 ms either
        // way), so this is the shape that reads better rather than a win.
        (
            (pad_ceil(self.m, bm) / bm) as u32,
            (pad_ceil(self.n, bn) / bn) as u32,
            self.batches as u32,
        )
    }
}

pub struct Kernels {
    /// The plain GEMM, used for every `nn.Linear` in the model.
    gemm: wgpu::ComputePipeline,
    /// The same kernel with per-workgroup batch offsets, used only by attention,
    /// where operands are indexed by `(band, head)`.
    gemm_batched: wgpu::ComputePipeline,
    /// The batched form with B read as `(n, k)`, which is what `Q @ K^T` needs.
    gemm_transb: wgpu::ComputePipeline,
    /// `gemm_f32_transb(false)`: same addressing, batch bases compiled out.
    /// Linear projections are `batches == 1` and the dead offsets cost ~15% on
    /// this driver. QK^T stays on [`Self::gemm_transb`].
    gemm_transb_plain: wgpu::ComputePipeline,
    /// The 64x128 tile, unbatched: an option for the convolutions, whose output
    /// channel counts are far below a 128-row tile.
    gemm_bm64: wgpu::ComputePipeline,
    /// The same tile with a coalesced `B` staging walk, for the patch matrices
    /// whose rows are megabytes apart.
    gemm_bm64_coalesced: wgpu::ComputePipeline,
    /// The batched form of `gemm_bm64`, which is what a convolution uses now
    /// that its per-batch blocks are stitched into one dispatch rather than
    /// bound one at a time: the batch offset has to stay out of the bind group
    /// (see `shaders::gemm_batched_bm64_bn128`).
    gemm_batched_bm64_bn128: wgpu::ComputePipeline,
    gemm_batched_bm64_bn128_coalesced: wgpu::ComputePipeline,
    gemm_batched_bm96_bn128: wgpu::ComputePipeline,
    /// The same tile with the im2col gather folded into its `B` staging: a
    /// convolution that never materialises a patch matrix.
    gemm_conv_direct: wgpu::ComputePipeline,
    /// `gemm_conv_direct` with the batch base terms emitted, so a forward that
    /// carries several segments through one pass gathers from each one's own
    /// block of the input. Batch 1 keeps the plain form (see
    /// `shaders::gemm_conv_direct_batched`).
    gemm_conv_direct_batched: wgpu::ComputePipeline,
    /// The same with a 64-wide N tile, for the frequency axis' 60-wide keys.
    gemm_transb_bn64: wgpu::ComputePipeline,
    /// Batched form with a 64-wide N tile: the AV product's `n = dim_head = 64`
    /// would otherwise leave half of a 128-wide tile computing zeros.
    gemm_batched_bn64: wgpu::ComputePipeline,
    /// The AV product with the softmax folded into its `A` staging: `A` is the
    /// raw score matrix and the per-row `(max, 1/Σexp)` come from
    /// [`Kernels::softmax_stats`].
    gemm_batched_bn64_row_exp: wgpu::ComputePipeline,
    /// Both tiles 64 wide, for the frequency axis' `60 x 60` attention problem:
    /// there a 128-row M tile computes more zeros than results.
    gemm_transb_bm64: wgpu::ComputePipeline,
    gemm_batched_bm64: wgpu::ComputePipeline,
    /// Single-batch transb with Linear bias in the epilogue.
    gemm_transb_plain_bias: wgpu::ComputePipeline,
    /// Plain GEMM with the `nn.Linear` bias folded into the store.
    gemm_bias: wgpu::ComputePipeline,
    /// Plain GEMM with bias and the feed-forward GELU folded into the store.
    gemm_gelu_bias: wgpu::ComputePipeline,
    /// Plain GEMM with the destination read back and added: the block's skip
    /// connection, without a separate `add_in_place` pass.
    gemm_residual: wgpu::ComputePipeline,
    /// Same, with the bias as well — the feed-forward's second projection.
    gemm_bias_residual: wgpu::ComputePipeline,
    rms_norm: wgpu::ComputePipeline,
    /// One warp per row, shuffle-reduced: the same normalisation without the
    /// tree's eight barriers. Built only for adapters with a 32-lane subgroup.
    rms_norm_warp: Option<wgpu::ComputePipeline>,
    gelu: wgpu::ComputePipeline,
    gelu_in_place: wgpu::ComputePipeline,
    /// Four-lane variant, used when the element count is a multiple of four.
    gelu_in_place_vec4: wgpu::ComputePipeline,
    glu: wgpu::ComputePipeline,
    /// The DConv's GLU and its per-channel LayerScale in one pass.
    glu_channel_affine: wgpu::ComputePipeline,
    /// The same with the DConv's residual folded in: `out = base + glu * gamma`.
    glu_channel_affine_add: wgpu::ComputePipeline,
    sigmoid_gate: wgpu::ComputePipeline,
    rope: wgpu::ComputePipeline,
    add_in_place: wgpu::ComputePipeline,
    /// `x *= y`, elementwise: DTTNet's decoder gates the upsampled tensor by the
    /// encoder's feature of the same resolution.
    mul_in_place: wgpu::ComputePipeline,
    /// Four-lane variant, used when the element count is a multiple of four.
    add_in_place_vec4: wgpu::ComputePipeline,
    transpose: wgpu::ComputePipeline,
    im2col: wgpu::ComputePipeline,
    /// `x = 0` over a range, on the device (see the shader's note on why the
    /// host-side clear is not an option at conv sizes).
    fill_zero: wgpu::ComputePipeline,
    /// A `(batch, rows, cols)` block copied from a tight row pitch to a padded
    /// one, which is how a conv-transpose's input becomes a GEMM operand.
    copy_pitched: wgpu::ComputePipeline,
    crop_rows: wgpu::ComputePipeline,
    add_row_bias_in_place: wgpu::ComputePipeline,
    channel_affine_act_in_place: wgpu::ComputePipeline,
    /// `out = base + act(y * scale + shift)`: the residual a transformer block
    /// ends with, in one pass instead of a scale, a copy and an add.
    channel_affine_act_add: wgpu::ComputePipeline,
    col2im: wgpu::ComputePipeline,
    group_norm: wgpu::ComputePipeline,
    /// The split form of `group_norm`, for slices too big for one workgroup:
    /// a per-segment reduce, a per-pair stats fold, then apply.
    group_norm_partial: wgpu::ComputePipeline,
    group_norm_combine: wgpu::ComputePipeline,
    group_norm_apply: wgpu::ComputePipeline,
    /// Original apply: every lane re-sums the pair's partials. A/B only.
    group_norm_apply_from_partials: wgpu::ComputePipeline,
    lstm_recur: wgpu::ComputePipeline,
    /// The register-resident variant, compiled per hidden width: its workgroup
    /// size is `8 * hidden`, so one pipeline cannot serve two layer widths.
    lstm_recur_regs: std::cell::RefCell<std::collections::HashMap<usize, wgpu::ComputePipeline>>,
    heads_permute: wgpu::ComputePipeline,
    tanh: wgpu::ComputePipeline,
    /// `x = tanh(x + bias)` in place; the bias is constant over blocks of rows.
    tanh_bias_in_place: wgpu::ComputePipeline,
    copy: wgpu::ComputePipeline,
    softmax: wgpu::ComputePipeline,
    softmax_in_place: wgpu::ComputePipeline,
    /// The same, one warp per row with shuffle reductions: for rows of at most
    /// 64 columns, where the tree's six barriers cost more than the arithmetic.
    /// Only built when the adapter can reduce across a 32-lane subgroup.
    softmax_warp: Option<wgpu::ComputePipeline>,
    /// Out-of-place scaled softmax, one warp per row, any width.
    softmax_scaled_warp: Option<wgpu::ComputePipeline>,
    /// The row scan alone: `(max, 1/Σexp)` per row, for the AV product that
    /// folds the `exp` into its own staging (see
    /// [`Kernels::softmax_stats`]).
    softmax_stats_warp: Option<wgpu::ComputePipeline>,
    /// Fused attention for `dim_head == 64`, replacing the score-matrix path.
    flash_attention: wgpu::ComputePipeline,
    /// 4096-point inverse real FFT, one workgroup per spectrogram frame.
    istft_irfft4096: wgpu::ComputePipeline,
    /// Overlap-add of those frames onto the waveform, with the Hann envelope.
    istft_ola: wgpu::ComputePipeline,
    /// Per-batch scalar affine, the epilogue's denormalise.
    batch_affine_in_place: wgpu::ComputePipeline,
    /// 4096-point real forward FFT of one STFT frame.
    stft_rfft4096: wgpu::ComputePipeline,
    /// Per-batch mean and unbiased std.
    batch_moments: wgpu::ComputePipeline,
    /// The split form of `batch_moments`, for planes too big for one
    /// workgroup: a per-segment reduce of `(sum, sum of squares)` and a
    /// per-batch fold of those partials.
    batch_moments_partial: wgpu::ComputePipeline,
    batch_moments_combine: wgpu::ComputePipeline,
    /// `(x - mean) / (1e-5 + std)` per batch item.
    batch_normalize_in_place: wgpu::ComputePipeline,
}

impl Kernels {
    pub fn new(gpu: &Gpu) -> Result<Self> {
        Ok(Self {
            gemm: gpu.pipeline("gemm_f32", &shaders::gemm_f32(false), "gemm")?,
            gemm_batched: gpu.pipeline(
                "gemm_f32_batched",
                &shaders::gemm_f32(true),
                "gemm",
            )?,
            gemm_transb: gpu.pipeline(
                "gemm_f32_transb",
                &shaders::gemm_f32_transb(true),
                "gemm",
            )?,
            gemm_transb_plain: gpu.pipeline(
                "gemm_f32_transb_plain",
                &shaders::gemm_f32_transb(false),
                "gemm",
            )?,
            gemm_bm64: gpu.pipeline("gemm_f32_bm64", &shaders::gemm_bm64(), "gemm")?,
            gemm_bm64_coalesced: gpu.pipeline(
                "gemm_f32_bm64_coalesced",
                &shaders::gemm_bm64_coalesced(),
                "gemm",
            )?,
            // The batched form of `gemm_bm64`, which is what a convolution
            // uses now that its per-batch blocks are stitched into one
            // dispatch rather than bound one at a time.
            gemm_batched_bm64_bn128: gpu.pipeline(
                "gemm_f32_batched_bm64_bn128",
                &shaders::gemm_batched_bm64_bn128(),
                "gemm",
            )?,
            gemm_batched_bm64_bn128_coalesced: gpu.pipeline(
                "gemm_f32_batched_bm64_bn128_coalesced",
                &shaders::gemm_batched_bm64_bn128_coalesced(),
                "gemm",
            )?,
            gemm_batched_bm96_bn128: gpu.pipeline(
                "gemm_f32_batched_bm96_bn128",
                &shaders::gemm_batched_bm96_bn128(),
                "gemm",
            )?,
            gemm_conv_direct: gpu.pipeline(
                "gemm_f32_conv_direct",
                &shaders::gemm_conv_direct(),
                "gemm",
            )?,
            gemm_conv_direct_batched: gpu.pipeline(
                "gemm_f32_conv_direct_batched",
                &shaders::gemm_conv_direct_batched(),
                "gemm",
            )?,
            gemm_transb_bn64: gpu.pipeline(
                "gemm_f32_transb_bn64",
                &shaders::gemm_transb_bn64(),
                "gemm",
            )?,
            gemm_batched_bn64: gpu.pipeline(
                "gemm_f32_batched_bn64",
                &shaders::gemm_batched_bn64(),
                "gemm",
            )?,
            gemm_batched_bn64_row_exp: gpu.pipeline(
                "gemm_f32_batched_bn64_row_exp",
                &shaders::gemm_batched_bn64_row_exp(),
                "gemm",
            )?,
            gemm_transb_bm64: gpu.pipeline(
                "gemm_f32_transb_bm64",
                &shaders::gemm_transb_bm64(),
                "gemm",
            )?,
            gemm_batched_bm64: gpu.pipeline(
                "gemm_f32_batched_bm64",
                &shaders::gemm_batched_bm64(),
                "gemm",
            )?,
            gemm_transb_plain_bias: gpu.pipeline(
                "gemm_f32_transb_plain_bias",
                &shaders::gemm_transb_plain_bias(),
                "gemm",
            )?,
            gemm_bias: gpu.pipeline("gemm_f32_bias", &shaders::gemm_bias(), "gemm")?,
            gemm_gelu_bias: gpu.pipeline(
                "gemm_f32_gelu_bias",
                &shaders::gemm_gelu_bias(),
                "gemm",
            )?,
            gemm_residual: gpu.pipeline(
                "gemm_f32_residual",
                &shaders::gemm_residual(),
                "gemm",
            )?,
            gemm_bias_residual: gpu.pipeline(
                "gemm_f32_bias_residual",
                &shaders::gemm_bias_residual(),
                "gemm",
            )?,
            rms_norm: gpu.pipeline("rms_norm", &shaders::rms_norm(), "rms_norm")?,
            rms_norm_warp: if gpu.info.shuffle_reduction {
                Some(gpu.pipeline("rms_norm_warp", &shaders::rms_norm_warp(), "rms_norm")?)
            } else {
                None
            },
            gelu: gpu.pipeline("gelu", &shaders::gelu(), "gelu")?,
            gelu_in_place: gpu.pipeline(
                "gelu_in_place",
                &shaders::gelu_in_place(),
                "gelu_in_place",
            )?,
            gelu_in_place_vec4: gpu.pipeline(
                "gelu_in_place_vec4",
                &shaders::gelu_in_place_vec4(),
                "gelu_in_place_vec4",
            )?,
            glu: gpu.pipeline("glu", &shaders::glu(), "glu")?,
            glu_channel_affine: gpu.pipeline(
                "glu_channel_affine",
                &shaders::glu_channel_affine(),
                "glu_channel_affine",
            )?,
            glu_channel_affine_add: gpu.pipeline(
                "glu_channel_affine_add",
                &shaders::glu_channel_affine_add(),
                "glu_channel_affine_add",
            )?,
            sigmoid_gate: gpu.pipeline(
                "sigmoid_gate",
                &shaders::sigmoid_gate(),
                "sigmoid_gate",
            )?,
            rope: gpu.pipeline("rope", &shaders::rope(), "rope")?,
            add_in_place: gpu.pipeline(
                "add_in_place",
                &shaders::add_in_place(),
                "add_in_place",
            )?,
            mul_in_place: gpu.pipeline(
                "mul_in_place",
                &shaders::mul_in_place(),
                "mul_in_place",
            )?,
            add_in_place_vec4: gpu.pipeline(
                "add_in_place_vec4",
                &shaders::add_in_place_vec4(),
                "add_in_place_vec4",
            )?,
            transpose: gpu.pipeline("transpose", &shaders::transpose(), "transpose")?,
            im2col: gpu.pipeline("im2col", &shaders::im2col(), "im2col")?,
            fill_zero: gpu.pipeline("fill_zero", &shaders::fill_zero(), "fill_zero")?,
            crop_rows: gpu.pipeline("crop_rows", &shaders::crop_rows(), "crop_rows")?,
            copy_pitched: gpu.pipeline(
                "copy_pitched",
                &shaders::copy_pitched(),
                "copy_pitched",
            )?,
            group_norm: gpu.pipeline("group_norm", &shaders::group_norm(), "group_norm")?,
            group_norm_partial: gpu.pipeline(
                "group_norm_partial",
                &shaders::group_norm_partial(),
                "group_norm_partial",
            )?,
            group_norm_combine: gpu.pipeline(
                "group_norm_combine",
                &shaders::group_norm_combine(),
                "group_norm_combine",
            )?,
            group_norm_apply: gpu.pipeline(
                "group_norm_apply",
                &shaders::group_norm_apply(),
                "group_norm_apply",
            )?,
            group_norm_apply_from_partials: gpu.pipeline(
                "group_norm_apply_from_partials",
                &shaders::group_norm_apply_from_partials(),
                "group_norm_apply",
            )?,
            lstm_recur: gpu.pipeline("lstm_recur", &shaders::lstm_recur(), "lstm_recur")?,
            lstm_recur_regs: std::cell::RefCell::new(std::collections::HashMap::new()),
            heads_permute: gpu.pipeline(
                "heads_permute",
                &shaders::heads_permute(),
                "heads_permute",
            )?,
            add_row_bias_in_place: gpu.pipeline(
                "add_row_bias_in_place",
                &shaders::add_row_bias_in_place(),
                "add_row_bias_in_place",
            )?,
            channel_affine_act_in_place: gpu.pipeline(
                "channel_affine_act_in_place",
                &shaders::channel_affine_act_in_place(),
                "channel_affine_act_in_place",
            )?,
            channel_affine_act_add: gpu.pipeline(
                "channel_affine_act_add",
                &shaders::channel_affine_act_add(),
                "channel_affine_act_add",
            )?,
            col2im: gpu.pipeline("col2im", &shaders::col2im(), "col2im")?,
            tanh: gpu.pipeline("tanh_activation", &shaders::tanh(), "tanh_activation")?,
            tanh_bias_in_place: gpu.pipeline(
                "tanh_bias_in_place",
                &shaders::tanh_bias_in_place(),
                "tanh_bias",
            )?,
            copy: gpu.pipeline("copy", &shaders::copy(), "copy")?,
            softmax: gpu.pipeline("softmax", &shaders::softmax(), "softmax")?,
            softmax_in_place: gpu.pipeline(
                "softmax_in_place",
                &shaders::softmax_in_place(),
                "softmax_in_place",
            )?,
            softmax_warp: if gpu.info.shuffle_reduction {
                Some(gpu.pipeline(
                    "softmax_in_place_warp",
                    &shaders::softmax_warp(),
                    "softmax_in_place",
                )?)
            } else {
                None
            },
            softmax_scaled_warp: if gpu.info.shuffle_reduction {
                Some(gpu.pipeline(
                    "softmax_scaled_warp",
                    &shaders::softmax_scaled_warp(),
                    "softmax",
                )?)
            } else {
                None
            },
            softmax_stats_warp: if gpu.info.shuffle_reduction {
                Some(gpu.pipeline(
                    "softmax_stats_warp",
                    &shaders::softmax_stats_warp(),
                    "softmax_stats",
                )?)
            } else {
                None
            },
            flash_attention: gpu.pipeline(
                "flash_attention",
                &shaders::flash_attention(),
                "flash_attention",
            )?,
            istft_irfft4096: gpu.pipeline(
                "istft_irfft4096",
                &shaders::istft_irfft4096(),
                "istft_irfft4096",
            )?,
            istft_ola: gpu.pipeline("istft_ola", &shaders::istft_ola(), "istft_ola")?,
            batch_affine_in_place: gpu.pipeline(
                "batch_affine_in_place",
                &shaders::batch_affine_in_place(),
                "batch_affine_in_place",
            )?,
            stft_rfft4096: gpu.pipeline(
                "stft_rfft4096",
                &shaders::stft_rfft4096(),
                "stft_rfft4096",
            )?,
            batch_moments: gpu.pipeline(
                "batch_moments",
                &shaders::batch_moments(),
                "batch_moments",
            )?,
            batch_moments_partial: gpu.pipeline(
                "batch_moments_partial",
                &shaders::batch_moments_partial(),
                "batch_moments_partial",
            )?,
            batch_moments_combine: gpu.pipeline(
                "batch_moments_combine",
                &shaders::batch_moments_combine(),
                "batch_moments_combine",
            )?,
            batch_normalize_in_place: gpu.pipeline(
                "batch_normalize_in_place",
                &shaders::batch_normalize_in_place(),
                "batch_normalize_in_place",
            )?,
        })
    }

    /// A 16-byte uniform slot in the per-forward pack. Offsets are unique, so
    /// every dispatch sees its own value; the bytes go out in one `write_buffer`
    /// at submit.
    fn params(&self, gpu: &Gpu, _arena: &mut Arena, values: [u32; 4]) -> Result<DevTensor> {
        gpu.push_uniform(bytemuck::cast_slice(&values))
    }

    /// Times the `nn.Linear` GEMM (the model's most expensive kernel family)
    /// against two mutants of its own source.
    ///
    /// The plateau is the open item in the handoff: the square-tile GEMM
    /// sustains ~3.3 TFLOP/s where a register-only FMA probe reaches 5.8, and
    /// no tile arithmetic explains the gap. Two mutants split it in two --
    /// `no_global` replaces every global `A`/`B` staging load with a constant
    /// (the shared staging, the LDS reads, the barriers and the FMAs all stay),
    /// and `no_barrier` drops only the `workgroupBarrier` calls. Both compute
    /// nonsense; they are timed, never checked.
    ///
    /// Probe, not a path: driven by
    /// `cargo test --release -p demucs-core --test gpu_gemm_plateau -- --ignored --nocapture`.
    pub fn plateau_probe(
        &self,
        gpu: &Gpu,
        repeats: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        // The Linear is single-batch, transposed, with the bias in the
        // epilogue: `gemm_transb_plain_bias` is its pipeline.
        let base = shaders::gemm_transb_plain_bias();
        // Each global load is pointed at element zero rather than replaced by a
        // constant: the bindings stay live (an auto layout drops what the
        // shader stops using, and the bind group would then be invalid) and the
        // loads still issue, but every one of them hits the same cache line.
        let strip = |source: &str, patterns: [&str; 2], reads_only: bool| -> String {
            let mut out = String::with_capacity(source.len());
            for line in source.split_inclusive('\n') {
                let mut rest = line;
                loop {
                    let hit = patterns
                        .iter()
                        .filter_map(|pat| rest.find(pat).map(|at| (at, *pat)))
                        .min_by_key(|(at, _)| *at);
                    let Some((start, pat)) = hit else { break };
                    let Some(len) = rest[start..].find(']') else {
                        break;
                    };
                    let after = &rest[start + len + 1..];
                    // A shared slice can be the target of a store as well as a
                    // load; only the loads are the traffic this mutant removes.
                    let store = reads_only && after.trim_start().starts_with('=');
                    out.push_str(&rest[..start]);
                    if store {
                        out.push_str(&rest[start..start + len + 1]);
                    } else {
                        out.push_str(pat);
                        out.push_str("0u]");
                    }
                    rest = after;
                }
                out.push_str(rest);
            }
            out
        };
        let no_global = strip(&base, ["A[", "B["], false);
        let no_barrier = base.replace("    workgroupBarrier();\n", "");
        // Shared-read mutant: every `As[...]`/`Bs[...]` load becomes the same
        // element of its row, so it still issues but its address stops moving —
        // and a load whose address is loop-invariant is free to float out of the
        // loop entirely, which is the point: it isolates the register-side read
        // traffic out of shared. A vec4 fragment is what would halve that count.
        let no_shared = strip(&base, ["As[", "Bs["], true);
        let loads = base.matches("A[").count() + base.matches("B[").count();
        let barriers = base.matches("workgroupBarrier();").count();
        println!(
            "plateau probe {m}x{n}x{k}: rewrote {loads} global loads, removed {barriers} barriers"
        );

        let pipelines = [
            ("real", gpu.pipeline("probe_real", &base, "gemm")?),
            (
                "no_global",
                gpu.pipeline("probe_no_global", &no_global, "gemm")?,
            ),
            (
                "no_barrier",
                gpu.pipeline("probe_no_barrier", &no_barrier, "gemm")?,
            ),
            (
                "no_shared",
                gpu.pipeline("probe_no_shared", &no_shared, "gemm")?,
            ),
        ];

        let lda = pad_ceil(k, shaders::BK);
        let ldb = pad_ceil(k, shaders::BN);
        let job = GemmJob {
            m,
            n,
            k,
            lda,
            ldb,
            ldc: n,
            batches: 1,
            inner_count: 1,
            a_outer: 0,
            a_inner: 0,
            b_outer: 0,
            b_inner: 0,
            c_outer: 0,
            c_inner: 0,
            transb: true,
        };
        let dims = GemmDims {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            lda: lda as u32,
            ldb: ldb as u32,
            ldc: n as u32,
            inner_count: 1,
            asa_outer: 0,
            asa_inner: 0,
            bsb_outer: 0,
            bsb_inner: 0,
            csc_outer: 0,
            csc_inner: 0,
            _pad: 0,
        };
        let mut arena = Arena::new(gpu, 1 << 30);
        let a = arena.tensor(gpu, &[pad_ceil(m, shaders::BM) * lda], "probe.a")?;
        let b = arena.tensor(gpu, &[pad_ceil(n, shaders::BN) * ldb], "probe.b")?;
        let c = arena.tensor(gpu, &[m * n], "probe.c")?;
        let bias = arena.tensor(gpu, &[n], "probe.bias")?;
        let grid = job.grid_with(shaders::BM, shaders::BN);
        let flops = 2.0 * (m as f64) * (n as f64) * (k as f64);

        for (label, pipeline) in &pipelines {
            let params = gpu.push_uniform(bytemuck::bytes_of(&dims))?;
            let group = bind_group(
                gpu,
                "plateau_probe",
                &pipeline.get_bind_group_layout(0),
                &[
                    (&a.buffer, a.offset, (a.len() * 4) as u64),
                    (&b.buffer, b.offset, (b.len() * 4) as u64),
                    (&c.buffer, c.offset, (c.len() * 4) as u64),
                    (&params.buffer, params.offset, 64),
                    (&bias.buffer, bias.offset, (bias.len() * 4) as u64),
                ],
            );
            let mut warm = Recorder::new(gpu);
            warm.dispatch(label, pipeline, &group, grid);
            warm.submit(gpu)?;
            gpu.device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| Error::Gpu(format!("probe warm-up poll: {e}")))?;

            let started = std::time::Instant::now();
            let mut recorder = Recorder::new(gpu);
            for _ in 0..repeats {
                recorder.dispatch(label, pipeline, &group, grid);
            }
            recorder.submit(gpu)?;
            gpu.device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| Error::Gpu(format!("probe poll: {e}")))?;
            let elapsed = started.elapsed().as_secs_f64();
            let per_dispatch = elapsed / repeats as f64;
            println!(
                "  {label:>10}: {:.3} ms/dispatch, {:.2} TFLOP/s",
                per_dispatch * 1e3,
                flops * repeats as f64 / elapsed / 1e12
            );
        }

        // What a submission boundary costs: the same dispatches, one per submit,
        // each with the wait the chunk loop pays. The difference against the
        // batched form is the fixed cost the chunk loop can only avoid by
        // submitting less often.
        let (_, pipeline) = &pipelines[0];
        let params = gpu.push_uniform(bytemuck::bytes_of(&dims))?;
        let group = bind_group(
            gpu,
            "plateau_probe_single",
            &pipeline.get_bind_group_layout(0),
            &[
                (&a.buffer, a.offset, (a.len() * 4) as u64),
                (&b.buffer, b.offset, (b.len() * 4) as u64),
                (&c.buffer, c.offset, (c.len() * 4) as u64),
                (&params.buffer, params.offset, 64),
                (&bias.buffer, bias.offset, (bias.len() * 4) as u64),
            ],
        );
        let started = std::time::Instant::now();
        for _ in 0..repeats {
            let mut recorder = Recorder::new(gpu);
            recorder.dispatch("probe_single", pipeline, &group, grid);
            recorder.submit(gpu)?;
            gpu.device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| Error::Gpu(format!("probe poll: {e}")))?;
        }
        let per_submit = started.elapsed().as_secs_f64() / repeats as f64;
        println!(
            "  submission boundary: {:.3} ms per 1-dispatch submit ({:.2} TFLOP/s)",
            per_submit * 1e3,
            flops / per_submit / 1e12
        );
        Ok(())
    }

    /// Times the `nn.Linear` GEMM (`A @ B^T + bias`, the `TransB` + `Bias` form)
    /// over the tiles the tree knows how to compile.
    ///
    /// The plateau probe puts the kernel's ceiling at 3.3 TFLOP/s against 5.8 for
    /// a register-only FMA probe, and rules out memory and barriers as the cause:
    /// it is latency-bound, so occupancy is the lever, and occupancy is set by
    /// the register tile — 128x128 with 16x16 threads gives every thread an 8x8
    /// accumulator, 64 registers that alone cap the workgroups per SM.
    ///
    /// The conv experiment that rejected the narrow tile used a shape whose grid
    /// did not fill the device (84 workgroups), which is a different question;
    /// these run the Linear shapes, where every tile gets a full grid.
    ///
    /// Probe, not a path: driven by
    /// `cargo test --release -p demucs-core --test gpu_gemm_plateau -- --ignored --nocapture`.
    pub fn tile_probe(
        &self,
        gpu: &Gpu,
        repeats: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        use shaders::{GemmEpilogue, Tile};
        let variants = [
            ("128x128 vec2", Tile::square(), false),
            ("128x128 vec4", Tile::square(), true),
            ("64x128 vec4", Tile::bm64_bn128(), true),
            ("96x128 vec4", Tile::bm96_bn128(), true),
            ("64x64 vec4", Tile::bm64_bn64(), true),
        ];

        let lda = pad_ceil(k, shaders::BK);
        let ldb = pad_ceil(k, shaders::BN);
        let job = GemmJob {
            m,
            n,
            k,
            lda,
            ldb,
            ldc: n,
            batches: 1,
            inner_count: 1,
            a_outer: 0,
            a_inner: 0,
            b_outer: 0,
            b_inner: 0,
            c_outer: 0,
            c_inner: 0,
            transb: true,
        };
        let dims = GemmDims {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            lda: lda as u32,
            ldb: ldb as u32,
            ldc: n as u32,
            inner_count: 1,
            asa_outer: 0,
            asa_inner: 0,
            bsb_outer: 0,
            bsb_inner: 0,
            csc_outer: 0,
            csc_inner: 0,
            _pad: 0,
        };
        let mut arena = Arena::new(gpu, 1 << 30);
        // Padded to the *widest* tile each operand is read with, so one set of
        // buffers serves every variant below.
        let a = arena.tensor(gpu, &[pad_ceil(m, shaders::BM) * lda], "tile.a")?;
        let b = arena.tensor(gpu, &[pad_ceil(n, shaders::BN) * ldb], "tile.b")?;
        let c = arena.tensor(gpu, &[m * n], "tile.c")?;
        let bias = arena.tensor(gpu, &[n], "tile.bias")?;
        let flops = 2.0 * (m as f64) * (n as f64) * (k as f64);
        println!("tile probe {m}x{n}x{k} (transb + bias):");

        for (label, tile, wide) in variants {
            let pipeline = gpu.pipeline(
                "tile_probe",
                &shaders::gemm_variant(tile, true, GemmEpilogue::Bias, wide),
                "gemm",
            )?;
            let params = gpu.push_uniform(bytemuck::bytes_of(&dims))?;
            let group = bind_group(
                gpu,
                "tile_probe",
                &pipeline.get_bind_group_layout(0),
                &[
                    (&a.buffer, a.offset, (a.len() * 4) as u64),
                    (&b.buffer, b.offset, (b.len() * 4) as u64),
                    (&c.buffer, c.offset, (c.len() * 4) as u64),
                    (&params.buffer, params.offset, 64),
                    (&bias.buffer, bias.offset, (bias.len() * 4) as u64),
                ],
            );
            // The grid follows the tile: the shader derives `m0`/`n0` from it.
            let grid = job.grid_with(tile.bm, tile.bn);
            let mut warm = Recorder::new(gpu);
            warm.dispatch(label, &pipeline, &group, grid);
            warm.submit(gpu)?;
            gpu.device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| Error::Gpu(format!("tile probe warm-up poll: {e}")))?;

            let started = std::time::Instant::now();
            let mut recorder = Recorder::new(gpu);
            for _ in 0..repeats {
                recorder.dispatch(label, &pipeline, &group, grid);
            }
            recorder.submit(gpu)?;
            gpu.device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| Error::Gpu(format!("tile probe poll: {e}")))?;
            let elapsed = started.elapsed().as_secs_f64();
            println!(
                "  {label:>16}: {:.3} ms/dispatch, {:.2} TFLOP/s  (grid {}x{})",
                elapsed / repeats as f64 * 1e3,
                flops * repeats as f64 / elapsed / 1e12,
                grid.0,
                grid.1
            );
        }
        Ok(())
    }

    /// `out[m, n] = x[m, k] @ w[k, n]`.
    ///
    /// `w` must already be zero-padded to `(round_up(k, BK), round_up(n, BN))` and
    /// `x` to `(round_up(m, BM), round_up(k, BK))` — the kernel's inner loop has
    /// no bounds checks and relies on the padding being zeros.
    pub fn linear(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        x: &DevTensor,
        w: &DevTensor,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<DevTensor> {
        let out = arena.tensor(gpu, &[m, n], "linear.out")?;
        self.linear_into(gpu, arena, recorder, x, w, &out, m, n, k)?;
        Ok(out)
    }

    /// Same, writing into a caller-provided output.
    ///
    /// This is the form the model uses for its hot path: a gigabyte-scale
    /// activation reallocated per chunk costs more than the kernel does, so the
    /// work buffers are allocated once for a given shape and rewritten.
    pub fn linear_into(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        x: &DevTensor,
        w: &DevTensor,
        out: &DevTensor,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        self.gemm_into(gpu, arena, recorder, x, w, out, GemmJob::simple(m, n, k))
    }

    /// One GEMM dispatch with an explicitly chosen pipeline and tile.
    ///
    /// The tile has to be stated rather than derived because the *grid* follows
    /// it — the shader computes `m0`/`n0` from its workgroup id and its own
    /// compiled-in tile, so a 64-row kernel dispatched on a 128-row grid computes
    /// half the rows and leaves the rest untouched.
    #[allow(clippy::too_many_arguments)]
    fn gemm_into_tile(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        a: &DevTensor,
        b: &DevTensor,
        c: &DevTensor,
        job: GemmJob,
        pipeline: &wgpu::ComputePipeline,
        tile: (usize, usize),
    ) -> Result<()> {
        if job.m == 0 || job.n == 0 || job.k == 0 {
            return Ok(());
        }
        let dims = GemmDims {
            m: job.m as u32,
            n: job.n as u32,
            k: job.k as u32,
            lda: job.lda as u32,
            ldb: job.ldb as u32,
            ldc: job.ldc as u32,
            inner_count: job.inner_count as u32,
            asa_outer: job.a_outer as u32,
            asa_inner: job.a_inner as u32,
            bsb_outer: job.b_outer as u32,
            bsb_inner: job.b_inner as u32,
            csc_outer: job.c_outer as u32,
            csc_inner: job.c_inner as u32,
            _pad: 0,
        };
        let params = gpu.push_uniform(bytemuck::bytes_of(&dims))?;
        let grid = job.grid_with(tile.0, tile.1);
        let group = bind_group(
            gpu,
            "gemm",
            &pipeline.get_bind_group_layout(0),
            &[
                (&a.buffer, a.offset, (a.len() * 4) as u64),
                (&b.buffer, b.offset, (b.len() * 4) as u64),
                (&c.buffer, c.offset, (c.len() * 4) as u64),
                (&params.buffer, params.offset, 64),
            ],
        );
        let name = match tile.0 {
            shaders::BM96 => "gemm_bm96",
            shaders::BM64 => "gemm_bm64",
            _ => "gemm",
        };
        recorder.dispatch(name, pipeline, &group, grid);
        Ok(())
    }

    /// `(max, 1/Σexp)` per row of the scaled softmax, without forming it.
    ///
    /// The scan is the one [`Kernels::softmax_scaled`] runs; this is the half of
    /// it that does not touch memory the AV product reads anyway. Needs the
    /// warp kernel's 32-lane subgroup reduction.
    pub fn softmax_stats(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        x: &DevTensor,
        stats: &DevTensor,
        rows: usize,
        cols: usize,
        scale: f32,
    ) -> Result<()> {
        let Some(pipeline) = self.softmax_stats_warp.as_ref() else {
            return Err(Error::Gpu(
                "softmax stats needs the 32-lane subgroup reduction".into(),
            ));
        };
        if stats.len() < 2 * rows {
            return Err(Error::Shape(format!(
                "softmax stats writes {} values for {rows} rows, the tensor holds {}",
                2 * rows,
                stats.len()
            )));
        }
        let params = self.params(gpu, arena, [rows as u32, cols as u32, scale.to_bits(), 0])?;
        let group = bind_group(
            gpu,
            "softmax_stats",
            &pipeline.get_bind_group_layout(0),
            &[
                (&x.buffer, x.offset, (x.len() * 4) as u64),
                (&stats.buffer, stats.offset, (stats.len() * 4) as u64),
                (&params.buffer, params.offset, 16),
            ],
        );
        let (gx, gy) = row_grid(rows.div_ceil(shaders::SOFTMAX_WARP_ROWS_PER_WG));
        recorder.dispatch("softmax_stats", pipeline, &group, (gx, gy, 1));
        Ok(())
    }

    /// The attention's `P·V` straight off the raw score matrix: the `A` staging
    /// applies `exp(a * scale - max_row) * inv_row` from `stats`, which is what
    /// lets [`Kernels::softmax_stats`] skip materialising the probabilities.
    ///
    /// `job` is the one the `(probabilities, v)` product would take; `stats`
    /// holds two floats per scores row, `(max, 1/Σexp)`.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_scores_av_into(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        scores: &DevTensor,
        stats: &DevTensor,
        b: &DevTensor,
        c: &DevTensor,
        job: GemmJob,
        scale: f32,
    ) -> Result<()> {
        if job.m == 0 || job.n == 0 || job.k == 0 {
            return Ok(());
        }
        if job.lda == 0 || job.a_inner % job.lda != 0 {
            return Err(Error::Shape(format!(
                "the folded softmax needs whole rows: a_inner {} is not a multiple of lda {}",
                job.a_inner, job.lda
            )));
        }
        let rows_per_batch = job.a_inner / job.lda;
        let need = 2 * job.batches * rows_per_batch;
        if stats.len() < need {
            return Err(Error::Shape(format!(
                "the folded softmax needs {need} statistics, the tensor holds {}",
                stats.len()
            )));
        }
        let dims = GemmDims {
            m: job.m as u32,
            n: job.n as u32,
            k: job.k as u32,
            lda: job.lda as u32,
            ldb: job.ldb as u32,
            ldc: job.ldc as u32,
            inner_count: job.inner_count as u32,
            asa_outer: job.a_outer as u32,
            asa_inner: job.a_inner as u32,
            bsb_outer: job.b_outer as u32,
            bsb_inner: job.b_inner as u32,
            csc_outer: job.c_outer as u32,
            csc_inner: job.c_inner as u32,
            _pad: 0,
        };
        let params = gpu.push_uniform(bytemuck::bytes_of(&dims))?;
        let ae = self.params(gpu, arena, [scale.to_bits(), rows_per_batch as u32, 0, 0])?;
        let group = bind_group(
            gpu,
            "gemm_av_row_exp",
            &self.gemm_batched_bn64_row_exp.get_bind_group_layout(0),
            &[
                (&scores.buffer, scores.offset, (scores.len() * 4) as u64),
                (&b.buffer, b.offset, (b.len() * 4) as u64),
                (&c.buffer, c.offset, (c.len() * 4) as u64),
                (&params.buffer, params.offset, 64),
                (&stats.buffer, stats.offset, (stats.len() * 4) as u64),
                (&ae.buffer, ae.offset, 16),
            ],
        );
        let grid = job.grid_with(shaders::BM, 64);
        recorder.dispatch(
            "gemm_av_row_exp",
            &self.gemm_batched_bn64_row_exp,
            &group,
            grid,
        );
        Ok(())
    }

    /// One convolution as an implicit GEMM: the same tile and the same compute
    /// loop as [`Kernels::gemm_into_tile`], with the im2col gather folded into its
    /// `B` staging (see `shaders::gemm_conv_direct`).
    ///
    /// `x` is the layer's input, `(batch, in_channels, h, w)`; the shader reads it
    /// as the patch matrix would have been. `batched` selects the variant with
    /// the batch base terms emitted — the plain form would gather every batch's
    /// output from batch 0's input.
    #[allow(clippy::too_many_arguments)]
    fn gemm_conv_into(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        weight: &DevTensor,
        x: &DevTensor,
        c: &DevTensor,
        job: GemmJob,
        shape: Im2ColShape,
        batched: bool,
    ) -> Result<()> {
        let (out_h, out_w) = shape.out_hw();
        for (name, value) in [
            ("in_channels", shape.in_channels),
            ("kh", shape.kernel.0),
            ("kw", shape.kernel.1),
            ("stride_w", shape.stride.1),
            ("stride_h", shape.stride.0),
            ("pad_w", shape.pad.1),
            ("pad_h", shape.pad.0),
            ("h", shape.h),
            ("w", shape.w),
        ] {
            if value > u16::MAX as usize {
                return Err(Error::Shape(format!(
                    "conv-direct parameter {name}={value} does not fit the kernel's 16-bit packing"
                )));
            }
        }
        let cd = self.params(
            gpu,
            arena,
            [
                out_w as u32,
                out_h as u32,
                shape.w as u32,
                shape.h as u32,
            ],
        )?;
        let ce = self.params(
            gpu,
            arena,
            [
                pack(shape.stride.1, shape.stride.0),
                pack(shape.pad.1, shape.pad.0),
                pack(shape.kernel.0, shape.kernel.1),
                shape.in_channels as u32,
            ],
        )?;
        // In the direct form the input itself is the patch matrix, so `B`'s batch
        // stride is the input's per-batch block, not the pitched patch plane.
        let per_batch_in = shape.in_channels * shape.h * shape.w;
        let job = if batched {
            GemmJob { b_outer: per_batch_in, ..job }
        } else {
            job
        };
        let dims = GemmDims {
            m: job.m as u32,
            n: job.n as u32,
            k: job.k as u32,
            lda: job.lda as u32,
            ldb: job.ldb as u32,
            ldc: job.ldc as u32,
            inner_count: job.inner_count as u32,
            asa_outer: job.a_outer as u32,
            asa_inner: job.a_inner as u32,
            bsb_outer: job.b_outer as u32,
            bsb_inner: job.b_inner as u32,
            csc_outer: job.c_outer as u32,
            csc_inner: job.c_inner as u32,
            _pad: 0,
        };
        let params = gpu.push_uniform(bytemuck::bytes_of(&dims))?;
        let per_batch_out = job.m * job.n;
        // Batch 0's block is bound for the plain form; the batched one needs the
        // whole tensor because its gather offsets into each batch's own block.
        let (batch_x, batch_c) = if batched {
            (x.clone(), c.clone())
        } else {
            (x.slice(0, vec![per_batch_in])?, c.slice(0, vec![per_batch_out])?)
        };
        let (name, pipeline) = if batched {
            ("gemm_conv_direct_batched", &self.gemm_conv_direct_batched)
        } else {
            ("gemm_conv_direct", &self.gemm_conv_direct)
        };
        let group = bind_group(
            gpu,
            name,
            &pipeline.get_bind_group_layout(0),
            &[
                (&weight.buffer, weight.offset, (weight.len() * 4) as u64),
                (&batch_x.buffer, batch_x.offset, (batch_x.len() * 4) as u64),
                (&batch_c.buffer, batch_c.offset, (batch_c.len() * 4) as u64),
                (&params.buffer, params.offset, 64),
                (&cd.buffer, cd.offset, 16),
                (&ce.buffer, ce.offset, 16),
            ],
        );
        let grid = job.grid_with(shaders::BM64, shaders::BN);
        recorder.dispatch(name, pipeline, &group, grid);
        Ok(())
    }

    /// The general form: one dispatch of the batched GEMM described by `job`.
    ///
    /// All operands are read and written in place; nothing is allocated here, so
    /// the caller decides what lives in the resident arena and what is per-chunk.
    pub fn gemm_into(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        a: &DevTensor,
        b: &DevTensor,
        c: &DevTensor,
        job: GemmJob,
    ) -> Result<()> {
        if job.m == 0 || job.n == 0 || job.k == 0 {
            return Ok(());
        }
        if job.batches > 65535 {
            return Err(Error::Gpu(format!(
                "{} batches overflows the z grid dimension",
                job.batches
            )));
        }
        if job.inner_count == 0 {
            return Err(Error::Gpu("inner_count must be at least 1".into()));
        }
        // Bound the highest element a *valid* row can reach. The base of the last
        // batch is the largest one, so it — not an independent maximum of each
        // level — is what the bound is built from.
        //
        // The kernel stages whole `BM`x`BN` tiles, so it also reads up to
        // `pad_ceil(m, BM) - 1` rows of A and `pad_ceil(n, BN) - 1` rows of B.
        // Those reads are deliberate and harmless: a padded row only feeds
        // accumulators whose index the epilogue never writes, and wgpu returns
        // zero for a storage read past the end of a buffer.
        let last = job.batches - 1;
        let last_outer = last / job.inner_count;
        let last_inner = last % job.inner_count;
        let a_need =
            job.a_outer * last_outer + job.a_inner * last_inner + (job.m - 1) * job.lda + job.k;
        let b_base = job.b_outer * last_outer + job.b_inner * last_inner;
        let b_need = if job.transb {
            // B is stored `(n, k)`: its rows are the n axis.
            b_base + (job.n - 1) * job.ldb + job.k
        } else {
            b_base + (job.k - 1) * job.ldb + job.n
        };
        let c_need = job.c_outer * last_outer
            + job.c_inner * last_inner
            + (job.m - 1) * job.ldc
            + job.n;
        for (label, have, need) in [
            ("A", a.len(), a_need),
            ("B", b.len(), b_need),
            ("C", c.len(), c_need),
        ] {
            if have < need {
                return Err(Error::Shape(format!(
                    "GEMM operand {label} holds {have} elements, the job indexes up to {need}"
                )));
            }
        }
        let dims = GemmDims {
            m: job.m as u32,
            n: job.n as u32,
            k: job.k as u32,
            lda: job.lda as u32,
            ldb: job.ldb as u32,
            ldc: job.ldc as u32,
            inner_count: job.inner_count.max(1) as u32,
            asa_outer: job.a_outer as u32,
            asa_inner: job.a_inner as u32,
            bsb_outer: job.b_outer as u32,
            bsb_inner: job.b_inner as u32,
            csc_outer: job.c_outer as u32,
            csc_inner: job.c_inner as u32,
            _pad: 0,
        };
        // The whole Dims struct, not just the first four fields: the batched
        // pipelines read asa/bsb/csc from the tail, and with a 16-byte upload
        // those strides come from uninitialised arena memory. Single-batch jobs
        // never noticed (their batch base is zero), so the first batched caller
        // of this entry point was the one that found it.
        let params = gpu.push_uniform(bytemuck::bytes_of(&dims))?;

        // The plain pipeline hard-codes a zero batch base, so it is only valid for
        // a single batch; everything else takes a batched form, which covers
        // `inner_count == 1` (a plain stride) as well as the two-level case.
        // The AV product's `n = dim_head = 64` takes a 64-wide N tile: half of
        // a 128-wide tile would compute nothing but zeros. So does `Q K^T` when
        // the sequence is the frequency axis' 60 bands, where the same argument
        // applies to the transposed form. The grid follows the tile the pipeline
        // was compiled with -- the shader derives `m0`/`n0` from it.
        let square = (shaders::BM, shaders::BN);
        let n64 = (shaders::BM, shaders::Tile::bn64().bn);
        let small = (
            shaders::Tile::bm64_bn64().bm,
            shaders::Tile::bm64_bn64().bn,
        );
        // Both choices are about the same thing: how much of the tile computes
        // zeros. A 64-wide N is right whenever `n <= 64` (the AV product's
        // `dim_head`, the frequency axis' keys); a 64-wide M only where the M
        // itself is tiny, which is the frequency axis and nothing else -- the
        // time axis' 801 queries would lose far more to the smaller tile's
        // worse loads-per-FMA than they gain from the padding.
        let narrow_m = job.m <= 64;
        let (pipeline, name, tile) = match (job.batches == 1, job.transb, job.n <= 64) {
            (_, true, true) if narrow_m => (&self.gemm_transb_bm64, "gemm_transb_bm64", small),
            (_, true, true) => (&self.gemm_transb_bn64, "gemm_transb_bn64", n64),
            (false, false, true) if narrow_m => {
                (&self.gemm_batched_bm64, "gemm_batched_bm64", small)
            }
            (false, false, true) => (&self.gemm_batched_bn64, "gemm_batched_bn64", n64),
            (true, false, _) => (&self.gemm, "gemm", square),
            (false, false, false) => (&self.gemm_batched, "gemm_batched", square),
            (true, true, false) if shaders::transb_plain_for_single_batch() => {
                (&self.gemm_transb_plain, "gemm_transb", square)
            }
            (_, true, false) => (&self.gemm_transb, "gemm_transb", square),
        };
        let grid = job.grid_with(tile.0, tile.1);
        let group = bind_group(
            gpu,
            "gemm",
            &pipeline.get_bind_group_layout(0),
            &[
                (&a.buffer, a.offset, (a.len() * 4) as u64),
                (&b.buffer, b.offset, (b.len() * 4) as u64),
                (&c.buffer, c.offset, (c.len() * 4) as u64),
                (&params.buffer, params.offset, 64),
            ],
        );
        recorder.dispatch(name, pipeline, &group, grid);
        Ok(())
    }

    /// `gemm_into` with the `nn.Linear` bias — and optionally the feed-forward
    /// GELU — folded into the store. Plain single-batch only: every `nn.Linear`
    /// is, and the epilogue pipelines are compiled for that form.
    ///
    /// The folded sequence replays the unfused one exactly (`acc + bias`, then
    /// erf of that sum), so the outputs are bit-identical and the separate
    /// `add_in_place` / `gelu_in_place` dispatches — a full extra
    /// read-modify-write of the widest activation in the model — disappear.
    ///
    /// `residual` folds the block's skip connection in as well: the epilogue
    /// reads the destination element back and adds it after the bias, which is
    /// the order the unfused `x += normed` added it in, so that too is
    /// bit-identical — and it removes the intermediate write entirely.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_into_epilogue(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        a: &DevTensor,
        b: &DevTensor,
        c: &DevTensor,
        job: GemmJob,
        bias: Option<&DevTensor>,
        gelu: bool,
        residual: bool,
    ) -> Result<()> {
        if job.m == 0 || job.n == 0 || job.k == 0 {
            return Ok(());
        }
        if job.batches != 1 {
            return Err(Error::Gpu(
                "the bias/GELU epilogue is compiled for single-batch GEMMs only".into(),
            ));
        }
        let dims = GemmDims {
            m: job.m as u32,
            n: job.n as u32,
            k: job.k as u32,
            lda: job.lda as u32,
            ldb: job.ldb as u32,
            ldc: job.ldc as u32,
            inner_count: 1,
            asa_outer: 0,
            asa_inner: 0,
            bsb_outer: 0,
            bsb_inner: 0,
            csc_outer: 0,
            csc_inner: 0,
            _pad: 0,
        };
        let params = gpu.push_uniform(bytemuck::bytes_of(&dims))?;
        // The grid has to follow whichever pipeline was chosen: the shader
        // derives `m0` from the workgroup id and its own `BM`, so a 256-row
        // kernel dispatched on a 128-row grid computes half the rows.
        let square = (shaders::BM, shaders::BN);
        // A narrow tile is NOT a way to raise occupancy for a small GEMM. Tried
        // on the mask estimator's `801 x 1536` (84 workgroups at 128x128, its
        // measured 1.19 TFLOP/s against the same kernel's 3.5 where the grid
        // fills): 64x64 quadruples the workgroup count but halves the
        // loads-per-FMA from 1:8 to 1:4, and the whole forward came out 1%
        // *slower* (1147 ms against 1136). The fix for that shape is fewer,
        // bigger dispatches -- the bands batched into one grid -- not a
        // different tile.
        let (pipeline, tile) = match (job.transb, bias.is_some(), gelu, residual) {
            (true, true, false, false) => (&self.gemm_transb_plain_bias, square),
            (true, _, _, _) => {
                return Err(Error::Gpu(
                    "the transb epilogue is bias-only, single-batch".into(),
                ));
            }
            (false, true, true, _) => (&self.gemm_gelu_bias, square),
            (false, true, false, false) => (&self.gemm_bias, square),
            (false, false, _, false) => {
                return Err(Error::Gpu("the plain epilogue is `gemm_into`".into()))
            }
            (false, false, _, true) => (&self.gemm_residual, square),
            (false, true, false, true) => (&self.gemm_bias_residual, square),
        };
        let grid = job.grid_with(tile.0, tile.1);
        // The bias binding is declared only when the epilogue uses it: a declared
        // but unbound binding fails pipeline validation, and declaring it for the
        // residual-only form would force a dummy tensor into every call.
        // The type annotation is what turns `&tensor.buffer` into the
        // `&wgpu::Buffer` the binding table wants: without it inference fixes
        // the tuple to `&Arc<Slot>` and the coercion never happens.
        let mut entries: Vec<(&wgpu::Buffer, u64, u64)> = vec![
            (&a.buffer, a.offset, (a.len() * 4) as u64),
            (&b.buffer, b.offset, (b.len() * 4) as u64),
            (&c.buffer, c.offset, (c.len() * 4) as u64),
            (&params.buffer, params.offset, 64),
        ];
        if let Some(bias) = bias {
            entries.push((&bias.buffer, bias.offset, (bias.len() * 4) as u64));
        }
        let group = bind_group(
            gpu,
            "gemm_epilogue",
            &pipeline.get_bind_group_layout(0),
            &entries,
        );
        recorder.dispatch("gemm_bias", pipeline, &group, grid);
        Ok(())
    }

    /// `F.normalize(x) * sqrt(dim) * gamma`, one workgroup per row.
    pub fn rms_norm(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        x: &DevTensor,
        gamma: &DevTensor,
        rows: usize,
        dim: usize,
    ) -> Result<DevTensor> {
        let out = arena.tensor(gpu, &[rows, dim], "rms_norm.out")?;
        self.rms_norm_into(gpu, arena, recorder, x, gamma, &out, rows, dim)?;
        Ok(out)
    }

    /// Same, into a caller-provided output.
    ///
    /// The model calls this four times per transformer block over tensors of tens
    /// of megabytes; allocating each time would run a chunk out of memory long
    /// before it ran out of work.
    pub fn rms_norm_into(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        x: &DevTensor,
        gamma: &DevTensor,
        out: &DevTensor,
        rows: usize,
        dim: usize,
    ) -> Result<()> {
        self.rms_norm_pitched(gpu, arena, recorder, x, gamma, out, rows, dim, dim, dim)
    }

    /// Same, with explicit row pitches.
    ///
    /// A norm whose feature width is not a whole number of GEMM reduction steps
    /// has to write its output pitched, with the padding cleared, so the
    /// consuming GEMM cannot read the next row's values as if they were features.
    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_pitched(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        x: &DevTensor,
        gamma: &DevTensor,
        out: &DevTensor,
        rows: usize,
        dim: usize,
        pitch_in: usize,
        pitch_out: usize,
    ) -> Result<()> {
        let params = self.params(
            gpu,
            arena,
            [rows as u32, dim as u32, pitch_in as u32, pitch_out as u32],
        )?;
        // Rows the warp kernel covers take it: its reduction is one shuffle
        // rather than eight barriers, and it keeps the row in registers instead
        // of reading it twice. Wider rows and adapters without a 32-lane
        // subgroup stay on the tree.
        let warp = match &self.rms_norm_warp {
            Some(pipeline) if dim <= shaders::RMS_WARP_COLS => Some(pipeline),
            _ => None,
        };
        let pipeline = warp.unwrap_or(&self.rms_norm);
        let group = bind_group(
            gpu,
            "rms_norm",
            &pipeline.get_bind_group_layout(0),
            &[
                (&x.buffer, x.offset, (x.len() * 4) as u64),
                (&gamma.buffer, gamma.offset, (gamma.len() * 4) as u64),
                (&out.buffer, out.offset, (out.len() * 4) as u64),
                (&params.buffer, params.offset, 16),
            ],
        );
        let per_wg = if warp.is_some() {
            shaders::RMS_WARP_ROWS_PER_WG
        } else {
            1
        };
        let (gx, gy) = row_grid(rows.div_ceil(per_wg));
        recorder.dispatch("rms_norm", pipeline, &group, (gx, gy, 1));
        Ok(())
    }

    pub fn gelu(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        x: &DevTensor,
    ) -> Result<DevTensor> {
        let out = arena.tensor(gpu, &x.shape, "gelu.out")?;
        self.gelu_into(gpu, arena, recorder, x, &out)?;
        Ok(out)
    }

    /// Same, into a caller-provided output.
    pub fn gelu_into(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        x: &DevTensor,
        out: &DevTensor,
    ) -> Result<()> {
        let count = x.len();
        let params = self.params(gpu, arena, [count as u32, 0, 0, 0])?;
        let group = bind_group(
            gpu,
            "gelu",
            &self.gelu.get_bind_group_layout(0),
            &[
                (&x.buffer, x.offset, (count * 4) as u64),
                (&out.buffer, out.offset, (count * 4) as u64),
                (&params.buffer, params.offset, 16),
            ],
        );
        // The shader derives its flat index from the workgroup, so the grid is
        // counted in workgroups and the second axis covers anything past the
        // per-axis cap.
        let (gx, gy) = row_grid(count.div_ceil(ROW_THREADS));
        recorder.dispatch("gelu", &self.gelu, &group, (gx, gy, 1));
        Ok(())
    }

    /// GELU written back over its input.
    ///
    /// A buffer cannot be a read-only and a read-write binding in the same
    /// dispatch, so the in-place form needs a shader of its own rather than
    /// pointing the out-of-place one at its own input.
    pub fn gelu_in_place(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        x: &DevTensor,
    ) -> Result<()> {
        let count = x.len();
        let params = self.params(gpu, arena, [count as u32, 0, 0, 0])?;
        // The vec4 variant measured slightly *slower* than the scalar one
        // (2.36 ms against 2.09 ms at full-chunk size) -- the erf chain is
        // ALU-bound, not instruction-issue-bound, so wider lanes only add
        // register pressure. The pipeline stays compiled for experiments.
        let group = bind_group(
            gpu,
            "gelu_in_place",
            &self.gelu_in_place.get_bind_group_layout(0),
            &[
                (&x.buffer, x.offset, (count * 4) as u64),
                (&params.buffer, params.offset, 16),
            ],
        );
        let (gx, gy) = row_grid(count.div_ceil(ROW_THREADS));
        recorder.dispatch("gelu_in_place", &self.gelu_in_place, &group, (gx, gy, 1));
        Ok(())
    }

    /// `nn.GLU(dim=-1)`: the last axis halves.
    pub fn glu(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        x: &DevTensor,
        rows: usize,
        half: usize,
    ) -> Result<DevTensor> {
        let out = arena.tensor(gpu, &[rows, half], "glu.out")?;
        self.glu_into(gpu, arena, recorder, x, &out, rows, half)?;
        Ok(out)
    }

    /// The DConv's `nn.GLU` and the `nn.LayerScale` that follows it in one pass:
    /// `out = glu(x) * scale + shift`.
    ///
    /// The two are adjacent elementwise passes over the same activation, so the
    /// fused store produces exactly what the pair does — the GLU's value is the
    /// same expression and the scale/shift is the same arithmetic, applied on
    /// the way out instead of on a second read-modify-write.
    #[allow(clippy::too_many_arguments)]
    pub fn glu_channel_affine_into(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        x: &DevTensor,
        scale: &DevTensor,
        shift: &DevTensor,
        out: &DevTensor,
        rows: usize,
        half: usize,
        channels: usize,
        plane: usize,
    ) -> Result<()> {
        let count = rows * half;
        let params = self.params(
            gpu,
            arena,
            [rows as u32, half as u32, channels as u32, plane as u32],
        )?;
        let group = bind_group(
            gpu,
            "glu_channel_affine",
            &self.glu_channel_affine.get_bind_group_layout(0),
            &[
                (&x.buffer, x.offset, (x.len() * 4) as u64),
                (&out.buffer, out.offset, (count * 4) as u64),
                (&scale.buffer, scale.offset, (scale.len() * 4) as u64),
                (&shift.buffer, shift.offset, (shift.len() * 4) as u64),
                (&params.buffer, params.offset, 16),
            ],
        );
        let (gx, gy) = row_grid(count.div_ceil(ROW_THREADS));
        recorder.dispatch("glu_channel_affine", &self.glu_channel_affine, &group, (gx, gy, 1));
        Ok(())
    }

    /// [`Kernels::glu_channel_affine_into`] with the DConv layer's residual in
    /// the same pass: `out = base + (glu(x) * scale + shift)`.
    ///
    /// The reference's `out = current + gamma * glu(x)` is a GLU, a LayerScale
    /// and an add over the same tensor; this is one pass over it. The
    /// parenthesised order is the three-pass one's, so the sums round the same
    /// way rather than merely to within an epsilon.
    #[allow(clippy::too_many_arguments)]
    pub fn glu_channel_affine_add_into(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        base: &DevTensor,
        x: &DevTensor,
        scale: &DevTensor,
        shift: &DevTensor,
        out: &DevTensor,
        rows: usize,
        half: usize,
        channels: usize,
        plane: usize,
    ) -> Result<()> {
        let count = rows * half;
        let params = self.params(
            gpu,
            arena,
            [rows as u32, half as u32, channels as u32, plane as u32],
        )?;
        let group = bind_group(
            gpu,
            "glu_channel_affine_add",
            &self.glu_channel_affine_add.get_bind_group_layout(0),
            &[
                (&x.buffer, x.offset, (x.len() * 4) as u64),
                (&out.buffer, out.offset, (count * 4) as u64),
                (&scale.buffer, scale.offset, (scale.len() * 4) as u64),
                (&shift.buffer, shift.offset, (shift.len() * 4) as u64),
                (&params.buffer, params.offset, 16),
                (&base.buffer, base.offset, (base.len() * 4) as u64),
            ],
        );
        let (gx, gy) = row_grid(count.div_ceil(ROW_THREADS));
        recorder.dispatch(
            "glu_channel_affine_add",
            &self.glu_channel_affine_add,
            &group,
            (gx, gy, 1),
        );
        Ok(())
    }

    /// Same, into a caller-provided output.
    pub fn glu_into(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        x: &DevTensor,
        out: &DevTensor,
        rows: usize,
        half: usize,
    ) -> Result<()> {
        let count = rows * half;
        let params = self.params(gpu, arena, [rows as u32, half as u32, 0, 0])?;
        let group = bind_group(
            gpu,
            "glu",
            &self.glu.get_bind_group_layout(0),
            &[
                (&x.buffer, x.offset, (x.len() * 4) as u64),
                (&out.buffer, out.offset, (count * 4) as u64),
                (&params.buffer, params.offset, 16),
            ],
        );
        let (gx, gy) = row_grid(count.div_ceil(ROW_THREADS));
        recorder.dispatch("glu", &self.glu, &group, (gx, gy, 1));
        Ok(())
    }

    pub fn tanh(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        x: &DevTensor,
    ) -> Result<DevTensor> {
        let out = arena.tensor(gpu, &x.shape, "tanh.out")?;
        self.tanh_into(gpu, arena, recorder, x, &out)?;
        Ok(out)
    }

    pub fn tanh_into(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        x: &DevTensor,
        out: &DevTensor,
    ) -> Result<()> {
        let count = x.len();
        let params = self.params(gpu, arena, [count as u32, 0, 0, 0])?;
        let group = bind_group(
            gpu,
            "tanh_activation",
            &self.tanh.get_bind_group_layout(0),
            &[
                (&x.buffer, x.offset, (count * 4) as u64),
                (&out.buffer, out.offset, (count * 4) as u64),
                (&params.buffer, params.offset, 16),
            ],
        );
        let (gx, gy) = row_grid(count.div_ceil(ROW_THREADS));
        recorder.dispatch("tanh", &self.tanh, &group, (gx, gy, 1));
        Ok(())
    }

    /// `x = tanh(x + Bias[row / frames][col])`, one dispatch over a batch of
    /// rows. Fused because the batched GEMM has no epilogue: this is where the
    /// mask estimator's per-band bias goes when its bands share a grid.
    pub fn tanh_bias_in_place(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        x: &DevTensor,
        bias: &DevTensor,
        cols: usize,
        frames: usize,
    ) -> Result<()> {
        let count = x.len();
        let params = self.params(gpu, arena, [count as u32, cols as u32, frames as u32, 0])?;
        let group = bind_group(
            gpu,
            "tanh_bias_in_place",
            &self.tanh_bias_in_place.get_bind_group_layout(0),
            &[
                (&x.buffer, x.offset, (count * 4) as u64),
                (&bias.buffer, bias.offset, (bias.len() * 4) as u64),
                (&params.buffer, params.offset, 16),
            ],
        );
        let (gx, gy) = row_grid(count.div_ceil(ROW_THREADS));
        recorder.dispatch("tanh_bias", &self.tanh_bias_in_place, &group, (gx, gy, 1));
        Ok(())
    }

    /// `out = in` over `count` elements.
    pub fn copy(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        x: &DevTensor,
        out: &DevTensor,
        count: usize,
    ) -> Result<()> {
        let params = self.params(gpu, arena, [count as u32, 0, 0, 0])?;
        let group = bind_group(
            gpu,
            "copy",
            &self.copy.get_bind_group_layout(0),
            &[
                (&x.buffer, x.offset, (count * 4) as u64),
                (&out.buffer, out.offset, (count * 4) as u64),
                (&params.buffer, params.offset, 16),
            ],
        );
        let (gx, gy) = row_grid(count.div_ceil(ROW_THREADS));
        recorder.dispatch("copy", &self.copy, &group, (gx, gy, 1));
        Ok(())
    }

    /// `out = x * sigmoid(gate)` where the gate is indexed per head.
    ///
    /// The row count is implicit: element `i` of a row-major
    /// `(rows, heads * dim_head)` tensor sits in head `(i % (heads*dim_head)) /
    /// dim_head` of row `i / (heads*dim_head)`, which the shader derives itself.
    pub fn sigmoid_gate(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        x: &DevTensor,
        gate: &DevTensor,
        heads: usize,
        dim_head: usize,
    ) -> Result<DevTensor> {
        let out = arena.tensor(gpu, &x.shape, "gate.out")?;
        self.sigmoid_gate_into(gpu, arena, recorder, x, gate, &out, heads, dim_head)?;
        Ok(out)
    }

    /// Same, writing into caller-owned memory.
    ///
    /// The model runs this once per axial transformer per layer, so the one-shot
    /// version's fresh allocation per call is enough to exhaust a card over a
    /// chunk.
    pub fn sigmoid_gate_into(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        x: &DevTensor,
        gate: &DevTensor,
        out: &DevTensor,
        heads: usize,
        dim_head: usize,
    ) -> Result<()> {
        let count = x.len();
        let params = self.params(
            gpu,
            arena,
            [count as u32, heads as u32, dim_head as u32, 0],
        )?;
        let group = bind_group(
            gpu,
            "sigmoid_gate",
            &self.sigmoid_gate.get_bind_group_layout(0),
            &[
                (&x.buffer, x.offset, (count * 4) as u64),
                (&gate.buffer, gate.offset, (gate.len() * 4) as u64),
                (&out.buffer, out.offset, (count * 4) as u64),
                (&params.buffer, params.offset, 16),
            ],
        );
        let (gx, gy) = row_grid(count.div_ceil(ROW_THREADS));
        recorder.dispatch("sigmoid_gate", &self.sigmoid_gate, &group, (gx, gy, 1));
        Ok(())
    }

    /// Interleaved RoPE applied to a head range of a fused QKV tensor.
    ///
    /// `qkv` is `(bands, frames, 3 * heads * dim_head)` row-major. Q is heads
    /// `0..heads` and K is `heads..2*heads`; V is untouched. The head range is a
    /// parameter rather than two separate kernels because the rotation is
    /// identical and only the offset differs.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_heads(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        qkv: &DevTensor,
        freqs: &DevTensor,
        bands: usize,
        frames: usize,
        heads: usize,
        dim_head: usize,
        head_begin: usize,
        head_count: usize,
    ) -> Result<()> {
        let row_stride = 3 * heads * dim_head;
        let total_rows = bands * frames;
        let params0 = gpu.push_uniform(bytemuck::cast_slice(&[
            frames as u32,
            row_stride as u32,
            dim_head as u32,
            head_begin as u32,
        ]))?;
        let params1 = gpu.push_uniform(bytemuck::cast_slice(&[
            total_rows as u32,
            head_count as u32,
            0u32,
            0u32,
        ]))?;
        let group = bind_group(
            gpu,
            "rope",
            &self.rope.get_bind_group_layout(0),
            &[
                (&qkv.buffer, qkv.offset, (qkv.len() * 4) as u64),
                (&freqs.buffer, freqs.offset, (freqs.len() * 4) as u64),
                (&params0.buffer, params0.offset, 16),
                (&params1.buffer, params1.offset, 16),
            ],
        );
        let pairs = total_rows * (dim_head / 2);
        recorder.dispatch(
            "rope",
            &self.rope,
            &group,
            (pairs.div_ceil(shaders::ROW_THREADS as usize) as u32, 1, 1),
        );
        Ok(())
    }

    /// `out += source`.
    ///
    /// `source` either matches `out` element for element (a residual) or is
    /// shorter and repeats along the last axis (a bias). One kernel serves both
    /// because the model needs both and two implementations would be two things
    /// to keep aligned.
    pub fn add_in_place(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        out: &DevTensor,
        source: &DevTensor,
    ) -> Result<()> {
        let count = out.len();
        let src_len = source.len();
        if src_len != count && count % src_len != 0 {
            return Err(Error::Shape(format!(
                "cannot add {} elements into {count}: the source must match or divide it",
                src_len
            )));
        }
        let broadcast = if src_len == count { 0 } else { src_len as u32 };
        let params = self.params(gpu, arena, [count as u32, broadcast, 0, 0])?;

        // Four lanes per thread when the count allows; the tail-less vec4 form
        // runs the residual and bias adds ~2.5x faster at full-chunk size.
        if count % 4 == 0 {
            let group = bind_group(
                gpu,
                "add_in_place_vec4",
                &self.add_in_place_vec4.get_bind_group_layout(0),
                &[
                    (&out.buffer, out.offset, (count * 4) as u64),
                    (&source.buffer, source.offset, (src_len * 4) as u64),
                    (&params.buffer, params.offset, 16),
                ],
            );
            let groups = count / 4;
            let (gx, gy) = row_grid(groups.div_ceil(ROW_THREADS));
            recorder.dispatch(
                "add_in_place_vec4",
                &self.add_in_place_vec4,
                &group,
                (gx, gy, 1),
            );
            return Ok(());
        }

        let group = bind_group(
            gpu,
            "add_in_place",
            &self.add_in_place.get_bind_group_layout(0),
            &[
                (&out.buffer, out.offset, (count * 4) as u64),
                (&source.buffer, source.offset, (src_len * 4) as u64),
                (&params.buffer, params.offset, 16),
            ],
        );
        let (gx, gy) = row_grid(count.div_ceil(ROW_THREADS));
        recorder.dispatch(
            "add_in_place",
            &self.add_in_place,
            &group,
            (gx, gy, 1),
        );
        Ok(())
    }

    /// `out *= source`, elementwise, over `count` elements of each.
    ///
    /// Separate from [`Kernels::add_in_place`] because the two are *different
    /// arithmetic*, not a parameter apart: DTTNet's decoder multiplies the
    /// upsampled tensor by the encoder feature at the same resolution, and a
    /// kernel that took an opcode for this would turn a typo into a model that
    /// still runs.
    pub fn mul_in_place(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        out: &DevTensor,
        source: &DevTensor,
        count: usize,
    ) -> Result<()> {
        if out.len() < count || source.len() < count {
            return Err(Error::Shape(format!(
                "mul_in_place covers {count} elements, the tensors hold {} and {}",
                out.len(),
                source.len()
            )));
        }
        if count > u32::MAX as usize {
            return Err(Error::Shape(format!(
                "mul_in_place covers {count} elements, more than a u32"
            )));
        }
        let params = self.params(gpu, arena, [count as u32, 0, 0, 0])?;
        let group = bind_group(
            gpu,
            "mul_in_place",
            &self.mul_in_place.get_bind_group_layout(0),
            &[
                (&out.buffer, out.offset, (count * 4) as u64),
                (&source.buffer, source.offset, (count * 4) as u64),
                (&params.buffer, params.offset, 16),
            ],
        );
        let (gx, gy) = row_grid(count.div_ceil(ROW_THREADS));
        recorder.dispatch("mul_in_place", &self.mul_in_place, &group, (gx, gy, 1));
        Ok(())
    }

    /// Folds a tap matrix back into an image: the second half of
    /// `nn.ConvTranspose2d`.
    ///
    /// `taps` is the GEMM's output, `(batch, pad_ceil(m, BM), pitch)` with
    /// `m = out_channels * kh * kw` and `pitch = pad_ceil(positions_in, BN)`;
    /// `out` is `(batch, out_channels, out_h, out_w)` tight. Every output pixel
    /// sums the taps that land on it, so no element is written twice.
    pub fn col2im_into(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        taps: &DevTensor,
        out: &DevTensor,
        shape: Col2ImShape,
        pitch: usize,
    ) -> Result<()> {
        let (out_h, out_w) = shape.out_hw();
        let positions_in = shape.positions_in();
        let m_pad = crate::gpu::kernels::pad_ceil(shape.m(), shaders::BM);
        if pitch < positions_in {
            return Err(Error::Shape(format!(
                "col2im pitch {pitch} is narrower than its {positions_in} input positions"
            )));
        }
        if pitch > u32::MAX as usize {
            return Err(Error::Shape(format!("col2im pitch {pitch} overflows a u32")));
        }
        if taps.len() < shape.batch * m_pad * pitch {
            return Err(Error::Shape(format!(
                "col2im reads {} tap elements, the tensor holds {}",
                shape.batch * m_pad * pitch,
                taps.len()
            )));
        }
        let wanted = shape.batch * shape.out_channels * shape.positions_out();
        if out.len() < wanted {
            return Err(Error::Shape(format!(
                "col2im writes {wanted} elements, the target holds {}",
                out.len()
            )));
        }

        // Plain 32-bit uniforms, like im2col: the last decoder's conv_tr has an
        // `out_w` at the full sample rate, which a packed pair capped at 65535.
        let gd = self.params(
            gpu,
            arena,
            [
                shape.out_channels as u32,
                shape.kernel.0 as u32,
                shape.kernel.1 as u32,
                out_h as u32,
            ],
        )?;
        let ge = self.params(
            gpu,
            arena,
            [
                out_w as u32,
                shape.stride.0 as u32,
                shape.stride.1 as u32,
                shape.in_h as u32,
            ],
        )?;
        let gf = self.params(
            gpu,
            arena,
            [
                shape.in_w as u32,
                shape.batch as u32,
                pitch as u32,
                m_pad as u32,
            ],
        )?;
        let group = bind_group(
            gpu,
            "col2im",
            &self.col2im.get_bind_group_layout(0),
            &[
                (&taps.buffer, taps.offset, (taps.len() * 4) as u64),
                (&out.buffer, out.offset, (out.len() * 4) as u64),
                (&gd.buffer, gd.offset, 16),
                (&ge.buffer, ge.offset, 16),
                (&gf.buffer, gf.offset, 16),
            ],
        );
        let total = shape.batch * shape.out_channels * shape.positions_out();
        let (gx, gy) = row_grid(total.div_ceil(ROW_THREADS));
        recorder.dispatch("col2im", &self.col2im, &group, (gx, gy, 1));
        Ok(())
    }

    /// `nn.GroupNorm` over the channel axis, statistics per `(row, group)`.
    ///
    /// One workgroup per `(row, group)`: the slice it normalises is one
    /// contiguous run of `per_group * len` elements per thread walk, so the
    /// reduction needs no cross-workgroup combine and the affine can be applied
    /// straight afterwards by the same workgroup, from the same values it just
    /// reduced. That is why this does not go through the `rms_norm` path: that
    /// one takes two dispatches (reduce, then scale) because its rows are wide
    /// enough to want a grid per row, and these slices are a few hundred
    /// elements.
    pub fn group_norm_into(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        x: &DevTensor,
        gamma: &DevTensor,
        beta: &DevTensor,
        out: &DevTensor,
        shape: GroupNormShape,
    ) -> Result<()> {
        if shape.rows == 0 || shape.groups == 0 || shape.per_group == 0 || shape.len == 0 {
            return Err(Error::Shape(
                "group_norm needs non-zero rows, groups, per_group and len".into(),
            ));
        }
        let channels = shape.channels();
        if gamma.len() < channels || beta.len() < channels {
            return Err(Error::Shape(format!(
                "group_norm has {channels} channels but {} gammas and {} betas",
                gamma.len(),
                beta.len()
            )));
        }
        let needed = shape.needed();
        if x.len() < needed || out.len() < needed {
            return Err(Error::Shape(format!(
                "group_norm covers {needed} elements, the tensors hold {} and {}",
                x.len(),
                out.len()
            )));
        }
        let workgroups = shape.rows * shape.groups;
        if workgroups.div_ceil(shaders::ROW_GRID_X) > shaders::ROW_GRID_X as usize {
            return Err(Error::Shape(format!(
                "group_norm needs {workgroups} workgroups, more than the grid can address"
            )));
        }
        let gd = self.params(
            gpu,
            arena,
            [
                shape.groups as u32,
                shape.per_group as u32,
                shape.len as u32,
                shape.row_stride as u32,
            ],
        )?;
        let ge = self.params(
            gpu,
            arena,
            [
                shape.rows as u32,
                shape.len_stride as u32,
                shape.channel_stride as u32,
                shape.eps.to_bits(),
            ],
        )?;
        // A slice larger than the single-workgroup form can walk in the time it
        // takes to launch it — the waveform DConv's `(1, 96, 85995)` measured
        // 23.7 ms as one workgroup against the host's sub-millisecond for the
        // same statistics — is cut into segments, one workgroup each, then a
        // per-pair stats fold and an apply. The single-workgroup form stays for
        // the DConv's few-hundred-element slices, where extra passes would cost
        // more than the first.
        let group_size = shape.per_group * shape.len;
        let segments = group_size.div_ceil(GROUP_NORM_SEGMENT_ELEMENTS).max(1);
        if segments > 1 {
            let pairs = shape.rows * shape.groups;
            let segment_len = group_size.div_ceil(segments);
            // Two floats per (pair, segment): written, never accumulated into,
            // so no clearing pass is needed.
            let partials = arena.alloc(
                gpu,
                (2 * pairs * segments * 4) as u64,
                "group_norm.partials",
            )?;
            let gs = self.params(gpu, arena, [segments as u32, segment_len as u32, 0, 0])?;
            // Two bind groups, because the two shaders see different tables:
            // the reduce reads the geometry uniforms, the apply also walks
            // gamma/beta and the output. Both bind `partials` the same way.
            let group_partial = bind_group(
                gpu,
                "group_norm_partial",
                &self.group_norm_partial.get_bind_group_layout(0),
                &[
                    (&x.buffer, x.offset, (x.len() * 4) as u64),
                    (&gd.buffer, gd.offset, 16),
                    (&ge.buffer, ge.offset, 16),
                    (&gs.buffer, gs.offset, 16),
                    (&partials.buffer, partials.offset, (partials.len() * 4) as u64),
                ],
            );
            let workgroups = pairs * segments;
            if workgroups.div_ceil(shaders::ROW_GRID_X) > shaders::ROW_GRID_X as usize {
                return Err(Error::Shape(format!(
                    "group_norm needs {workgroups} workgroups, more than the grid can address"
                )));
            }
            let (gx, gy) = row_grid(workgroups);
            recorder.dispatch(
                "group_norm_partial",
                &self.group_norm_partial,
                &group_partial,
                (gx, gy, 1),
            );
            if shaders::group_norm_combine_for(pairs, segments) {
                // Two floats per pair: mean and the already-formed scale.
                let stats = arena.alloc(gpu, (2 * pairs * 4) as u64, "group_norm.stats")?;
                let group_combine = bind_group(
                    gpu,
                    "group_norm_combine",
                    &self.group_norm_combine.get_bind_group_layout(0),
                    &[
                        (&partials.buffer, partials.offset, (partials.len() * 4) as u64),
                        (&stats.buffer, stats.offset, (stats.len() * 4) as u64),
                        (&gd.buffer, gd.offset, 16),
                        (&ge.buffer, ge.offset, 16),
                        (&gs.buffer, gs.offset, 16),
                    ],
                );
                let group_apply = bind_group(
                    gpu,
                    "group_norm_apply",
                    &self.group_norm_apply.get_bind_group_layout(0),
                    &[
                        (&x.buffer, x.offset, (x.len() * 4) as u64),
                        (&gamma.buffer, gamma.offset, (gamma.len() * 4) as u64),
                        (&beta.buffer, beta.offset, (beta.len() * 4) as u64),
                        (&out.buffer, out.offset, (out.len() * 4) as u64),
                        (&gd.buffer, gd.offset, 16),
                        (&ge.buffer, ge.offset, 16),
                        (&gs.buffer, gs.offset, 16),
                        (&stats.buffer, stats.offset, (stats.len() * 4) as u64),
                    ],
                );
                let combine_wgs = pairs.div_ceil(ROW_THREADS);
                let (cx, cy) = row_grid(combine_wgs);
                recorder.dispatch(
                    "group_norm_combine",
                    &self.group_norm_combine,
                    &group_combine,
                    (cx, cy, 1),
                );
                recorder.dispatch(
                    "group_norm_apply",
                    &self.group_norm_apply,
                    &group_apply,
                    (gx, gy, 1),
                );
            } else {
                let group_apply = bind_group(
                    gpu,
                    "group_norm_apply",
                    &self.group_norm_apply_from_partials.get_bind_group_layout(0),
                    &[
                        (&x.buffer, x.offset, (x.len() * 4) as u64),
                        (&gamma.buffer, gamma.offset, (gamma.len() * 4) as u64),
                        (&beta.buffer, beta.offset, (beta.len() * 4) as u64),
                        (&out.buffer, out.offset, (out.len() * 4) as u64),
                        (&gd.buffer, gd.offset, 16),
                        (&ge.buffer, ge.offset, 16),
                        (&gs.buffer, gs.offset, 16),
                        (&partials.buffer, partials.offset, (partials.len() * 4) as u64),
                    ],
                );
                recorder.dispatch(
                    "group_norm_apply",
                    &self.group_norm_apply_from_partials,
                    &group_apply,
                    (gx, gy, 1),
                );
            }
            return Ok(());
        }
        let group = bind_group(
            gpu,
            "group_norm",
            &self.group_norm.get_bind_group_layout(0),
            &[
                (&x.buffer, x.offset, (x.len() * 4) as u64),
                (&gamma.buffer, gamma.offset, (gamma.len() * 4) as u64),
                (&beta.buffer, beta.offset, (beta.len() * 4) as u64),
                (&out.buffer, out.offset, (out.len() * 4) as u64),
                (&gd.buffer, gd.offset, 16),
                (&ge.buffer, ge.offset, 16),
            ],
        );
        let (gx, gy) = row_grid(workgroups);
        recorder.dispatch("group_norm", &self.group_norm, &group, (gx, gy, 1));
        Ok(())
    }

    /// One direction of an LSTM's recurrence over the whole time axis, in one
    /// dispatch. See [`shaders::lstm_recur`] for why it is shaped this way.
    ///
    /// `projected` is the already-computed `W_ih x + b_ih` for every step —
    /// `(rows, time, 4 * hidden)` with `proj_pitch` as its row pitch — and `out`
    /// is `(rows, time, out_pitch)`. Both directions of a BiLSTM write the same
    /// output, at `out_offset` 0 and `hidden`, so the caller can run them as two
    /// dispatches into one buffer with no combine step.
    pub fn lstm_recur_into(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        projected: &DevTensor,
        w_hh: &DevTensor,
        b_hh: &DevTensor,
        out: &DevTensor,
        job: LstmJob,
    ) -> Result<()> {
        // The register variant holds `hidden / 2` weights per thread, so its
        // workgroup size depends on the layer's width; a width that does not fit
        // (or `DEMUCS_LSTM_REGS=0`) falls back to the shared-memory kernel.
        let fits = job.hidden >= 4
            && job.hidden % 2 == 0
            && job.hidden <= shaders::LSTM_REG_MAX_HIDDEN;
        if fits && std::env::var("DEMUCS_LSTM_REGS").map(|v| v != "0").unwrap_or(true) {
            self.lstm_checked(job, projected, w_hh, b_hh, out)?;
            let pipeline = self.lstm_regs_pipeline(gpu, job.hidden)?;
            return self.lstm_dispatch(
                gpu, arena, recorder, &pipeline, "lstm_recur_regs", projected, w_hh, b_hh, out, job,
            );
        }
        self.lstm_checked(job, projected, w_hh, b_hh, out)?;
        self.lstm_dispatch(
            gpu, arena, recorder, &self.lstm_recur, "lstm_recur", projected, w_hh, b_hh, out, job,
        )
    }

    /// The register variant's pipeline, built on first use per hidden width.
    ///
    /// The shader's workgroup size is `8 * hidden`, so a different width is a
    /// different pipeline; building one costs milliseconds, which is why it is
    /// cached rather than rebuilt per call.
    fn lstm_regs_pipeline(
        &self,
        gpu: &Gpu,
        hidden: usize,
    ) -> Result<std::cell::Ref<'_, wgpu::ComputePipeline>> {
        if !self.lstm_recur_regs.borrow().contains_key(&hidden) {
            let pipeline = gpu.pipeline(
                "lstm_recur_regs",
                &shaders::lstm_recur_regs(hidden),
                "lstm_recur_regs",
            )?;
            self.lstm_recur_regs.borrow_mut().insert(hidden, pipeline);
        }
        Ok(std::cell::Ref::map(self.lstm_recur_regs.borrow(), |map| {
            map.get(&hidden).expect("just inserted")
        }))
    }

    /// Bounds both LSTM variants share.
    #[allow(clippy::too_many_arguments)]
    fn lstm_checked(
        &self,
        job: LstmJob,
        projected: &DevTensor,
        w_hh: &DevTensor,
        b_hh: &DevTensor,
        out: &DevTensor,
    ) -> Result<()> {
        if job.hidden == 0 || job.time == 0 || job.rows == 0 {
            return Err(Error::Shape("lstm_recur needs non-zero dimensions".into()));
        }
        if job.hidden > shaders::LSTM_MAX_HIDDEN {
            return Err(Error::Gpu(format!(
                "lstm_recur is compiled for hidden <= {}, this layer has {}",
                shaders::LSTM_MAX_HIDDEN,
                job.hidden
            )));
        }
        let gates = 4 * job.hidden;
        let proj_needed = (job.rows - 1) * job.time * job.proj_pitch
            + (job.time - 1) * job.proj_pitch
            + gates;
        if projected.len() < proj_needed {
            return Err(Error::Shape(format!(
                "lstm_recur reads {proj_needed} projected elements, the tensor holds {}",
                projected.len()
            )));
        }
        if w_hh.len() < gates * job.hidden || b_hh.len() < gates {
            return Err(Error::Shape(format!(
                "lstm_recur needs a ({gates}, {}) weight and a {gates}-element bias, \
                 the tensors hold {} and {}",
                job.hidden,
                w_hh.len(),
                b_hh.len()
            )));
        }
        let out_needed = (job.rows - 1) * job.time * job.out_pitch
            + (job.time - 1) * job.out_pitch
            + job.out_offset
            + job.hidden;
        if out.len() < out_needed {
            return Err(Error::Shape(format!(
                "lstm_recur writes {out_needed} elements, the target holds {}",
                out.len()
            )));
        }
        for (name, value) in [
            ("hidden", job.hidden),
            ("time", job.time),
            ("out_pitch", job.out_pitch),
            ("proj_pitch", job.proj_pitch),
        ] {
            if value > u16::MAX as usize {
                return Err(Error::Shape(format!(
                    "lstm_recur parameter {name}={value} does not fit the kernel's 16-bit packing"
                )));
            }
        }

        Ok(())
    }

    /// The dispatch both variants share: same bindings, same grid rule.
    #[allow(clippy::too_many_arguments)]
    fn lstm_dispatch(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        pipeline: &wgpu::ComputePipeline,
        label: &str,
        projected: &DevTensor,
        w_hh: &DevTensor,
        b_hh: &DevTensor,
        out: &DevTensor,
        job: LstmJob,
    ) -> Result<()> {
        let gd = self.params(
            gpu,
            arena,
            [
                job.hidden as u32,
                job.time as u32,
                job.reverse as u32,
                job.out_pitch as u32,
            ],
        )?;
        // One workgroup covers a fixed number of rows, so the grid is the group
        // count rather than the row count: each shader derives its row range from
        // its workgroup id times its own constant.
        let rows_per_group = if label == "lstm_recur_regs" {
            shaders::LSTM_REG_ROWS
        } else {
            shaders::LSTM_ROWS
        };
        let (gx, gy) = row_grid(job.rows.div_ceil(rows_per_group));
        let ge = self.params(gpu, arena, [job.out_offset as u32, job.rows as u32, gx, job.proj_pitch as u32])?;
        let group = bind_group(
            gpu,
            label,
            &pipeline.get_bind_group_layout(0),
            &[
                (&projected.buffer, projected.offset, (projected.len() * 4) as u64),
                (&w_hh.buffer, w_hh.offset, (w_hh.len() * 4) as u64),
                (&b_hh.buffer, b_hh.offset, (b_hh.len() * 4) as u64),
                (&out.buffer, out.offset, (out.len() * 4) as u64),
                (&gd.buffer, gd.offset, 16),
                (&ge.buffer, ge.offset, 16),
            ],
        );
        recorder.dispatch(label, pipeline, &group, (gx, gy, 1));
        Ok(())
    }

    /// `BandSequenceModelModule`'s `(b, c, t, f) <-> (b * heads, f, t, c / heads)`
    /// permutation, run forwards or backwards.
    ///
    /// Both directions are the same kernel because the two index maps are
    /// inverses; `inverse` selects which side of the copy the map applies to.
    /// One kernel rather than two is not a style choice: a permutation and its
    /// inverse written separately are two chances to get an axis order wrong, and
    /// this one is not symmetric — the module's output has the time and frequency
    /// axes swapped relative to its input.
    #[allow(clippy::too_many_arguments)]
    pub fn heads_permute_into(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        x: &DevTensor,
        out: &DevTensor,
        batches: usize,
        heads: usize,
        per_head: usize,
        t: usize,
        f: usize,
        inverse: bool,
    ) -> Result<()> {
        if batches == 0 || heads == 0 || per_head == 0 || t == 0 || f == 0 {
            return Err(Error::Shape(
                "heads_permute needs non-zero batches, heads, per_head, t and f".into(),
            ));
        }
        let count = batches * heads * per_head * t * f;
        if x.len() < count || out.len() < count {
            return Err(Error::Shape(format!(
                "heads_permute moves {count} elements, the tensors hold {} and {}",
                x.len(),
                out.len()
            )));
        }
        for (name, value) in [
            ("batches", batches),
            ("heads", heads),
            ("per_head", per_head),
            ("t", t),
            ("f", f),
        ] {
            if value > u16::MAX as usize {
                return Err(Error::Shape(format!(
                    "heads_permute parameter {name}={value} does not fit the kernel's 16-bit packing"
                )));
            }
        }
        if count > u32::MAX as usize {
            return Err(Error::Shape(format!(
                "heads_permute moves {count} elements, more than a u32"
            )));
        }
        let gd = self.params(
            gpu,
            arena,
            [batches as u32, heads as u32, per_head as u32, t as u32],
        )?;
        let ge = self.params(gpu, arena, [f as u32, count as u32, inverse as u32, 0])?;
        let group = bind_group(
            gpu,
            "heads_permute",
            &self.heads_permute.get_bind_group_layout(0),
            &[
                (&x.buffer, x.offset, (x.len() * 4) as u64),
                (&out.buffer, out.offset, (out.len() * 4) as u64),
                (&gd.buffer, gd.offset, 16),
                (&ge.buffer, ge.offset, 16),
            ],
        );
        let (gx, gy) = row_grid(count.div_ceil(ROW_THREADS));
        recorder.dispatch("heads_permute", &self.heads_permute, &group, (gx, gy, 1));
        Ok(())
    }

    /// `x = act(x * scale[c] + shift[c])`, in place, over `(batch, channels,
    /// plane)`.
    ///
    /// The caller folds a normalisation's parameters into `scale`/`shift` once
    /// at load time (a BatchNorm in eval mode is exactly this), so inference is
    /// one pass rather than a normalise followed by an activation.
    pub fn channel_affine_act_in_place(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        x: &DevTensor,
        scale: &DevTensor,
        shift: &DevTensor,
        batch: usize,
        channels: usize,
        plane: usize,
        activation: Activation,
    ) -> Result<()> {
        if scale.len() < channels || shift.len() < channels {
            return Err(Error::Shape(format!(
                "an affine of {channels} channels was given {} scales and {} shifts",
                scale.len(),
                shift.len()
            )));
        }
        if plane == 0 {
            return Err(Error::Shape("affine needs a non-zero plane".into()));
        }
        let total = batch * channels * plane;
        if x.len() < total {
            return Err(Error::Shape(format!(
                "affine covers {total} elements, the tensor holds {}",
                x.len()
            )));
        }
        // The channel of element `i` is `(i / plane) % channels`, which walks the
        // batches correctly without knowing how many there are.
        let params = self.params(
            gpu,
            arena,
            [
                channels as u32,
                plane as u32,
                activation.code(),
                total as u32,
            ],
        )?;
        let group = bind_group(
            gpu,
            "channel_affine_act_in_place",
            &self.channel_affine_act_in_place.get_bind_group_layout(0),
            &[
                (&x.buffer, x.offset, (x.len() * 4) as u64),
                (&scale.buffer, scale.offset, (scale.len() * 4) as u64),
                (&shift.buffer, shift.offset, (shift.len() * 4) as u64),
                (&params.buffer, params.offset, 16),
            ],
        );
        let (gx, gy) = row_grid(total.div_ceil(ROW_THREADS));
        recorder.dispatch(
            "channel_affine_act_in_place",
            &self.channel_affine_act_in_place,
            &group,
            (gx, gy, 1),
        );
        Ok(())
    }

    /// `out = base + act(x * scale + shift)`, over a `(batch, channels, plane)`
    /// tensor whose channel index is `(i / plane) % channels`.
    ///
    /// This is the residual a transformer block ends with. Written as the
    /// reference has it — `x = act(x * gamma)` in place, copy the block input,
    /// add — it is three passes over the same tensor; as one pass it is the same
    /// arithmetic with the same rounding (`fl(base + fl(x * gamma + shift))`,
    /// and `shift` is the zero tensor the LayerScale came with), so the only
    /// difference is the traffic.
    #[allow(clippy::too_many_arguments)]
    pub fn channel_affine_act_add_into(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        base: &DevTensor,
        x: &DevTensor,
        scale: &DevTensor,
        shift: &DevTensor,
        out: &mut DevTensor,
        batch: usize,
        channels: usize,
        plane: usize,
        activation: Activation,
    ) -> Result<()> {
        if scale.len() < channels || shift.len() < channels {
            return Err(Error::Shape(format!(
                "an affine of {channels} channels was given {} scales and {} shifts",
                scale.len(),
                shift.len()
            )));
        }
        if plane == 0 {
            return Err(Error::Shape("affine needs a non-zero plane".into()));
        }
        let total = batch * channels * plane;
        for (label, tensor) in [("x", x), ("base", base), ("out", out)] {
            if tensor.len() < total {
                return Err(Error::Shape(format!(
                    "affine covers {total} elements, `{label}` holds {}",
                    tensor.len()
                )));
            }
        }
        let params = self.params(
            gpu,
            arena,
            [
                channels as u32,
                plane as u32,
                activation.code(),
                total as u32,
            ],
        )?;
        let group = bind_group(
            gpu,
            "channel_affine_act_add",
            &self.channel_affine_act_add.get_bind_group_layout(0),
            &[
                (&x.buffer, x.offset, (x.len() * 4) as u64),
                (&scale.buffer, scale.offset, (scale.len() * 4) as u64),
                (&shift.buffer, shift.offset, (shift.len() * 4) as u64),
                (&params.buffer, params.offset, 16),
                (&base.buffer, base.offset, (base.len() * 4) as u64),
                (&out.buffer, out.offset, (out.len() * 4) as u64),
            ],
        );
        let (gx, gy) = row_grid(total.div_ceil(ROW_THREADS));
        recorder.dispatch(
            "channel_affine_act_add",
            &self.channel_affine_act_add,
            &group,
            (gx, gy, 1),
        );
        Ok(())
    }

    /// `x[row, :] += bias[row % bias_rows]`, in place, over a `(rows, cols)`
    /// tensor.
    ///
    /// The convolution's bias varies along its output channels, which the GEMM
    /// lays out as rows — the opposite of every other bias in these models, so
    /// it cannot ride the existing epilogues or `add_in_place`. `bias_rows` is
    /// the period of that pattern: `rows` itself for every per-tensor caller,
    /// or the convolution's output-channel count when `rows` is
    /// `batch * out_channels` over a whole batched output.
    pub fn add_row_bias_in_place(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        x: &DevTensor,
        bias: &DevTensor,
        rows: usize,
        cols: usize,
        bias_rows: usize,
    ) -> Result<()> {
        if bias_rows == 0 {
            return Err(Error::Shape("a row bias repeats every 0 rows".into()));
        }
        if bias.len() < bias_rows {
            return Err(Error::Shape(format!(
                "a row bias of {bias_rows} values was given {}",
                bias.len()
            )));
        }
        if x.len() < rows * cols {
            return Err(Error::Shape(format!(
                "add_row_bias needs {} elements, the tensor holds {}",
                rows * cols,
                x.len()
            )));
        }
        let params = self.params(gpu, arena, [rows as u32, cols as u32, bias_rows as u32, 0])?;
        let group = bind_group(
            gpu,
            "add_row_bias_in_place",
            &self.add_row_bias_in_place.get_bind_group_layout(0),
            &[
                (&x.buffer, x.offset, (x.len() * 4) as u64),
                (&bias.buffer, bias.offset, (bias.len() * 4) as u64),
                (&params.buffer, params.offset, 16),
            ],
        );
        let (gx, gy) = row_grid((rows * cols).div_ceil(ROW_THREADS));
        recorder.dispatch("add_row_bias_in_place", &self.add_row_bias_in_place, &group, (gx, gy, 1));
        Ok(())
    }

    /// One `nn.Conv2d` layer: the im2col gather followed by the validated GEMM.
    ///
    /// `weight` is the layer's `(out_channels, k)` matrix, **already zero-padded**
    /// to `(pad_ceil(out_channels, BM), pad_ceil(k, BK))` — the GEMM's A operand
    /// is read without bounds checks. `patches` is the scratch buffer
    /// `(pad_ceil(k, BK), pad_ceil(positions, BN))`: this op gathers into it,
    /// and then issues one batched GEMM over all of `shape.batch`, so the patch
    /// margin columns are never written by the gather but are always read by the
    /// GEMM.
    ///
    /// `out` is `(batch, out_channels, positions)` tight. `bias`, when given, is
    /// the layer's `(out_channels)` vector and is applied after the GEMM.
    ///
    /// This is [`Kernels::conv2d_gather_into`] then [`Kernels::conv2d_gemm_into`];
    /// a caller that needs the two halves apart — a row-batched DConv, whose
    /// gather has to be chunked to stay under the im2col grid's y cap while its
    /// GEMM can cover every row in one dispatch — calls them directly.
    #[allow(clippy::too_many_arguments)]
    pub fn conv2d_into(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        x: &DevTensor,
        weight: &DevTensor,
        bias: Option<&DevTensor>,
        patches: &DevTensor,
        out: &DevTensor,
        shape: Im2ColShape,
        out_channels: usize,
    ) -> Result<()> {
        self.conv2d_gather_into(gpu, arena, recorder, x, patches, shape)?;
        self.conv2d_gemm_into(gpu, arena, recorder, weight, x, patches, out, shape, out_channels, bias)
    }

    /// The gather half of [`Kernels::conv2d_into`]: `x` gathered into `patches`,
    /// which must hold `shape.pitched_len_with(pitch, k_pad)` elements for the
    /// geometry `shape` implies.
    pub fn conv2d_gather_into(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        x: &DevTensor,
        patches: &DevTensor,
        shape: Im2ColShape,
    ) -> Result<()> {
        let k = shape.k();
        let k_pad = crate::gpu::kernels::pad_ceil(k, shaders::BK);
        let pitch = crate::gpu::kernels::pad_ceil(shape.positions(), shaders::BN);
        if patches.len() < shape.pitched_len_with(pitch, k_pad) {
            return Err(Error::Shape(format!(
                "conv2d needs a patch scratch of {} elements (batch {} x k {} x pitch {}), the tensor holds {}",
                shape.pitched_len_with(pitch, k_pad),
                shape.batch,
                k_pad,
                pitch,
                patches.len()
            )));
        }
        // Deliberately no clear. The gather never writes the tile margins, so the
        // GEMM reads whatever was in the scratch for two groups of elements; the
        // patch rows past `k` and the columns past `positions`. Neither can reach
        // a stored output: the column margins only feed accumulators whose `n`
        // index the epilogue guards, and the row margins only ever multiply the
        // *weight* matrix's own k padding, which was zeroed when it was loaded.
        // `tests/gpu_conv2d.rs::poisoned_patch_margins_cannot_reach_the_output`
        // fills the scratch with a large sentinel and checks the output is
        // unchanged, which is that argument as a test rather than a comment.
        //
        // The clear that used to be here was also not free: `Arena::clear` is a
        // host-to-device `write_buffer`, which at a conv-sized scratch (~600 MB
        // for DTTNet's first block) is over a second per layer.
        self.im2col_into(gpu, arena, recorder, x, patches, shape, pitch, k_pad)
    }

    /// The GEMM half of [`Kernels::conv2d_into`]: one batched dispatch over
    /// `shape.batch` patch blocks, then one bias pass over the whole output.
    ///
    /// `patches` and `out` are bound **whole**, and the per-batch strides
    /// (`k_pad * pitch`, `out_channels * positions`) ride the job. The
    /// alternative — slicing `out` per batch — would put the batch offset into
    /// the bind group, where the 32-byte storage alignment does not always
    /// hold: the freq encoder.3 DConv lays its conv1 output out
    /// `(rows, 6, 33)` at a real segment's frame count, and 6 * 33 = 198 is not
    /// a multiple of 8, so batch 1's block lands mid-alignment and the driver
    /// rejects hundreds of bind groups without naming the allocation.
    ///
    /// `x` is the conv's input, used only by the `DEMUCS_CONV_DIRECT` form,
    /// which re-derives the gather inside the GEMM instead of reading
    /// `patches`.
    #[allow(clippy::too_many_arguments)]
    pub fn conv2d_gemm_into(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        weight: &DevTensor,
        x: &DevTensor,
        patches: &DevTensor,
        out: &DevTensor,
        shape: Im2ColShape,
        out_channels: usize,
        bias: Option<&DevTensor>,
    ) -> Result<()> {
        let positions = shape.positions();
        let k = shape.k();
        let k_pad = crate::gpu::kernels::pad_ceil(k, shaders::BK);
        let pitch = crate::gpu::kernels::pad_ceil(positions, shaders::BN);
        let weight_len = crate::gpu::kernels::pad_ceil(out_channels, shaders::BM) * k_pad;
        if weight.len() < weight_len {
            return Err(Error::Shape(format!(
                "conv2d needs a padded weight of {weight_len} elements, the tensor holds {}",
                weight.len()
            )));
        }
        if out.len() < shape.batch * out_channels * positions {
            return Err(Error::Shape(format!(
                "conv2d writes {} elements, the target holds {}",
                shape.batch * out_channels * positions,
                out.len()
            )));
        }

        // One GEMM dispatch for the whole batch: `patches` and `out` are bound
        // whole and the per-batch strides ride the job. Slicing `out` per batch
        // instead would put the batch offset into the bind group, and the
        // 32-byte storage alignment does not always hold there — the freq
        // encoder.3 DConv lays its conv1 output out `(rows, 6, 33)` at a real
        // segment's frame count, and 6 * 33 = 198 is not a multiple of 8, so
        // batch 1's block lands mid-alignment and the driver rejects hundreds
        // of bind groups without naming the allocation. The gather has been
        // one batched dispatch all along, so this also collapses `batches`
        // dispatches into one.
        let per_batch = k_pad * pitch;
        let job = GemmJob {
            m: out_channels,
            n: positions,
            k,
            lda: k_pad,
            ldb: pitch,
            ldc: positions,
            batches: shape.batch,
            inner_count: 1,
            a_outer: 0,
            a_inner: 0,
            b_outer: per_batch,
            b_inner: 0,
            c_outer: out_channels * positions,
            c_inner: 0,
            transb: false,
        };
        // The gather folded into the GEMM: needs one output row per tile so the
        // staging can derive `(oy, ox)` once. That holds only when a tile's
        // positions cannot straddle a row boundary — `out_w` a multiple of `BN`,
        // not merely `>= BN`: a tile starting at `n0 = 128` with `out_w = 200`
        // spans row 0's tail and row 1's head, and every position past the
        // boundary gathers from the wrong row's input. Any batch count is fine
        // — the batched variant of the tile offsets the gather by each batch's
        // own block of the input, so a multi-batch conv never silently computes
        // batch 0 for every block and never materialises a patch matrix either.
        if shaders::conv_direct_for_convs() && shape.out_hw().1 % shaders::BN == 0
        {
            let batched = shape.batch > 1;
            self.gemm_conv_into(gpu, arena, recorder, weight, x, out, job, shape, batched)?;
            if let Some(bias) = bias {
                self.add_row_bias_in_place(
                    gpu,
                    arena,
                    recorder,
                    out,
                    bias,
                    shape.batch * out_channels,
                    positions,
                    out_channels,
                )?;
            }
            return Ok(());
        }
        if out_channels <= 64 && shaders::bm64_for_convs() {
            // The patch matrix's rows are `positions` apart, so which staging
            // walk wins is a property of this operand, not of the GEMM; see
            // `shaders::gemm_bm64_coalesced`. `DEMUCS_GEMM_B_WALK` switches it,
            // which is how the two were compared.
            let coalesced = shaders::coalesced_b_for_convs();
            let pipeline = if coalesced {
                &self.gemm_batched_bm64_bn128_coalesced
            } else {
                &self.gemm_batched_bm64_bn128
            };
            self.gemm_into_tile(
                gpu,
                arena,
                recorder,
                weight,
                patches,
                out,
                job,
                pipeline,
                (shaders::BM64, shaders::BN),
            )?;
        } else if out_channels <= 192 && shaders::bm96_for_convs() {
            self.gemm_into_tile(
                gpu,
                arena,
                recorder,
                weight,
                patches,
                out,
                job,
                &self.gemm_batched_bm96_bn128,
                (shaders::BM96, shaders::BN),
            )?;
        } else {
            self.gemm_into(gpu, arena, recorder, weight, patches, out, job)?;
        }
        if let Some(bias) = bias {
            // One pass over the whole `(batch, out_channels, positions)`
            // output: the bias repeats every `out_channels` rows.
            self.add_row_bias_in_place(
                gpu,
                arena,
                recorder,
                out,
                bias,
                shape.batch * out_channels,
                positions,
                out_channels,
            )?;
        }
        Ok(())
    }

    /// `(batch, rows, cols)` copied from one row pitch to another, with the
    /// destination's extra rows per batch left alone.
    ///
    /// Used only to turn a conv-transpose's input into a GEMM operand; see
    /// [`Kernels::conv_transpose2d_into`] for why that copy is unavoidable.
    #[allow(clippy::too_many_arguments)]
    pub fn copy_pitched_into(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        x: &DevTensor,
        out: &DevTensor,
        batch: usize,
        rows: usize,
        cols: usize,
        src_pitch: usize,
        dst_pitch: usize,
        dst_rows: usize,
    ) -> Result<()> {
        if rows > dst_rows {
            return Err(Error::Shape(format!(
                "copy_pitched has {rows} rows per batch but the destination holds {dst_rows}"
            )));
        }
        for (name, value) in [
            ("batch", batch),
            ("rows", rows),
            ("cols", cols),
            ("src_pitch", src_pitch),
            ("dst_pitch", dst_pitch),
            ("dst_rows", dst_rows),
        ] {
            if value > u32::MAX as usize {
                return Err(Error::Shape(format!(
                    "copy_pitched parameter {name}={value} overflows a u32"
                )));
            }
        }
        let src_needed = batch * rows * src_pitch;
        if x.len() < src_needed {
            return Err(Error::Shape(format!(
                "copy_pitched reads {src_needed} input elements, the tensor holds {}",
                x.len()
            )));
        }
        let dst_needed = batch * dst_rows * dst_pitch;
        if out.len() < dst_needed {
            return Err(Error::Shape(format!(
                "copy_pitched writes {dst_needed} elements, the target holds {}",
                out.len()
            )));
        }
        let gd = self.params(
            gpu,
            arena,
            [batch as u32, rows as u32, cols as u32, src_pitch as u32],
        )?;
        let ge = self.params(gpu, arena, [dst_rows as u32, dst_pitch as u32, 0, 0])?;
        let group = bind_group(
            gpu,
            "copy_pitched",
            &self.copy_pitched.get_bind_group_layout(0),
            &[
                (&x.buffer, x.offset, (x.len() * 4) as u64),
                (&out.buffer, out.offset, (out.len() * 4) as u64),
                (&gd.buffer, gd.offset, 16),
                (&ge.buffer, ge.offset, 16),
            ],
        );
        let (gx, gy) = row_grid((batch * rows * cols).div_ceil(ROW_THREADS));
        recorder.dispatch("copy_pitched", &self.copy_pitched, &group, (gx, gy, 1));
        Ok(())
    }

    /// Keeps `row_offset .. row_offset + dst_rows` of each batch's rows.
    /// The frequency branch's transposed-conv crop.
    #[allow(clippy::too_many_arguments)]
    pub fn crop_rows_into(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        x: &DevTensor,
        out: &DevTensor,
        batch: usize,
        src_batch_stride: usize,
        dst_rows: usize,
        cols: usize,
        row_offset: usize,
    ) -> Result<()> {
        let total = batch * dst_rows * cols;
        if out.len() < total {
            return Err(Error::Shape(format!(
                "crop_rows writes {total} elements, the target holds {}",
                out.len()
            )));
        }
        if src_batch_stride * batch > x.len() {
            return Err(Error::Shape(format!(
                "crop_rows reads {} elements per the given strides, the tensor holds {}",
                src_batch_stride * batch,
                x.len()
            )));
        }
        let gd = self.params(
            gpu,
            arena,
            [src_batch_stride as u32, dst_rows as u32, cols as u32, row_offset as u32],
        )?;
        let ge = self.params(gpu, arena, [batch as u32, 0, 0, 0])?;
        let group = bind_group(
            gpu,
            "crop_rows",
            &self.crop_rows.get_bind_group_layout(0),
            &[
                (&x.buffer, x.offset, (x.len() * 4) as u64),
                (&out.buffer, out.offset, (out.len() * 4) as u64),
                (&gd.buffer, gd.offset, 16),
                (&ge.buffer, ge.offset, 16),
            ],
        );
        let (gx, gy) = row_grid(total.div_ceil(ROW_THREADS));
        recorder.dispatch("crop_rows", &self.crop_rows, &group, (gx, gy, 1));
        Ok(())
    }

    /// Zeroes `count` elements on the device.
    pub fn fill_zero_into(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        x: &DevTensor,
        count: usize,
    ) -> Result<()> {
        if count > u32::MAX as usize {
            return Err(Error::Shape(format!(
                "fill_zero covers {count} elements, more than a u32"
            )));
        }
        if x.len() < count {
            return Err(Error::Shape(format!(
                "fill_zero covers {count} elements, the tensor holds {}",
                x.len()
            )));
        }
        let params = self.params(gpu, arena, [count as u32, 0, 0, 0])?;
        let group = bind_group(
            gpu,
            "fill_zero",
            &self.fill_zero.get_bind_group_layout(0),
            &[
                (&x.buffer, x.offset, (x.len() * 4) as u64),
                (&params.buffer, params.offset, 16),
            ],
        );
        let (gx, gy) = row_grid(count.div_ceil(ROW_THREADS));
        recorder.dispatch("fill_zero", &self.fill_zero, &group, (gx, gy, 1));
        Ok(())
    }

    /// `x[i] = x[i] * scale[i / plane] + shift[i / plane]`, in place.
    ///
    /// `scale` and `shift` have one value per batch item; `plane` is the number
    /// of elements that share a batch index (`x.len() / batch`).
    pub fn batch_affine_in_place(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        x: &DevTensor,
        scale: &DevTensor,
        shift: &DevTensor,
        plane: usize,
    ) -> Result<()> {
        let count = x.len();
        if plane == 0 || count % plane != 0 {
            return Err(Error::Shape(format!(
                "batch affine: {count} elements are not a multiple of plane {plane}"
            )));
        }
        let batch = count / plane;
        if scale.len() < batch || shift.len() < batch {
            return Err(Error::Shape(format!(
                "batch affine: scale/shift need {batch} values"
            )));
        }
        let params = self.params(gpu, arena, [count as u32, plane as u32, 0, 0])?;
        let group = bind_group(
            gpu,
            "batch_affine_in_place",
            &self.batch_affine_in_place.get_bind_group_layout(0),
            &[
                (&x.buffer, x.offset, (count * 4) as u64),
                (&scale.buffer, scale.offset, (scale.len() * 4) as u64),
                (&shift.buffer, shift.offset, (shift.len() * 4) as u64),
                (&params.buffer, params.offset, 16),
            ],
        );
        let (gx, gy) = row_grid(count.div_ceil(ROW_THREADS));
        recorder.dispatch(
            "batch_affine_in_place",
            &self.batch_affine_in_place,
            &group,
            (gx, gy, 1),
        );
        Ok(())
    }

    /// Device `_spec` + CaC pack: `padded` is `_spec`'s reflect-padded mix
    /// `(batch, channels, padded_len)`. `out` is `(batch, 2*channels, nfft/2, le)`.
    pub fn stft_cac_into(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        padded: &DevTensor,
        window: &DevTensor,
        out: &DevTensor,
        batch: usize,
        channels: usize,
        padded_len: usize,
        le: usize,
        bins: usize,
        hop: usize,
        nfft: usize,
    ) -> Result<()> {
        if nfft != 4096 {
            return Err(Error::Shape(format!("stft_cac needs nfft=4096, got {nfft}")));
        }
        if padded.shape != [batch, channels, padded_len] {
            return Err(Error::Shape(format!(
                "stft padded shape {:?} != [{batch}, {channels}, {padded_len}]",
                padded.shape
            )));
        }
        if out.shape != [batch, channels * 2, bins, le] {
            return Err(Error::Shape(format!(
                "stft out shape {:?} != [{batch}, {}, {bins}, {le}]",
                out.shape,
                channels * 2
            )));
        }
        let gd = self.params(
            gpu,
            arena,
            [le as u32, bins as u32, channels as u32, hop as u32],
        )?;
        let ge = self.params(
            gpu,
            arena,
            [padded_len as u32, (nfft / 2) as u32, batch as u32, 0],
        )?;
        let group = bind_group(
            gpu,
            "stft_rfft4096",
            &self.stft_rfft4096.get_bind_group_layout(0),
            &[
                (&padded.buffer, padded.offset, (padded.len() * 4) as u64),
                (&window.buffer, window.offset, (nfft * 4) as u64),
                (&out.buffer, out.offset, (out.len() * 4) as u64),
                (&gd.buffer, gd.offset, 16),
                (&ge.buffer, ge.offset, 16),
            ],
        );
        recorder.dispatch(
            "stft_rfft4096",
            &self.stft_rfft4096,
            &group,
            (le.max(1) as u32, (batch * channels).max(1) as u32, 1),
        );
        Ok(())
    }

    /// Mean and unbiased std of each batch item's plane, written to `mean`/`std`.
    pub fn batch_moments_into(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        x: &DevTensor,
        mean: &DevTensor,
        std: &DevTensor,
        plane: usize,
    ) -> Result<()> {
        let count = x.len();
        if plane == 0 || count % plane != 0 {
            return Err(Error::Shape(format!(
                "batch moments: {count} is not a multiple of plane {plane}"
            )));
        }
        let batch = count / plane;
        if batch == 0 {
            return Ok(());
        }
        // One workgroup against the whole GPU is fine for a small plane and
        // terrible for the frequency branch's 2.75M-element one (2.68 ms
        // measured, 4 GB/s), so the plane is cut into segments once it is
        // bigger than a workgroup should walk. `DEMUCS_BATCH_MOMENTS_SEG=0`
        // keeps the single-workgroup form.
        let segment_elements = shaders::batch_moments_segment_elements();
        let segments = if segment_elements == 0 {
            1
        } else {
            plane.div_ceil(segment_elements).max(1)
        };
        if segments > 1 {
            let segment_len = plane.div_ceil(segments);
            let partials = arena.alloc(
                gpu,
                (2 * batch * segments * 4) as u64,
                "batch_moments.partials",
            )?;
            let gd = self.params(
                gpu,
                arena,
                [
                    plane as u32,
                    segments as u32,
                    segment_len as u32,
                    batch as u32,
                ],
            )?;
            let partial_group = bind_group(
                gpu,
                "batch_moments_partial",
                &self.batch_moments_partial.get_bind_group_layout(0),
                &[
                    (&x.buffer, x.offset, (count * 4) as u64),
                    (&partials.buffer, partials.offset, (partials.len() * 4) as u64),
                    (&gd.buffer, gd.offset, 16),
                ],
            );
            let (gx, gy) = row_grid(batch * segments);
            recorder.dispatch(
                "batch_moments_partial",
                &self.batch_moments_partial,
                &partial_group,
                (gx, gy, 1),
            );
            let gc = self.params(gpu, arena, [plane as u32, batch as u32, segments as u32, 0])?;
            let combine_group = bind_group(
                gpu,
                "batch_moments_combine",
                &self.batch_moments_combine.get_bind_group_layout(0),
                &[
                    (&partials.buffer, partials.offset, (partials.len() * 4) as u64),
                    (&mean.buffer, mean.offset, (mean.len() * 4) as u64),
                    (&std.buffer, std.offset, (std.len() * 4) as u64),
                    (&gc.buffer, gc.offset, 16),
                ],
            );
            let (cx, cy) = row_grid(batch.div_ceil(shaders::ROW_THREADS));
            recorder.dispatch(
                "batch_moments_combine",
                &self.batch_moments_combine,
                &combine_group,
                (cx, cy, 1),
            );
            return Ok(());
        }
        let params = self.params(gpu, arena, [plane as u32, batch as u32, 0, 0])?;
        let group = bind_group(
            gpu,
            "batch_moments",
            &self.batch_moments.get_bind_group_layout(0),
            &[
                (&x.buffer, x.offset, (count * 4) as u64),
                (&mean.buffer, mean.offset, (mean.len() * 4) as u64),
                (&std.buffer, std.offset, (std.len() * 4) as u64),
                (&params.buffer, params.offset, 16),
            ],
        );
        recorder.dispatch(
            "batch_moments",
            &self.batch_moments,
            &group,
            (batch.max(1) as u32, 1, 1),
        );
        Ok(())
    }

    /// `x = (x - mean) / (1e-5 + std)` per batch item.
    pub fn batch_normalize_in_place(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        x: &DevTensor,
        mean: &DevTensor,
        std: &DevTensor,
        plane: usize,
    ) -> Result<()> {
        let count = x.len();
        if plane == 0 || count % plane != 0 {
            return Err(Error::Shape(format!(
                "batch normalize: {count} is not a multiple of plane {plane}"
            )));
        }
        let params = self.params(gpu, arena, [count as u32, plane as u32, 0, 0])?;
        let group = bind_group(
            gpu,
            "batch_normalize_in_place",
            &self.batch_normalize_in_place.get_bind_group_layout(0),
            &[
                (&x.buffer, x.offset, (count * 4) as u64),
                (&mean.buffer, mean.offset, (mean.len() * 4) as u64),
                (&std.buffer, std.offset, (std.len() * 4) as u64),
                (&params.buffer, params.offset, 16),
            ],
        );
        let (gx, gy) = row_grid(count.div_ceil(ROW_THREADS));
        recorder.dispatch(
            "batch_normalize_in_place",
            &self.batch_normalize_in_place,
            &group,
            (gx, gy, 1),
        );
        Ok(())
    }

    /// Device `_ispec` for the CaC frequency branch: 4096-point inverse FFT of
    /// every frame, overlap-add with the Hann envelope, trim to `length`.
    ///
    /// `spec` is `(batch, 4*sources, nfft/2, frames)` after denormalise.
    /// `window` is the periodic Hann of `nfft`. `out` is `(batch, 2*sources, length)`.
    pub fn ispec_cac_into(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        spec: &DevTensor,
        window: &DevTensor,
        out: &DevTensor,
        batch: usize,
        sources: usize,
        frames: usize,
        bins: usize,
        length: usize,
    ) -> Result<()> {
        const NFFT: usize = 4096;
        if window.len() < NFFT {
            return Err(Error::Shape("istft window is shorter than nfft=4096".into()));
        }
        if bins != NFFT / 2 {
            return Err(Error::Shape(format!(
                "ispec_cac expected {} bins, got {bins}",
                NFFT / 2
            )));
        }
        let packed = sources * 4;
        let packed_out = sources * 2;
        if spec.shape != [batch, packed, bins, frames] {
            return Err(Error::Shape(format!(
                "ispec_cac spec shape {:?} != [{batch}, {packed}, {bins}, {frames}]",
                spec.shape
            )));
        }
        if out.len() < batch * packed_out * length {
            return Err(Error::Shape(format!(
                "ispec_cac out holds {} elements, need {}",
                out.len(),
                batch * packed_out * length
            )));
        }
        let hop = NFFT / 4;
        let pad = hop / 2 * 3;
        let rows = batch * packed_out;
        let frame_buf = arena.tensor(
            gpu,
            &[rows, frames, NFFT],
            "istft.frames",
        )?;
        let irfft_params = self.params(
            gpu,
            arena,
            [frames as u32, bins as u32, packed as u32, batch as u32],
        )?;
        let irfft_group = bind_group(
            gpu,
            "istft_irfft4096",
            &self.istft_irfft4096.get_bind_group_layout(0),
            &[
                (&spec.buffer, spec.offset, (spec.len() * 4) as u64),
                (&window.buffer, window.offset, (NFFT * 4) as u64),
                (&frame_buf.buffer, frame_buf.offset, (frame_buf.len() * 4) as u64),
                (&irfft_params.buffer, irfft_params.offset, 16),
            ],
        );
        recorder.dispatch(
            "istft_irfft4096",
            &self.istft_irfft4096,
            &irfft_group,
            (frames.max(1) as u32, rows.max(1) as u32, 1),
        );

        let ola_gd = self.params(
            gpu,
            arena,
            [frames as u32, hop as u32, NFFT as u32, pad as u32],
        )?;
        let ola_ge = self.params(
            gpu,
            arena,
            [length as u32, packed_out as u32, batch as u32, 0],
        )?;
        let ola_group = bind_group(
            gpu,
            "istft_ola",
            &self.istft_ola.get_bind_group_layout(0),
            &[
                (&frame_buf.buffer, frame_buf.offset, (frame_buf.len() * 4) as u64),
                (&window.buffer, window.offset, (NFFT * 4) as u64),
                (&out.buffer, out.offset, (out.len() * 4) as u64),
                (&ola_gd.buffer, ola_gd.offset, 16),
                (&ola_ge.buffer, ola_ge.offset, 16),
            ],
        );
        let samples = batch * packed_out * length;
        let (gx, gy) = row_grid(samples.div_ceil(ROW_THREADS));
        recorder.dispatch("istft_ola", &self.istft_ola, &ola_group, (gx, gy, 1));
        Ok(())
    }

    /// One `nn.ConvTranspose2d` layer: the tap matrix, then the col2im gather.
    ///
    /// The tap matrix is `(out_channels * kh * kw, in_channels) @
    /// (in_channels, positions_in)` — the *same* GEMM the forward convolutions
    /// use, with the weights in `(oc, ky, kx)`-major order — and
    /// [`Kernels::col2im_into`] then sums the taps that land on each output
    /// pixel. The host reference instead runs one GEMM per kernel tap followed by
    /// a strided accumulate, which on the device would be `kh * kw` dispatches
    /// per layer plus either atomics or a per-tap scratch; composing the two
    /// existing ops keeps this to one GEMM and one gather per batch and reuses
    /// kernels that already have host-reference tests.
    ///
    /// **The input copy.** The GEMM's `B` operand has to be padded — rows to
    /// `BK`, columns to `BN` — and every read inside a tile has to land in a
    /// buffer, so `x` cannot be handed over as a tight `(batch, in_channels,
    /// positions)` tensor. Two ways to fix that: have every producer of a
    /// conv-transpose input write the padded layout, or copy here. The copy wins
    /// because the padding a tensor needs depends on its *consumer*: these
    /// activations are produced by convolutions whose output is read back as
    /// `(channels, h, w)` by the next layer's im2col gather, so a producer cannot
    /// emit both layouts without either duplicating every convolution's epilogue
    /// or making every consumer's layout a function of the graph. One extra pass
    /// over an activation costs ~1% of the layer it feeds.
    ///
    /// **The copy's margins are not zeroed, on purpose.** `copy_pitched_into`
    /// writes only the `cols` real columns of each row, so the tap GEMM reads
    /// uninitialised memory for two groups of elements: `B` rows past
    /// `in_channels` (when that is not a multiple of `BK`) and `B` columns past
    /// `positions_in` (when that is not a multiple of `BN`). Neither can affect a
    /// stored output. The column margins only feed accumulators whose `n` index
    /// is past `positions_in`, and the GEMM's epilogue guards those. The row
    /// margins only ever multiply `A`'s own `k` padding, and `A` is a weight
    /// matrix that was zero-padded when it was loaded — so every product
    /// involving them is `0 * garbage = 0`. Zeroing the margins here would be a
    /// second full pass over the scratch for no change in the result.
    #[allow(clippy::too_many_arguments)]
    pub fn conv_transpose2d_into(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        x: &DevTensor,
        weight: &DevTensor,
        bias: Option<&DevTensor>,
        padded_input: &DevTensor,
        taps: &DevTensor,
        out: &DevTensor,
        shape: Col2ImShape,
    ) -> Result<()> {
        let in_channels = shape.in_channels;
        let positions_in = shape.positions_in();
        let k_pad = pad_ceil(in_channels, shaders::BK);
        let pitch = pad_ceil(positions_in, shaders::BN);
        let m = shape.m();
        let m_pad = pad_ceil(m, shaders::BM);
        let (out_h, out_w) = shape.out_hw();
        let positions_out = out_h * out_w;
        if in_channels == 0 || positions_in == 0 || m == 0 {
            return Err(Error::Shape("conv_transpose2d needs non-zero dimensions".into()));
        }
        let weight_len = m_pad * k_pad;
        if weight.len() < weight_len {
            return Err(Error::Shape(format!(
                "conv_transpose2d needs a padded weight of {weight_len} elements, the tensor holds {}",
                weight.len()
            )));
        }
        let input_len = shape.batch * in_channels * positions_in;
        if x.len() < input_len {
            return Err(Error::Shape(format!(
                "conv_transpose2d needs {input_len} input elements, the tensor holds {}",
                x.len()
            )));
        }
        let padded_len = shape.batch * k_pad * pitch;
        if padded_input.len() < padded_len {
            return Err(Error::Shape(format!(
                "conv_transpose2d needs a padded input of {padded_len} elements, the tensor holds {}",
                padded_input.len()
            )));
        }
        let tap_len = shape.batch * m_pad * pitch;
        if taps.len() < tap_len {
            return Err(Error::Shape(format!(
                "conv_transpose2d needs a tap scratch of {tap_len} elements, the tensor holds {}",
                taps.len()
            )));
        }
        let wanted = shape.batch * shape.out_channels * positions_out;
        if out.len() < wanted {
            return Err(Error::Shape(format!(
                "conv_transpose2d writes {wanted} elements, the target holds {}",
                out.len()
            )));
        }

        self.copy_pitched_into(
            gpu,
            arena,
            recorder,
            x,
            padded_input,
            shape.batch,
            in_channels,
            positions_in,
            positions_in,
            pitch,
            k_pad,
        )?;

        let job = GemmJob {
            m,
            n: positions_in,
            k: in_channels,
            lda: k_pad,
            ldb: pitch,
            ldc: pitch,
            batches: 1,
            inner_count: 1,
            a_outer: 0,
            a_inner: 0,
            b_outer: 0,
            b_inner: 0,
            c_outer: 0,
            c_inner: 0,
            transb: false,
        };
        for batch in 0..shape.batch {
            let batch_input = padded_input.slice(batch * k_pad * pitch, vec![k_pad * pitch])?;
            let batch_taps = taps.slice(batch * m_pad * pitch, vec![m_pad * pitch])?;
            let batch_out = out.slice(
                batch * shape.out_channels * positions_out,
                vec![shape.out_channels * positions_out],
            )?;
            self.gemm_into(
                gpu,
                arena,
                recorder,
                weight,
                &batch_input,
                &batch_taps,
                job,
            )?;
            self.col2im_into(
                gpu,
                arena,
                recorder,
                &batch_taps,
                &batch_out,
                Col2ImShape {
                    batch: 1,
                    ..shape
                },
                pitch,
            )?;
            // The bias belongs to the output channel, and the gather has already
            // written every element, so this is the host's `bias + taps` in the
            // other addition order rather than an initialisation.
            if let Some(bias) = bias {
                self.add_row_bias_in_place(
                    gpu,
                    arena,
                    recorder,
                    &batch_out,
                    bias,
                    shape.out_channels,
                    positions_out,
                    shape.out_channels,
                )?;
            }
        }
        Ok(())
    }

    /// Shape of an `nn.Conv2d` call, as the im2col gather needs to see it.
    ///
    /// `pitch` is the destination's row stride: `shape.positions()` for a tight
    /// buffer, or `pad_ceil(shape.positions(), BN)` to hand the result straight
    /// to the GEMM, whose inner loop reads whole tiles and relies on the margin
    /// columns being zero. Margins are never written, so the caller zeroes the
    /// buffer first.
    pub fn im2col_into(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        x: &DevTensor,
        out: &DevTensor,
        shape: Im2ColShape,
        pitch: usize,
        rows: usize,
    ) -> Result<()> {
        let (out_h, out_w) = shape.out_hw();
        let input_len = shape.batch * shape.in_channels * shape.h * shape.w;
        let positions = shape.positions();
        if pitch < positions {
            return Err(Error::Shape(format!(
                "im2col pitch {pitch} is narrower than its {positions} positions"
            )));
        }
        if pitch > u32::MAX as usize {
            return Err(Error::Shape(format!("im2col pitch {pitch} overflows a u32")));
        }
        if rows < shape.k() {
            return Err(Error::Shape(format!(
                "im2col has {} rows per batch but the gather writes {}",
                rows,
                shape.k()
            )));
        }
        if x.len() < input_len {
            return Err(Error::Shape(format!(
                "im2col needs {input_len} input elements, the tensor holds {}",
                x.len()
            )));
        }
        let needed = shape.pitched_len_with(pitch, rows);
        if out.len() < needed {
            return Err(Error::Shape(format!(
                "im2col writes {needed} elements at pitch {pitch}, the target holds {}",
                out.len()
            )));
        }

        // Plain 32-bit uniforms: the waveform branch feeds the full sample
        // rate through here, and a 16-bit packing capped `w` at 65535.
        let gd = self.params(
            gpu,
            arena,
            [
                shape.in_channels as u32,
                shape.kernel.0 as u32,
                shape.kernel.1 as u32,
                shape.h as u32,
            ],
        )?;
        let ge = self.params(
            gpu,
            arena,
            [shape.w as u32, out_h as u32, out_w as u32, shape.batch as u32],
        )?;
        let gf = self.params(
            gpu,
            arena,
            [
                shape.stride.0 as u32,
                shape.stride.1 as u32,
                shape.pad.0 as u32,
                shape.pad.1 as u32,
            ],
        )?;
        let gg = self.params(gpu, arena, [pitch as u32, rows as u32, 0, 0])?;
        let group = bind_group(
            gpu,
            "im2col",
            &self.im2col.get_bind_group_layout(0),
            &[
                (&x.buffer, x.offset, (x.len() * 4) as u64),
                (&out.buffer, out.offset, (out.len() * 4) as u64),
                (&gd.buffer, gd.offset, 16),
                (&ge.buffer, ge.offset, 16),
                (&gf.buffer, gf.offset, 16),
                (&gg.buffer, gg.offset, 16),
            ],
        );
        // The dispatch carries (position block, patch row of a batch, output row):
        // no flat index, so the shader needs no division to find out where it is.
        let (out_h, out_w) = shape.out_hw();
        let gx = out_w.div_ceil(ROW_THREADS);
        let gy = rows * shape.batch;
        for (name, value) in [("x", gx), ("y", gy), ("z", out_h)] {
            if value > 65535 {
                return Err(Error::Shape(format!(
                    "im2col needs {value} workgroups on the {name} grid axis, over the 65535 cap"
                )));
            }
        }
        recorder.dispatch("im2col", &self.im2col, &group, (gx.max(1) as u32, gy.max(1) as u32, out_h.max(1) as u32));
        Ok(())
    }

    /// `(rows, cols) -> (cols, rows)`, tiled through shared memory.
    ///
    /// The two axial transformers want the sequence on different axes, and one
    /// layout flip per transformer pair is cheaper than a strided variant of every
    /// op in the frequency branch.
    /// Swaps the outer two axes of a `(batch, rows, cols, width)` tensor, which
    /// is the model's `(chunks, ...) -> (..., chunks, ...)`-style layout flip.
    ///
    /// `width` is the innermost run that travels as a unit — the model's axial
    /// transformers swap the band and frame axes while leaving the feature axis
    /// untouched, which a plain matrix transpose cannot express. Because
    /// consecutive threads walk consecutive `d` and `d` is contiguous on both
    /// sides, the elementwise form needs no shared-memory tiling to stay
    /// coalesced; a plain matrix transpose is the `width = 1` case.
    pub fn transpose(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        x: &DevTensor,
        out: &DevTensor,
        batch: usize,
        rows: usize,
        cols: usize,
        width: usize,
    ) -> Result<()> {
        let elements = batch * rows * cols * width;
        if x.len() < elements || out.len() < elements {
            return Err(Error::Shape(format!(
                "transpose {batch}x{rows}x{cols}x{width} needs {elements} elements on both sides"
            )));
        }
        let params = self.params(
            gpu,
            arena,
            [batch as u32, rows as u32, cols as u32, width as u32],
        )?;
        let group = bind_group(
            gpu,
            "transpose",
            &self.transpose.get_bind_group_layout(0),
            &[
                (&x.buffer, x.offset, (x.len() * 4) as u64),
                (&out.buffer, out.offset, (out.len() * 4) as u64),
                (&params.buffer, params.offset, 16),
            ],
        );
        let (gx, gy) = row_grid(elements.div_ceil(ROW_THREADS));
        recorder.dispatch("transpose", &self.transpose, &group, (gx, gy, 1));
        Ok(())
    }

    /// Softmax in place over the last axis, one workgroup per row.
    ///
    /// Requires its own pipeline: the out-of-place form binds the same buffer as
    /// read-only and read-write, which wgpu rejects, so an in-place call cannot
    /// reuse it.
    pub fn softmax_in_place(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        x: &DevTensor,
        rows: usize,
        cols: usize,
        pitch: usize,
        scale: f32,
    ) -> Result<()> {
        if pitch < cols {
            return Err(Error::Shape(format!(
                "softmax pitch {pitch} is narrower than its {cols} columns"
            )));
        }
        // The kernel caches the row in registers, `SLOTS` values per lane: the
        // bound is lanes x slots, and the lanes are the kernel's, not the op's.
        let capacity = shaders::SOFTMAX_SLOTS as usize * shaders::SOFTMAX_LANES;
        if cols > capacity {
            return Err(Error::Shape(format!(
                "softmax rows of {cols} columns exceed the kernel's register cache \
                 ({capacity} = {} slots x {} lanes); widen SOFTMAX_SLOTS",
                shaders::SOFTMAX_SLOTS,
                shaders::SOFTMAX_LANES
            )));
        }
        let params = self.params(gpu, arena, [rows as u32, cols as u32, scale.to_bits(), pitch as u32])?;
        // Narrow rows take the warp-per-row kernel: its reduction is shuffles
        // rather than six barriers, and a 60-column row has nothing else to
        // hide them behind. The tree stays for wide rows and for adapters that
        // cannot promise a 32-lane subgroup.
        let warp = match &self.softmax_warp {
            Some(pipeline) if cols <= shaders::SOFTMAX_WARP_COLS && pitch <= shaders::SOFTMAX_WARP_COLS => {
                Some(pipeline)
            }
            _ => None,
        };
        let pipeline = warp.unwrap_or(&self.softmax_in_place);
        let group = bind_group(
            gpu,
            "softmax_in_place",
            &pipeline.get_bind_group_layout(0),
            &[
                (&x.buffer, x.offset, (x.len() * 4) as u64),
                (&params.buffer, params.offset, 16),
            ],
        );
        // The warp kernel covers eight rows per workgroup, the tree one.
        let per_wg = if warp.is_some() {
            shaders::SOFTMAX_WARP_ROWS_PER_WG
        } else {
            shaders::SOFTMAX_ROWS_PER_WG
        };
        let (gx, gy) = row_grid(rows.div_ceil(per_wg));
        recorder.dispatch("softmax_in_place", pipeline, &group, (gx, gy, 1));
        Ok(())
    }

    /// Softmax over the last axis, one workgroup per row, with an optional scale
    /// applied to every element first.
    ///
    /// The scale rides in the uniform rather than being pre-applied to the scores
    /// buffer, which would cost a gigabyte-scale read-modify-write per attention
    /// call.
    pub fn softmax_scaled(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        x: &DevTensor,
        out: &DevTensor,
        rows: usize,
        cols: usize,
        scale: f32,
    ) -> Result<()> {
        let params = self.params(gpu, arena, [rows as u32, cols as u32, scale.to_bits(), 0])?;
        if let Some(pipeline) = self.softmax_scaled_warp.as_ref() {
            let group = bind_group(
                gpu,
                "softmax",
                &pipeline.get_bind_group_layout(0),
                &[
                    (&x.buffer, x.offset, (x.len() * 4) as u64),
                    (&out.buffer, out.offset, (out.len() * 4) as u64),
                    (&params.buffer, params.offset, 16),
                ],
            );
            let (gx, gy) = row_grid(rows.div_ceil(shaders::SOFTMAX_WARP_ROWS_PER_WG));
            recorder.dispatch("softmax", pipeline, &group, (gx, gy, 1));
            return Ok(());
        }
        let group = bind_group(
            gpu,
            "softmax",
            &self.softmax.get_bind_group_layout(0),
            &[
                (&x.buffer, x.offset, (x.len() * 4) as u64),
                (&out.buffer, out.offset, (out.len() * 4) as u64),
                (&params.buffer, params.offset, 16),
            ],
        );
        let (gx, gy) = row_grid(rows);
        recorder.dispatch("softmax", &self.softmax, &group, (gx, gy, 1));
        Ok(())
    }

    /// Softmax over the last axis, one workgroup per row.
    pub fn softmax(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        x: &DevTensor,
        rows: usize,
        cols: usize,
    ) -> Result<DevTensor> {
        let out = arena.tensor(gpu, &[rows, cols], "softmax.out")?;
        self.softmax_scaled(gpu, arena, recorder, x, &out, rows, cols, 1.0)?;
        Ok(out)
    }

    /// Fused `softmax(Q Kᵀ √d⁻¹) V` straight out of the fused QKV tensor.
    ///
    /// Requires `dim_head == 64` (the shader's accumulator block is compiled
    /// for it); callers fall back to the score-matrix path otherwise. One
    /// dispatch covers every (batch, head) pair and Q block, and no score
    /// scratch exists at all — which is the point: at a full chunk the scores
    /// are gigabyte-scale traffic the fused kernel simply does not do.
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attention_into(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        qkv: &DevTensor,
        out: &DevTensor,
        batches: usize,
        seq: usize,
        heads: usize,
        dim_head: usize,
    ) -> Result<()> {
        if dim_head != 64 {
            return Err(Error::Gpu(format!(
                "flash attention is compiled for dim_head 64, got {dim_head}"
            )));
        }
        let row_stride = 3 * heads * dim_head;
        let inner = heads * dim_head;
        let scale = 1.0 / (dim_head as f32).sqrt();
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct FlashDims {
            seq: u32,
            row_stride: u32,
            inner: u32,
            heads: u32,
            scale_bits: u32,
        }
        let dims = FlashDims {
            seq: seq as u32,
            row_stride: row_stride as u32,
            inner: inner as u32,
            heads: heads as u32,
            scale_bits: scale.to_bits(),
        };
        let params = gpu.push_uniform(bytemuck::bytes_of(&dims))?;
        let group = bind_group(
            gpu,
            "flash_attention",
            &self.flash_attention.get_bind_group_layout(0),
            &[
                (&qkv.buffer, qkv.offset, (qkv.len() * 4) as u64),
                (&out.buffer, out.offset, (out.len() * 4) as u64),
                (&params.buffer, params.offset, 32),
            ],
        );
        let pairs = batches * heads;
        let q_blocks = seq.div_ceil(64);
        recorder.dispatch(
            "flash_attention",
            &self.flash_attention,
            &group,
            (pairs as u32, q_blocks as u32, 1),
        );
        Ok(())
    }
}

/// Geometry of one axial attention pass.
///
/// The operands live in a fused QKV tensor laid out `(bands, frames,
/// 3 * heads * dim_head)` — the layout the reference produces by reshaping its
/// `to_qkv` output, so no reordering pass is needed anywhere.
#[derive(Debug, Clone, Copy)]
pub struct AttentionShape {
    /// Outer batch: the mel band axis for the time transformer, the time axis for
    /// the frequency one.
    pub bands: usize,
    pub frames: usize,
    pub heads: usize,
    pub dim_head: usize,
}

impl AttentionShape {
    pub fn inner(&self) -> usize {
        self.heads * self.dim_head
    }

    pub fn row(&self) -> usize {
        3 * self.inner()
    }

    /// Row pitch of the score scratch, rounded so the AV product's K dimension is
    /// a whole number of reduction steps.
    ///
    /// This is not tidiness: the GEMM stages K in blocks of `BK` and reads a full
    /// block past the end of the logical K, so an unpadded `frames` of 801 would
    /// let the last stage pull the next row's values in as if they were scores.
    /// Rounded up here, those cells hold zeros.
    pub fn scores_pitch(&self) -> usize {
        pad_ceil(self.frames, shaders::BK)
    }

    pub fn scores_len(&self) -> usize {
        self.bands * self.heads * self.frames * self.scores_pitch()
    }

    pub fn output_len(&self) -> usize {
        self.bands * self.frames * self.inner()
    }
}

impl Kernels {
    /// `softmax(Q K^T / sqrt(d)) V`, written out as `(bands, frames, inner)`.
    ///
    /// `qkv` is only read, so the caller can keep it as the input to whatever
    /// follows; `scores` is caller-provided scratch because at a full chunk the
    /// score tensor is over a gigabyte and belongs to the per-chunk buffers rather
    /// than to this call.
    ///
    /// Q, K and V are taken as offset views of the fused tensor rather than
    /// copies — three dispatches, no reordering pass.
    pub fn attention(
        &self,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        qkv: &DevTensor,
        scores: &DevTensor,
        out: &DevTensor,
        shape: AttentionShape,
    ) -> Result<()> {
        let AttentionShape {
            bands,
            frames,
            heads,
            dim_head,
        } = shape;
        let inner = shape.inner();
        let row = shape.row();
        let pitch = shape.scores_pitch();
        let batches = bands * heads;

        if scores.len() < shape.scores_len() {
            return Err(Error::Shape(format!(
                "score scratch holds {} elements, needs {}",
                scores.len(),
                shape.scores_len()
            )));
        }
        if out.len() < shape.output_len() {
            return Err(Error::Shape(format!(
                "attention output holds {} elements, needs {}",
                out.len(),
                shape.output_len()
            )));
        }

        let q = qkv.slice(0, vec![qkv.len()])?;
        let k = qkv.slice(inner, vec![qkv.len() - inner])?;
        let v = qkv.slice(2 * inner, vec![qkv.len() - 2 * inner])?;

        // Scores: (frames x dim_head) @ (dim_head x frames), with K read
        // transposed out of its [frame][dim_head] storage. The score tensor is
        // batched contiguously by (band, head), so the outer stride covers the
        // whole head run.
        let (scores_outer, scores_inner) = GemmJob::contiguous_batch(frames * pitch, heads);
        self.gemm_into(
            gpu,
            arena,
            recorder,
            &q,
            &k,
            scores,
            GemmJob {
                m: frames,
                n: frames,
                k: dim_head,
                lda: row,
                ldb: row,
                ldc: pitch,
                batches,
                inner_count: heads,
                a_outer: frames * row,
                a_inner: dim_head,
                b_outer: frames * row,
                b_inner: dim_head,
                c_outer: scores_outer,
                c_inner: scores_inner,
                transb: true,
            },
        )?;

        // Softmax with the attention scale folded in, in place, clearing the
        // pitch padding so the AV product's K sees zeros there.
        let scale = 1.0 / (dim_head as f32).sqrt();
        self.softmax_in_place(
            gpu,
            arena,
            recorder,
            scores,
            batches * frames,
            frames,
            pitch,
            scale,
        )?;

        // AV: (frames x frames) @ (frames x dim_head), written head-major into the
        // output rows so the following projection reads it without a reorder. The
        // left operand is the score tensor, which is batched contiguously by
        // (band, head) — so its outer stride covers the whole head run too.
        self.gemm_into(
            gpu,
            arena,
            recorder,
            scores,
            &v,
            out,
            GemmJob {
                m: frames,
                n: dim_head,
                k: frames,
                lda: pitch,
                ldb: row,
                ldc: inner,
                batches,
                inner_count: heads,
                a_outer: scores_outer,
                a_inner: scores_inner,
                b_outer: frames * row,
                b_inner: dim_head,
                c_outer: frames * inner,
                c_inner: dim_head,
                transb: false,
            },
        )?;
        Ok(())
    }
}

/// Elements per workgroup in the split `group_norm`: a slice larger than this
/// is reduced in segments rather than by one workgroup, because one workgroup
/// is 8 warps against the whole GPU and the waveform DConv's slices run to
/// 8.2 million elements.
const GROUP_NORM_SEGMENT_ELEMENTS: usize = 4096;

/// A grid that covers `rows` row-wise workgroups without exceeding the 65535 cap
/// on any dispatch dimension.
fn row_grid(rows: usize) -> (u32, u32) {
    let x = rows.min(shaders::ROW_GRID_X);
    let y = rows.div_ceil(shaders::ROW_GRID_X);
    (x.max(1) as u32, y.max(1) as u32)
}

/// Rounds `value` up to the next multiple of `tile`.
pub fn pad_ceil(value: usize, tile: usize) -> usize {
    value.div_ceil(tile) * tile
}

/// The row pitch the GEMM expects for a `cols`-wide operand.
pub fn pad_pitch(cols: usize, tile: usize) -> usize {
    pad_ceil(cols, tile)
}

/// Pads `(rows, cols)` into `(pad_ceil(rows, row_tile), pad_ceil(cols, col_tile))`.
///
/// The zeros are load-bearing: they are what the kernel's unguarded tile reads
/// land on.
pub fn pad_matrix_for(
    data: &[f32],
    rows: usize,
    cols: usize,
    row_tile: usize,
    col_tile: usize,
) -> Result<(Vec<f32>, usize, usize)> {
    let padded_rows = pad_ceil(rows, row_tile);
    let padded_cols = pad_ceil(cols, col_tile);
    if data.len() < rows * cols {
        return Err(Error::Shape(format!(
            "matrix holds {} elements, expected {}",
            data.len(),
            rows * cols
        )));
    }
    let mut out = vec![0.0f32; padded_rows * padded_cols];
    for r in 0..rows {
        out[r * padded_cols..r * padded_cols + cols]
            .copy_from_slice(&data[r * cols..r * cols + cols]);
    }
    Ok((out, padded_rows, padded_cols))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padding_rounds_up_to_tile_multiples() {
        assert_eq!(pad_ceil(384, 16), 384);
        assert_eq!(pad_ceil(385, 16), 400);
        assert_eq!(pad_ceil(1, 128), 128);
    }

    #[test]
    fn padding_zero_fills_the_margins() {
        // 2x3 into a 4x4 tile grid.
        let data: Vec<f32> = (1..=6).map(|v| v as f32).collect();
        let (padded, rows, cols) = pad_matrix_for(&data, 2, 3, 4, 4).unwrap();
        assert_eq!((rows, cols), (4, 4));
        assert_eq!(padded.len(), 16);
        assert_eq!(&padded[0..3], &[1.0, 2.0, 3.0]);
        assert_eq!(padded[3], 0.0, "the margin must be zero, not stale data");
        assert_eq!(&padded[4..7], &[4.0, 5.0, 6.0]);
        assert!(padded[8..].iter().all(|v| *v == 0.0));
    }

    #[test]
    fn padding_rejects_short_input() {
        assert!(pad_matrix_for(&[1.0, 2.0], 2, 3, 4, 4).is_err());
    }
}
