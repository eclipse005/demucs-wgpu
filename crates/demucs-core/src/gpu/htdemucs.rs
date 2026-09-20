//! The `HTDemucs` forward pass on the GPU.
//!
//! Status: the **encoder half** (frequency and waveform branches, through the
//! bottleneck) runs on the device; the transformer, decoders and the complex
//! packing stay on the host for now. `GpuHtdemucsRunner::forward_encoders`
//! returns the two bottleneck tensors, and the device-vs-host test compares
//! every traced stage by name — so each further stage can be moved over and
//! checked independently of everything after it.
//!
//! Split of work, as in the two prior ports: `torch.stft` / `torch.istft` and
//! the complex packing stay on the host (2.5 % of the reference's time, already
//! exact); one `Recorder` builds the whole device graph and submits it once.
//!
//! **Weights come from the host model**, which is the architecture description
//! already aligned against PyTorch, so this file cannot disagree with it about
//! what the model *is* — only about layout, and layout is exactly what the
//! device-vs-host test measures.

use ndarray::{Array3, Array4};

use crate::demucs::host::{one_dimensional_positions, two_dimensional_positions, Htdemucs, TraceSink};
use crate::demucs::weights::{
    ConvW, DConvW, EncLayerW, HtdemucsWeights, LayerNormW, TransformerLayerW,
};
use crate::error::{Error, Result};
use crate::gpu::arena::{Arena, DevTensor, Recorder};
use crate::gpu::kernels::{
    pad_ceil, pad_matrix_for, Activation, Col2ImShape, GemmJob, GroupNormShape, Im2ColShape,
    Kernels,
};
use crate::gpu::shaders::{BK as BK_C, BN as BN_C};
use crate::ops::Linear;
use crate::gpu::shaders::BK;
use crate::gpu::Gpu;

/// `nn.GroupNorm` / `nn.LayerNorm` default epsilon, as on the host.
const NORM_EPS: f32 = 1e-5;
/// Weights live in their own arena and are uploaded once, so their pool never
/// grows: the budget is only an upper bound for the report.
const WEIGHT_ARENA_BYTES: u64 = 1 << 30;
/// The work arena's free lists are capped here. It bounds what the pool
/// *retains* across chunks, not what one chunk allocates: recycling is what
/// holds a chunk to its live set, and the cap only stops a long session with
/// drifting shapes from accumulating. A 7.8 s chunk peaks near 1.1 GiB live;
/// 1 GiB of idle unique sizes on top of that is what Task Manager reported as
/// ~3.5 GiB committed. Large planes share buffers (see `REUSE_SIZE_RATIO`),
/// so 1 GiB of retained free lists is enough for the scratch + softmax pair
/// plus a cushion of medium activations.
const WORK_ARENA_BYTES: u64 = 1 << 30;

// --------------------------------------------------------------------- weights

/// A forward convolution, `(pad(oc), pad(ic*kh*kw))` in PyTorch's `(oc, ic, kh, kw)`
/// order — the same order the im2col gather emits its patch rows in, so no
/// repack is needed on upload.
struct GpuConv {
    weight: DevTensor,
    bias: DevTensor,
    in_channels: usize,
    out_channels: usize,
    kernel: (usize, usize),
    stride: (usize, usize),
    pad: (usize, usize),
}

/// A transposed convolution for the decoder upsampling, packed the way
/// `conv_transpose2d_into` reads it: rows `(oc, ky, kx)` times columns
/// `in_channels`, zero-padded to the GEMM tiles.
struct GpuConvT {
    weight: DevTensor,
    bias: DevTensor,
    in_channels: usize,
    out_channels: usize,
    kernel: (usize, usize),
    stride: (usize, usize),
}

impl GpuConv {
    fn k(&self) -> usize {
        self.in_channels * self.kernel.0 * self.kernel.1
    }
}

/// One `DConv` sub-layer: `conv1 -> GroupNorm(1) -> GELU -> conv2(1x1) ->
/// GroupNorm(1) -> GLU -> LayerScale`, the residual added by the caller because
/// the branch input has to stay alive until then.
struct GpuDConvLayer {
    /// Widened for dilation exactly as the host does: `k=3, dilation=2` becomes
    /// `k=5` with zeros on the odd taps, `pad = dilation * (k/2)` — the device
    /// im2col has no dilation either.
    conv1: GpuConv,
    /// Ones of `2C` length: the affine scale that folds conv2's bias in, since
    /// the batched GEMM has no epilogue.
    conv2_ones: DevTensor,
    norm1_scale: DevTensor,
    norm1_shift: DevTensor,
    conv2: GpuConv,
    norm2_scale: DevTensor,
    norm2_shift: DevTensor,
    /// LayerScale's per-channel factor; the affine kernel indexes the shift by
    /// channel too, so a matching-length zero vector rides along with it.
    gamma: DevTensor,
    gamma_zero: DevTensor,
    channels: usize,
    hidden: usize,
}

struct GpuDConv {
    layers: Vec<GpuDConvLayer>,
}

struct GpuEnc {
    conv: GpuConv,
    rewrite: GpuConv,
    dconv: GpuDConv,
}

/// A transformer projection: PyTorch's `(out, in)` weight, uploaded as-is for
/// a `transb` GEMM (the kernel reads B as `(n, k)` through its strides), with
/// the bias folded by the caller's affine or the epilogue.
struct GpuProj {
    weight: DevTensor,
    bias: DevTensor,
    out_features: usize,
    in_features: usize,
}

/// One transformer layer. `qkv_fused` is set on classic layers (one
/// `(3*dim, dim)` projection covering q, k, v in that order, which is also the
/// flash-attention kernel's layout); cross layers project q and k/v separately
/// because their inputs differ.
struct GpuTransformerLayer {
    is_cross: bool,
    /// Separate q/k/v projections on every layer: the score-matrix attention
    /// wants each operand as its own `(tokens, dim)` tensor, and slicing a
    /// fused `(tokens, 3 * dim)` GEMM output into contiguous thirds mixes
    /// tokens (the fused rows are `q_t | k_t | v_t` per token).
    q_proj: GpuProj,
    k_proj: GpuProj,
    v_proj: GpuProj,
    out_proj: GpuProj,
    linear1: GpuProj,
    linear2: GpuProj,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: Option<LayerNorm>,
    /// `MyGroupNorm(1, dim)` over `(dim, tokens)` per batch.
    norm_out: GroupNorm,
    gamma1: DevTensor,
    gamma2: DevTensor,
    dim: usize,
}

#[derive(Clone)]
struct LayerNorm {
    scale: DevTensor,
    shift: DevTensor,
    dim: usize,
}

#[derive(Clone)]
struct GroupNorm {
    scale: DevTensor,
    shift: DevTensor,
    dim: usize,
}

struct GpuTransformer {
    norm_in: LayerNorm,
    norm_in_t: LayerNorm,
    layers: Vec<GpuTransformerLayer>,
    layers_t: Vec<GpuTransformerLayer>,
    /// All-ones, long enough for the widest projection's bias fold (the FFN's
    /// hidden = 4*dim, or the fused QKV's 3*dim — whichever is larger).
    ones: DevTensor,
    /// Zeros, `dim` long: the identity-affine shift for LayerScale.
    zeros: DevTensor,
}

pub struct GpuWeights {
    encoder: Vec<GpuEnc>,
    tencoder: Vec<GpuEnc>,
    /// The frequency embedding, pre-scaled by `freq_emb_scale` and flattened to
    /// one entry per `(freq, channel)` row, ready for `add_row_bias_in_place`
    /// on the permuted `(f, c, t)` tensor (whose rows are exactly `f * c`).
    freq_emb_rows: DevTensor,
    /// Shared constants: a one-element 1.0 (LayerScale's "no shift") and 0.0.
    unit: DevTensor,
    zero: DevTensor,
    transformer: GpuTransformer,
    /// Channel up/downsamplers around the transformer.
    up_freq: GpuConv,
    up_time: GpuConv,
    down_freq: GpuConv,
    down_time: GpuConv,
    decoder: Vec<GpuDec>,
    tdecoder: Vec<GpuDec>,
}

/// One decoder layer: skip-sum rewrite, DConv, transposed-conv upsample.
struct GpuDec {
    rewrite: GpuConv,
    dconv: GpuDConv,
    conv_tr: GpuConvT,
}

// ----------------------------------------------------------------------- loader

fn upload_conv(
    gpu: &Gpu,
    arena: &mut Arena,
    w: &ConvW,
    stride: (usize, usize),
    pad: (usize, usize),
    label: &str,
) -> Result<GpuConv> {
    let (in_channels, out_channels) = (w.shape[1], w.shape[0]);
    let kernel = if w.shape.len() == 4 {
        (w.shape[2], w.shape[3])
    } else {
        (1, w.shape[2])
    };
    let k = in_channels * kernel.0 * kernel.1;
    debug_assert_eq!(w.weight.len(), out_channels * k);
    let (padded, _, _) = pad_matrix_for(&w.weight, out_channels, k, crate::gpu::shaders::BM, BK)?;
    Ok(GpuConv {
        weight: arena.upload(
            gpu,
            &[pad_ceil(out_channels, crate::gpu::shaders::BM), pad_ceil(k, BK)],
            &padded,
            label,
        )?,
        bias: arena.upload(gpu, &[out_channels], &w.bias, label)?,
        in_channels,
        out_channels,
        kernel,
        stride,
        pad,
    })
}

/// Widens a `k`-tap weight set for `dilation` (zeros on the odd taps) so the
/// device im2col needs no dilation support — the same trick the host uses.
fn upload_widened_conv(
    gpu: &Gpu,
    arena: &mut Arena,
    layer: &crate::demucs::weights::DConvLayerW,
    dilation: usize,
    label: &str,
) -> Result<GpuConv> {
    let k = layer.conv1.kernel()[0];
    let width = (k - 1) * dilation + 1;
    let out_channels = layer.conv1.out_channels();
    let in_channels = layer.conv1.in_channels();
    let mut widened = vec![0.0f32; out_channels * in_channels * width];
    for oc in 0..out_channels {
        for ic in 0..in_channels {
            for j in 0..k {
                widened[(oc * in_channels + ic) * width + j * dilation] =
                    layer.conv1.weight[(oc * in_channels + ic) * k + j];
            }
        }
    }
    let w = ConvW {
        weight: widened,
        bias: layer.conv1.bias.clone(),
        shape: vec![out_channels, in_channels, width],
    };
    upload_conv(gpu, arena, &w, (1, 1), (0, dilation * (k / 2)), label)
}

fn upload_dconv(gpu: &Gpu, arena: &mut Arena, d: &DConvW, label: &str) -> Result<GpuDConv> {
    let mut layers = Vec::with_capacity(d.layers.len());
    for (depth, layer) in d.layers.iter().enumerate() {
        let dilation = 1usize << depth;
        let conv1 = upload_widened_conv(gpu, arena, layer, dilation, label)?;
        let conv2 = upload_conv(gpu, arena, &layer.conv2, (1, 1), (0, 0), label)?;
        let conv2_channels = layer.conv2.out_channels();
        layers.push(GpuDConvLayer {
            conv1,
            conv2_ones: arena.upload(
                gpu,
                &[conv2_channels],
                &vec![1.0f32; conv2_channels],
                label,
            )?,
            norm1_scale: arena.upload(
                gpu,
                &[layer.norm1_weight.len()],
                &layer.norm1_weight,
                label,
            )?,
            norm1_shift: arena.upload(
                gpu,
                &[layer.norm1_bias.len()],
                &layer.norm1_bias,
                label,
            )?,
            conv2,
            norm2_scale: arena.upload(
                gpu,
                &[layer.norm2_weight.len()],
                &layer.norm2_weight,
                label,
            )?,
            norm2_shift: arena.upload(
                gpu,
                &[layer.norm2_bias.len()],
                &layer.norm2_bias,
                label,
            )?,
            gamma: arena.upload(gpu, &[layer.gamma.len()], &layer.gamma, label)?,
            gamma_zero: arena.upload(
                gpu,
                &[layer.gamma.len()],
                &vec![0.0f32; layer.gamma.len()],
                label,
            )?,
            channels: d.channels,
            hidden: layer.conv1.out_channels(),
        });
    }
    Ok(GpuDConv { layers })
}

fn upload_encoder(gpu: &Gpu, arena: &mut Arena, layer: &EncLayerW, label: &str) -> Result<GpuEnc> {
    // The frequency branch's conv is `Conv2d(k=(8,1), s=(4,1), p=(2,0))`; the
    // waveform branch's `Conv1d(k=8, s=4, p=2)` runs as a `(1, k)` kernel over a
    // `(b, c, 1, t)` view — the same lowering the host uses.
    let freq = layer.conv.shape.len() == 4;
    let conv = if freq {
        upload_conv(gpu, arena, &layer.conv, (4, 1), (2, 0), label)?
    } else {
        upload_conv(gpu, arena, &layer.conv, (1, 4), (0, 2), label)?
    };
    // Every encoder rewrite is 1x1 (`context_enc = 0`), so no padding.
    let rewrite = upload_conv(gpu, arena, &layer.rewrite, (1, 1), (0, 0), label)?;
    Ok(GpuEnc {
        conv,
        rewrite,
        dconv: upload_dconv(gpu, arena, &layer.dconv, label)?,
    })
}

/// Uploads a decoder's transposed convolution. The checkpoint stores
/// `(in_channels, out_channels, kh, kw)`; the GEMM wants one row per
/// `(oc, ky, kx)` tap across `in_channels` columns.
fn upload_conv_tr(
    gpu: &Gpu,
    arena: &mut Arena,
    w: &ConvW,
    stride: (usize, usize),
    label: &str,
) -> Result<GpuConvT> {
    // A transposed convolution's checkpoint shape is (in, out, kh, kw) — the
    // reverse of a plain convolution's (out, in, ...).
    let (in_channels, out_channels) = (w.shape[0], w.shape[1]);
    let kernel = if w.shape.len() == 4 {
        (w.shape[2], w.shape[3])
    } else {
        (1, w.shape[2])
    };
    let (kh, kw) = kernel;
    let taps = kh * kw;
    debug_assert_eq!(w.weight.len(), in_channels * out_channels * taps);
    let m = out_channels * taps;
    let mut ordered = vec![0.0f32; m * in_channels];
    for oc in 0..out_channels {
        for tap in 0..taps {
            for ic in 0..in_channels {
                let src = ((ic * out_channels + oc) * taps) + tap;
                let dst = (oc * taps + tap) * in_channels + ic;
                ordered[dst] = w.weight[src];
            }
        }
    }
    let (padded, _, _) = pad_matrix_for(&ordered, m, in_channels, crate::gpu::shaders::BM, BK)?;
    Ok(GpuConvT {
        weight: arena.upload(
            gpu,
            &[pad_ceil(m, crate::gpu::shaders::BM), pad_ceil(in_channels, BK)],
            &padded,
            label,
        )?,
        bias: arena.upload(gpu, &[out_channels], &w.bias, label)?,
        in_channels,
        out_channels,
        kernel,
        stride,
    })
}

/// Uploads a `nn.Linear` for a `transb` GEMM: the weight stays in PyTorch's
/// `(out, in)` layout (the kernel reads B as `(n, k)` through its strides) and
/// the bias rides separately.
fn upload_proj(gpu: &Gpu, arena: &mut Arena, layer: &Linear, label: &str) -> Result<GpuProj> {
    let (out_features, in_features) = (layer.out_features(), layer.in_features());
    let weight = layer.weight_t.t().to_owned(); // (out, in), standard layout
    let padded_rows = pad_ceil(out_features, crate::gpu::shaders::BN);
    let padded_cols = pad_ceil(in_features, BK_C);
    let mut padded = vec![0.0f32; padded_rows * padded_cols];
    for r in 0..out_features {
        padded[r * padded_cols..r * padded_cols + in_features]
            .copy_from_slice(&weight.as_slice().expect("standard layout")[r * in_features..(r + 1) * in_features]);
    }
    Ok(GpuProj {
        weight: arena.upload(gpu, &[padded_rows, padded_cols], &padded, label)?,
        bias: arena.upload(
            gpu,
            &[out_features],
            layer.bias.as_ref().expect("linear bias").as_slice().expect("contiguous"),
            label,
        )?,
        out_features,
        in_features,
    })
}

/// Concatenates q, k, v into the fused `(3*dim, dim)` projection the reference's
/// `in_proj_weight` uses — the order the flash-attention kernel expects.
fn fused_qkv(layer: &TransformerLayerW) -> Vec<f32> {
    let dim = layer.q_weight.len() / layer.q_bias.len().max(1) * 0 + layer.q_bias.len();
    let rows = 3 * dim;
    let mut fused = vec![0.0f32; rows * dim];
    for (index, weight) in [&layer.q_weight, &layer.k_weight, &layer.v_weight].into_iter().enumerate() {
        fused[index * dim * dim..(index + 1) * dim * dim].copy_from_slice(weight);
    }
    let _ = rows;
    fused
}

fn linear_from_rows(weight: &[f32], bias: &[f32]) -> Linear {
    let dim = bias.len();
    let view = ndarray::ArrayView2::from_shape((dim, dim), weight).expect("square projection");
    Linear::new(
        view,
        Some(ndarray::ArrayView1::from_shape(dim, bias).expect("projection bias")),
    )
}

fn upload_transformer_layer(
    gpu: &Gpu,
    arena: &mut Arena,
    layer: &TransformerLayerW,
    label: &str,
) -> Result<GpuTransformerLayer> {
    let dim = layer.q_bias.len();
    let hidden = layer.linear1.out_features();
    let norm_out = GroupNorm {
        scale: arena.upload(gpu, &[dim], &layer.norm_out.weight, label)?,
        shift: arena.upload(gpu, &[dim], &layer.norm_out.bias, label)?,
        dim,
    };
    // LayerScale has no shift; a zero vector of `dim` rides along because the
    // affine kernel indexes both by channel.
    let zeros = arena.upload(gpu, &[dim], &vec![0.0f32; dim], label)?;

    Ok(GpuTransformerLayer {
        is_cross: layer.is_cross,
        q_proj: upload_proj(gpu, arena, &linear_from_rows(&layer.q_weight, &layer.q_bias), label)?,
        k_proj: upload_proj(gpu, arena, &linear_from_rows(&layer.k_weight, &layer.k_bias), label)?,
        v_proj: upload_proj(gpu, arena, &linear_from_rows(&layer.v_weight, &layer.v_bias), label)?,
        out_proj: upload_proj(gpu, arena, &layer.out_proj, label)?,
        linear1: upload_proj(gpu, arena, &layer.linear1, label)?,
        linear2: upload_proj(gpu, arena, &layer.linear2, label)?,
        norm1: upload_ln(gpu, arena, &layer.norm1, label)?,
        norm2: upload_ln(gpu, arena, &layer.norm2, label)?,
        norm3: match &layer.norm3 {
            Some(norm) => Some(upload_ln(gpu, arena, norm, label)?),
            None => None,
        },
        norm_out,
        gamma1: arena.upload(gpu, &[dim], &layer.gamma1, label)?,
        gamma2: arena.upload(gpu, &[dim], &layer.gamma2, label)?,
        dim,
    })
}

fn upload_ln(
    gpu: &Gpu,
    arena: &mut Arena,
    norm: &LayerNormW,
    label: &str,
) -> Result<LayerNorm> {
    Ok(LayerNorm {
        scale: arena.upload(gpu, &[norm.weight.len()], &norm.weight, label)?,
        shift: arena.upload(gpu, &[norm.bias.len()], &norm.bias, label)?,
        dim: norm.weight.len(),
    })
}

/// A shared one-element 1.0 affine scale and the transformer weights.
fn upload_transformer(
    gpu: &Gpu,
    arena: &mut Arena,
    host: &Htdemucs,
) -> Result<GpuTransformer> {
    let weights = &host.weights.transformer;
    let mut layers = Vec::new();
    let mut layers_t = Vec::new();
    for index in 0..host.config.t_layers {
        layers.push(upload_transformer_layer(
            gpu,
            arena,
            &weights.layers[index],
            &format!("xt.layer.{index}"),
        )?);
        layers_t.push(upload_transformer_layer(
            gpu,
            arena,
            &weights.layers_t[index],
            &format!("xt.layer_t.{index}"),
        )?);
    }
    Ok(GpuTransformer {
        norm_in: upload_ln(gpu, arena, &weights.norm_in, "xt.norm_in")?,
        norm_in_t: upload_ln(gpu, arena, &weights.norm_in_t, "xt.norm_in_t")?,
        layers,
        layers_t,
        ones: arena.upload(
            gpu,
            &[2048],
            &vec![1.0f32; 2048],
            "xt.ones",
        )?,
        zeros: arena.upload(
            gpu,
            &[host.arch.transformer_channels],
            &vec![0.0f32; host.arch.transformer_channels],
            "xt.zeros",
        )?,
    })
}

impl GpuWeights {
    pub fn load(host: &Htdemucs, gpu: &Gpu, arena: &mut Arena) -> Result<Self> {
        let weights: &HtdemucsWeights = &host.weights;
        let mut encoder = Vec::new();
        let mut tencoder = Vec::new();
        for (index, layer) in weights.encoder.iter().enumerate() {
            encoder.push(upload_encoder(gpu, arena, layer, &format!("encoder.{index}"))?);
        }
        for (index, layer) in weights.tencoder.iter().enumerate() {
            tencoder.push(upload_encoder(gpu, arena, layer, &format!("tencoder.{index}"))?);
        }

        let embedding = weights.freq_emb.as_slice().ok_or_else(|| {
            Error::Gpu("the frequency embedding is not in standard layout".into())
        })?;
        // `weights.freq_emb` is already scaled by `emb_scale` (the loader did
        // it); the forward adds `freq_emb_scale * emb`, so only that factor
        // remains. Scaling twice made the embedding 10x too large.
        let rows = weights.freq_emb.dim().0;
        let columns = weights.freq_emb.dim().1;
        // Transposed to (channels, freqs) row-major: the row-bias add runs on
        // the (c, f, t) tensor whose rows are (c, f) pairs.
        let scale = host.config.freq_emb as f32;
        let mut expanded = vec![0.0f32; rows * columns];
        for f in 0..rows {
            for c in 0..columns {
                expanded[c * rows + f] = embedding[f * columns + c] * scale;
            }
        }
        let freq_emb_rows = arena.upload(gpu, &[rows * columns], &expanded, "freq_emb")?;

        let transformer = upload_transformer(gpu, arena, host)?;
        // The channel samplers are 1x1 convs over the channel axis: (out, in)
        // weights applied as one GEMM over the flattened (in, positions) input.
        let up_freq_w = {
            let w3 = host.weights.channel_upsampler.weight_t.t().to_owned().insert_axis(ndarray::Axis(2));
            ConvW { weight: w3.iter().copied().collect(), bias: host.weights.channel_upsampler.bias.as_ref().unwrap().to_vec(), shape: vec![512, 384, 1] }
        };
        let up_freq = upload_conv(gpu, arena, &up_freq_w, (1, 1), (0, 0), "up.freq")?;
        let up_time_w = {
            let w3 = host.weights.channel_upsampler_t.weight_t.t().to_owned().insert_axis(ndarray::Axis(2));
            ConvW { weight: w3.iter().copied().collect(), bias: host.weights.channel_upsampler_t.bias.as_ref().unwrap().to_vec(), shape: vec![512, 384, 1] }
        };
        let up_time = upload_conv(gpu, arena, &up_time_w, (1, 1), (0, 0), "up.time")?;
        let down_freq_w = {
            let w3 = host.weights.channel_downsampler.weight_t.t().to_owned().insert_axis(ndarray::Axis(2));
            ConvW { weight: w3.iter().copied().collect(), bias: host.weights.channel_downsampler.bias.as_ref().unwrap().to_vec(), shape: vec![384, 512, 1] }
        };
        let down_freq = upload_conv(gpu, arena, &down_freq_w, (1, 1), (0, 0), "down.freq")?;
        let down_time_w = {
            let w3 = host.weights.channel_downsampler_t.weight_t.t().to_owned().insert_axis(ndarray::Axis(2));
            ConvW { weight: w3.iter().copied().collect(), bias: host.weights.channel_downsampler_t.bias.as_ref().unwrap().to_vec(), shape: vec![384, 512, 1] }
        };
        let down_time = upload_conv(gpu, arena, &down_time_w, (1, 1), (0, 0), "down.time")?;

        // Decoder layers. The freq branch's rewrite is a (3, 3) conv with pad
        // 1; the time branch's is a width-3 conv1d with pad 1 (as (1, 3)).
        // Both branches upsample along their spatial axis with a transposed
        // conv, cropped by kernel/4 afterwards.
        let stride = host.config.stride;
        let decoder = host
            .weights
            .decoder
            .iter()
            .enumerate()
            .map(|(index, layer)| {
                let (rewrite, conv_tr) = if layer.conv_tr.shape.len() == 4 {
                    let k0 = layer.rewrite.shape[2];
                    let k1 = layer.rewrite.shape[3];
                    (
                        upload_conv(gpu, arena, &layer.rewrite, (1, 1), (k0 / 2, k1 / 2), "dec.rewrite")?,
                        upload_conv_tr(gpu, arena, &layer.conv_tr, (stride, 1), "dec.conv_tr")?,
                    )
                } else {
                    let k = layer.rewrite.shape[2];
                    (
                        upload_conv(gpu, arena, &layer.rewrite, (1, 1), (0, k / 2), "dec.rewrite")?,
                        upload_conv_tr(gpu, arena, &layer.conv_tr, (1, stride), "dec.conv_tr")?,
                    )
                };
                Ok(GpuDec {
                    rewrite,
                    dconv: upload_dconv(gpu, arena, &layer.dconv, "dec.dconv")?,
                    conv_tr,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let tdecoder = host
            .weights
            .tdecoder
            .iter()
            .enumerate()
            .map(|(index, layer)| {
                let k = layer.rewrite.shape[2];
                Ok(GpuDec {
                    rewrite: upload_conv(gpu, arena, &layer.rewrite, (1, 1), (0, k / 2), "tdec.rewrite")?,
                    dconv: upload_dconv(gpu, arena, &layer.dconv, "tdec.dconv")?,
                    conv_tr: upload_conv_tr(gpu, arena, &layer.conv_tr, (1, stride), "tdec.conv_tr")?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            encoder,
            tencoder,
            freq_emb_rows,
            unit: arena.upload(gpu, &[1], &[1.0], "unit")?,
            zero: arena.upload(gpu, &[1], &[0.0], "zero")?,
            transformer,
            up_freq,
            up_time,
            down_freq,
            down_time,
            decoder,
            tdecoder,
        })
    }
}

// ------------------------------------------------------------------ stage glue

/// Grows `scratch` in place when `needed` elements would not fit.
///
/// The gather's plane is `rows * k_pad * pitch` and every row's patches live
/// simultaneously — the GEMM reads them all after the gathers — so a conv that
/// does not fit cannot be sliced smaller without splitting the conv itself. A
/// growth allocates a fresh buffer; the dead one goes back to the pool, so
/// steady state is one buffer of the largest size the shapes demand.
fn scratch_for(
    arena: &mut Arena,
    gpu: &Gpu,
    scratch: &mut DevTensor,
    needed: usize,
    label: &str,
) -> Result<()> {
    if scratch.len() >= needed {
        return Ok(());
    }
    *scratch = arena.alloc(gpu, (needed * 4) as u64, label)?;
    Ok(())
}

/// Elements of an im2col plane for `conv` on a `(batch, c, h, w)` (or
/// `(batch, c, w)` with `h = 1`) input. Matches `conv_stage`.
fn conv_plane_elems(batch: usize, h: usize, w: usize, conv: &GpuConv) -> usize {
    let shape = Im2ColShape {
        batch,
        in_channels: conv.in_channels,
        h,
        w,
        kernel: conv.kernel,
        stride: conv.stride,
        pad: conv.pad,
    };
    batch * pad_ceil(conv.k(), BK) * pad_ceil(shape.positions(), BN_C)
}

/// Largest im2col plane across a DConv's widened conv1 layers. Matches
/// `dconv_stage`.
fn dconv_plane_elems(rows: usize, time: usize, dconv: &GpuDConv) -> usize {
    dconv
        .layers
        .iter()
        .map(|layer| rows * pad_ceil(layer.conv1.k(), BK) * pad_ceil(time, BN_C))
        .max()
        .unwrap_or(0)
}

/// Largest im2col / DConv plane the encoder stack will ask `scratch` to hold.
fn encoder_scratch_elems(
    batch: usize,
    bins: usize,
    frames: usize,
    samples: usize,
    encoder: &[GpuEnc],
    tencoder: &[GpuEnc],
) -> usize {
    let mut max_elems = 0usize;
    let mut freq = bins;
    for layer in encoder {
        max_elems = max_elems.max(conv_plane_elems(batch, freq, frames, &layer.conv));
        let f_out = freq / 4;
        max_elems = max_elems.max(dconv_plane_elems(batch * f_out, frames, &layer.dconv));
        max_elems = max_elems.max(conv_plane_elems(batch, f_out, frames, &layer.rewrite));
        freq = f_out;
    }
    let mut time = samples;
    for layer in tencoder {
        let padded = if time % 4 != 0 {
            time + 4 - time % 4
        } else {
            time
        };
        max_elems = max_elems.max(conv_plane_elems(batch, 1, padded, &layer.conv));
        let t_out = Im2ColShape {
            batch,
            in_channels: layer.conv.in_channels,
            h: 1,
            w: padded,
            kernel: layer.conv.kernel,
            stride: layer.conv.stride,
            pad: layer.conv.pad,
        }
        .out_hw()
        .1;
        max_elems = max_elems.max(dconv_plane_elems(batch, t_out, &layer.dconv));
        max_elems = max_elems.max(conv_plane_elems(batch, 1, t_out, &layer.rewrite));
        time = t_out;
    }
    max_elems
}

/// Bottleneck spatial size and the per-layer time lengths the decoder crops
/// to — the same values `run_encoder_stack` records in `EncoderStack`.
fn bottleneck_shapes(
    batch: usize,
    bins: usize,
    frames: usize,
    samples: usize,
    encoder: &[GpuEnc],
    tencoder: &[GpuEnc],
) -> (usize, usize, usize, Vec<usize>) {
    let freq_h = bins / 4usize.pow(encoder.len() as u32);
    let mut time = samples;
    let mut lengths_t = Vec::with_capacity(tencoder.len());
    for layer in tencoder {
        lengths_t.push(time);
        let padded = if time % 4 != 0 {
            time + 4 - time % 4
        } else {
            time
        };
        time = Im2ColShape {
            batch,
            in_channels: layer.conv.in_channels,
            h: 1,
            w: padded,
            kernel: layer.conv.kernel,
            stride: layer.conv.stride,
            pad: layer.conv.pad,
        }
        .out_hw()
        .1;
    }
    (freq_h, frames, time, lengths_t)
}

/// Largest im2col / DConv plane the decoder stack will ask `scratch` to hold.
fn decoder_scratch_elems(
    batch: usize,
    freq_h: usize,
    freq_t: usize,
    time_samples: usize,
    lengths_t: &[usize],
    decoder: &[GpuDec],
    tdecoder: &[GpuDec],
    stride: usize,
) -> usize {
    let mut max_elems = 0usize;
    let mut f = freq_h;
    let t = freq_t;
    let mut samples = time_samples;
    let depth = decoder.len();
    for (index, (layer, layer_t)) in decoder.iter().zip(tdecoder.iter()).enumerate() {
        max_elems = max_elems.max(conv_plane_elems(batch, f, t, &layer.rewrite));
        max_elems = max_elems.max(dconv_plane_elems(batch * f, t, &layer.dconv));
        let kh = layer.conv_tr.kernel.0;
        let out_f = (f - 1) * stride + kh;
        f = out_f - 2 * (kh / 4);
        max_elems = max_elems.max(conv_plane_elems(batch, 1, samples, &layer_t.rewrite));
        max_elems = max_elems.max(dconv_plane_elems(batch, samples, &layer_t.dconv));
        samples = lengths_t
            .get(depth - 1 - index)
            .copied()
            .unwrap_or(samples);
    }
    max_elems
}

/// One convolution with its bias on a `(batch, c, h, w)` input. `batch` is the
/// tensor's own leading axis — a batched forward carries several segments of
/// the same shape through one pass.
fn conv_stage(
    kernels: &Kernels,
    gpu: &Gpu,
    arena: &mut Arena,
    recorder: &mut Recorder,
    scratch: &mut DevTensor,
    x: &DevTensor,
    conv: &GpuConv,
    out: &DevTensor,
) -> Result<()> {
    let dims = &x.shape;
    let batch = dims[0];
    // The waveform branch passes a 3-D `(batch, c, t)` view (its height axis is
    // 1 and already folded away); the frequency branch passes 4-D.
    let (h, w) = match dims.len() {
        4 => (dims[2], dims[3]),
        3 => (1, dims[2]),
        other => panic!("conv_stage needs a 3-D or 4-D input, got rank {other}"),
    };
    let shape = Im2ColShape {
        batch,
        in_channels: conv.in_channels,
        h,
        w,
        kernel: conv.kernel,
        stride: conv.stride,
        pad: conv.pad,
    };
    if std::env::var("DEMUCS_SHAPE_DEBUG").is_ok() {
        eprintln!(
            "[conv] batch={batch} ic={} oc={} k={:?} h={h} w={w} stride={:?} pad={:?} positions={}",
            conv.in_channels, conv.out_channels, conv.kernel, conv.stride, conv.pad, shape.positions()
        );
    }
    // The whole plane, not one chunk of it: every batch's patches live at once.
    let needed =
        batch * pad_ceil(conv.k(), BK) * pad_ceil(shape.positions(), BN_C);
    scratch_for(arena, gpu, scratch, needed, "conv.scratch")?;
    kernels.conv2d_into(
        gpu,
        arena,
        recorder,
        x,
        &conv.weight,
        Some(&conv.bias),
        scratch,
        out,
        shape,
        conv.out_channels,
    )
}

/// `(1, c, f, t) -> (c*f, t)`-row-major `(f, c, t)`: one transpose of the
/// `(c, f, t)` view swaps the channel and frequency axes, which is the layout
/// the reference runs its DConv in. `x` is `(batch, c, f, t)` and `out` is
/// `(batch, f, c, t)`: the batch axis rides along untouched, never folded into
/// the swap.
fn channels_to_leading_freq(
    kernels: &Kernels,
    gpu: &Gpu,
    arena: &mut Arena,
    recorder: &mut Recorder,
    x: &DevTensor,
    out: &DevTensor,
) -> Result<()> {
    let dims = &x.shape;
    // (batch, rows = c, cols = f, width = t) -> (batch, f, c, t)
    kernels.transpose(gpu, arena, recorder, x, out, dims[0], dims[1], dims[2], dims[3])
}

/// Inverse of [`channels_to_leading_freq`]: `x` is the `(batch * f, c, t)` the
/// DConv consumes (rows baked into one axis) and `out` is `(batch, c, f, t)`.
fn leading_freq_to_channels(
    kernels: &Kernels,
    gpu: &Gpu,
    arena: &mut Arena,
    recorder: &mut Recorder,
    x: &DevTensor,
    out: &DevTensor,
) -> Result<()> {
    let dims = &x.shape;
    // (batch, rows = f, cols = c, width = t) -> (batch, c, f, t)
    kernels.transpose(gpu, arena, recorder, x, out, out.shape[0], out.shape[2], dims[1], dims[2])
}

/// The `DConv` residual branch on a `(rows, channels, time)` tensor.
///
/// `conv2` is a batched GEMM — one `(2C, hidden) @ (hidden, time)` per row — so
/// the output lands channel-major exactly as the layout wants. `input` is the
/// *pre-branch* tensor and stays alive for the residual add, so the caller must
/// not recycle its arena slot until this returns.
#[allow(clippy::too_many_arguments)]
fn dconv_stage(
    kernels: &Kernels,
    gpu: &Gpu,
    arena: &mut Arena,
    mut recorder: &mut Recorder,
    scratch: &mut DevTensor,
    input: &DevTensor,
    dconv: &GpuDConv,
    rows: usize,
    time: usize,
    trace: &mut dyn TraceSink,
) -> Result<DevTensor> {
    let mut current = input.clone();
    for (depth, layer) in dconv.layers.iter().enumerate() {
        let name = format!("dconv.{depth}");

        // conv1: (hidden, c*k) @ (c*k, t) per row, tight layout.
        let conv1_out =
            arena.tensor(gpu, &[rows, layer.hidden, time], &format!("{name}.conv1"))?;
        // The DConv is a Conv1d over each row: the freq branch's permuted
        // tensor is `rows` independent (channels, time) convolutions, and the
        // waveform branch's is one. `rows` is the *batch* here — treating it as
        // the spatial height makes the gather read the tensor as
        // (channels, rows, time), which mixes every row.
        //
        // The im2col grid's y axis is `batch * patch_rows`, capped at 65535, so
        // the gather has to be chunked (encoder.0's first DConv: 512 rows x 240
        // patch rows = 122880, well over). The GEMM does not: it covers every
        // row in one batched dispatch, which is what keeps the per-row offsets
        // out of the bind groups — `hidden * time` is 6 * 33 = 198 at a real
        // segment's frame count, not a multiple of 8, so a per-row bind lands
        // mid-alignment and the driver rejects hundreds of bind groups. The
        // chunk's block of the scratch is `span * k_pad` rows of `pitch`, whose
        // offset is a multiple of 8 because `k_pad` and `pitch` are multiples
        // of `BK` and `BN`.
        let k_pad = pad_ceil(layer.conv1.k(), BK);
        let pitch = pad_ceil(time, BN_C);
        // Every row's patches live simultaneously (the GEMM reads them all
        // after the gathers), so the scratch has to hold the whole
        // `rows * k_pad * pitch` plane, not one chunk of it. Growing is fine
        // here: the per-layer plane is computed before the gather loop and the
        // previous buffer returns to the pool.
        let needed = rows * k_pad * pitch;
        scratch_for(arena, gpu, scratch, needed, "dconv.scratch")?;
        let chunk_rows = (60_000 / k_pad.max(1)).max(1);
        let mut row = 0usize;
        while row < rows {
            let span = chunk_rows.min(rows - row);
            // Each chunk is a contiguous (span, c, time) slab of `current`.
            let in_slab = current.slice(
                row * layer.conv1.in_channels * time,
                vec![span, layer.conv1.in_channels, time],
            )?;
            let patch_slab = scratch.slice(row * k_pad * pitch, vec![span * k_pad * pitch])?;
            kernels.conv2d_gather_into(
                gpu,
                arena,
                &mut recorder,
                &in_slab,
                &patch_slab,
                Im2ColShape {
                    batch: span,
                    in_channels: layer.conv1.in_channels,
                    h: 1,
                    w: time,
                    kernel: layer.conv1.kernel,
                    stride: (1, 1),
                    pad: layer.conv1.pad,
                },
            )?;
            row += span;
        }
        kernels.conv2d_gemm_into(
            gpu,
            arena,
            &mut recorder,
            &layer.conv1.weight,
            &current,
            scratch,
            &conv1_out,
            Im2ColShape {
                batch: rows,
                in_channels: layer.conv1.in_channels,
                h: 1,
                w: time,
                kernel: layer.conv1.kernel,
                stride: (1, 1),
                pad: layer.conv1.pad,
            },
            layer.hidden,
            Some(&layer.conv1.bias),
        )?;
        if trace.wants(&format!("{name}.conv1")) {
            record_tensor(gpu, &mut recorder, &conv1_out, &format!("{name}.conv1"), trace)?;
        }

        // GroupNorm(1, hidden) + GELU, tight layout: the GEMM takes the logical
        // width as its B row stride (the kernel reads strides, not just tiles).
        let normed = arena.tensor(gpu, &[rows, layer.hidden, time], &format!("{name}.norm1"))?;
        kernels.group_norm_into(
            gpu,
            arena,
            &mut recorder,
            &conv1_out,
            &layer.norm1_scale,
            &layer.norm1_shift,
            &normed,
            GroupNormShape {
                rows,
                groups: 1,
                per_group: layer.hidden,
                len: time,
                row_stride: layer.hidden * time,
                len_stride: 1,
                channel_stride: time,
                eps: NORM_EPS,
            },
        )?;
        kernels.gelu_in_place(gpu, arena, &mut recorder, &normed)?;
        if trace.wants(&format!("{name}.norm1")) {
            record_tensor(gpu, &mut recorder, &normed, &format!("{name}.norm1"), trace)?;
        }

        // conv2: C[b] = W @ X[b], one (2C, hidden) @ (hidden, time) per row,
        // batched over rows. A is the shared weight; B are the normed rows; C
        // is one (channels, time) matrix per batch with tight rows.
        let channels = layer.channels * 2;
        let conv2_out = arena.tensor(gpu, &[rows, channels, time], &format!("{name}.conv2"))?;
        if std::env::var("DEMUCS_CONV2_PROBE").is_ok() {
            // Probe: a plain copy instead of the GEMM. If the trace then shows
            // norm1's values, the plumbing is fine and the GEMM dispatch itself
            // is what fails.
            kernels.copy(gpu, arena, &mut recorder, &normed, &conv2_out, normed.len())?;
        } else {
        kernels.gemm_into(
            gpu,
            arena,
            &mut recorder,
            &layer.conv2.weight,
            &normed,
            &conv2_out,
            GemmJob {
                m: channels,
                n: time,
                k: layer.hidden,
                lda: pad_ceil(layer.hidden, BK),
                ldb: time,
                ldc: time,
                batches: rows,
                inner_count: 1,
                a_outer: 0,
                a_inner: 0,
                b_outer: layer.hidden * time,
                b_inner: 0,
                c_outer: channels * time,
                c_inner: 0,
                transb: false,
            },
        )?;
        }

        if trace.wants(&format!("{name}.norm1.after")) {
            // Re-read the GEMM's B operand after the dispatch: if it changed,
            // something aliased it.
            record_tensor(
                gpu,
                &mut recorder,
                &normed,
                &format!("{name}.norm1.after"),
                trace,
            )?;
        }

        // conv2's bias: the batched GEMM has no epilogue, so fold it in with
        // an identity affine (scale 1, shift = bias, per output channel).
        kernels.channel_affine_act_in_place(
            gpu,
            arena,
            &mut recorder,
            &conv2_out,
            &layer.conv2_ones,
            &layer.conv2.bias,
            rows,
            channels,
            time,
            Activation::Identity,
        )?;

        if trace.wants(&format!("{name}.conv2")) {
            // Recorded after the bias fold: the host's conv module output
            // includes its bias.
            record_tensor(gpu, &mut recorder, &conv2_out, &format!("{name}.conv2"), trace)?;
        }

        // GroupNorm(1, 2C) + GLU + LayerScale + residual add.
        let normed2 = arena.tensor(gpu, &[rows, channels, time], &format!("{name}.norm2"))?;
        kernels.group_norm_into(
            gpu,
            arena,
            recorder,
            &conv2_out,
            &layer.norm2_scale,
            &layer.norm2_shift,
            &normed2,
            GroupNormShape {
                rows,
                groups: 1,
                per_group: channels,
                len: time,
                row_stride: channels * time,
                len_stride: 1,
                channel_stride: time,
                eps: NORM_EPS,
            },
        )?;
        let gated = kernels.glu(gpu, arena, &mut recorder, &normed2, rows, layer.channels * time)?;
        if trace.wants(&format!("{name}.glu")) {
            record_tensor(gpu, &mut recorder, &gated, &format!("{name}.glu"), trace)?;
        }
        kernels.channel_affine_act_in_place(
            gpu,
            arena,
            recorder,
            &gated,
            &layer.gamma,
            &layer.gamma_zero,
            rows,
            layer.channels,
            time,
            Activation::Identity,
        )?;
        if trace.wants(&format!("{name}.out")) {
            record_tensor(gpu, &mut recorder, &gated, &format!("{name}.out"), trace)?;
        }

        if trace.wants(&format!("{name}.scaled")) {
            record_tensor(gpu, &mut recorder, &gated, &format!("{name}.scaled"), trace)?;
        }
        let next = arena.tensor(gpu, &[rows, layer.channels, time], &format!("{name}.add"))?;
        // The residual accumulates across depths (the reference's
        // `out = out + y`), so the base is `current`, not the branch input.
        kernels.copy(gpu, arena, &mut recorder, &current, &next, current.len())?;
        if trace.wants(&format!("{name}.residual")) {
            record_tensor(gpu, &mut recorder, &next, &format!("{name}.residual"), trace)?;
        }
        kernels.add_in_place(gpu, arena, &mut recorder, &next, &gated)?;
        if trace.wants(&format!("{name}.added")) {
            record_tensor(gpu, &mut recorder, &next, &format!("{name}.added"), trace)?;
        }
        current = next;
    }
    Ok(current)
}

/// `nn.LayerNorm` over the channel axis of a `(tokens, dim)` tensor, expressed
/// in the group_norm kernel's terms: one group of `dim` channels at `len == 1`,
/// so the group's statistics cover exactly the channel vector and the affine
/// index walks the channels.
fn layer_norm_stage(
    kernels: &Kernels,
    gpu: &Gpu,
    arena: &mut Arena,
    recorder: &mut Recorder,
    x: &DevTensor,
    out: &DevTensor,
    norm: &LayerNorm,
    tokens: usize,
) -> Result<()> {
    kernels.group_norm_into(
        gpu,
        arena,
        recorder,
        x,
        &norm.scale,
        &norm.shift,
        out,
        GroupNormShape {
            rows: tokens,
            groups: 1,
            per_group: norm.dim,
            len: 1,
            row_stride: norm.dim,
            len_stride: 1,
            channel_stride: 1,
            eps: NORM_EPS,
        },
    )
}

/// `MyGroupNorm(1, dim)` on a `(tokens, dim)` tensor: the reference transposes
/// to `(dim, tokens)`, normalises over the whole (dim, tokens) plane per batch,
/// and applies a per-channel affine. The kernel's index formula
/// `(local/len)*channel_stride + (local%len)*len_stride` reproduces the
/// `(token, channel)` layout with `len_stride = dim, channel_stride = 1`.
///
/// `tokens` is the *total* row count (`batch * per-segment`): the reference
/// normalises each batch's plane on its own, so the group rows are the
/// segments, not the whole tensor.
fn group_norm_positions_stage(
    kernels: &Kernels,
    gpu: &Gpu,
    arena: &mut Arena,
    recorder: &mut Recorder,
    x: &DevTensor,
    out: &DevTensor,
    norm: &GroupNorm,
    tokens: usize,
    batch: usize,
) -> Result<()> {
    let per_segment = tokens / batch;
    kernels.group_norm_into(
        gpu,
        arena,
        recorder,
        x,
        &norm.scale,
        &norm.shift,
        out,
        GroupNormShape {
            rows: batch,
            groups: 1,
            per_group: norm.dim,
            len: per_segment,
            row_stride: norm.dim * per_segment,
            len_stride: norm.dim,
            channel_stride: 1,
            eps: NORM_EPS,
        },
    )
}

/// A projection GEMM with its bias: `(tokens, in) @ (in, out)` read through the
/// weight's `(out, in)` layout via `transb`, then the bias folded with an
/// identity affine (the batched GEMM has no epilogue). `ones` is a shared
/// all-ones vector; the kernel reads its first `out_features` entries.
#[allow(clippy::too_many_arguments)]
/// A projection GEMM with its bias: `(tokens, in) @ (in, out)` read through the
/// weight's `(out, in)` layout via `transb`, then the bias folded with an
/// identity affine. `ones` is a shared all-ones vector; the kernel reads its
/// first `out_features` entries.
#[allow(clippy::too_many_arguments)]
fn proj_stage(
    kernels: &Kernels,
    gpu: &Gpu,
    arena: &mut Arena,
    recorder: &mut Recorder,
    x: &DevTensor,
    proj: &GpuProj,
    out: &DevTensor,
    tokens: usize,
    ones: &DevTensor,
) -> Result<()> {
    kernels.gemm_into(
        gpu,
        arena,
        recorder,
        x,
        &proj.weight,
        out,
        GemmJob {
            m: tokens,
            n: proj.out_features,
            k: proj.in_features,
            lda: proj.in_features,
            ldb: pad_ceil(proj.in_features, BK_C),
            ldc: proj.out_features,
            batches: 1,
            inner_count: 1,
            a_outer: 0,
            a_inner: 0,
            b_outer: 0,
            b_inner: 0,
            c_outer: 0,
            c_inner: 0,
            transb: true,
        },
    )?;
    kernels.channel_affine_act_in_place(
        gpu,
        arena,
        recorder,
        out,
        ones,
        &proj.bias,
        tokens,
        proj.out_features,
        1,
        Activation::Identity,
    )
}

impl GpuHtdemucsRunner {
    /// One attention block: head split, Q K^T per head (transb), softmax with
    /// the 1/sqrt(d_head) scale, AV per head, head merge.
    ///
    /// `q` comes from one branch and `k`/`v` from the other in the cross layers;
    /// the classic layers pass the same tensor for all three.
    ///
    /// `batch` segments are stacked on the token axis, so every attention
    /// pattern is per segment and the head axis is really `(segment, head)`.
    /// The GEMMs address that as two levels — `batches = batch * heads` with
    /// `inner_count = heads` — so one chunk's tokens never attend to another's,
    /// and each output element is the same computation it is at batch 1.
    #[allow(clippy::too_many_arguments)]
    fn attention_stage(
        &self,
        kernels: &Kernels,
        gpu: &Gpu,
        arena: &mut Arena,
        recorder: &mut Recorder,
        q: &DevTensor,
        k: &DevTensor,
        v: &DevTensor,
        batch: usize,
        q_tokens: usize,
        k_tokens: usize,
        out: &DevTensor,
        label: &str,
    ) -> Result<()> {
        let dim = q.shape[1];
        let heads = self.config.t_heads;
        let d_head = dim / heads;
        // The split lands (batch, heads, tokens, d_head) — batch-major, which is
        // what the two-level batch addressing reads: outer = segment, inner =
        // head.
        let mut split = |x: &DevTensor, tokens: usize, label: String| -> Result<DevTensor> {
            let tmp = arena.tensor(gpu, &[x.shape[0], heads, d_head], &format!("{label}.split"))?;
            kernels.transpose(gpu, arena, recorder, x, &tmp, batch, tokens, heads, d_head)?;
            Ok(tmp.with_shape(vec![batch * heads, tokens, d_head])?)
        };
        let q_h = split(q, q_tokens, format!("{label}.qh"))?;
        let k_h = split(k, k_tokens, format!("{label}.kh"))?;
        let v_h = split(v, k_tokens, format!("{label}.vh"))?;

        let scores = arena.tensor(
            gpu,
            &[batch * heads, q_tokens, k_tokens],
            &format!("{label}.scores"),
        )?;
        kernels.gemm_into(
            gpu, arena, recorder, &q_h, &k_h, &scores,
            GemmJob {
                m: q_tokens,
                n: k_tokens,
                k: d_head,
                lda: d_head,
                ldb: d_head,
                ldc: k_tokens,
                batches: batch * heads,
                inner_count: heads,
                a_outer: heads * q_tokens * d_head,
                a_inner: q_tokens * d_head,
                b_outer: heads * k_tokens * d_head,
                b_inner: k_tokens * d_head,
                c_outer: heads * q_tokens * k_tokens,
                c_inner: q_tokens * k_tokens,
                transb: true,
            },
        )?;
        let scaled = arena.tensor(
            gpu,
            &[batch * heads, q_tokens, k_tokens],
            &format!("{label}.softmax"),
        )?;
        kernels.softmax_scaled(
            gpu, arena, recorder, &scores, &scaled, batch * heads * q_tokens, k_tokens,
            1.0 / (d_head as f32).sqrt(),
        )?;
        let context = arena.tensor(
            gpu,
            &[batch * heads, q_tokens, d_head],
            &format!("{label}.ctx"),
        )?;
        kernels.gemm_into(
            gpu, arena, recorder, &scaled, &v_h, &context,
            GemmJob {
                m: q_tokens,
                n: d_head,
                k: k_tokens,
                lda: k_tokens,
                ldb: d_head,
                ldc: d_head,
                batches: batch * heads,
                inner_count: heads,
                a_outer: heads * q_tokens * k_tokens,
                a_inner: q_tokens * k_tokens,
                b_outer: heads * k_tokens * d_head,
                b_inner: k_tokens * d_head,
                c_outer: heads * q_tokens * d_head,
                c_inner: q_tokens * d_head,
                transb: false,
            },
        )?;
        // Head merge: (batch, heads, tokens, d_head) -> (batch, tokens, dim).
        kernels.transpose(gpu, arena, recorder, &context, out, batch, heads, q_tokens, d_head)
    }
}

impl GpuHtdemucsRunner {
    /// The cross-transformer, from the two bottleneck tensors in their branch
    /// layouts to the same layouts after the channel downsampler's input.
    ///
    /// Token order is t-major (`token = t * freqs + freq`), matching the
    /// reference's rearrange and the host traces. The pos embeddings are
    /// computed on the host per chunk (they depend on the token counts) and
    /// uploaded; everything else is device work.
    #[allow(clippy::too_many_arguments)]
    fn forward_transformer(
        &self,
        arena: &mut Arena,
        freq_bottleneck: &DevTensor,
        time_bottleneck: &DevTensor,
        pos_spec: &Array3<f32>,
        pos_time: &Array3<f32>,
        recorder: &mut Recorder,
        trace: &mut dyn TraceSink,
    ) -> Result<(DevTensor, DevTensor)> {
        let mut recorder = recorder;
        let gpu = &self.gpu;
        let kernels = &self.kernels;
        let t = &self.weights.transformer;
        let dim = t.norm_in.dim;
        let batch = freq_bottleneck.shape[0];
        let freq_tokens = freq_bottleneck.shape[2] * freq_bottleneck.shape[3];
        let time_tokens = time_bottleneck.shape[2];
        if std::env::var("DEMUCS_SHAPE_DEBUG").is_ok() {
            eprintln!("[shapes] freq_bottleneck={:?} batch={batch} freq_tokens={freq_tokens} time_tokens={time_tokens}", freq_bottleneck.shape);
        }

        // Token-major (t*freqs + freq), per batch segment: two transposes.
        // Step 1: (c, f, t) -> (f, c, t) — swap the channel and frequency axes.
        let spec_step1 = arena.tensor(gpu, &[batch, freq_bottleneck.shape[2], freq_bottleneck.shape[1], freq_bottleneck.shape[3]], "xt.spec.step1")?;
        kernels.transpose(gpu, arena, &mut recorder, freq_bottleneck, &spec_step1,
            batch, freq_bottleneck.shape[1], freq_bottleneck.shape[2], freq_bottleneck.shape[3])?;
        // Step 2: (f*c, t) -> (t, f*c) — swap the (f*c) and t axes.
        // The (44, 4096) output re-labels to (352, 512) = (tokens, dim).
        let f = freq_bottleneck.shape[2];
        let c_ch = freq_bottleneck.shape[1];
        let t_n = freq_bottleneck.shape[3];
        let mut spec_raw = arena.tensor(gpu, &[batch, t_n, f * c_ch], "xt.spec.tokens")?;
        kernels.transpose(gpu, arena, &mut recorder, &spec_step1, &spec_raw,
            batch, f * c_ch, t_n, 1)?;
        let mut spec = spec_raw.with_shape(vec![batch * freq_tokens, dim])?;
        let mut spec_cur = arena.tensor(gpu, &[batch * freq_tokens, dim], "xt.spec.normin")?;
        layer_norm_stage(kernels, gpu, arena, &mut recorder, &spec, &spec_cur, &t.norm_in, batch * freq_tokens)?;
        // The pos embeddings are the chunk's first host write *after* dispatches
        // are pending on the recorder, so they get a fresh, never-pooled buffer:
        // a pooled one may still be read by those dispatches, and a
        // `write_buffer` lands ahead of the whole pending submission.
        let pos_spec_dev = arena.upload_dedicated(
            gpu, pos_spec.shape(), pos_spec.as_slice().expect("standard layout"), "xt.pos.spec",
        )?;
        // The pos embedding is one segment's worth; add_in_place repeats it
        // along the leading (token) axis, so every batch segment gets its own
        // copy of the same positions.
        kernels.add_in_place(gpu, arena, &mut recorder, &spec_cur, &pos_spec_dev)?;
        let mut time = arena.tensor(gpu, &[batch, time_tokens, dim], "xt.time.tokens")?;
        kernels.transpose(gpu, arena, &mut recorder, time_bottleneck, &time,
            batch, time_bottleneck.shape[1], time_tokens, 1)?;
        let mut time_cur = arena.tensor(gpu, &[batch * time_tokens, dim], "xt.time.normin")?;
        layer_norm_stage(kernels, gpu, arena, &mut recorder, &time, &time_cur, &t.norm_in_t, batch * time_tokens)?;
        let pos_time_dev = arena.upload_dedicated(
            gpu, pos_time.shape(), pos_time.as_slice().expect("standard layout"), "xt.pos.time",
        )?;
        kernels.add_in_place(gpu, arena, &mut recorder, &time_cur, &pos_time_dev)?;
        drop(spec);
        drop(time);

        if trace.wants("crosstransformer.norm_in") {
            record_tensor(gpu, &mut recorder, &spec_cur, "crosstransformer.norm_in", trace)?;
        }
        if trace.wants("crosstransformer.norm_in_t") {
            record_tensor(gpu, &mut recorder, &time_cur, "crosstransformer.norm_in_t", trace)?;
        }

        for index in 0..self.config.t_layers {
            let layer = &t.layers[index];
            let layer_t = &t.layers_t[index];
            let name = format!("crosstransformer.layers.{index}");
            let name_t = format!("crosstransformer.layers_t.{index}");
            if layer.is_cross {
                let old_spec = spec_cur.clone();
                spec_cur = self.run_cross_layer(
                    &mut recorder, arena, layer, &spec_cur, &time_cur,
                    batch * freq_tokens, batch * time_tokens, batch, &name, trace,
                )?;
                time_cur = self.run_cross_layer(
                    &mut recorder, arena, layer_t, &time_cur, &old_spec,
                    batch * time_tokens, batch * freq_tokens, batch, &name_t, trace,
                )?;
            } else {
                spec_cur = self.run_self_layer(
                    &mut recorder, arena, layer, &spec_cur, batch * freq_tokens, batch, &name, trace,
                )?;
                time_cur = self.run_self_layer(
                    &mut recorder, arena, layer_t, &time_cur, batch * time_tokens, batch, &name_t, trace,
                )?;
            }
        }

        // Back to the branch layouts. The tokens are t-major (token = t*F + f);
        // the (c, f, t) the downsampler wants walks f-major, so first transpose
        // the token matrix viewed as (t, f, c) into (f, t, c) — that re-indexes
        // the rows to f-major — then the usual (tokens, dim) -> (dim, tokens).
        // Both keep the batch axis in front.
        let (f_n, t_n) = (freq_bottleneck.shape[2], freq_bottleneck.shape[3]);
        let mut spec_step1 = arena.tensor(gpu, &[batch * freq_tokens, dim], "xt.spec.out1")?;
        kernels.transpose(gpu, arena, &mut recorder,
            &spec_cur.with_shape(vec![batch, t_n, f_n, dim])?, &spec_step1, batch, t_n, f_n, dim)?;
        let mut spec_out = arena.tensor(gpu, &freq_bottleneck.shape, "xt.spec.out")?;
        kernels.transpose(gpu, arena, &mut recorder, &spec_step1, &spec_out,
            batch, freq_tokens, dim, 1)?;
        let mut time_out = arena.tensor(gpu, &time_bottleneck.shape, "xt.time.out")?;
        kernels.transpose(gpu, arena, &mut recorder, &time_cur, &time_out,
            batch, time_tokens, dim, 1)?;
        Ok((spec_out, time_out))
    }

    /// The encoder half + channel samplers + cross-transformer, from the
    /// normalised inputs to the two tensors the decoders consume. One submit
    /// per stage group; the decoders and the complex packing stay on host.
    pub fn forward_encoder_and_transformer(
        &self,
        freq_mag: &Array4<f32>,
        wave: &Array3<f32>,
        trace: &mut dyn TraceSink,
    ) -> Result<(Array4<f32>, Array3<f32>)> {
        let mut recorder = Recorder::new(&self.gpu);
        let (freq_dev, time_dev, _stack) =
            self.run_to_decoder_inputs(freq_mag, wave, &mut recorder, trace)?;
        let (freq_staging, freq_bytes) = stage_readback(&self.gpu, &mut recorder, &freq_dev)?;
        let (time_staging, time_bytes) = stage_readback(&self.gpu, &mut recorder, &time_dev)?;
        recorder.submit(&self.gpu)?;
        let freq_out = array4_from_bytes(
            &self.gpu.mapped_bytes(&freq_staging, freq_bytes)?,
            &freq_dev.shape,
        )?;
        let time_out = array3_from_bytes(
            &self.gpu.mapped_bytes(&time_staging, time_bytes)?,
            &time_dev.shape,
        )?;
        Ok((freq_out, time_out))
    }

    /// Encoders, samplers, transformer and downsamplers; the decoder inputs
    /// stay on device so the decoder pass can chain without a round trip.
    ///
    /// `recorder` is the chunk's single recorder: everything recorded here rides
    /// one submission with the rest of the chunk (see
    /// [`GpuHtdemucsRunner::forward_branch_outputs`]).
    fn run_to_decoder_inputs(
        &self,
        freq_mag: &Array4<f32>,
        wave: &Array3<f32>,
        recorder: &mut Recorder,
        trace: &mut dyn TraceSink,
    ) -> Result<(DevTensor, DevTensor, EncoderStack)> {
        let stack = self.run_encoder_stack(freq_mag, wave, recorder, trace)?;
        let batch = stack.freq_enc.shape[0];
        let freq_dims = stack.freq_enc.shape.clone();
        let time_samples = stack.time_enc.shape[2];
        let freq_flat = freq_dims[2] * freq_dims[3];
        if std::env::var("DEMUCS_SHAPE_DEBUG").is_ok() {
            eprintln!("[shapes] freq_dims={freq_dims:?} batch={batch} time_samples={time_samples} freq_flat={freq_flat}");
        }

        // Stage 1: channel upsamplers.
        let (freq_up, time_up) = {
            let arena = &mut *self.work.borrow_mut();
            let gpu = &self.gpu;
            let freq_up_in = stack.freq_enc.with_shape(vec![batch, 384, 1, freq_flat])?;
            let freq_up_out = arena.tensor(gpu, &[batch, 512, 1, freq_flat], "xt.up.freq")?;
            let mut su = arena.alloc(gpu, (batch * 384 * pad_ceil(freq_flat, BN_C) * 4) as u64, "scratch.uf")?;
            conv_stage(&self.kernels, gpu, arena, recorder, &mut su, &freq_up_in, &self.weights.up_freq, &freq_up_out)?;
            let time_up_in = stack.time_enc.with_shape(vec![batch, 384, 1, time_samples])?;
            let time_up_out = arena.tensor(gpu, &[batch, 512, 1, time_samples], "xt.up.time")?;
            let mut st = arena.alloc(gpu, (batch * 384 * pad_ceil(time_samples, BN_C) * 4) as u64, "scratch.ut")?;
            conv_stage(&self.kernels, gpu, arena, recorder, &mut st, &time_up_in, &self.weights.up_time, &time_up_out)?;
            (
                freq_up_out.with_shape(vec![batch, 512, freq_dims[2], freq_dims[3]])?,
                time_up_out.with_shape(vec![batch, 512, time_samples])?,
            )
        }; // arena borrow dropped

        // Pos embeddings, computed on the host per chunk. The device's two-step
        // transpose produces t-major tokens matching the reference's layout.
        // One segment's worth is enough: the add broadcasts along the leading
        // (token) axis, so every batch segment gets the same positions.
        let pos_spec = two_dimensional_positions(
            512, freq_dims[2], freq_dims[3], self.config.t_max_period as f32,
        );
        let pos_time = one_dimensional_positions(
            time_samples, 512, self.config.t_max_period as f32,
        );

        // Stage 2: the cross-transformer.
        let (spec_out, time_out) = {
            let arena = &mut *self.work.borrow_mut();
            self.forward_transformer(arena, &freq_up, &time_up, &pos_spec, &pos_time, recorder, trace)?
        };

        // Stage 3: channel downsamplers.
        let (freq_dec_in_dev, time_dec_in_dev) = {
            let arena = &mut *self.work.borrow_mut();
            let gpu = &self.gpu;
            let spec_down_in = spec_out.with_shape(vec![batch, 512, 1, freq_flat])?;
            let spec_down_out = arena.tensor(gpu, &[batch, 384, 1, freq_flat], "xt.down.freq")?;
            let mut sd = arena.alloc(gpu, (batch * 512 * pad_ceil(freq_flat, BN_C) * 4) as u64, "scratch.df")?;
            conv_stage(&self.kernels, gpu, arena, recorder, &mut sd, &spec_down_in, &self.weights.down_freq, &spec_down_out)?;
            let time_down_in = time_out.with_shape(vec![batch, 512, 1, time_samples])?;
            let time_down_out = arena.tensor(gpu, &[batch, 384, 1, time_samples], "xt.down.time")?;
            let mut st = arena.alloc(gpu, (batch * 512 * pad_ceil(time_samples, BN_C) * 4) as u64, "scratch.dt")?;
            conv_stage(&self.kernels, gpu, arena, recorder, &mut st, &time_down_in, &self.weights.down_time, &time_down_out)?;
            (
                spec_down_out.with_shape(vec![batch, 384, freq_dims[2], freq_dims[3]])?,
                time_down_out.with_shape(vec![batch, 384, time_samples])?,
            )
        };
        Ok((freq_dec_in_dev, time_dec_in_dev, stack))
    }

    /// The two decoder stacks on device: skip sums, rewrite convolutions,
    /// DConv branches and transposed-conv upsamples with their crops. Returns
    /// the final branch outputs, pre-epilogue (denormalise, CaC masking and
    /// the STFT inverse stay on the host).
    fn run_decoders(
        &self,
        freq_dec_in: DevTensor,
        time_dec_in: DevTensor,
        mut stack: EncoderStack,
        recorder: &mut Recorder,
        trace: &mut dyn TraceSink,
    ) -> Result<(DevTensor, DevTensor)> {
        let mut recorder = recorder;
        let arena = &mut *self.work.borrow_mut();
        let gpu = &self.gpu;
        let kernels = &self.kernels;
        let stride = self.config.stride;
        let batch = freq_dec_in.shape[0];
        // Sized to the largest plane this decoder will gather (decoder.3's 3×3
        // rewrite at a 7.8 s segment is ~74 M elements / 297 MiB). `scratch_for`
        // still grows if the estimate is short. The encoder's matching buffer
        // is idle in the pool and, being within 4×, is reused here.
        let scratch_elements = decoder_scratch_elems(
            batch,
            freq_dec_in.shape[2],
            freq_dec_in.shape[3],
            time_dec_in.shape[2],
            &stack.lengths_t,
            &self.weights.decoder,
            &self.weights.tdecoder,
            stride,
        )
        .max(1);
        let mut scratch = arena.alloc(gpu, (scratch_elements * 4) as u64, "dec.scratch")?;

        let mut x = freq_dec_in;
        let mut xt = time_dec_in;
        let depth = self.config.depth;
        for index in 0..depth {
            let layer = &self.weights.decoder[index];
            let layer_t = &self.weights.tdecoder[index];
            let last = index == depth - 1;

            // ---- frequency branch ----
            let skip = stack.saved_freq.pop().expect("freq skip");
            kernels.add_in_place(gpu, arena, &mut recorder, &x, &skip)?;
            let (f, t) = (x.shape[2], x.shape[3]);
            let rewritten = arena.tensor(
                gpu,
                &[batch, layer.rewrite.out_channels, f, t],
                &format!("decoder.{index}.rewrite"),
            )?;
            conv_stage(kernels, gpu, arena, &mut recorder, &mut scratch, &x, &layer.rewrite, &rewritten)?;
            if trace.wants(&format!("decoder.{index}.rewrite")) {
                record_tensor(gpu, &mut recorder, &rewritten, &format!("decoder.{index}.rewrite"), trace)?;
            }
            let c2 = layer.rewrite.out_channels / 2;
            let gated = kernels.glu(
                gpu, arena, &mut recorder, &rewritten, batch, c2 * f * t,
            )?;
            let freq = gated.with_shape(vec![batch, c2, f, t])?;
            let permuted = arena.tensor(gpu, &[batch * f, c2, t], &format!("decoder.{index}.perm"))?;
            channels_to_leading_freq(kernels, gpu, arena, &mut recorder, &freq, &permuted)?;
            let branched = dconv_stage(
                kernels, gpu, arena, &mut recorder, &mut scratch, &permuted,
                &layer.dconv, batch * f, t, trace,
            )?;
            if trace.wants(&format!("decoder.{index}.dconv")) {
                record_tensor(gpu, &mut recorder, &branched, &format!("decoder.{index}.dconv"), trace)?;
            }
            let back = arena.tensor(gpu, &[batch, c2, f, t], &format!("decoder.{index}.unperm"))?;
            leading_freq_to_channels(kernels, gpu, arena, &mut recorder, &branched, &back)?;

            // Transposed-conv upsample along the frequency axis.
            let (ic, oc, kh, kw) = (
                layer.conv_tr.in_channels, layer.conv_tr.out_channels,
                layer.conv_tr.kernel.0, layer.conv_tr.kernel.1,
            );
            let out_f = (f - 1) * stride + kh;
            let up = arena.tensor(gpu, &[batch, oc, out_f, t], &format!("decoder.{index}.up"))?;
            let shape = Col2ImShape {
                batch, in_channels: ic, out_channels: oc,
                in_h: f, in_w: t, kernel: (kh, kw), stride: (stride, 1),
            };
            let k_pad = pad_ceil(ic, BK_C);
            let pitch = pad_ceil(f * t, BN_C);
            let m_pad = pad_ceil(shape.m(), crate::gpu::shaders::BM);
            let padded = arena.alloc(gpu, (batch * k_pad * pitch * 4) as u64, &format!("dec.padded.{index}"))?;
            let taps = arena.alloc(gpu, (batch * m_pad * pitch * 4) as u64, &format!("dec.taps.{index}"))?;
            kernels.conv_transpose2d_into(
                gpu, arena, &mut recorder, &back, &layer.conv_tr.weight, Some(&layer.conv_tr.bias),
                &padded, &taps, &up, shape,
            )?;
            if trace.wants(&format!("decoder.{index}.conv_tr")) {
                record_tensor(gpu, &mut recorder, &up, &format!("decoder.{index}.conv_tr"), trace)?;
            }
            // The frequency crop: keep kernel/4 off each edge.
            let pad = kh / 4;
            let kept = out_f - 2 * pad;
            let cropped = arena.tensor(gpu, &[batch, oc, kept, t], &format!("decoder.{index}.crop"))?;
            kernels.crop_rows_into(
                gpu, arena, &mut recorder, &up, &cropped,
                batch * oc, out_f * t, kept, t, pad,
            )?;
            if !last {
                kernels.gelu_in_place(gpu, arena, &mut recorder, &cropped)?;
            }
            if trace.wants(&format!("decoder.{index}#0")) {
                record_tensor(gpu, &mut recorder, &cropped, &format!("decoder.{index}#0"), trace)?;
            }
            x = cropped;

            // ---- waveform branch ----
            let skip_t = stack.saved_t.pop().expect("time skip");
            kernels.add_in_place(gpu, arena, &mut recorder, &xt, &skip_t)?;
            let samples = xt.shape[2];
            let rewritten_t = arena.tensor(
                gpu,
                &[batch, layer_t.rewrite.out_channels, samples],
                &format!("tdecoder.{index}.rewrite"),
            )?;
            conv_stage(kernels, gpu, arena, &mut recorder, &mut scratch, &xt, &layer_t.rewrite, &rewritten_t)?;
            let c2t = layer_t.rewrite.out_channels / 2;
            let gated_t = kernels.glu(
                gpu, arena, &mut recorder, &rewritten_t, batch, c2t * samples,
            )?;
            let freq_t = gated_t.with_shape(vec![batch, c2t, samples])?;
            let pre_t = dconv_stage(
                kernels, gpu, arena, &mut recorder, &mut scratch, &freq_t,
                &layer_t.dconv, batch, samples, trace,
            )?;

            let (ic_t, oc_t, kw_t) = (
                layer_t.conv_tr.in_channels, layer_t.conv_tr.out_channels, layer_t.conv_tr.kernel.1,
            );
            let up_samples = (samples - 1) * stride + kw_t;
            let up_t = arena.tensor(gpu, &[batch, oc_t, 1, up_samples], &format!("tdecoder.{index}.up"))?;
            let shape_t = Col2ImShape {
                batch, in_channels: ic_t, out_channels: oc_t,
                in_h: 1, in_w: samples, kernel: (1, kw_t), stride: (1, stride),
            };
            let k_pad_t = pad_ceil(ic_t, BK_C);
            let pitch_t = pad_ceil(samples, BN_C);
            let m_pad_t = pad_ceil(shape_t.m(), crate::gpu::shaders::BM);
            let padded_t = arena.alloc(gpu, (batch * k_pad_t * pitch_t * 4) as u64, &format!("tdec.padded.{index}"))?;
            let taps_t = arena.alloc(gpu, (batch * m_pad_t * pitch_t * 4) as u64, &format!("tdec.taps.{index}"))?;
            kernels.conv_transpose2d_into(
                gpu, arena, &mut recorder, &pre_t, &layer_t.conv_tr.weight, Some(&layer_t.conv_tr.bias),
                &padded_t, &taps_t, &up_t, shape_t,
            )?;
            let pad_t = kw_t / 4;
            let length = stack.lengths_t.pop().expect("time length");
            let cropped_t = arena.tensor(gpu, &[batch, oc_t, length], &format!("tdecoder.{index}.crop"))?;
            kernels.crop_rows_into(
                gpu, arena, &mut recorder, &up_t, &cropped_t,
                batch * oc_t, up_samples, length, 1, pad_t,
            )?;
            if !last {
                kernels.gelu_in_place(gpu, arena, &mut recorder, &cropped_t)?;
            }
            if trace.wants(&format!("tdecoder.{index}#0")) {
                record_tensor(gpu, &mut recorder, &cropped_t, &format!("tdecoder.{index}#0"), trace)?;
            }
            xt = cropped_t;
        }

        Ok((x, xt))
    }

    /// The full device path from the normalised inputs to the two branches'
    /// final outputs, before the host's denormalise/CaC/STFT epilogue.
    ///
    /// One recorder for the whole chunk — every stage's dispatches and the two
    /// readback copies ride the same submission. Submitting per stage instead
    /// drains the queue each time, and the GPU idles while the host records the
    /// next stage; the single submit keeps it fed end to end.
    pub fn forward_branch_outputs(
        &self,
        freq_mag: &Array4<f32>,
        wave: &Array3<f32>,
        trace: &mut dyn TraceSink,
    ) -> Result<(Array4<f32>, Array3<f32>)> {
        let mut recorder = Recorder::new(&self.gpu);
        let (freq_dev, time_dev, stack) =
            self.run_to_decoder_inputs(freq_mag, wave, &mut recorder, trace)?;
        let (freq_out, time_out) = self.run_decoders(freq_dev, time_dev, stack, &mut recorder, trace)?;
        let (freq_staging, freq_bytes) = stage_readback(&self.gpu, &mut recorder, &freq_out)?;
        let (time_staging, time_bytes) = stage_readback(&self.gpu, &mut recorder, &time_out)?;
        recorder.submit(&self.gpu)?;
        let freq_read = array4_from_bytes(
            &self.gpu.mapped_bytes(&freq_staging, freq_bytes)?,
            &freq_out.shape,
        )?;
        let time_read = array3_from_bytes(
            &self.gpu.mapped_bytes(&time_staging, time_bytes)?,
            &time_out.shape,
        )?;
        Ok((freq_read, time_read))
    }

    /// A classic (self-attention) layer: fused QKV projection, flash attention,
    /// out projection, the two feed-forward halves, the three norms and the
    /// residual/LayerScale pairs.
    ///
    /// `tokens` is the *total* row count (`batch * per-segment tokens`); every
    /// per-row op covers all of them, and `batch` only tells the attention how
    /// many segments' token blocks to keep apart.
    fn run_self_layer(
        &self,
        mut recorder: &mut Recorder,
        arena: &mut Arena,
        layer: &GpuTransformerLayer,
        x: &DevTensor,
        tokens: usize,
        batch: usize,
        name: &str,
        trace: &mut dyn TraceSink,
    ) -> Result<DevTensor> {
        let gpu = &self.gpu;
        let kernels = &self.kernels;
        let ones = &self.weights.transformer.ones;
        let dim = layer.dim;
        let heads = self.config.t_heads;

        let normed = arena.tensor(gpu, &[tokens, dim], &format!("{name}.norm1"))?;
        layer_norm_stage(kernels, gpu, arena, &mut recorder, x, &normed, &layer.norm1, tokens)?;
        if trace.wants(&format!("{name}.norm1")) {
            record_tensor(gpu, &mut recorder, &normed, &format!("{name}.norm1"), trace)?;
        }

        // Separate q/k/v projections: the score-matrix attention consumes each
        // as its own (tokens, dim) tensor.
        let q = arena.tensor(gpu, &[tokens, dim], &format!("{name}.q"))?;
        proj_stage(kernels, gpu, arena, &mut recorder, &normed, &layer.q_proj, &q, tokens, ones)?;
        let k = arena.tensor(gpu, &[tokens, dim], &format!("{name}.k"))?;
        proj_stage(kernels, gpu, arena, &mut recorder, &normed, &layer.k_proj, &k, tokens, ones)?;
        let v = arena.tensor(gpu, &[tokens, dim], &format!("{name}.v"))?;
        proj_stage(kernels, gpu, arena, &mut recorder, &normed, &layer.v_proj, &v, tokens, ones)?;
        if trace.wants(&format!("{name}.attn.q")) {
            record_tensor(gpu, &mut recorder, &q, &format!("{name}.attn.q"), trace)?;
            record_tensor(gpu, &mut recorder, &k, &format!("{name}.attn.k"), trace)?;
            record_tensor(gpu, &mut recorder, &v, &format!("{name}.attn.v"), trace)?;
        }
        let mut attended = arena.tensor(gpu, &[tokens, dim], &format!("{name}.attn.raw"))?;
        self.attention_stage(kernels, gpu, arena, &mut recorder, &q, &k, &v,
            batch, tokens / batch, tokens / batch, &mut attended, name)?;
        // The host's `self_attn#0` includes out_proj; apply it before the trace
        // and the LayerScale.
        let attended_p = arena.tensor(gpu, &[tokens, dim], &format!("{name}.attn"))?;
        proj_stage(kernels, gpu, arena, &mut recorder, &attended, &layer.out_proj, &attended_p, tokens, ones)?;
        if trace.wants(&format!("{name}.attn")) {
            record_tensor(gpu, &mut recorder, &attended_p, &format!("{name}.attn"), trace)?;
        }

        // The affine is in place on `attended_p`, which then holds gamma1*attn.
        kernels.channel_affine_act_in_place(
            gpu, arena, &mut recorder, &attended_p, &layer.gamma1, &self.weights.transformer.zeros,
            tokens, dim, 1, Activation::Identity,
        )?;
        let mut res1 = arena.tensor(gpu, &[tokens, dim], &format!("{name}.res1"))?;
        kernels.copy(gpu, arena, &mut recorder, x, &res1, x.len())?;
        kernels.add_in_place(gpu, arena, &mut recorder, &res1, &attended_p)?;
        if trace.wants(&format!("{name}.res1")) {
            record_tensor(gpu, &mut recorder, &res1, &format!("{name}.res1"), trace)?;
        }

        let normed2 = arena.tensor(gpu, &[tokens, dim], &format!("{name}.norm2"))?;
        layer_norm_stage(kernels, gpu, arena, &mut recorder, &res1, &normed2, &layer.norm2, tokens)?;
        if trace.wants(&format!("{name}.norm2")) {
            record_tensor(gpu, &mut recorder, &normed2, &format!("{name}.norm2"), trace)?;
        }
        let hidden = arena.tensor(gpu, &[tokens, layer.linear1.out_features], &format!("{name}.hidden"))?;
        proj_stage(kernels, gpu, arena, &mut recorder, &normed2, &layer.linear1, &hidden, tokens, ones)?;
        if trace.wants(&format!("{name}.linear1")) {
            record_tensor(gpu, &mut recorder, &hidden, &format!("{name}.linear1"), trace)?;
        }
        kernels.gelu_in_place(gpu, arena, &mut recorder, &hidden)?;
        let fed = arena.tensor(gpu, &[tokens, dim], &format!("{name}.fed"))?;
        proj_stage(kernels, gpu, arena, &mut recorder, &hidden, &layer.linear2, &fed, tokens, ones)?;
        if trace.wants(&format!("{name}.fed")) {
            record_tensor(gpu, &mut recorder, &fed, &format!("{name}.fed"), trace)?;
        }

        // In place on `fed`, which then holds gamma2*ffn.
        kernels.channel_affine_act_in_place(
            gpu, arena, &mut recorder, &fed, &layer.gamma2, &self.weights.transformer.zeros,
            tokens, dim, 1, Activation::Identity,
        )?;
        let mut res2 = arena.tensor(gpu, &[tokens, dim], &format!("{name}.res2"))?;
        kernels.copy(gpu, arena, &mut recorder, &res1, &res2, res1.len())?;
        kernels.add_in_place(gpu, arena, &mut recorder, &res2, &fed)?;
        let final_out = arena.tensor(gpu, &[tokens, dim], &format!("{name}.normout"))?;
        group_norm_positions_stage(kernels, gpu, arena, &mut recorder, &res2, &final_out, &layer.norm_out, tokens, batch)?;

        if trace.wants(name) {
            record_tensor(gpu, &mut recorder, &final_out, name, trace)?;
        }
        Ok(final_out)
    }

    /// A cross-attention layer: q from one branch, k/v from the other, through
    /// the score-matrix path (the flash kernel wants q and k in one tensor).
    ///
    /// `q_tokens`/`k_tokens` are *total* row counts (`batch * per-segment`).
    #[allow(clippy::too_many_arguments)]
    fn run_cross_layer(
        &self,
        mut recorder: &mut Recorder,
        arena: &mut Arena,
        layer: &GpuTransformerLayer,
        q_in: &DevTensor,
        k_in: &DevTensor,
        q_tokens: usize,
        k_tokens: usize,
        batch: usize,
        name: &str,
        trace: &mut dyn TraceSink,
    ) -> Result<DevTensor> {
        let gpu = &self.gpu;
        let kernels = &self.kernels;
        let ones = &self.weights.transformer.ones;
        let dim = layer.dim;
        let heads = self.config.t_heads;
        let d_head = dim / heads;

        let normed_q = arena.tensor(gpu, &[q_tokens, dim], &format!("{name}.norm1"))?;
        layer_norm_stage(kernels, gpu, arena, &mut recorder, q_in, &normed_q, &layer.norm1, q_tokens)?;
        let normed_k = arena.tensor(gpu, &[k_tokens, dim], &format!("{name}.norm2"))?;
        layer_norm_stage(kernels, gpu, arena, &mut recorder, k_in, &normed_k, &layer.norm2, k_tokens)?;
        if trace.wants(&format!("{name}.norm1")) {
            eprintln!("[dbg] recording {name}.norm1");
            record_tensor(gpu, &mut recorder, &normed_q, &format!("{name}.norm1"), trace)?;
            record_tensor(gpu, &mut recorder, &normed_k, &format!("{name}.norm2"), trace)?;
        }

        let q = arena.tensor(gpu, &[q_tokens, dim], &format!("{name}.q"))?;
        proj_stage(kernels, gpu, arena, &mut recorder, &normed_q, &layer.q_proj, &q, q_tokens, ones)?;
        let k = arena.tensor(gpu, &[k_tokens, dim], &format!("{name}.k"))?;
        proj_stage(kernels, gpu, arena, &mut recorder, &normed_k, &layer.k_proj, &k, k_tokens, ones)?;
        let v = arena.tensor(gpu, &[k_tokens, dim], &format!("{name}.v"))?;
        proj_stage(kernels, gpu, arena, &mut recorder, &normed_k, &layer.v_proj, &v, k_tokens, ones)?;
        if trace.wants(&format!("{name}.attn.q")) {
            record_tensor(gpu, &mut recorder, &q, &format!("{name}.attn.q"), trace)?;
            record_tensor(gpu, &mut recorder, &k, &format!("{name}.attn.k"), trace)?;
            record_tensor(gpu, &mut recorder, &v, &format!("{name}.attn.v"), trace)?;
        }

        let mut attn_out = arena.tensor(gpu, &[q_tokens, dim], &format!("{name}.attn.raw"))?;
        self.attention_stage(kernels, gpu, arena, &mut recorder, &q, &k, &v,
            batch, q_tokens / batch, k_tokens / batch, &mut attn_out, name)?;
        let attn_proj = arena.tensor(gpu, &[q_tokens, dim], &format!("{name}.attn"))?;
        proj_stage(kernels, gpu, arena, &mut recorder, &attn_out, &layer.out_proj, &attn_proj, q_tokens, ones)?;

        // In place on `attn_proj`, which then holds gamma1*attn.
        kernels.channel_affine_act_in_place(
            gpu, arena, &mut recorder, &attn_proj, &layer.gamma1, &self.weights.transformer.zeros,
            q_tokens, dim, 1, Activation::Identity,
        )?;
        let mut res1 = arena.tensor(gpu, &[q_tokens, dim], &format!("{name}.res1"))?;
        kernels.copy(gpu, arena, &mut recorder, q_in, &res1, q_in.len())?;
        kernels.add_in_place(gpu, arena, &mut recorder, &res1, &attn_proj)?;

        let norm3 = layer.norm3.as_ref().ok_or_else(|| Error::Gpu("cross layer without norm3".into()))?;
        let normed3 = arena.tensor(gpu, &[q_tokens, dim], &format!("{name}.norm3"))?;
        layer_norm_stage(kernels, gpu, arena, &mut recorder, &res1, &normed3, norm3, q_tokens)?;
        let hidden = arena.tensor(gpu, &[q_tokens, layer.linear1.out_features], &format!("{name}.hidden"))?;
        proj_stage(kernels, gpu, arena, &mut recorder, &normed3, &layer.linear1, &hidden, q_tokens, ones)?;
        kernels.gelu_in_place(gpu, arena, &mut recorder, &hidden)?;
        let fed = arena.tensor(gpu, &[q_tokens, dim], &format!("{name}.fed"))?;
        proj_stage(kernels, gpu, arena, &mut recorder, &hidden, &layer.linear2, &fed, q_tokens, ones)?;

        // In place on `fed`, which then holds gamma2*ffn.
        kernels.channel_affine_act_in_place(
            gpu, arena, &mut recorder, &fed, &layer.gamma2, &self.weights.transformer.zeros,
            q_tokens, dim, 1, Activation::Identity,
        )?;
        let mut res2 = arena.tensor(gpu, &[q_tokens, dim], &format!("{name}.res2"))?;
        kernels.copy(gpu, arena, &mut recorder, &res1, &res2, res1.len())?;
        kernels.add_in_place(gpu, arena, &mut recorder, &res2, &fed)?;
        let final_out = arena.tensor(gpu, &[q_tokens, dim], &format!("{name}.normout"))?;
        group_norm_positions_stage(kernels, gpu, arena, &mut recorder, &res2, &final_out, &layer.norm_out, q_tokens, batch)?;

        if trace.wants(name) {
            record_tensor(gpu, &mut recorder, &final_out, name, trace)?;
        }
        Ok(final_out)
    }
}

/// The runner: device weights, one work arena, one `Kernels`.
pub struct GpuHtdemucsRunner {
    gpu: Gpu,
    kernels: Kernels,
    /// Weights, uploaded once at construction.
    _weight_arena: Arena,
    weights: GpuWeights,
    /// Reset per chunk by [`GpuHtdemucsRunner::forward_encoders`]. Interior
    /// mutability: the stage methods take `&self` (they only read weights and
    /// config) while allocating from the work arena.
    work: std::cell::RefCell<Arena>,
    /// The transformer's head count and token layout come from the config.
    config: crate::demucs::config::HtdemucsConfig,
}

impl GpuHtdemucsRunner {
    pub fn new(host: &Htdemucs) -> Result<Self> {
        Self::with_selector(host, crate::gpu::DeviceSelector::Auto)
    }

    /// The adapter this runner was built on, for logs and `backend_tag`.
    pub fn device_name(&self) -> &str {
        &self.gpu.info.adapter.name
    }

    /// Builds the runner on a specific adapter, so a machine with both a
    /// discrete and an integrated GPU can be measured on each (`demucs
    /// adapters` lists what a selector accepts).
    pub fn with_selector(
        host: &Htdemucs,
        selector: crate::gpu::DeviceSelector,
    ) -> Result<Self> {
        let gpu = Gpu::with_selector(selector)?;
        let kernels = Kernels::new(&gpu)?;
        let mut weight_arena = Arena::new(&gpu, WEIGHT_ARENA_BYTES);
        let weights = GpuWeights::load(host, &gpu, &mut weight_arena)?;
        let work = std::cell::RefCell::new(Arena::new(&gpu, WORK_ARENA_BYTES));
        let config = host.config.clone();
        Ok(Self {
            gpu,
            kernels,
            _weight_arena: weight_arena,
            weights,
            work,
            config,
        })
    }

    pub fn adapter(&self) -> String {
        self.gpu.info.describe()
    }

    /// The frequency and waveform encoders, from the normalised magnitude /
    /// waveform inputs to the two bottleneck tensors.
    ///
    /// `freq_mag` is `(batch, 4, nfft/2, frames)` — the host's `_magnitude`,
    /// already `(x - mean) / (1e-5 + std)` normalised. `wave` is
    /// `(batch, 2, samples)`, the waveform branch's normalised input. The
    /// outputs are the two tensors the host transformer would consume:
    /// `(batch, 384, 8, frames)` and `(batch, 384, time)`.
    ///
    /// `batch` segments of the same shape ride through one pass: every op here
    /// is per-segment independent, so the results are the segment-wise ones.
    pub fn forward_encoders(
        &self,
        freq_mag: &Array4<f32>,
        wave: &Array3<f32>,
        trace: &mut dyn TraceSink,
    ) -> Result<(Array4<f32>, Array3<f32>)> {
        let gpu = &self.gpu;
        let mut recorder = Recorder::new(gpu);
        let stack = self.run_encoder_stack(freq_mag, wave, &mut recorder, trace)?;
        let (freq_staging, freq_bytes) = stage_readback(gpu, &mut recorder, &stack.freq_enc)?;
        let (time_staging, time_bytes) = stage_readback(gpu, &mut recorder, &stack.time_enc)?;
        recorder.submit(gpu)?;
        let freq_out = array4_from_bytes(
            &gpu.mapped_bytes(&freq_staging, freq_bytes)?,
            &stack.freq_enc.shape,
        )?;
        let time_out = array3_from_bytes(
            &gpu.mapped_bytes(&time_staging, time_bytes)?,
            &stack.time_enc.shape,
        )?;
        Ok((freq_out, time_out))
    }

    /// The two encoder stacks through their bottlenecks, everything still on
    /// device, with the skip connections the decoders consume.
    ///
    /// `recorder` is the caller's: a stage API (`forward_encoders`) submits it
    /// once after the readback copies, and the full path hands it its own
    /// single-submission recorder.
    fn run_encoder_stack(
        &self,
        freq_mag: &Array4<f32>,
        wave: &Array3<f32>,
        recorder: &mut Recorder,
        trace: &mut dyn TraceSink,
    ) -> Result<EncoderStack> {
        let mut recorder = recorder;
        let arena = &mut *self.work.borrow_mut();
        arena.reset();

        let gpu = &self.gpu;
        let kernels = &self.kernels;
        let dims = freq_mag.shape();
        let (batch, _channels, bins, frames) = (dims[0], dims[1], dims[2], dims[3]);

        let mut freq = arena.upload(gpu, freq_mag.shape(), freq_mag.as_slice().expect("standard layout"), "freq.in")?;
        let mut time = arena.upload(
            gpu,
            wave.shape(),
            wave.as_slice().expect("standard layout"),
            "time.in",
        )?;

        // One scratch sized to the larger of the encoder and decoder planes
        // (decoder.3's 3×3 rewrite is the global max, ~283 MiB at 7.8 s).
        // Sharing the size means the pool keeps a single jumbo that also
        // covers attention scores, instead of an 180 MiB encoder buffer and a
        // 283 MiB decoder buffer both sitting idle and filling the cap.
        let enc_elems = encoder_scratch_elems(
            batch,
            bins,
            frames,
            time.shape[2],
            &self.weights.encoder,
            &self.weights.tencoder,
        );
        let (bot_h, bot_t, bot_samples, lengths_t) = bottleneck_shapes(
            batch,
            bins,
            frames,
            time.shape[2],
            &self.weights.encoder,
            &self.weights.tencoder,
        );
        let dec_elems = decoder_scratch_elems(
            batch,
            bot_h,
            bot_t,
            bot_samples,
            &lengths_t,
            &self.weights.decoder,
            &self.weights.tdecoder,
            self.config.stride,
        );
        let scratch_elements = enc_elems.max(dec_elems).max(1);
        let mut scratch = arena.alloc(gpu, (scratch_elements * 4) as u64, "im2col.scratch")?;

        let mut saved_freq: Vec<DevTensor> = Vec::new();
        let mut saved_t: Vec<DevTensor> = Vec::new();
        for (index, layer) in self.weights.encoder.iter().enumerate() {
            // conv + GELU.
            let (oc, f_out) = (layer.conv.out_channels, freq.shape[2] / 4);
            let conv_out = arena.tensor(gpu, &[batch, oc, f_out, frames], &format!("encoder.{index}.conv"))?;
            conv_stage(kernels, gpu, arena, &mut recorder, &mut scratch, &freq, &layer.conv, &conv_out)?;
            if trace.wants(&format!("encoder.{index}.conv")) {
                record_tensor(gpu, &mut recorder, &conv_out, &format!("encoder.{index}.conv"), trace)?;
            }
            kernels.gelu_in_place(gpu, arena, &mut recorder, &conv_out)?;

            // DConv in the (f, c, t) order the reference uses. `rows` is the
            // bands of every segment: the DConv batches over them.
            let rows = batch * f_out;
            let permuted = arena.tensor(gpu, &[rows, oc, frames], &format!("encoder.{index}.perm"))?;
            channels_to_leading_freq(kernels, gpu, arena, &mut recorder, &conv_out, &permuted)?;
            if trace.wants(&format!("encoder.{index}.pre_emb")) {
                record_tensor(gpu, &mut recorder, &permuted, &format!("encoder.{index}.pre_emb"), trace)?;
            }
            let branched = dconv_stage(
                kernels,
                gpu,
                arena,
                &mut recorder,
                &mut scratch,
                &permuted,
                &layer.dconv,
                rows,
                frames,
                trace,
            )?;
            if trace.wants(&format!("encoder.{index}.dconv")) {
                record_tensor(gpu, &mut recorder, &branched, &format!("encoder.{index}.dconv"), trace)?;
            }
            let back = arena.tensor(gpu, &[batch, oc, f_out, frames], &format!("encoder.{index}.unperm"))?;
            leading_freq_to_channels(kernels, gpu, arena, &mut recorder, &branched, &back)?;


            // rewrite (1x1) + GLU.
            let rewritten = arena.tensor(
                gpu,
                &[batch, layer.rewrite.out_channels, f_out, frames],
                &format!("encoder.{index}.rewrite"),
            )?;
            conv_stage(kernels, gpu, arena, &mut recorder, &mut scratch, &back, &layer.rewrite, &rewritten)?;
            if trace.wants(&format!("encoder.{index}.rewrite")) {
                record_tensor(gpu, &mut recorder, &rewritten, &format!("encoder.{index}.rewrite"), trace)?;
            }
            let gated = kernels.glu(
                gpu,
                arena,
                &mut recorder,
                &rewritten,
                batch,
                layer.rewrite.out_channels / 2 * f_out * frames,
            )?;
            // The GLU kernel emits `(rows, half)`; the next encoder layer wants
            // the 4-D shape back (same element count, so this is a re-label).
            freq = gated.with_shape(vec![batch, layer.rewrite.out_channels / 2, f_out, frames])?;
            if index == 0 {
                // The reference adds the frequency embedding to encoder.0's
                // output — after the whole layer (rewrite + GLU included). The
                // (c, f, t) rows are (c, f) pairs and the embedding is constant
                // across t: one row bias covers it, repeated per segment by the
                // shader's `row % bias_rows`.
                kernels.add_row_bias_in_place(
                    gpu,
                    arena,
                    &mut recorder,
                    &freq,
                    &self.weights.freq_emb_rows,
                    batch * (layer.rewrite.out_channels / 2) * f_out,
                    frames,
                    (layer.rewrite.out_channels / 2) * f_out,
                )?;
            }

            if trace.wants(&format!("encoder.{index}")) {
                record_tensor(gpu, &mut recorder, &freq, &format!("encoder.{index}"), trace)?;
            }
            saved_freq.push(freq.clone());
        }

        let mut lengths_t: Vec<usize> = Vec::new();
        for (index, layer) in self.weights.tencoder.iter().enumerate() {
            let samples = time.shape[2];
            lengths_t.push(samples);
            let padded = if samples % 4 != 0 {
                let wide = arena.tensor(
                    gpu,
                    &[batch, layer.conv.in_channels, samples + 4 - samples % 4],
                    &format!("tencoder.{index}.pad"),
                )?;
                // The pad columns are read by the conv's gather as if they were
                // zero, and `copy_pitched_into` writes only the real columns —
                // a fresh buffer was zero because wgpu zeroes new buffers, but a
                // recycled one is not, and the pool hands this buffer back
                // every chunk. Zero it on the device first: a device-side pass
                // over a few MB is nothing next to the conv, while the host
                // `Arena::clear` would be a PCIe round trip at ~520 MB/s.
                // Zeroing the whole tensor rather than the tail because the
                // tail is not a contiguous region — it is the last few columns
                // of every row.
                kernels.fill_zero_into(gpu, arena, &mut recorder, &wide, wide.len())?;
                // Per-row copy: the wider destination has a larger row stride,
                // so a flat copy would shift every channel after the first.
                kernels.copy_pitched_into(
                    gpu,
                    arena,
                    &mut recorder,
                    &time,
                    &wide,
                    batch,
                    layer.conv.in_channels,
                    samples,
                    samples,
                    wide.shape[2],
                    layer.conv.in_channels,
                )?;
                wide
            } else {
                time.clone()
            };
            let length = padded.shape[2];
            let (oc, t_out) = (layer.conv.out_channels, (length + 2 * 2 - 8) / 4 + 1);
            let conv_out = arena.tensor(gpu, &[batch, oc, 1, t_out], &format!("tencoder.{index}.conv"))?;
            conv_stage(kernels, gpu, arena, &mut recorder, &mut scratch, &padded, &layer.conv, &conv_out)?;
            let conv_out = conv_out.slice(0, vec![batch, oc, t_out])?;
            if trace.wants(&format!("tencoder.{index}.conv")) {
                record_tensor(gpu, &mut recorder, &conv_out, &format!("tencoder.{index}.conv"), trace)?;
            }
            kernels.gelu_in_place(gpu, arena, &mut recorder, &conv_out)?;
            if trace.wants(&format!("tencoder.{index}.conv.gelu")) {
                record_tensor(gpu, &mut recorder, &conv_out, &format!("tencoder.{index}.conv.gelu"), trace)?;
            }

            // The waveform branch has no permute: the reference runs its DConv
            // straight on `(batch, channels, time)`, so `rows` here is the batch
            // and `oc` is the *channel* count. Calling this with rows=oc — as an
            // earlier version did — convolves each channel as if it were an
            // independent row over the whole time axis: 48x the work and a
            // 25M-element im2col for a 529k-element input.
            let branched = dconv_stage(
                kernels,
                gpu,
                arena,
                &mut recorder,
                &mut scratch,
                &conv_out,
                &layer.dconv,
                batch,
                t_out,
                trace,
            )?;

            let rewritten = arena.tensor(
                gpu,
                &[batch, layer.rewrite.out_channels, t_out],
                &format!("tencoder.{index}.rewrite"),
            )?;
            conv_stage(kernels, gpu, arena, &mut recorder, &mut scratch, &branched, &layer.rewrite, &rewritten)?;
            if trace.wants(&format!("tencoder.{index}.rewrite")) {
                record_tensor(gpu, &mut recorder, &rewritten, &format!("tencoder.{index}.rewrite"), trace)?;
            }
            let gated_t = kernels.glu(
                gpu,
                arena,
                &mut recorder,
                &rewritten,
                batch,
                layer.rewrite.out_channels / 2 * t_out,
            )?;
            time = gated_t.with_shape(vec![batch, layer.rewrite.out_channels / 2, t_out])?;

            if trace.wants(&format!("tencoder.{index}")) {
                record_tensor(gpu, &mut recorder, &time, &format!("tencoder.{index}"), trace)?;
            }
            saved_t.push(time.clone());
        }

        // No submit here: this rides the caller's recorder, which the full path
        // submits once for the whole chunk (the stage APIs create and submit
        // their own).
        Ok(EncoderStack {
            freq_enc: freq,
            time_enc: time,
            saved_freq,
            saved_t,
            lengths_t,
        })
    }
}

/// The encoder half's outputs plus everything the decoders consume, still on
/// device. The tensors live in the work arena; they stay valid until the next
/// `arena.reset()`.
struct EncoderStack {
    /// The frequency branch's bottleneck, the channel upsampler's input.
    freq_enc: DevTensor,
    /// The waveform branch's bottleneck, the channel upsampler's input.
    time_enc: DevTensor,
    /// Per depth, pushed in encoder order — the decoders pop from the back.
    saved_freq: Vec<DevTensor>,
    saved_t: Vec<DevTensor>,
    /// Per depth, the sample count each time decoder crops to.
    lengths_t: Vec<usize>,
}

/// Reads a device tensor back as an `(1, len)`-shaped trace record.
fn record_tensor(
    gpu: &Gpu,
    recorder: &mut Recorder,
    tensor: &DevTensor,
    name: &str,
    trace: &mut dyn TraceSink,
) -> Result<()> {
    let bytes = (tensor.len() * 4) as u64;
    let staging = gpu.staging(bytes);
    recorder.copy_to_staging(tensor, &staging, bytes);
    let taken = std::mem::replace(recorder, Recorder::new(gpu));
    taken.submit(gpu)?;
    let data = gpu.mapped_bytes(&staging, bytes)?;
    let values: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
    let view = ndarray::ArrayViewD::from_shape(ndarray::IxDyn(&[1, values.len()]), &values)
        .map_err(|e| Error::Shape(e.to_string()))?;
    trace.record(name, view);
    Ok(())
}

/// Records a readback of `tensor` into a fresh staging buffer on the recorder's
/// own encoder, so the copy rides the same submission as the dispatches that
/// produced it. The caller submits and then maps with [`Gpu::mapped_bytes`].
fn stage_readback(
    gpu: &Gpu,
    recorder: &mut Recorder,
    tensor: &DevTensor,
) -> Result<(wgpu::Buffer, u64)> {
    let bytes = (tensor.len() * 4) as u64;
    let staging = gpu.staging(bytes);
    recorder.copy_to_staging(tensor, &staging, bytes);
    Ok((staging, bytes))
}

/// Reshapes mapped staging bytes into a rank-4 array.
fn array4_from_bytes(data: &[u8], shape: &[usize]) -> Result<Array4<f32>> {
    let values: Vec<f32> = bytemuck::cast_slice(data).to_vec();
    Array4::from_shape_vec((shape[0], shape[1], shape[2], shape[3]), values)
        .map_err(|e| Error::Shape(format!("readback: {e}")))
}

/// Reshapes mapped staging bytes into a rank-3 array.
fn array3_from_bytes(data: &[u8], shape: &[usize]) -> Result<Array3<f32>> {
    let values: Vec<f32> = bytemuck::cast_slice(data).to_vec();
    Array3::from_shape_vec((shape[0], shape[1], shape[2]), values)
        .map_err(|e| Error::Shape(format!("readback: {e}")))
}

/// The full separation path on device: the host front end (STFT, packing,
/// per-branch normalisation), the device encoder/transformer/decoder stacks,
/// and the host epilogue (denormalise, CaC masking, inverse STFT, branch sum).
///
/// `mix` may stack several same-shaped chunks on its batch axis — the split
/// path pads every chunk to the segment length, so that is the natural shape
/// for a batched pass — and the result carries the same batch.
pub fn separate_gpu(
    host: &crate::demucs::host::Htdemucs,
    runner: &GpuHtdemucsRunner,
    mix: &Array3<f32>,
    trace: &mut dyn TraceSink,
) -> Result<Array4<f32>> {
    let batch = mix.shape()[0];
    let sources = host.config.sources.len();
    let training_length = host.training_length();

    // Front end: spectrogram, complex-as-channels packing, normalisation. The
    // stats are per batch element, so the loops run over all of them.
    let stage_timing = std::env::var("DEMUCS_STAGE_TIMING").is_ok();
    let front = std::time::Instant::now();
    let z = host.spec.spec(mix)?;
    let mag = crate::demucs::spec::pack_complex_as_channels(&z);
    let norm = host.normalise(&mag, mix);
    let mut mag_norm = mag.clone();
    for bi in 0..batch {
        let (offset, scale) = (norm.mean[bi], norm.std[bi]);
        mag_norm
            .slice_mut(ndarray::s![bi, .., .., ..])
            .mapv_inplace(|v| (v - offset) / (1e-5 + scale));
    }
    let mut wave_norm = mix.clone();
    for bi in 0..batch {
        let (offset, scale) = (norm.mean_t[bi], norm.std_t[bi]);
        wave_norm
            .slice_mut(ndarray::s![bi, .., ..])
            .mapv_inplace(|v| (v - offset) / (1e-5 + scale));
    }

    // Device stacks.
    let model = std::time::Instant::now();
    if stage_timing {
        eprintln!("[stg] front end {:?}", front.elapsed());
    }
    let (freq_out, time_out) = runner.forward_branch_outputs(&mag_norm, &wave_norm, trace)?;
    if stage_timing {
        eprintln!("[stg] device+readback {:?}", model.elapsed());
        let (labels, ms, ops) = crate::gpu::arena::host_timing_take();
        for slot in 0..labels.len() {
            eprintln!(
                "[stg]   host {:>10} {:>8.2} ms in {:>5} ops ({:.3} ms/op)",
                labels[slot], ms[slot], ops[slot],
                ms[slot] / ops[slot].max(1) as f64
            );
        }
    }
    let epilogue = std::time::Instant::now();

    // Epilogue.
    let mut x = freq_out;
    for bi in 0..batch {
        // The reference denormalises the mask input with  —
        // no epsilon, unlike the forward normalisation.
        let (mean, std) = (norm.mean[bi], norm.std[bi]);
        x.slice_mut(ndarray::s![bi, .., .., ..])
            .mapv_inplace(|v| (v + mean / std) * std);
    }
    if trace.wants("frequency_branch_out") {
        trace.record("_mask.in", x.view().into_dyn());
    }
    let waveform = {
        let zout = crate::demucs::spec::unpack_channels_as_complex(&x, sources)?;
        let zout = crate::demucs::spec::pad_for_ispec(&zout);
        host.spec.ispec(&zout, training_length, 0)?
    };
    if trace.wants("freq_ispec") {
        trace.record("_ispec.out", waveform.view().into_dyn());
    }

    let audio_channels = mix.shape()[1];
    let mut time = time_out
        .view()
        .into_shape_with_order((batch, sources, audio_channels, training_length))
        .expect("contiguous view")
        .to_owned();
    for bi in 0..batch {
        let (mean, std) = (norm.mean_t[bi], norm.std_t[bi]);
        time.slice_mut(ndarray::s![bi, .., .., ..])
            .mapv_inplace(|v| (v + mean / std) * std);
    }

    if stage_timing {
        eprintln!("[stg] epilogue {:?}", epilogue.elapsed());
    }
    Ok(time + waveform)
}

