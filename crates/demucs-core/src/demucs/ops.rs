//! Host kernels used by the `HTDemucs` forward.
//!
//! These are the numerical reference the wgpu kernels are checked against, so
//! each one states the PyTorch call it mirrors. Layouts are `(b, c, ...)` with
//! the last axis contiguous, exactly like the reference tensors.

use ndarray::{Array1, Array2, Array3, Array4, ArrayBase, ArrayView1, ArrayView2, Axis, s};
use rayon::prelude::*;

use crate::demucs::weights::ConvW;
use crate::error::{Error, Result};
use crate::ops::{gelu_erf, softmax_in_place, Linear};

/// Output positions per GEMM tile when a convolution is lowered to im2col.
const POSITIONS_PER_TILE: usize = 8192;

/// Largest patch matrix built in one go, in elements (128 MB).
///
/// The gather is the convolution's memory cost: `k * positions` floats written
/// and then read by the GEMM. Blocking keeps that bounded while still giving the
/// parallel loops enough work per block.
const PATCH_BLOCK_ELEMENTS: usize = 32 << 20;

/// `nn.Conv2d` with arbitrary stride/padding/dilation on a `(b, c, h, w)` input.
///
/// Three passes per block of output positions: a parallel im2col gather, a
/// parallel implicit GEMM, and a parallel bias + write-back. The earlier version
/// parallelised over the batch only, which for this model (batch 1) meant every
/// convolution ran on a single core — a 3x3 convolution over a 512x336 map cost
/// 322 ms that way.
pub fn conv2d<S: ndarray::Data<Elem = f32> + Sync>(
    x: &ndarray::ArrayBase<S, ndarray::Ix4>,
    w: &ConvW,
    stride: (usize, usize),
    pad: (usize, usize),
    dilation: (usize, usize),
) -> Array4<f32> {
    let (b, in_channels, h, width) = x.dim();
    // A `Conv1d` weight is `(out, in, k)`; it is applied as a `(1, k)` kernel.
    let (kh, kw) = match w.kernel().len() {
        1 => (1, w.kernel()[0]),
        2 => (w.kernel()[0], w.kernel()[1]),
        other => panic!("conv2d needs a 1-D or 2-D kernel, got {other} axes"),
    };
    let (sh, sw) = stride;
    let (ph, pw) = pad;
    let (dh, dw) = dilation;
    let oc = w.out_channels();

    let effective_h = dh * (kh - 1) + 1;
    let effective_w = dw * (kw - 1) + 1;
    let out_h = (h + 2 * ph).saturating_sub(effective_h) / sh + 1;
    let out_w = (width + 2 * pw).saturating_sub(effective_w) / sw + 1;
    let k = in_channels * kh * kw;
    let positions_all = out_h * out_w;
    let plane = positions_all;
    let weight = ArrayView2::from_shape((oc, k), &w.weight).expect("contiguous weight");

    // A 1x1 convolution *is* a GEMM: the patch matrix is the input, so the
    // gather and its 100 MB of traffic disappear. The weight here is `(oc, ic)`
    // and the input views as `(ic, positions)`, so the product lands in the
    // output's order already — no transpose on either side. Runs the bias in
    // the same write-back pass: a separate sweep would double the output
    // traffic for the DConv's 1x1s.
    if kh == 1 && kw == 1 && sh == 1 && sw == 1 && ph == 0 && pw == 0 {
        debug_assert_eq!(out_h, h);
        debug_assert_eq!(out_w, width);
        let mut flat = vec![0.0f32; b * oc * plane];
        // One GEMM per leading index, the leading indices in parallel.
        //
        // This loop *is* the DConv's 1x1 cost: `dconv.conv2` is 4096 of these a
        // segment (32 scopes of 128) and `dec.rewrite.conv` another 512, each of
        // them a `(96, 6) @ (6, 336)` — 387 kFLOP, 0.07 ms, 5 GFLOP/s, against a
        // memory floor two orders of magnitude below that. Serial, the crate's
        // per-call setup is what the time goes on; the rows have nothing to do
        // with each other, so they may as well all be in flight at once. Nested
        // inside rayon the crate's own dispatch does not oversubscribe — it is
        // the same pool — it just stops being the only thing running.
        flat.par_chunks_mut(oc * plane)
            .enumerate()
            .for_each(|(bi, dst)| {
                let input = x.slice(s![bi, .., .., ..]);
                let input_2d = input
                    .to_shape((in_channels, plane))
                    .expect("standard layout");
                let patch_view =
                    ArrayView2::from_shape((in_channels, plane), input_2d.as_slice().unwrap())
                        .expect("exact size");
                // (oc, k) @ (k, plane) + bias in one GEMM. Splitting the output
                // into channel blocks was measured *slower* (165 -> 218 ms across
                // the segment): each block pays the crate's own parallel dispatch,
                // and twelve small GEMMs lose to one big one here.
                let product = matmul_with_row_bias(&weight, &patch_view, &w.bias);
                dst.copy_from_slice(product.as_slice().expect("standard layout"));
            });
        return Array4::from_shape_vec((b, oc, out_h, out_w), flat)
            .expect("the output shape matches");
    }

    let mut flat = vec![0.0f32; b * oc * plane];
    let rows_per_block = (PATCH_BLOCK_ELEMENTS / (k * out_w).max(1)).max(1);
    let blocks: Vec<(usize, usize)> = {
        let mut blocks = Vec::new();
        let mut row = 0;
        while row < out_h {
            let rows = rows_per_block.min(out_h - row);
            blocks.push((row, rows));
            row += rows;
        }
        blocks
    };

    for bi in 0..b {
        let input = x.slice(s![bi, .., .., ..]);
        for &(row0, rows_here) in &blocks {
            let positions = rows_here * out_w;
            // 1. im2col: one row of the patch matrix per (ic, ky, kx) tap.
            let mut patches = vec![0.0f32; k * positions];
            patches
                .par_chunks_mut(positions)
                .enumerate()
                .for_each(|(krow, dst)| {
                    let ic = krow / (kh * kw);
                    let tap = krow % (kh * kw);
                    let (ky, kx) = (tap / kw, tap % kw);
                    let src = input.slice(s![ic, .., ..]);
                    let mut p = 0usize;
                    for oy in row0..row0 + rows_here {
                        let iy = (oy * sh + ky * dh) as isize - ph as isize;
                        if iy < 0 || iy >= h as isize {
                            p += out_w;
                            continue;
                        }
                        let src_row = src.slice(s![iy as usize, ..]);
                        for ox in 0..out_w {
                            let ix = (ox * sw + kx * dw) as isize - pw as isize;
                            dst[p] = if ix < 0 || ix >= width as isize {
                                0.0
                            } else {
                                src_row[ix as usize]
                            };
                            p += 1;
                        }
                    }
                });

            // 2. the convolution as one GEMM, `(oc, k) x (k, positions)`.
            let patch_view =
                ArrayView2::from_shape((k, positions), &patches).expect("exact size");
            let product = matmul(&weight, &patch_view);

            // 3. bias and write-back, one contiguous run per output channel.
            let base = bi * oc * plane + row0 * out_w;
            flat.par_chunks_mut(plane)
                .enumerate()
                .skip(bi * oc)
                .take(oc)
                .for_each(|(channel, dst)| {
                    let o = channel % oc;
                    let bias = w.bias[o];
                    let values = &product.as_slice().expect("standard layout")
                        [o * positions..(o + 1) * positions];
                    let target = &mut dst[row0 * out_w..row0 * out_w + positions];
                    for (slot, value) in target.iter_mut().zip(values.iter()) {
                        *slot = *value + bias;
                    }
                });
            let _ = base;
        }
    }
    Array4::from_shape_vec((b, oc, out_h, out_w), flat).expect("the output shape matches")
}

/// Direct (no im2col) 1-D convolution for `stride == 1`, which is what every
/// `DConv` uses.
///
/// The im2col path gathers `in_channels * k * positions` values — 99 MB for the
/// frequency branch's `k=3` DConv, to do 0.3 GFLOP — and that gather was the
/// single most expensive op in the segment. Here the input is read once per
/// (row, time block) and reused across output channels from L1.
/// Split a `(rows, in, t)` input and a `(out, in, k)` weight into independent
/// jobs that each produce one `(oc_block, out_t)` tile of the output.
///
/// The jobs carry raw pointers — they are only valid for the caller's slices —
/// so the only caller is [`conv1d_stride1_into`], which joins them before
/// returning. This exists because a closure cannot borrow the disjoint output
/// ranges rayon would need.
struct DirectConvJobs<'a> {
    source: &'a [f32],
    target: *mut f32,
    weight: &'a [f32],
    bias: &'a [f32],
    rows: usize,
    in_channels: usize,
    t: usize,
    oc: usize,
    k: usize,
    pad: usize,
    dilation: usize,
    out_len: usize,
    block_size: usize,
    chan_block: usize,
}

// The pointers are joined before the function returns, and each job touches
// only its own output tile.
unsafe impl<'a> Sync for DirectConvJobs<'a> {}

impl<'a> DirectConvJobs<'a> {
    /// `target_base` is the absolute element offset of `(row, c0, t0)` in the
    /// output; the tiles are disjoint, so no two jobs write the same slot.
    fn run(&self, target_base: usize, c0: usize, channels_here: usize, t0: usize, width: usize) {
        let DirectConvJobs {
            source,
            target,
            weight,
            bias,
            rows,
            in_channels: c,
            t,
            oc,
            k,
            pad,
            dilation,
            out_len,
            ..
        } = *self;
        let target = unsafe { std::slice::from_raw_parts_mut(target, rows * oc * out_len) };
        let row = target_base / (oc * out_len);
        let row_base = row * c * t;
        // The tile is small enough to sit in L2 (`chan_block * width` floats).
        let mut accumulators = vec![0.0f32; channels_here * width];
        for (channel, run) in accumulators.chunks_mut(width).enumerate() {
            run.fill(bias[c0 + channel]);
        }
        for ic in 0..c {
            let input = &source[row_base + ic * t..row_base + (ic + 1) * t];
            for j in 0..k {
                let offset = t0 as isize + (j * dilation) as isize - pad as isize;
                let first = (0isize.max(-offset)) as usize;
                let last = width.min((t as isize - offset).max(0) as usize);
                for (channel, run) in accumulators.chunks_mut(width).enumerate() {
                    let weight = weight[((c0 + channel) * c + ic) * k + j];
                    if weight == 0.0 {
                        continue;
                    }
                    for (i, slot) in run[first..last].iter_mut().enumerate() {
                        *slot += weight * input[(offset + first as isize + i as isize) as usize];
                    }
                }
            }
        }
        for channel in 0..channels_here {
            target[target_base + channel * out_len..target_base + channel * out_len + width]
                .copy_from_slice(&accumulators[channel * width..(channel + 1) * width]);
        }
    }
}

pub fn conv1d_stride1_into(
    x: &ndarray::ArrayView3<f32>,
    w: &ConvW,
    out: &mut Array3<f32>,
    pad: usize,
    dilation: usize,
) {
    let (rows, c, t) = x.dim();
    let oc = w.out_channels();
    let k = w.kernel()[0];
    let out_t = t + 2 * pad - (dilation * (k - 1) + 1) + 1;
    debug_assert_eq!(out.dim(), (rows, oc, out_t));
    let source = x.as_standard_layout();
    let source = source.as_slice().expect("standard layout");
    let weight = &w.weight;

    // A block of time long enough to amortise the halo, small enough that the
    // accumulators stay in L1 (`oc * tb` floats) and that even the layers with
    // eight rows produce more tasks than there are threads. At 2048 the
    // `rows == 8` layers ran on 8 of 20 cores and cost more than they saved.
    let block_size = match std::env::var("DEMUCS_CONV1D_BLOCK") {
        Ok(value) => value.parse::<usize>().unwrap_or(256),
        Err(_) => 256,
    };
    let block_size = block_size.max(32);
    let blocks = out_t.div_ceil(block_size);
    // Channel block small enough that one task's tile sits in L2
    // (`chan_block * width` floats); time blocks give the rows == 8 layers
    // more tasks than there are threads (at 2048 they ran on 8 of 20 cores).
    let chan_block: usize = 8;
    let chan_blocks = oc.div_ceil(chan_block);
    let jobs = DirectConvJobs {
        source,
        target: out.as_slice_mut().expect("standard layout").as_mut_ptr(),
        weight,
        bias: &w.bias,
        rows,
        in_channels: c,
        t,
        oc,
        k,
        pad,
        dilation,
        out_len: out_t,
        block_size,
        chan_block,
    };
    let tasks: Vec<(usize, usize, usize)> = (0..rows)
        .flat_map(|row| {
            (0..chan_blocks).flat_map(move |cb| (0..blocks).map(move |block| (row, cb, block)))
        })
        .collect();
    tasks.par_iter().for_each(|&(row, cb, block)| {
        let t0 = block * block_size;
        let width = (t0 + block_size).min(out_t) - t0;
        let c0 = cb * chan_block;
        let channels_here = (c0 + chan_block).min(oc) - c0;
        jobs.run(
            row * oc * out_t + c0 * out_t + t0,
            c0,
            channels_here,
            t0,
            width,
        );
    });
}

/// `nn.Conv1d` on a `(b, c, t)` input, lowered to a `(b, c, 1, t)` 2-D convolution.
pub fn conv1d(x: &Array3<f32>, w: &ConvW, stride: usize, pad: usize, dilation: usize) -> Array3<f32> {
    let (b, c, t) = x.dim();
    // The direct path wins exactly where the im2col patch matrix was the
    // bottleneck: a `k >= 3` kernel over *many short rows*, which is the
    // frequency branch's DConv ((b*f, c, 336) with b*f = 8..512). There it is
    // 22x faster in isolation (1.4 ms vs 31 ms). It loses for `k == 1` (already a
    // GEMM) and for the waveform branch's single long row, where one row means
    // few parallel tasks and the per-task accumulators spill out of L1: routing
    // those through it cost 1.25x on the whole segment.
    // ... 22x in isolation, and 1.12x on the whole segment. It was *slower* on
    // the segment until the caching allocator's free list stopped scanning
    // linearly: the per-task accumulators push thousands of small blocks, and
    // every later allocation in the model then paid a scan over all of them —
    // which showed up as slowdowns in stages that never call this code.
    let min_rows: usize = std::env::var("DEMUCS_CONV1D_MINROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let direct = stride == 1
        && w.kernel().len() == 1
        && w.kernel()[0] >= 3
        && b >= min_rows
        && std::env::var("DEMUCS_CONV1D_DIRECT").map_or(true, |v| v != "0");
    if direct {
        let k = w.kernel()[0];
        let out_t = t + 2 * pad - (dilation * (k - 1) + 1) + 1;
        let mut out = Array3::<f32>::zeros((b, w.out_channels(), out_t));
        conv1d_stride1_into(&x.view(), w, &mut out, pad, dilation);
        return out;
    }
    let input = x
        .view()
        .into_shape_with_order((b, c, 1, t))
        .expect("contiguous view");
    let out = conv2d(&input, w, (1, stride), (0, pad), (1, dilation));
    let (_, oc, _, out_t) = out.dim();
    out.into_shape_with_order((b, oc, out_t))
        .expect("reshape back")
}

/// `nn.ConvTranspose2d` with no padding and no output padding; the caller crops.
///
/// One GEMM per `(ky, kx)` tap produces an `(out_channels, positions_in)` tile,
/// which is then accumulated into the output — parallel over the output channels
/// (the batch is 1, so parallelising over it, as this used to, meant one core).
/// With `kw == 1` and `sw == 1`, which is every transposed convolution in this
/// model, the accumulation is a contiguous row add rather than a scatter.
pub fn conv_transpose2d<S: ndarray::Data<Elem = f32> + Sync>(
    x: &ndarray::ArrayBase<S, ndarray::Ix4>,
    w: &ConvW,
    stride: (usize, usize),
) -> Array4<f32> {
    let (b, in_channels, h, width) = x.dim();
    let (kh, kw) = match w.kernel().len() {
        1 => (1, w.kernel()[0]),
        2 => (w.kernel()[0], w.kernel()[1]),
        other => panic!("conv_transpose2d needs a 1-D or 2-D kernel, got {other} axes"),
    };
    let (sh, sw) = stride;
    // `nn.ConvTranspose2d` stores its weight as `(in_channels, out_channels, kh, kw)`,
    // the transpose of what `nn.Conv2d` uses.
    if w.shape[0] != in_channels {
        panic!(
            "conv_transpose2d: input has {in_channels} channels but the weight has {}",
            w.shape[0]
        );
    }
    let oc = w.shape[1];
    let out_h = (h - 1) * sh + kh;
    let out_w = (width - 1) * sw + kw;
    let positions = h * width;

    // One `(oc, ic)` matrix per tap, in tap order.
    let mut taps = vec![vec![0.0f32; oc * in_channels]; kh * kw];
    for ky in 0..kh {
        for kx in 0..kw {
            let tap = &mut taps[ky * kw + kx];
            for ic in 0..in_channels {
                for channel in 0..oc {
                    // The stored weight is `(in_channels, out_channels, kh, kw)`.
                    tap[channel * in_channels + ic] =
                        w.weight[((ic * oc + channel) * kh + ky) * kw + kx];
                }
            }
        }
    }

    let mut out = Array4::<f32>::zeros((b, oc, out_h, out_w));
    for (channel, value) in w.bias.iter().enumerate() {
        out.slice_mut(s![.., channel, .., ..]).fill(*value);
    }

    for bi in 0..b {
        let input = x.slice(s![bi, .., .., ..]);
        let input_2d = input
            .to_shape((in_channels, positions))
            .expect("standard layout")
            .to_owned();
        for (tap_index, weights) in taps.iter().enumerate() {
            let (ky, kx) = (tap_index / kw, tap_index % kw);
            let weight =
                ArrayView2::from_shape((oc, in_channels), weights).expect("tap is contiguous");
            let product = matmul(&weight, &input_2d);
            let values = product.as_slice().expect("standard layout");

            let mut plane = out.slice_mut(s![bi, .., .., ..]);
            plane
                .axis_iter_mut(Axis(0))
                .into_par_iter()
                .enumerate()
                .for_each(|(channel, mut target)| {
                    let source = &values[channel * positions..(channel + 1) * positions];
                    for iy in 0..h {
                        let src = &source[iy * width..(iy + 1) * width];
                        let oy = iy * sh + ky;
                        let ox = kx;
                        // `sw == 1 && kw == 1` (every transposed convolution here)
                        // writes a contiguous run; the general case steps.
                        if sw == 1 {
                            let row = &mut target.slice_mut(s![oy, ox..ox + width]);
                            for (slot, value) in row.into_iter().zip(src.iter()) {
                                *slot += *value;
                            }
                        } else {
                            for (ix, value) in src.iter().enumerate() {
                                target[[oy, ix * sw + ox]] += *value;
                            }
                        }
                    }
                });
        }
    }
    out
}

/// `nn.ConvTranspose1d` on a `(b, c, t)` input.
pub fn conv_transpose1d(x: &Array3<f32>, w: &ConvW, stride: usize) -> Array3<f32> {
    let (b, c, t) = x.dim();
    let input = x
        .view()
        .into_shape_with_order((b, c, 1, t))
        .expect("contiguous view");
    let out = conv_transpose2d(&input, w, (1, stride));
    let (_, oc, _, out_t) = out.dim();
    out.into_shape_with_order((b, oc, out_t))
        .expect("reshape back")
}

/// `nn.GroupNorm(groups, c, eps=1e-5)` on `(b, c, ...)`, normalising each group
/// over every remaining axis with the biased variance.
pub fn group_norm<D: ndarray::Dimension>(
    x: &ndarray::Array<f32, D>,
    groups: usize,
    weight: &[f32],
    bias: &[f32],
) -> Result<ndarray::Array<f32, D>> {
    let shape = x.shape().to_vec();
    if shape.len() < 2 {
        return Err(Error::Shape("group_norm needs at least (batch, channels)".into()));
    }
    let (b, c) = (shape[0], shape[1]);
    if c % groups != 0 || weight.len() != c || bias.len() != c {
        return Err(Error::Shape(format!(
            "group_norm: {c} channels, {groups} groups, {} affine entries",
            weight.len()
        )));
    }
    let per_group = c / groups;
    let inner: usize = shape[2..].iter().product();
    let mut out = x.to_owned();
    let flat = out
        .as_slice_mut()
        .ok_or_else(|| Error::Shape("group_norm needs a standard-layout tensor".into()))?;
    let _ = b;

    // Each row is independent, so hand one (row, group) pair to each task.
    flat.par_chunks_mut(per_group * inner)
        .enumerate()
        .for_each(|(index, slice)| {
            let group = index % groups;
            let _ = group;
            {
            // Two passes: moments (sum and sum of squares together, in f64 so
            // the wave branch's 4 M-element reductions do not lose digits), then
            // the normalise-and-affine. A separate variance pass would be a third
            // sweep over the same memory.
            let count = slice.len() as f64;
            let mut sum = 0.0f64;
            let mut sum_sq = 0.0f64;
            for value in slice.iter() {
                let v = *value as f64;
                sum += v;
                sum_sq += v * v;
            }
            let mean = sum / count;
            let variance = (sum_sq / count - mean * mean).max(0.0);
            let inv = 1.0 / ((variance + 1e-5).sqrt() as f32);
            let mean = mean as f32;
            // One run per channel: hoisting the channel index out of the inner
            // loop removes an integer division per element and lets the loop
            // vectorise.
            for (channel, run) in slice.chunks_mut(inner).enumerate() {
                let scale = inv * weight[group * per_group + channel];
                let shift = bias[group * per_group + channel] - mean * scale;
                for value in run.iter_mut() {
                    *value = *value * scale + shift;
                }
            }
            }
        });
    Ok(out)
}

/// `nn.LayerNorm(dim, eps=1e-5)` over the last axis of a `(rows, dim)` tensor.
pub fn layer_norm_rows(x: &Array2<f32>, weight: &[f32], bias: &[f32]) -> Result<Array2<f32>> {
    let (rows, dim) = x.dim();
    if weight.len() != dim || bias.len() != dim {
        return Err(Error::Shape(format!(
            "layer_norm: dim {dim}, {} affine entries",
            weight.len()
        )));
    }
    let mut out = Array2::<f32>::zeros((rows, dim));
    out.axis_iter_mut(Axis(0))
        .into_par_iter()
        .zip(x.axis_iter(Axis(0)).into_par_iter())
        .for_each(|(mut out_row, in_row)| {
            let mean = in_row.iter().sum::<f32>() / dim as f32;
            let var = in_row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
            let inv = 1.0 / (var + 1e-5).sqrt();
            for i in 0..dim {
                out_row[i] = (in_row[i] - mean) * inv * weight[i] + bias[i];
            }
        });
    Ok(out)
}

/// `F.glu(x, dim=1)` on `(b, 2c, ...)`: first half gated by the sigmoid of the second.
pub fn glu<D: ndarray::Dimension>(x: &ndarray::Array<f32, D>) -> Result<ndarray::Array<f32, D>> {
    let shape = x.shape().to_vec();
    if shape.len() < 2 || shape[1] % 2 != 0 {
        return Err(Error::Shape(format!(
            "GLU needs an even channel count, found {shape:?}"
        )));
    }
    let (b, c) = (shape[0], shape[1]);
    let half = c / 2;
    let inner: usize = shape[2..].iter().product();
    let mut out_shape = shape.clone();
    out_shape[1] = half;
    let mut out = vec![0.0f32; b * half * inner];
    let source = x
        .as_slice()
        .ok_or_else(|| Error::Shape("GLU needs a standard-layout tensor".into()))?;
    // One task per output row (batch, channel): the two halves of the input row
    // are `inner`-sized blocks `half * inner` apart.
    out.par_chunks_mut(inner)
        .enumerate()
        .for_each(|(row, dst)| {
            let bi = row / half;
            let channel = row % half;
            let value_base = (bi * c + channel) * inner;
            let gate_base = value_base + half * inner;
            for (i, slot) in dst.iter_mut().enumerate() {
                let gate = source[gate_base + i];
                *slot = source[value_base + i] * (1.0 / (1.0 + (-gate).exp()));
            }
        });
    ndarray::Array::from_shape_vec(ndarray::IxDyn(&out_shape), out)
        .map_err(|e| Error::Shape(format!("GLU output: {e}")))?
        .into_dimensionality::<D>()
        .map_err(|e| Error::Shape(format!("GLU output rank: {e}")))
}

/// `F.gelu` (exact, erf based) applied elementwise.
pub fn gelu_in_place(values: &mut [f32]) {
    values.par_iter_mut().for_each(|value| *value = gelu_erf(*value));
}

/// `glu(group_norm(x))` in one pass: the DConv's `norm2 -> GLU` pair.
///
/// The two are separate sweeps today — the norm writes the whole `(2C, inner)`
/// plane and the GLU reads it back to write half of it, which for the DConv's
/// 32 calls a segment is 132 MB of traffic a fused form does not pay.
///
/// The statistics are [`group_norm`]'s own (same chunks, same f64 accumulation
/// order) and the two elementwise expressions are its and [`glu`]'s own, so the
/// result is bit-identical to the pair it replaces rather than merely close.
pub fn group_norm_glu<D: ndarray::Dimension>(
    x: &ndarray::Array<f32, D>,
    groups: usize,
    weight: &[f32],
    bias: &[f32],
) -> Result<ndarray::Array<f32, D>> {
    let shape = x.shape().to_vec();
    if shape.len() < 2 {
        return Err(Error::Shape("group_norm_glu needs at least (batch, channels)".into()));
    }
    let (b, c) = (shape[0], shape[1]);
    if groups == 0 || c % groups != 0 || weight.len() != c || bias.len() != c {
        return Err(Error::Shape(format!(
            "group_norm_glu: {c} channels, {groups} groups, {} affine entries",
            weight.len()
        )));
    }
    let per_group = c / groups;
    if per_group % 2 != 0 {
        return Err(Error::Shape(format!(
            "group_norm_glu needs an even per-group channel count, found {per_group}"
        )));
    }
    let half = per_group / 2;
    let inner: usize = shape[2..].iter().product();
    let source = x
        .as_slice()
        .ok_or_else(|| Error::Shape("group_norm_glu needs a standard-layout tensor".into()))?;
    let mut out_shape = shape.clone();
    out_shape[1] = c / 2;
    let mut out = vec![0.0f32; b * half * groups * inner];

    // One task per (row, group) pair, in the order `group_norm` walks them.
    out.par_chunks_mut(half * inner)
        .enumerate()
        .for_each(|(pair, dst)| {
            let group = pair % groups;
            let chunk = &source[pair * per_group * inner..(pair + 1) * per_group * inner];
            let count = chunk.len() as f64;
            let mut sum = 0.0f64;
            let mut sum_sq = 0.0f64;
            for value in chunk.iter() {
                let v = *value as f64;
                sum += v;
                sum_sq += v * v;
            }
            let mean = sum / count;
            let variance = (sum_sq / count - mean * mean).max(0.0);
            let inv = 1.0 / ((variance + 1e-5).sqrt() as f32);
            let mean = mean as f32;
            let scale = |channel: usize| inv * weight[group * per_group + channel];
            let shift = |channel: usize| bias[group * per_group + channel] - mean * scale(channel);
            for k in 0..half {
                let (value_scale, value_shift) = (scale(k), shift(k));
                let (gate_scale, gate_shift) = (scale(k + half), shift(k + half));
                let values = &chunk[k * inner..(k + 1) * inner];
                let gates = &chunk[(k + half) * inner..(k + half + 1) * inner];
                let dst = &mut dst[k * inner..(k + 1) * inner];
                for i in 0..inner {
                    let value = values[i] * value_scale + value_shift;
                    let gate = gates[i] * gate_scale + gate_shift;
                    dst[i] = value * (1.0 / (1.0 + (-gate).exp()));
                }
            }
        });
    ndarray::Array::from_shape_vec(ndarray::IxDyn(&out_shape), out)
        .map_err(|e| Error::Shape(format!("group_norm_glu output: {e}")))?
        .into_dimensionality::<D>()
        .map_err(|e| Error::Shape(format!("group_norm_glu output rank: {e}")))
}

/// `a += b` over two equally shaped tensors.
pub fn add_in_place_nd<S: ndarray::Data<Elem = f32> + Sync, D: ndarray::Dimension>(
    target: &mut ndarray::Array<f32, D>,
    other: &ndarray::ArrayBase<S, D>,
) {
    ndarray::Zip::from(target)
        .and(other)
        .par_for_each(|target, other| *target += *other);
}

/// `a + b` into a fresh tensor.
pub fn add_nd<
    S1: ndarray::Data<Elem = f32> + Sync,
    S2: ndarray::Data<Elem = f32> + Sync,
    D: ndarray::Dimension,
>(
    a: &ndarray::ArrayBase<S1, D>,
    b: &ndarray::ArrayBase<S2, D>,
) -> ndarray::Array<f32, D> {
    let mut out = ndarray::Array::<f32, D>::zeros(a.raw_dim());
    ndarray::Zip::from(&mut out)
        .and(a)
        .and(b)
        .par_for_each(|out, a, b| *out = *a + *b);
    out
}

/// An `(m, n)` matrix with uninitialised contents.
///
/// Every caller here writes all `m * n` elements before reading any — `gemm` runs
/// with `read_dst = false`, which overwrites the output — so skipping the
/// zero-fill removes a full pass over the output. That pass is not free: the
/// attention's score matrix is 231 MB, and `Array2::zeros` memset it on every
/// call.
///
/// SAFETY: the buffer is allocated and its length set without initialising the
/// elements. Reading an element before writing it would be undefined behaviour,
/// so every call site must fill the matrix first (which the GEMM that follows
/// does).
pub fn uninit_matrix(m: usize, n: usize) -> Array2<f32> {
    if std::env::var("DEMUCS_UNINIT").is_ok_and(|v| v == "0") {
        return Array2::<f32>::zeros((m, n));
    }
    let mut buffer = Vec::<f32>::with_capacity(m * n);
    unsafe {
        buffer.set_len(m * n);
        Array2::from_shape_vec_unchecked((m, n), buffer)
    }
}

/// Which GEMM implementation to use. `ndarray` is kept so the switch can be
/// A/B'd with one environment variable on the same binary.
pub fn use_ndarray_gemm() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| {
        std::env::var("DEMUCS_GEMM").is_ok_and(|v| v.eq_ignore_ascii_case("ndarray"))
    })
}

/// `dst = alpha * (a @ b) + beta * dst`, with both operands taken as (row, col)
/// strided views.
///
/// `b_transposed` says `b` is stored as `(n, k)`, which the attention needs: the
/// `gemm` crate reads it through its strides, so no copy is made.
pub fn gemm_strided(
    dst: &mut Array2<f32>,
    a: &ndarray::ArrayView2<f32>,
    b: &ndarray::ArrayView2<f32>,
    b_transposed: bool,
) {
    let (m, k) = a.dim();
    let n = if b_transposed { b.dim().0 } else { b.dim().1 };
    debug_assert_eq!(dst.dim(), (m, n));
    let (b_rs, b_cs) = if b_transposed {
        (1isize, b.strides()[0])
    } else {
        (b.strides()[0], b.strides()[1])
    };
    let (a_rs, a_cs) = (a.strides()[0], a.strides()[1]);
    unsafe {
        gemm::gemm(
            m,
            n,
            k,
            dst.as_mut_ptr(),
            1,
            n as isize,
            false,
            a.as_ptr(),
            a_cs,
            a_rs,
            b.as_ptr(),
            b_cs,
            b_rs,
            // `gemm`'s first scalar multiplies `dst` and the second the product
            // (`dst := alpha*dst + beta*lhs*rhs`), the opposite of the reading its
            // parameter names suggest. Swapping them makes every GEMM a no-op
            // that returns whatever the output buffer held.
            0.0,
            1.0,
            false,
            false,
            false,
            gemm::Parallelism::Rayon(0),
        );
    }
}

/// `(m, k) @ (k, n) + bias`, with the bias folded into the output write-back.
///
/// The standalone bias sweep this replaces was a full extra pass over the
/// output: on the 21 MB DConv 1x1 it cost more than the GEMM itself.
pub fn matmul_with_row_bias<SA, SB>(
    a: &ndarray::ArrayBase<SA, ndarray::Ix2>,
    b: &ndarray::ArrayBase<SB, ndarray::Ix2>,
    bias: &[f32],
) -> Array2<f32>
where
    SA: ndarray::Data<Elem = f32> + Sync,
    SB: ndarray::Data<Elem = f32> + Sync,
{
    // One GEMM, then the bias in the same write-back pass the bias sweep used
    // to be — except the GEMM already had to write every output element, so the
    // bias rides along for free instead of costing a second full sweep.
    let mut out = matmul(a, b);
    let (m, n) = out.dim();
    debug_assert_eq!(bias.len(), m);
    let dst = out.as_slice_mut().expect("standard layout");
    dst.par_chunks_mut(n)
        .enumerate()
        .for_each(|(row, out_row)| {
            let shift = bias[row];
            for slot in out_row.iter_mut() {
                *slot += shift;
            }
        });
    out
}

/// `(rows, in) @ (in, out) + bias`, the shared body of [`Linear::forward`]
/// and [`linear_3d`].
///
/// GEMM plus bias as one call: the bias used to be a second full sweep over the
/// same output (a `+=` per row after the multiply), so every linear in the model
/// paid twice for the output write bandwidth.
pub fn linear_forward<S: ndarray::Data<Elem = f32> + Sync>(
    x: &ArrayBase<S, ndarray::Ix2>,
    weight_t: &Array2<f32>,
    bias: Option<&Array1<f32>>,
) -> Array2<f32> {
    let (_rows, k) = x.dim();
    debug_assert_eq!(k, weight_t.dim().0);
    let n = weight_t.dim().1;
    let gemm_out = matmul(x, weight_t);
    match bias {
        None => gemm_out,
        Some(bias) => {
            let mut out = gemm_out;
            let bias = bias.as_slice().expect("contiguous bias");
            out.as_slice_mut()
                .expect("standard layout")
                .par_chunks_mut(n)
                .for_each(|row| {
                    for (slot, shift) in row.iter_mut().zip(bias.iter()) {
                        *slot += *shift;
                    }
                });
            out
        }
    }
}

/// `(m, k) x (k, n)`.
///
/// Uses the `gemm` crate's AVX microkernels, which the two prior ports in this
/// repo also settled on; `ndarray`'s own `general_mat_mul` is about 3x slower on
/// these shapes and the transformer is a third of the segment. The old path
/// stays reachable through `DEMUCS_GEMM=ndarray` for A/B.
pub fn matmul<SA: ndarray::Data<Elem = f32> + Sync, SB: ndarray::Data<Elem = f32> + Sync>(
    a: &ndarray::ArrayBase<SA, ndarray::Ix2>,
    b: &ndarray::ArrayBase<SB, ndarray::Ix2>,
) -> Array2<f32> {
    let (m, k) = a.dim();
    debug_assert_eq!(k, b.dim().0);
    let n = b.dim().1;
    if use_ndarray_gemm() {
        let mut out = Array2::<f32>::zeros((m, n));
        ndarray_matmul_into(&mut out, a, b);
        return out;
    }
    let mut out = uninit_matrix(m, n);
    let a_view = a.view();
    let b_view = b.view();
    gemm_strided(&mut out, &a_view, &b_view, false);
    out
}

/// The previous implementation, kept for the A/B switch: split the rows when
/// there are enough of them, otherwise the columns.
fn ndarray_matmul_into<SA: ndarray::Data<Elem = f32> + Sync, SB: ndarray::Data<Elem = f32> + Sync>(
    out: &mut Array2<f32>,
    a: &ndarray::ArrayBase<SA, ndarray::Ix2>,
    b: &ndarray::ArrayBase<SB, ndarray::Ix2>,
) {
    let (m, _k) = a.dim();
    let n = b.dim().1;
    let threads = rayon::current_num_threads().max(1);
    if m >= 8 * threads {
        let block = m.div_ceil(threads).max(32);
        out.axis_chunks_iter_mut(Axis(0), block)
            .zip(a.axis_chunks_iter(Axis(0), block))
            .collect::<Vec<_>>()
            .into_par_iter()
            .for_each(|(mut out_chunk, a_chunk)| {
                ndarray::linalg::general_mat_mul(1.0, &a_chunk, b, 0.0, &mut out_chunk);
            });
    } else {
        let block = n.div_ceil(threads).max(64);
        out.axis_chunks_iter_mut(Axis(1), block)
            .zip(b.axis_chunks_iter(Axis(1), block))
            .collect::<Vec<_>>()
            .into_par_iter()
            .for_each(|(mut out_chunk, b_chunk)| {
                ndarray::linalg::general_mat_mul(1.0, a, &b_chunk, 0.0, &mut out_chunk);
            });
    }
}

/// `(rows, k) x (k, n)` with the second operand transposed, i.e. `A @ B^T`.
pub fn matmul_bt<SA: ndarray::Data<Elem = f32> + Sync, SB: ndarray::Data<Elem = f32> + Sync>(
    a: &ndarray::ArrayBase<SA, ndarray::Ix2>,
    b: &ndarray::ArrayBase<SB, ndarray::Ix2>,
) -> Array2<f32> {
    let (m, k) = a.dim();
    debug_assert_eq!(k, b.dim().1);
    let n = b.dim().0;
    if use_ndarray_gemm() {
        let mut out = Array2::<f32>::zeros((m, n));
        let bt = b.t();
        let threads = rayon::current_num_threads().max(1);
        let block = (m / threads).max(1).max(64);
        out.axis_chunks_iter_mut(Axis(0), block)
            .zip(a.axis_chunks_iter(Axis(0), block))
            .collect::<Vec<_>>()
            .into_par_iter()
            .for_each(|(mut out_chunk, a_chunk)| {
                ndarray::linalg::general_mat_mul(1.0, &a_chunk, &bt, 0.0, &mut out_chunk);
            });
        return out;
    }
    let mut out = uninit_matrix(m, n);
    // The `gemm` crate reads a transposed operand through strides, but its
    // packed path only triggers for the non-transposed case; transposing the
    // smaller operand once is cheaper than the strided reads (measured in
    // `demucs kernels`: 24.9 ms -> 11.6 ms for the attention's score matrix).
    if std::env::var("DEMUCS_SCORES_TRANSPOSE").is_ok_and(|v| v == "0") {
        gemm_strided(&mut out, &a.view(), &b.view(), true);
        return out;
    }
    let transposed = transpose_owned_2d(b);
    gemm_strided(&mut out, &a.view(), &transposed.view(), false);
    out
}

/// A parallel 2-D transpose, used to turn a `(n, k)` operand into `(k, n)` so the
/// GEMM can take its packed path.
pub fn transpose_owned_2d<S: ndarray::Data<Elem = f32> + Sync>(
    x: &ndarray::ArrayBase<S, ndarray::Ix2>,
) -> Array2<f32> {
    let (rows, cols) = x.dim();
    let mut out = Array2::<f32>::zeros((cols, rows));
    let source = x.view();
    {
        let source = source.as_standard_layout();
        let source = source.as_slice().expect("standard layout");
        out.as_slice_mut()
            .expect("standard layout")
            .par_chunks_mut(rows)
            .enumerate()
            .for_each(|(col, dst)| {
                for (row, slot) in dst.iter_mut().enumerate() {
                    *slot = source[row * cols + col];
                }
            });
    }
    out
}

/// Plain `Linear` on a `(b, t, c)` input, keeping the batch axes.
pub fn linear_3d(layer: &Linear, x: &ndarray::Array3<f32>) -> Array3<f32> {
    let (b, t, c) = x.dim();
    let flat = x
        .view()
        .into_shape_with_order((b * t, c))
        .expect("contiguous view");
    let out = layer.forward(&flat);
    out.into_shape_with_order((b, t, layer.out_features()))
        .expect("reshape back")
}

/// `nn.MultiheadAttention` for one (batch, head) pair: `softmax(q k^T / sqrt(d)) v`.
///
/// `q` is `(n_q, d_head)`, `k` and `v` are `(n_k, d_head)`; the returned scores
/// matrix (`n_q, n_k`) is written into `scores_out` so callers can trace it.
pub fn attention_head(
    q: &Array2<f32>,
    k: &Array2<f32>,
    v: &Array2<f32>,
    scale: f32,
    scores_out: Option<&mut Array2<f32>>,
) -> Array2<f32> {
    // NOTE: a fused softmax+AV was tried and measured *slower* (1469 -> 1758
    // ms/segment): the hand-written AV loop runs at scalar speed while the GEMM
    // it replaces hits 1.1 TFLOP/s. Keep the matrices whole.
    let (n_q, _d_head) = q.dim();
    let n_k = k.dim().0;
    let scaled = scaled_rows(q, scale);
    let mut scores = matmul_bt(&scaled, k);
    softmax_rows(&mut scores);
    let _ = (n_q, n_k);
    let out = matmul(&scores, v);
    if let Some(target) = scores_out {
        target.assign(&scores);
    }
    out
}

/// `q * scale` into a fresh matrix, in parallel.
fn scaled_rows(q: &Array2<f32>, scale: f32) -> Array2<f32> {
    let mut scaled = q.to_owned();
    scaled
        .as_slice_mut()
        .expect("standard layout")
        .par_iter_mut()
        .for_each(|value| *value *= scale);
    scaled
}

/// In-place softmax over every row of a `(rows, cols)` matrix.
pub fn softmax_rows(x: &mut Array2<f32>) {
    let rows: Vec<&mut [f32]> = x
        .axis_iter_mut(Axis(0))
        .map(|row| row.into_slice().expect("contiguous row"))
        .collect();
    rows.into_par_iter().for_each(|row| softmax_in_place(row));
}

/// `torch.arange(n)` as `f32`.
pub fn arange(n: usize) -> Vec<f32> {
    (0..n).map(|i| i as f32).collect()
}

/// Applies a per-channel scale to a `(b, c, ...)` tensor: `LayerScale`.
pub fn layer_scale_in_place<S: ndarray::DataMut<Elem = f32>, D: ndarray::Dimension>(
    x: &mut ndarray::ArrayBase<S, D>,
    gamma: &[f32],
    channels_axis: usize,
) {
    for (index, mut lane) in x.lanes_mut(Axis(channels_axis)).into_iter().enumerate() {
        let g = gamma[index];
        lane.mapv_inplace(|v| v * g);
    }
}

/// `F.pad` on the last axis with zeros.
pub fn pad_last_axis(x: &Array2<f32>, left: usize, right: usize) -> Array2<f32> {
    let (rows, width) = x.dim();
    let mut out = Array2::<f32>::zeros((rows, left + width + right));
    out.slice_mut(s![.., left..left + width]).assign(x);
    out
}

/// `F.pad` with zeros on the last axis of a 3-D tensor.
pub fn pad_last_axis_3d(x: &Array3<f32>, left: usize, right: usize) -> Array3<f32> {
    let (a, b, width) = x.dim();
    let mut out = Array3::<f32>::zeros((a, b, left + width + right));
    out.slice_mut(s![.., .., left..left + width]).assign(x);
    out
}

/// `torch.std` with `unbiased=True` (the sample standard deviation) over a slice.
pub fn std_unbiased(values: &[f32]) -> f32 {
    let n = values.len();
    if n < 2 {
        return f32::NAN;
    }
    let mean = values.iter().map(|v| *v as f64).sum::<f64>() / n as f64;
    let var = values
        .iter()
        .map(|v| {
            let d = *v as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / (n - 1) as f64;
    var.sqrt() as f32
}

/// `(b, c, f, t) -> (b * f, c, t)`, the reference's
/// `y.permute(0, 2, 1, 3).reshape(-1, c, t)`.
///
/// `ndarray`'s `permuted_axes(...).as_standard_layout()` does the same thing but
/// single-threaded, and this transpose runs twice per DConv — 32 times per
/// segment on tensors of up to 8 M elements.
pub fn channels_to_leading_freq_into<S: ndarray::Data<Elem = f32> + Sync>(
    x: &ndarray::ArrayBase<S, ndarray::Ix4>,
    out: &mut Array3<f32>,
) -> Result<()> {
    let (b, c, f, t) = x.dim();
    if out.dim() != (b * f, c, t) {
        return Err(Error::Shape(format!(
            "permute target is {:?}, expected {:?}",
            out.dim(),
            (b * f, c, t)
        )));
    }
    let source = x
        .as_slice()
        .ok_or_else(|| Error::Shape("permute needs a standard-layout input".into()))?;
    let target = out
        .as_slice_mut()
        .ok_or_else(|| Error::Shape("permute needs a standard-layout output".into()))?;
    target
        .par_chunks_mut(c * t)
        .enumerate()
        .for_each(|(row, dst)| {
            // `row` runs over `b * f`, fastest axis last.
            let bi = row / f;
            let fi = row % f;
            for ci in 0..c {
                let src = &source[((bi * c + ci) * f + fi) * t..((bi * c + ci) * f + fi) * t + t];
                dst[ci * t..(ci + 1) * t].copy_from_slice(src);
            }
        });
    Ok(())
}

/// Inverse of [`channels_to_leading_freq_into`]:
/// `(b * f, c, t) -> (b, c, f, t)`.
pub fn leading_freq_to_channels_into<S: ndarray::Data<Elem = f32> + Sync>(
    x: &ndarray::ArrayBase<S, ndarray::Ix3>,
    b: usize,
    c: usize,
    f: usize,
    out: &mut Array4<f32>,
) -> Result<()> {
    let t = x.dim().2;
    if x.dim() != (b * f, c, t) || out.dim() != (b, c, f, t) {
        return Err(Error::Shape("permute-back received the wrong shapes".into()));
    }
    let source = x
        .as_slice()
        .ok_or_else(|| Error::Shape("permute needs a standard-layout input".into()))?;
    let target = out
        .as_slice_mut()
        .ok_or_else(|| Error::Shape("permute needs a standard-layout output".into()))?;
    target
        .par_chunks_mut(f * t)
        .enumerate()
        .for_each(|(bc, dst)| {
            let bi = bc / c;
            let ci = bc % c;
            for fi in 0..f {
                let src = &source[((bi * f + fi) * c + ci) * t..((bi * f + fi) * c + ci) * t + t];
                dst[fi * t..(fi + 1) * t].copy_from_slice(src);
            }
        });
    Ok(())
}

/// `x.mean()` over a slice, in `f64` to match the reference accumulation order
/// closely enough that the difference stays far below the fp32 epsilon.
pub fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        return f32::NAN;
    }
    (values.iter().map(|v| *v as f64).sum::<f64>() / values.len() as f64) as f32
}

/// Sentinel used by the tests to build reference activations.
pub fn view3<'a>(x: &'a Array4<f32>, batch: usize) -> ndarray::ArrayView3<'a, f32> {
    x.slice(s![batch, .., .., ..])
}

/// Convenience: a `(rows, cols)` view over a flat slice.
pub fn as_matrix(values: &[f32], rows: usize, cols: usize) -> ArrayView2<'_, f32> {
    ArrayView2::from_shape((rows, cols), values).expect("exact size")
}

/// Convenience: a `(len,)` view over a flat slice.
pub fn as_vector(values: &[f32]) -> ArrayView1<'_, f32> {
    ArrayView1::from_shape(values.len(), values).expect("exact size")
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn conv(weight: Vec<f32>, bias: Vec<f32>, shape: Vec<usize>) -> ConvW {
        ConvW {
            weight,
            bias,
            shape,
        }
    }

    #[test]
    fn conv1d_matches_a_hand_written_correlation() {
        // One input channel, kernel [1, 2], stride 1, no padding.
        let x = array![[[1.0f32, 2.0, 3.0, 4.0]]];
        let w = conv(vec![1.0, 2.0], vec![0.5], vec![1, 1, 2]);
        let out = conv1d(&x, &w, 1, 0, 1);
        assert_eq!(out.dim(), (1, 1, 3));
        assert!((out[[0, 0, 0]] - (1.0 * 1.0 + 2.0 * 2.0 + 0.5)).abs() < 1e-6);
        assert!((out[[0, 0, 1]] - (1.0 * 2.0 + 2.0 * 3.0 + 0.5)).abs() < 1e-6);
        assert!((out[[0, 0, 2]] - (1.0 * 3.0 + 2.0 * 4.0 + 0.5)).abs() < 1e-6);
    }

    #[test]
    fn conv1d_pads_with_zeros_and_respects_stride() {
        let x = array![[[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]]];
        let w = conv(vec![1.0, 1.0, 1.0], vec![0.0], vec![1, 1, 3]);
        // pad=1, stride=2 => taps centred at -1, 1, 3
        let out = conv1d(&x, &w, 2, 1, 1);
        assert_eq!(out.dim(), (1, 1, 3));
        assert!((out[[0, 0, 0]] - (0.0 + 1.0 + 2.0)).abs() < 1e-6);
        assert!((out[[0, 0, 1]] - (2.0 + 3.0 + 4.0)).abs() < 1e-6);
        assert!((out[[0, 0, 2]] - (4.0 + 5.0 + 6.0)).abs() < 1e-6);
    }

    #[test]
    fn conv1d_dilation_spreads_the_taps() {
        let x = array![[[1.0f32, 2.0, 3.0, 4.0, 5.0]]];
        let w = conv(vec![1.0, 10.0], vec![0.0], vec![1, 1, 2]);
        // Effective kernel 3 with pad 2 keeps the length; out[i] = x[i-2] + 10 x[i].
        let out = conv1d(&x, &w, 1, 2, 2);
        assert_eq!(out.dim(), (1, 1, 7));
        assert!((out[[0, 0, 0]] - (0.0 + 10.0 * 1.0)).abs() < 1e-6);
        assert!((out[[0, 0, 2]] - (1.0 + 10.0 * 3.0)).abs() < 1e-6);
        assert!((out[[0, 0, 6]] - (5.0 + 10.0 * 0.0)).abs() < 1e-6);
    }

    #[test]
    fn conv_transpose1d_scatters_taps() {
        // kernel 2, stride 2: out[i*2 + k] = sum_c x[c, i] * w[c, k]
        let x = array![[[1.0f32, 2.0]]];
        let w = conv(vec![1.0, 10.0], vec![0.0], vec![1, 1, 2]);
        let out = conv_transpose1d(&x, &w, 2);
        assert_eq!(out.dim(), (1, 1, 4));
        assert!((out[[0, 0, 0]] - 1.0).abs() < 1e-6);
        assert!((out[[0, 0, 1]] - 10.0).abs() < 1e-6);
        assert!((out[[0, 0, 2]] - 2.0).abs() < 1e-6);
        assert!((out[[0, 0, 3]] - 20.0).abs() < 1e-6);
    }

    #[test]
    fn conv2d_handles_a_two_dimensional_kernel() {
        // 2x2 kernel over a 3x3 input, no padding, stride 1 -> 2x2 output.
        let x = Array4::from_shape_vec(
            (1, 1, 3, 3),
            (1..=9).map(|v| v as f32).collect(),
        )
        .unwrap();
        let w = conv(vec![1.0, 0.0, 0.0, 0.0], vec![0.0], vec![1, 1, 2, 2]);
        let out = conv2d(&x, &w, (1, 1), (0, 0), (1, 1));
        assert_eq!(out.dim(), (1, 1, 2, 2));
        assert_eq!(out[[0, 0, 0, 0]], 1.0);
        assert_eq!(out[[0, 0, 0, 1]], 2.0);
        assert_eq!(out[[0, 0, 1, 0]], 4.0);
    }

    #[test]
    fn group_norm_normalises_each_group_over_every_other_axis() {
        // Two channels, one group: mean/var come from both channels and both steps.
        let x = Array3::from_shape_vec((1, 2, 2), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let out = group_norm(&x, 1, &[1.0, 1.0], &[0.0, 0.0]).unwrap();
        let mean = 2.5f32;
        let var = ((1.0 - 2.5f32).powi(2)
            + (2.0 - 2.5f32).powi(2)
            + (3.0 - 2.5f32).powi(2)
            + (4.0 - 2.5f32).powi(2))
            / 4.0;
        let inv = 1.0 / (var + 1e-5).sqrt();
        assert!((out[[0, 0, 0]] - (1.0 - mean) * inv).abs() < 1e-5);
        assert!((out[[0, 1, 1]] - (4.0 - mean) * inv).abs() < 1e-5);
    }

    #[test]
    fn group_norm_applies_the_affine_per_channel() {
        let x = Array3::from_shape_vec((1, 2, 2), vec![0.0f32, 1.0, 0.0, 1.0]).unwrap();
        let out = group_norm(&x, 1, &[2.0, 3.0], &[1.0, -1.0]).unwrap();
        // Both channels share the statistics, so only the affine differs.
        let inv = 1.0 / (0.25f32 + 1e-5).sqrt();
        let normalised = (0.0 - 0.5) * inv;
        assert!((out[[0, 0, 0]] - (normalised * 2.0 + 1.0)).abs() < 1e-5);
        assert!((out[[0, 1, 0]] - (normalised * 3.0 - 1.0)).abs() < 1e-5);
        assert!((out[[0, 0, 1]] - ((-normalised) * 2.0 + 1.0)).abs() < 1e-5);
    }

    #[test]
    fn group_norm_splits_channels_into_independent_groups() {
        // Two groups over four channels: channels {0,1} and {2,3} are separate.
        let x = Array3::from_shape_vec(
            (1, 4, 1),
            vec![1.0f32, 2.0, 100.0, 200.0],
        )
        .unwrap();
        let out = group_norm(&x, 2, &[1.0, 1.0, 1.0, 1.0], &[0.0, 0.0, 0.0, 0.0]).unwrap();
        // Group 1 has mean 1.5 and the pair is symmetric, so +-1/sqrt(0.25+eps).
        assert!((out[[0, 0, 0]] + out[[0, 1, 0]]).abs() < 1e-5);
        assert!((out[[0, 2, 0]] + out[[0, 3, 0]]).abs() < 1e-5);
        // Group 2 spans a much larger range but is normalised the same way.
        assert!((out[[0, 3, 0]] - out[[0, 1, 0]]).abs() < 1e-4);
    }

    #[test]
    fn layer_norm_matches_a_hand_computed_row() {
        let x = array![[1.0f32, 2.0, 3.0, 4.0]];
        let out = layer_norm_rows(&x, &[1.0; 4], &[0.0; 4]).unwrap();
        let mean = 2.5f32;
        let var = 1.25f32;
        let inv = 1.0 / (var + 1e-5).sqrt();
        assert!((out[[0, 0]] - (1.0 - mean) * inv).abs() < 1e-5);
        assert!((out[[0, 3]] - (4.0 - mean) * inv).abs() < 1e-5);
    }

    #[test]
    fn glu_gates_the_first_half_with_the_second() {
        let x = Array3::from_shape_vec((1, 4, 2), vec![1.0f32, 1.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0])
            .unwrap();
        let out = glu(&x).unwrap();
        assert_eq!(out.dim(), (1, 2, 2));
        assert!((out[[0, 0, 0]] - 0.5).abs() < 1e-6);
        assert!((out[[0, 1, 1]] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn matmul_bt_matches_an_explicit_transpose_product() {
        let a = array![[1.0f32, 2.0], [3.0, 4.0]];
        let b = array![[1.0f32, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let out = matmul_bt(&a, &b);
        assert_eq!(out.dim(), (2, 3));
        assert!((out[[0, 0]] - 1.0).abs() < 1e-6);
        assert!((out[[1, 2]] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn attention_head_is_a_scaled_softmax_times_v() {
        let q = array![[1.0f32, 0.0]];
        let k = array![[1.0f32, 0.0], [0.0, 1.0]];
        let v = array![[2.0f32, 0.0], [0.0, 8.0]];
        let out = attention_head(&q, &k, &v, 1.0, None);
        // scores = [1, 0] -> softmax = [0.731, 0.269]
        let a = 1.0f32.exp() / (1.0f32.exp() + 1.0);
        assert!((out[[0, 0]] - a * 2.0).abs() < 1e-5);
        assert!((out[[0, 1]] - (1.0 - a) * 8.0).abs() < 1e-5);
    }

    #[test]
    fn std_unbiased_matches_numpy_style_sample_deviation() {
        let values = [1.0f32, 2.0, 3.0, 4.0];
        // population std is sqrt(1.25); sample std is sqrt(5/3)
        assert!((std_unbiased(&values) - (5.0f32 / 3.0).sqrt()).abs() < 1e-6);
    }
}
