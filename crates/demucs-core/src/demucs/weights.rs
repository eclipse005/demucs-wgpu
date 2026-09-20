//! Typed weights for `HTDemucs`, loaded from either container.
//!
//! The layouts are kept exactly as PyTorch stores them (row-major, and for
//! convolutions `(out_channels, in_channels, k...)`), because the host kernels
//! and the device kernels both want to be able to reinterpret a slice without a
//! repacking step. The one exception is the transformer's `in_proj_weight`, which
//! [`TransformerLayerW`] splits into three `(dim, dim)` matrices at load time.

use std::path::{Path, PathBuf};

use ndarray::{Array1, Array2, ArrayD};

use crate::ckpt::Checkpoint;
use crate::demucs::config::{HtdemucsArch, HtdemucsConfig};
use crate::error::{Error, Result};
use crate::ops::Linear;
use crate::safetensors::SafeTensors;

/// Weights loaded from a `.th` archive or a `safetensors` file.
#[derive(Debug)]
pub struct LoadedWeights {
    pub checkpoint: Checkpoint,
    pub config: HtdemucsConfig,
    /// `safetensors` metadata, empty for `.th` archives.
    pub metadata: std::collections::BTreeMap<String, String>,
}

/// [`load_weights`] from bytes instead of a path. The caller says which format
/// the bytes are: torch zip (`.th`) or `safetensors`.
pub fn load_weights_from_bytes(bytes: &[u8], torch_zip: bool) -> Result<LoadedWeights> {
    if torch_zip {
        let checkpoint = Checkpoint::from_bytes(bytes)?;
        Ok(LoadedWeights {
            checkpoint,
            config: HtdemucsConfig::default(),
            metadata: Default::default(),
        })
    } else {
        let parsed = SafeTensors::parse(bytes)?;
        let config = match parsed.metadata.get("kwargs") {
            Some(kwargs) => HtdemucsConfig::from_kwargs_json(kwargs)?,
            None => HtdemucsConfig::default(),
        };
        let checkpoint = Checkpoint::from_tensors(parsed.tensors.clone(), PathBuf::from("<bytes>"));
        Ok(LoadedWeights {
            checkpoint,
            config,
            metadata: parsed.metadata.clone(),
        })
    }
}

/// Reads a checkpoint by extension: `.th`/`.pth` are torch zips, everything else
/// is assumed to be `safetensors`. The config is taken from the file when it
/// carries one and from the built-in defaults otherwise (both supported
/// checkpoints ship the same architecture).
pub fn load_weights(path: impl AsRef<Path>) -> Result<LoadedWeights> {
    let path = path.as_ref();
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if extension == "th" || extension == "pth" || extension == "pt" {
        let checkpoint = Checkpoint::load(path)?;
        // The `.th` package stores `kwargs` as pickled Python objects, which the
        // pickle VM does not reconstruct; the two checkpoints in scope use the
        // defaults verbatim.
        Ok(LoadedWeights {
            checkpoint,
            config: HtdemucsConfig::default(),
            metadata: Default::default(),
        })
    } else {
        let parsed = SafeTensors::load(path)?;
        let config = match parsed.metadata.get("kwargs") {
            Some(kwargs) => HtdemucsConfig::from_kwargs_json(kwargs)?,
            None => HtdemucsConfig::default(),
        };
        let checkpoint =
            Checkpoint::from_tensors(parsed.tensors.clone(), path.to_path_buf());
        Ok(LoadedWeights {
            checkpoint,
            config,
            metadata: parsed.metadata,
        })
    }
}

/// A convolution's raw weights, in PyTorch layout.
#[derive(Debug, Clone)]
pub struct ConvW {
    /// `(out_channels, in_channels, k...)`, flattened row-major.
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    /// The full PyTorch shape, e.g. `[48, 4, 8, 1]`.
    pub shape: Vec<usize>,
}

impl ConvW {
    pub fn out_channels(&self) -> usize {
        self.shape[0]
    }

    pub fn in_channels(&self) -> usize {
        self.shape[1]
    }

    /// Kernel extents after the channel pair, e.g. `[8, 1]` or `[3]`.
    pub fn kernel(&self) -> &[usize] {
        &self.shape[2..]
    }
}

/// One `DConv` residual branch: `depth` repetitions of
/// `Conv1d -> GroupNorm(1) -> GELU -> Conv1d(1x1) -> GroupNorm(1) -> GLU -> LayerScale`.
#[derive(Debug, Clone)]
pub struct DConvLayerW {
    pub conv1: ConvW,
    pub norm1_weight: Vec<f32>,
    pub norm1_bias: Vec<f32>,
    pub conv2: ConvW,
    pub norm2_weight: Vec<f32>,
    pub norm2_bias: Vec<f32>,
    pub gamma: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct DConvW {
    pub layers: Vec<DConvLayerW>,
    /// `channels` of the branch, i.e. the width the LayerScale multiplies.
    pub channels: usize,
}

#[derive(Debug, Clone)]
pub struct EncLayerW {
    pub conv: ConvW,
    pub rewrite: ConvW,
    pub dconv: DConvW,
}

#[derive(Debug, Clone)]
pub struct DecLayerW {
    pub conv_tr: ConvW,
    pub rewrite: ConvW,
    pub dconv: DConvW,
}

/// One transformer encoder layer, classic or cross attention.
#[derive(Debug, Clone)]
pub struct TransformerLayerW {
    pub is_cross: bool,
    /// Query/key/value projections, split out of `in_proj_weight`.
    pub q_weight: Vec<f32>,
    pub k_weight: Vec<f32>,
    pub v_weight: Vec<f32>,
    pub q_bias: Vec<f32>,
    pub k_bias: Vec<f32>,
    pub v_bias: Vec<f32>,
    pub out_proj: Linear,
    pub linear1: Linear,
    pub linear2: Linear,
    pub norm1: LayerNormW,
    pub norm2: LayerNormW,
    /// Only present on cross layers (it normalises the key branch).
    pub norm3: Option<LayerNormW>,
    /// `MyGroupNorm(1, dim)` applied on `(B, T, C)`.
    pub norm_out: GroupNormW,
    pub gamma1: Vec<f32>,
    pub gamma2: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct LayerNormW {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct GroupNormW {
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub groups: usize,
}

#[derive(Debug, Clone)]
pub struct CrossTransformerW {
    pub norm_in: LayerNormW,
    pub norm_in_t: LayerNormW,
    /// The spectrogram branch's layers, then the waveform branch's.
    pub layers: Vec<TransformerLayerW>,
    pub layers_t: Vec<TransformerLayerW>,
}

#[derive(Debug, Clone)]
pub struct HtdemucsWeights {
    pub encoder: Vec<EncLayerW>,
    pub tencoder: Vec<EncLayerW>,
    pub decoder: Vec<DecLayerW>,
    pub tdecoder: Vec<DecLayerW>,
    /// `ScaledEmbedding` weights already multiplied by `emb_scale`, as
    /// `ScaledEmbedding.forward` does: `(freqs, channels)`.
    pub freq_emb: Array2<f32>,
    pub channel_upsampler: Linear,
    pub channel_upsampler_t: Linear,
    pub channel_downsampler: Linear,
    pub channel_downsampler_t: Linear,
    pub transformer: CrossTransformerW,
}

fn shaped(checkpoint: &Checkpoint, name: &str, expected: &[usize]) -> Result<ArrayD<f32>> {
    let tensor = checkpoint.get_required(name)?;
    let found = tensor.shape().to_vec();
    if found != expected {
        return Err(Error::WeightShape {
            name: name.to_string(),
            found,
            expected: expected.to_vec(),
        });
    }
    Ok(tensor.to_owned())
}

fn conv(checkpoint: &Checkpoint, prefix: &str) -> Result<ConvW> {
    let weight = checkpoint.get_required(&format!("{prefix}.weight"))?;
    let shape = weight.shape().to_vec();
    if shape.len() < 3 {
        return Err(Error::WeightShape {
            name: format!("{prefix}.weight"),
            found: shape.clone(),
            expected: vec![0, 0, 0],
        });
    }
    let bias = checkpoint
        .get_required(&format!("{prefix}.bias"))?
        .iter()
        .copied()
        .collect();
    Ok(ConvW {
        weight: weight.iter().copied().collect(),
        bias,
        shape,
    })
}

fn linear(checkpoint: &Checkpoint, prefix: &str) -> Result<Linear> {
    let weight = checkpoint.get_required(&format!("{prefix}.weight"))?;
    let shape = weight.shape();
    if shape.len() != 2 {
        return Err(Error::WeightShape {
            name: format!("{prefix}.weight"),
            found: shape.to_vec(),
            expected: vec![0, 0],
        });
    }
    let bias = checkpoint
        .get_required(&format!("{prefix}.bias"))?
        .iter()
        .copied()
        .collect::<Vec<f32>>();
    let view = weight
        .view()
        .into_dimensionality::<ndarray::Ix2>()
        .expect("2-D weight");
    Ok(Linear::new(view, Some(Array1::from(bias).view())))
}

/// A 1x1 convolution is a linear layer with an extra trailing axis of size one.
fn conv_1x1_as_linear(checkpoint: &Checkpoint, prefix: &str) -> Result<Linear> {
    let weight = checkpoint.get_required(&format!("{prefix}.weight"))?;
    let shape = weight.shape().to_vec();
    if shape.len() != 3 || shape[2] != 1 {
        return Err(Error::WeightShape {
            name: format!("{prefix}.weight"),
            found: shape,
            expected: vec![0, 0, 1],
        });
    }
    let flat: Vec<f32> = weight.iter().copied().collect();
    let view = ndarray::ArrayView2::from_shape((shape[0], shape[1]), &flat)
        .expect("flattened 1x1 weight");
    let bias = checkpoint
        .get_required(&format!("{prefix}.bias"))?
        .iter()
        .copied()
        .collect::<Vec<f32>>();
    Ok(Linear::new(view, Some(Array1::from(bias).view())))
}

fn layer_norm(checkpoint: &Checkpoint, prefix: &str) -> Result<LayerNormW> {
    Ok(LayerNormW {
        weight: checkpoint
            .get_required(&format!("{prefix}.weight"))?
            .iter()
            .copied()
            .collect(),
        bias: checkpoint
            .get_required(&format!("{prefix}.bias"))?
            .iter()
            .copied()
            .collect(),
    })
}

fn group_norm(checkpoint: &Checkpoint, prefix: &str, groups: usize) -> Result<GroupNormW> {
    Ok(GroupNormW {
        weight: checkpoint
            .get_required(&format!("{prefix}.weight"))?
            .iter()
            .copied()
            .collect(),
        bias: checkpoint
            .get_required(&format!("{prefix}.bias"))?
            .iter()
            .copied()
            .collect(),
        groups,
    })
}

fn dconv(checkpoint: &Checkpoint, prefix: &str, channels: usize, depth: usize, compress: usize) -> Result<DConvW> {
    let hidden = channels / compress;
    let mut layers = Vec::with_capacity(depth);
    for d in 0..depth {
        let at = format!("{prefix}.layers.{d}");
        let conv1 = conv(checkpoint, &format!("{at}.0"))?;
        if conv1.shape != vec![hidden, channels, 3] {
            return Err(Error::WeightShape {
                name: format!("{at}.0.weight"),
                found: conv1.shape.clone(),
                expected: vec![hidden, channels, 3],
            });
        }
        let conv2 = conv(checkpoint, &format!("{at}.3"))?;
        if conv2.shape != vec![2 * channels, hidden, 1] {
            return Err(Error::WeightShape {
                name: format!("{at}.3.weight"),
                found: conv2.shape.clone(),
                expected: vec![2 * channels, hidden, 1],
            });
        }
        layers.push(DConvLayerW {
            conv1,
            norm1_weight: checkpoint
                .get_required(&format!("{at}.1.weight"))?
                .iter()
                .copied()
                .collect(),
            norm1_bias: checkpoint
                .get_required(&format!("{at}.1.bias"))?
                .iter()
                .copied()
                .collect(),
            conv2,
            norm2_weight: checkpoint
                .get_required(&format!("{at}.4.weight"))?
                .iter()
                .copied()
                .collect(),
            norm2_bias: checkpoint
                .get_required(&format!("{at}.4.bias"))?
                .iter()
                .copied()
                .collect(),
            gamma: checkpoint
                .get_required(&format!("{at}.6.scale"))?
                .iter()
                .copied()
                .collect(),
        });
    }
    Ok(DConvW { layers, channels })
}

fn encoder_layer(
    checkpoint: &Checkpoint,
    prefix: &str,
    channels: usize,
    compress: usize,
    depth: usize,
) -> Result<EncLayerW> {
    Ok(EncLayerW {
        conv: conv(checkpoint, &format!("{prefix}.conv"))?,
        rewrite: conv(checkpoint, &format!("{prefix}.rewrite"))?,
        dconv: dconv(checkpoint, &format!("{prefix}.dconv"), channels, depth, compress)?,
    })
}

fn decoder_layer(
    checkpoint: &Checkpoint,
    prefix: &str,
    channels: usize,
    compress: usize,
    depth: usize,
) -> Result<DecLayerW> {
    Ok(DecLayerW {
        conv_tr: conv(checkpoint, &format!("{prefix}.conv_tr"))?,
        rewrite: conv(checkpoint, &format!("{prefix}.rewrite"))?,
        dconv: dconv(checkpoint, &format!("{prefix}.dconv"), channels, depth, compress)?,
    })
}

/// Splits `nn.MultiheadAttention`'s packed `in_proj_weight` into q/k/v.
fn attention(checkpoint: &Checkpoint, prefix: &str, dim: usize) -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>, Linear)> {
    let packed = shaped(checkpoint, &format!("{prefix}.in_proj_weight"), &[3 * dim, dim])?;
    let flat: Vec<f32> = packed.iter().copied().collect();
    let slice = |index: usize| flat[index * dim * dim..(index + 1) * dim * dim].to_vec();
    let bias = shaped(checkpoint, &format!("{prefix}.in_proj_bias"), &[3 * dim])?;
    let bias: Vec<f32> = bias.iter().copied().collect();
    let bias_slice = |index: usize| bias[index * dim..(index + 1) * dim].to_vec();
    let out_proj = Linear::new(
        shaped(checkpoint, &format!("{prefix}.out_proj.weight"), &[dim, dim])?
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .expect("2-D out_proj"),
        Some(
            shaped(checkpoint, &format!("{prefix}.out_proj.bias"), &[dim])?
                .view()
                .into_dimensionality::<ndarray::Ix1>()
                .expect("1-D out_proj bias"),
        ),
    );
    Ok((
        vec![slice(0), slice(1), slice(2)],
        vec![bias_slice(0), bias_slice(1), bias_slice(2)],
        out_proj,
    ))
}

fn transformer_layer(
    checkpoint: &Checkpoint,
    prefix: &str,
    dim: usize,
    hidden: usize,
    is_cross: bool,
) -> Result<TransformerLayerW> {
    let attn = if is_cross { "cross_attn" } else { "self_attn" };
    let (weights, biases, out_proj) = attention(checkpoint, &format!("{prefix}.{attn}"), dim)?;
    let [q_weight, k_weight, v_weight] = <[Vec<f32>; 3]>::try_from(weights).expect("three projections");
    let [q_bias, k_bias, v_bias] = <[Vec<f32>; 3]>::try_from(biases).expect("three biases");
    Ok(TransformerLayerW {
        is_cross,
        q_weight,
        k_weight,
        v_weight,
        q_bias,
        k_bias,
        v_bias,
        out_proj,
        linear1: linear(checkpoint, &format!("{prefix}.linear1"))?,
        linear2: linear(checkpoint, &format!("{prefix}.linear2"))?,
        norm1: layer_norm(checkpoint, &format!("{prefix}.norm1"))?,
        norm2: layer_norm(checkpoint, &format!("{prefix}.norm2"))?,
        norm3: if is_cross {
            Some(layer_norm(checkpoint, &format!("{prefix}.norm3"))?)
        } else {
            None
        },
        norm_out: group_norm(checkpoint, &format!("{prefix}.norm_out"), 1)?,
        gamma1: checkpoint
            .get_required(&format!("{prefix}.gamma_1.scale"))?
            .iter()
            .copied()
            .collect(),
        gamma2: checkpoint
            .get_required(&format!("{prefix}.gamma_2.scale"))?
            .iter()
            .copied()
            .collect(),
    })
    .map(|layer| {
        debug_assert_eq!(layer.linear1.in_features(), dim);
        debug_assert_eq!(layer.linear1.out_features(), hidden);
        layer
    })
}

impl HtdemucsWeights {
    pub fn load(checkpoint: &Checkpoint, config: &HtdemucsConfig) -> Result<Self> {
        let arch = HtdemucsArch::from_config(config)?;
        let depth = config.depth;
        let compress = 8; // dconv_comp
        let dconv_depth = 2;

        let mut encoder = Vec::with_capacity(depth);
        let mut tencoder = Vec::with_capacity(depth);
        let mut decoder = Vec::with_capacity(depth);
        let mut tdecoder = Vec::with_capacity(depth);
        for spec in &arch.encoder {
            encoder.push(encoder_layer(
                checkpoint,
                &format!("encoder.{}", spec.index),
                spec.chout,
                compress,
                dconv_depth,
            )?);
        }
        for spec in &arch.tencoder {
            tencoder.push(encoder_layer(
                checkpoint,
                &format!("tencoder.{}", spec.index),
                spec.chout,
                compress,
                dconv_depth,
            )?);
        }
        // `arch.decoder` is in execution order but the checkpoint names follow
        // construction order, so index back through the table.
        for spec in &arch.decoder {
            // `HDecLayer` builds its `DConv` on the *input* channels.
            decoder.push(decoder_layer(
                checkpoint,
                &format!("decoder.{}", depth - 1 - spec.index),
                spec.chin,
                compress,
                dconv_depth,
            )?);
        }
        for spec in &arch.tdecoder {
            tdecoder.push(decoder_layer(
                checkpoint,
                &format!("tdecoder.{}", depth - 1 - spec.index),
                spec.chin,
                compress,
                dconv_depth,
            )?);
        }

        let embedding = shaped(
            checkpoint,
            "freq_emb.embedding.weight",
            &[arch.freq_emb_rows, config.channels],
        )?;
        let freq_emb = embedding
            .mapv(|v| v * config.emb_scale as f32)
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| Error::Config(format!("frequency embedding: {e}")))?;

        let dim = arch.transformer_channels;
        let hidden = arch.transformer_hidden;
        let mut layers = Vec::with_capacity(config.t_layers);
        let mut layers_t = Vec::with_capacity(config.t_layers);
        for index in 0..config.t_layers {
            let is_cross = index % 2 == 1;
            layers.push(transformer_layer(
                checkpoint,
                &format!("crosstransformer.layers.{index}"),
                dim,
                hidden,
                is_cross,
            )?);
            layers_t.push(transformer_layer(
                checkpoint,
                &format!("crosstransformer.layers_t.{index}"),
                dim,
                hidden,
                is_cross,
            )?);
        }

        Ok(Self {
            encoder,
            tencoder,
            decoder,
            tdecoder,
            freq_emb,
            channel_upsampler: conv_1x1_as_linear(checkpoint, "channel_upsampler")?,
            channel_upsampler_t: conv_1x1_as_linear(checkpoint, "channel_upsampler_t")?,
            channel_downsampler: conv_1x1_as_linear(checkpoint, "channel_downsampler")?,
            channel_downsampler_t: conv_1x1_as_linear(checkpoint, "channel_downsampler_t")?,
            transformer: CrossTransformerW {
                norm_in: layer_norm(checkpoint, "crosstransformer.norm_in")?,
                norm_in_t: layer_norm(checkpoint, "crosstransformer.norm_in_t")?,
                layers,
                layers_t,
            },
        })
    }

    /// Number of `f32` values held, for reporting.
    pub fn parameter_count(&self) -> usize {
        fn conv_count(c: &ConvW) -> usize {
            c.weight.len() + c.bias.len()
        }
        fn dconv_count(d: &DConvW) -> usize {
            let mut total = 0;
            for layer in &d.layers {
                total += conv_count(&layer.conv1) + conv_count(&layer.conv2);
                total += layer.norm1_weight.len() + layer.norm1_bias.len();
                total += layer.norm2_weight.len() + layer.norm2_bias.len();
                total += layer.gamma.len();
            }
            total
        }
        let mut total = 0;
        for layer in self.encoder.iter().chain(&self.tencoder) {
            total += conv_count(&layer.conv) + conv_count(&layer.rewrite) + dconv_count(&layer.dconv);
        }
        for layer in self.decoder.iter().chain(&self.tdecoder) {
            total += conv_count(&layer.conv_tr) + conv_count(&layer.rewrite) + dconv_count(&layer.dconv);
        }
        total += self.freq_emb.len();
        for head in [&self.channel_upsampler, &self.channel_upsampler_t,
                     &self.channel_downsampler, &self.channel_downsampler_t] {
            total += head.weight_t.len() + head.bias.as_ref().map_or(0, |b| b.len());
        }
        total += self.transformer.norm_in.weight.len() + self.transformer.norm_in.bias.len();
        total += self.transformer.norm_in_t.weight.len() + self.transformer.norm_in_t.bias.len();
        for layer in self.transformer.layers.iter().chain(&self.transformer.layers_t) {
            total += layer.q_weight.len() + layer.k_weight.len() + layer.v_weight.len();
            total += layer.q_bias.len() + layer.k_bias.len() + layer.v_bias.len();
            total += layer.out_proj.weight_t.len();
            total += layer.out_proj.bias.as_ref().map_or(0, |b| b.len());
            total += layer.linear1.weight_t.len() + layer.linear1.bias.as_ref().map_or(0, |b| b.len());
            total += layer.linear2.weight_t.len() + layer.linear2.bias.as_ref().map_or(0, |b| b.len());
            for norm in [Some(&layer.norm1), Some(&layer.norm2), layer.norm3.as_ref()].into_iter().flatten() {
                total += norm.weight.len() + norm.bias.len();
            }
            total += layer.norm_out.weight.len() + layer.norm_out.bias.len();
            total += layer.gamma1.len() + layer.gamma2.len();
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loading_a_missing_checkpoint_reports_the_path() {
        let error = match load_weights("no-such-file.th") {
            Ok(_) => panic!("loading a missing file should fail"),
            Err(error) => error,
        };
        assert!(format!("{error}").contains("no-such-file.th"), "{error}");
    }
}
