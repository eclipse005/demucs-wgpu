//! `HTDemucs` configuration and the derived layer table.
//!
//! The values mirror `demucs/htdemucs.py` defaults as instantiated by the two
//! checkpoints in scope (`htdemucs` / `955717e8` and `htdemucs_ft`): both ship
//! the identical architecture, and the HuggingFace `safetensors` metadata spells
//! it out in its `kwargs` entry.

use serde::Deserialize;

use crate::error::{Error, Result};

/// Architecture hyper-parameters that actually affect the forward pass.
///
/// Everything else in the checkpoint's `kwargs` is a training-time switch
/// (`wiener_iters`, `rescale`, `multi_freqs_depth`, the CAPE/sparse-attention
/// knobs) and is deliberately absent.
#[derive(Debug, Clone, PartialEq)]
pub struct HtdemucsConfig {
    pub sources: Vec<String>,
    pub audio_channels: usize,
    pub samplerate: usize,
    /// Training segment, in seconds. `int(segment * samplerate)` is the length
    /// `forward` pads every chunk to.
    pub segment: f64,
    pub channels: usize,
    pub growth: usize,
    pub nfft: usize,
    pub depth: usize,
    pub stride: usize,
    pub kernel_size: usize,
    pub context: usize,
    pub freq_emb: f64,
    pub emb_scale: f64,
    pub cac: bool,
    pub bottom_channels: usize,
    pub t_layers: usize,
    pub t_heads: usize,
    pub t_hidden_scale: f64,
    pub t_max_period: f64,
    pub t_weight_pos_embed: f64,
}

impl Default for HtdemucsConfig {
    fn default() -> Self {
        Self {
            sources: ["drums", "bass", "other", "vocals"]
                .iter()
                .map(|s| s.to_string())
                .collect(),
            audio_channels: 2,
            samplerate: 44100,
            segment: 39.0 / 5.0,
            channels: 48,
            growth: 2,
            nfft: 4096,
            depth: 4,
            stride: 4,
            kernel_size: 8,
            context: 1,
            freq_emb: 0.2,
            emb_scale: 10.0,
            cac: true,
            bottom_channels: 512,
            t_layers: 5,
            t_heads: 8,
            t_hidden_scale: 4.0,
            t_max_period: 10000.0,
            t_weight_pos_embed: 1.0,
        }
    }
}

/// The `kwargs` blob as it appears in a `safetensors` header.
///
/// Only the fields the forward needs are read; unknown ones are ignored so a
/// checkpoint with extra training switches still loads.
#[derive(Debug, Clone, Deserialize)]
struct RawKwargs {
    #[serde(default)]
    sources: Option<Vec<String>>,
    #[serde(default)]
    audio_channels: Option<usize>,
    #[serde(default)]
    samplerate: Option<usize>,
    #[serde(default)]
    segment: Option<FractionValue>,
    #[serde(default)]
    channels: Option<usize>,
    #[serde(default)]
    growth: Option<usize>,
    #[serde(default)]
    nfft: Option<usize>,
    #[serde(default)]
    depth: Option<usize>,
    #[serde(default)]
    stride: Option<usize>,
    #[serde(default)]
    kernel_size: Option<usize>,
    #[serde(default)]
    context: Option<usize>,
    #[serde(default)]
    freq_emb: Option<f64>,
    #[serde(default)]
    emb_scale: Option<f64>,
    #[serde(default)]
    cac: Option<bool>,
    #[serde(default)]
    bottom_channels: Option<usize>,
    #[serde(default)]
    t_layers: Option<usize>,
    #[serde(default)]
    t_heads: Option<usize>,
    #[serde(default)]
    t_hidden_scale: Option<f64>,
    #[serde(default)]
    t_max_period: Option<f64>,
    #[serde(default)]
    t_weight_pos_embed: Option<f64>,
}

/// `segment` is a `Fraction` in the checkpoint and a plain number in hand-written
/// configs, so accept both.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
enum FractionValue {
    Float(f64),
    Object { numerator: i64, denominator: i64 },
}

impl FractionValue {
    fn value(&self) -> f64 {
        match self {
            FractionValue::Float(v) => *v,
            FractionValue::Object {
                numerator,
                denominator,
            } => {
                if *denominator == 0 {
                    f64::NAN
                } else {
                    *numerator as f64 / *denominator as f64
                }
            }
        }
    }
}

impl HtdemucsConfig {
    /// Reads the `kwargs` JSON blob out of a `safetensors` metadata entry.
    pub fn from_kwargs_json(text: &str) -> Result<Self> {
        let raw: RawKwargs = serde_json::from_str(text)
            .map_err(|e| Error::Config(format!("checkpoint kwargs are not valid JSON: {e}")))?;
        let mut config = HtdemucsConfig::default();
        if let Some(v) = raw.sources {
            config.sources = v;
        }
        if let Some(v) = raw.audio_channels {
            config.audio_channels = v;
        }
        if let Some(v) = raw.samplerate {
            config.samplerate = v;
        }
        if let Some(v) = raw.segment {
            config.segment = v.value();
        }
        if let Some(v) = raw.channels {
            config.channels = v;
        }
        if let Some(v) = raw.growth {
            config.growth = v;
        }
        if let Some(v) = raw.nfft {
            config.nfft = v;
        }
        if let Some(v) = raw.depth {
            config.depth = v;
        }
        if let Some(v) = raw.stride {
            config.stride = v;
        }
        if let Some(v) = raw.kernel_size {
            config.kernel_size = v;
        }
        if let Some(v) = raw.context {
            config.context = v;
        }
        if let Some(v) = raw.freq_emb {
            config.freq_emb = v;
        }
        if let Some(v) = raw.emb_scale {
            config.emb_scale = v;
        }
        if let Some(v) = raw.cac {
            config.cac = v;
        }
        if let Some(v) = raw.bottom_channels {
            config.bottom_channels = v;
        }
        if let Some(v) = raw.t_layers {
            config.t_layers = v;
        }
        if let Some(v) = raw.t_heads {
            config.t_heads = v;
        }
        if let Some(v) = raw.t_hidden_scale {
            config.t_hidden_scale = v;
        }
        if let Some(v) = raw.t_max_period {
            config.t_max_period = v;
        }
        if let Some(v) = raw.t_weight_pos_embed {
            config.t_weight_pos_embed = v;
        }
        Ok(config)
    }

    pub fn hop_length(&self) -> usize {
        self.nfft / 4
    }

    /// `int(segment * samplerate)`: the length every chunk is padded to.
    pub fn training_length(&self) -> usize {
        (self.segment * self.samplerate as f64) as usize
    }

    /// Channels entering each encoder layer of the frequency branch, which is
    /// `audio_channels * 2` for `cac` and grows by `growth` per layer.
    pub fn freq_encoder_in(&self, index: usize) -> usize {
        let first = self.audio_channels * if self.cac { 2 } else { 1 };
        first * self.growth.pow(index as u32) / 1
    }

    pub fn arch(&self) -> Result<HtdemucsArch> {
        HtdemucsArch::from_config(self)
    }

    /// Rejects configurations outside the two supported checkpoints, rather than
    /// silently computing something else.
    pub fn validate(&self) -> Result<()> {
        let unsupported = |what: &str| {
            Err(Error::UnsupportedModel(format!(
                "htdemucs config: {what} is not supported by this port"
            )))
        };
        if !self.cac {
            return unsupported("cac = false (magnitude masking)");
        }
        if self.nfft != 4096 {
            return unsupported("nfft != 4096");
        }
        if self.kernel_size != 8 || self.stride != 4 {
            return unsupported("kernel_size/stride other than 8/4");
        }
        if self.audio_channels != 2 {
            return unsupported("audio_channels != 2");
        }
        if self.sources.len() != 4 {
            return unsupported("a source count other than 4");
        }
        // The freq branch must still have more than one bin at the last layer,
        // otherwise `HEncLayer(empty=...)` kicks in and the two branches merge
        // inside the encoder, a topology this port does not implement.
        let mut freqs = self.nfft / 2;
        for _ in 0..self.depth {
            if freqs > 1 && freqs <= self.kernel_size {
                return unsupported("a `last_freq` layer (the branches would merge in the encoder)");
            }
            if freqs > 1 {
                freqs /= self.stride;
            }
        }
        Ok(())
    }
}

/// One entry of the derived encoder/decoder layer table.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerSpec {
    pub index: usize,
    /// True for the frequency branch (4-D tensors), false for the waveform one.
    pub freq: bool,
    /// Frequency bins entering this layer (frequency branch only).
    pub freqs_in: usize,
    pub chin: usize,
    pub chout: usize,
    pub kernel_size: usize,
    pub stride: usize,
    /// Padding applied by the convolution for the encoder, and the amount of the
    /// transposed convolution's output that is cropped away for the decoder.
    pub pad: usize,
    pub rewrite: bool,
    pub rewrite_context: usize,
    pub dconv: bool,
}

/// The full layer table plus the transformer's shape parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct HtdemucsArch {
    pub config: HtdemucsConfig,
    /// Encoder layers, index 0 (outermost, most frequency bins) first.
    pub encoder: Vec<LayerSpec>,
    /// Waveform encoder layers.
    pub tencoder: Vec<LayerSpec>,
    /// Decoder layers in *execution* order: `decoder[0]` is the innermost,
    /// mirroring `nn.ModuleList.insert(0, ...)`.
    pub decoder: Vec<LayerSpec>,
    pub tdecoder: Vec<LayerSpec>,
    /// Number of frequency bins at the bottleneck.
    pub bottleneck_freqs: usize,
    /// Channels at the bottleneck (frequency / waveform branch).
    pub bottleneck_channels: usize,
    pub bottleneck_channels_t: usize,
    /// Transformer width (384 before `bottom_channels`, 512 with it).
    pub transformer_channels: usize,
    /// Input width of the channel up/down samplers (384).
    pub pre_transformer_channels: usize,
    pub transformer_hidden: usize,
    /// Channels of the final decoder output: `sources * audio_channels * 2`.
    pub output_channels: usize,
    /// `freq_emb` embedding rows.
    pub freq_emb_rows: usize,
}

impl HtdemucsArch {
    pub fn from_config(config: &HtdemucsConfig) -> Result<Self> {
        config.validate()?;
        let depth = config.depth;
        let stride = config.stride;
        let kernel_size = config.kernel_size;

        let mut freqs = config.nfft / 2;
        let mut chin = config.audio_channels;
        let mut chin_z = chin * if config.cac { 2 } else { 1 };
        let mut chout = config.channels;
        let mut chout_z = config.channels;

        let mut encoder = Vec::with_capacity(depth);
        let mut tencoder = Vec::with_capacity(depth);
        let mut decoder_specs = Vec::with_capacity(depth);
        let mut tdecoder_specs = Vec::with_capacity(depth);
        let mut freq_emb_rows = 0usize;

        for index in 0..depth {
            let freq = freqs > 1;
            debug_assert!(freq, "validate() rejects a bottleneck with freqs == 1");
            let (ker, stri) = (kernel_size, stride);
            let pad = ker / 4; // `pad=True` for every layer in this family
            encoder.push(LayerSpec {
                index,
                freq: true,
                freqs_in: freqs,
                chin: chin_z,
                chout: chout_z,
                kernel_size: ker,
                stride: stri,
                pad,
                rewrite: true,
                rewrite_context: 0,
                dconv: true,
            });
            tencoder.push(LayerSpec {
                index,
                freq: false,
                freqs_in: 1,
                chin,
                chout,
                kernel_size: ker,
                stride: stri,
                pad,
                rewrite: true,
                rewrite_context: 0,
                dconv: true,
            });

            if index == 0 {
                chin = config.audio_channels * config.sources.len();
                chin_z = chin * if config.cac { 2 } else { 1 };
            }
            decoder_specs.push(LayerSpec {
                index,
                freq: true,
                freqs_in: freqs / stride,
                chin: chout_z,
                chout: chin_z,
                kernel_size: ker,
                stride: stri,
                pad,
                rewrite: true,
                rewrite_context: config.context,
                dconv: true,
            });
            tdecoder_specs.push(LayerSpec {
                index,
                freq: false,
                freqs_in: 1,
                chin: chout,
                chout: chin,
                kernel_size: ker,
                stride: stri,
                pad,
                rewrite: true,
                rewrite_context: config.context,
                dconv: true,
            });

            if index == 0 {
                freq_emb_rows = freqs / stride;
            }
            chin = chout;
            chin_z = chout_z;
            chout = config.growth * chout;
            chout_z = config.growth * chout_z;
            freqs /= stride;
        }

        // `nn.ModuleList.insert(0, ...)` reverses the decoder's storage order.
        decoder_specs.reverse();
        tdecoder_specs.reverse();

        let bottleneck_channels = encoder.last().unwrap().chout;
        let bottleneck_channels_t = tencoder.last().unwrap().chout;
        let pre_transformer_channels = bottleneck_channels;
        let transformer_channels = if config.bottom_channels > 0 {
            config.bottom_channels
        } else {
            pre_transformer_channels
        };

        Ok(Self {
            config: config.clone(),
            encoder,
            tencoder,
            decoder: decoder_specs,
            tdecoder: tdecoder_specs,
            bottleneck_freqs: freqs,
            bottleneck_channels,
            bottleneck_channels_t,
            transformer_channels,
            pre_transformer_channels,
            transformer_hidden: (transformer_channels as f64 * config.t_hidden_scale) as usize,
            output_channels: config.sources.len() * config.audio_channels * 2,
            freq_emb_rows,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FT_KWARGS: &str = r#"{"sources": ["drums", "bass", "other", "vocals"], "audio_channels": 2, "samplerate": 44100, "segment": {"_type": "fraction", "numerator": 39, "denominator": 5}, "channels": 48, "channels_time": null, "growth": 2, "nfft": 4096, "wiener_iters": 0, "end_iters": 0, "wiener_residual": false, "cac": true, "depth": 4, "rewrite": true, "multi_freqs": [], "multi_freqs_depth": 3, "freq_emb": 0.2, "emb_scale": 10, "emb_smooth": true, "kernel_size": 8, "stride": 4, "time_stride": 2, "context": 1, "context_enc": 0, "norm_starts": 4, "norm_groups": 4, "dconv_mode": 3, "dconv_depth": 2, "dconv_comp": 8, "dconv_init": 0.001, "bottom_channels": 512, "t_layers": 5, "t_hidden_scale": 4.0, "t_heads": 8, "t_dropout": 0.02, "t_layer_scale": true, "t_gelu": true, "t_emb": "sin", "t_max_positions": 10000, "t_max_period": 10000.0, "t_weight_pos_embed": 1.0, "t_sparse_self_attn": false, "t_sparse_cross_attn": false, "t_cross_first": false, "rescale": 0.1}"#;

    #[test]
    fn the_checkpoint_kwargs_parse_into_the_expected_config() {
        let config = HtdemucsConfig::from_kwargs_json(FT_KWARGS).unwrap();
        assert_eq!(config, HtdemucsConfig::default());
        assert_eq!(config.training_length(), 343980);
        assert_eq!(config.hop_length(), 1024);
    }

    #[test]
    fn a_plain_float_segment_is_accepted() {
        let config = HtdemucsConfig::from_kwargs_json(r#"{"segment": 7.8}"#).unwrap();
        assert!((config.segment - 7.8).abs() < 1e-12);
    }

    #[test]
    fn the_layer_table_matches_the_reference_shapes() {
        let arch = HtdemucsConfig::default().arch().unwrap();
        let shapes: Vec<(usize, usize, usize, usize)> = arch
            .encoder
            .iter()
            .map(|l| (l.freqs_in, l.chin, l.chout, l.kernel_size))
            .collect();
        assert_eq!(
            shapes,
            vec![
                (2048, 4, 48, 8),
                (512, 48, 96, 8),
                (128, 96, 192, 8),
                (32, 192, 384, 8),
            ]
        );
        let time_shapes: Vec<(usize, usize)> = arch
            .tencoder
            .iter()
            .map(|l| (l.chin, l.chout))
            .collect();
        assert_eq!(time_shapes, vec![(2, 48), (48, 96), (96, 192), (192, 384)]);

        // Execution order: the deepest layer first, ending at the 16-channel
        // (4 sources x 2 channels x complex) output.
        let dec: Vec<(usize, usize)> = arch.decoder.iter().map(|l| (l.chin, l.chout)).collect();
        assert_eq!(dec, vec![(384, 192), (192, 96), (96, 48), (48, 16)]);
        let tdec: Vec<(usize, usize)> = arch.tdecoder.iter().map(|l| (l.chin, l.chout)).collect();
        assert_eq!(tdec, vec![(384, 192), (192, 96), (96, 48), (48, 8)]);

        assert_eq!(arch.bottleneck_freqs, 8);
        assert_eq!(arch.bottleneck_channels, 384);
        assert_eq!(arch.transformer_channels, 512);
        assert_eq!(arch.transformer_hidden, 2048);
        assert_eq!(arch.freq_emb_rows, 512);
        assert_eq!(arch.output_channels, 16);
    }
}
