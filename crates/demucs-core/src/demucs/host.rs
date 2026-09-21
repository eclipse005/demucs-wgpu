//! Host (CPU) implementation of `HTDemucs.forward`.
//!
//! A literal transcription of `demucs/htdemucs.py` + `hdemucs.py` +
//! `transformer.py`, kept structurally identical so activations can be compared
//! layer by layer against `tools/dump_demucs.py`. It is the numerical reference
//! for the wgpu path, not a performance target.

use ndarray::{Array2, Array3, Array4, ArrayViewD, s};
use rayon::prelude::*;

use crate::demucs::config::{HtdemucsArch, HtdemucsConfig};
use crate::demucs::ops::{
    add_in_place_nd, add_nd, channels_to_leading_freq_into, conv1d, conv2d,
    conv_transpose1d, conv_transpose2d, gelu_in_place, glu, group_norm, group_norm_glu,
    leading_freq_to_channels_into, layer_norm_rows, linear_3d, uninit_array,
};
use crate::demucs::spec::{
    pack_complex_as_channels, pad_for_ispec, unpack_channels_as_complex, DemucsSpec,
};
use crate::demucs::weights::{
    CrossTransformerW, DConvW, DecLayerW, EncLayerW, GroupNormW, HtdemucsWeights, LayerNormW,
    TransformerLayerW,
};
use crate::error::{Error, Result};

/// Stage timings for the host forward, enabled by `DEMUCS_PROFILE=1`.
///
/// The project rule is "measure before guessing": the host path is 7x behind
/// torch+CPU, and which op is responsible is not something to reason about from
/// the source. Enabled from the environment so a normal run pays one relaxed
/// atomic load per stage and nothing else.
pub mod profile {
    use std::sync::{Mutex, OnceLock};
    use std::time::Instant;

    /// One table for every thread.
    ///
    /// It used to be thread-local, which silently dropped the stages that run
    /// inside a rayon region: `attention_head`'s own split into QK, softmax and
    /// the AV product is entirely on worker threads, so those rows never
    /// appeared in the report and the whole 220 ms stage was unattributable.
    /// The lock costs nothing while profiling is off (the scopes never touch the
    /// table) and is noise next to the stages being measured while it is on.
    fn totals() -> &'static Mutex<Vec<(String, f64, usize)>> {
        static TOTALS: OnceLock<Mutex<Vec<(String, f64, usize)>>> = OnceLock::new();
        TOTALS.get_or_init(|| Mutex::new(Vec::new()))
    }

    pub fn enabled() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var("DEMUCS_PROFILE").is_ok_and(|v| v != "0"))
    }

    /// Accumulates into the shared table while it is alive.
    pub struct Scope {
        name: &'static str,
        index: Option<usize>,
        started: Option<Instant>,
    }

    impl Drop for Scope {
        fn drop(&mut self) {
            if let Some(started) = self.started {
                let seconds = started.elapsed().as_secs_f64();
                let key = match self.index {
                    Some(index) => format!("{}.{}", self.name, index),
                    None => self.name.to_string(),
                };
                if let Ok(mut totals) = totals().lock() {
                    match totals.iter_mut().find(|(name, _, _)| *name == key) {
                        Some(entry) => {
                            entry.1 += seconds;
                            entry.2 += 1;
                        }
                        None => totals.push((key, seconds, 1)),
                    }
                }
            }
        }
    }

    pub fn scope(name: &'static str) -> Scope {
        Scope {
            name,
            index: None,
            started: if enabled() { Some(Instant::now()) } else { None },
        }
    }

    /// Same, but the table key carries a layer index.
    pub fn scope_index(name: &'static str, index: usize) -> Scope {
        Scope {
            name,
            index: Some(index),
            started: if enabled() { Some(Instant::now()) } else { None },
        }
    }

    /// The accumulated table, sorted by total time, and reset.
    pub fn take() -> Vec<(String, f64, usize)> {
        let mut entries = match totals().lock() {
            Ok(mut totals) => std::mem::take(&mut *totals),
            Err(_) => Vec::new(),
        };
        entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        entries
    }

    pub fn reset() {
        if let Ok(mut totals) = totals().lock() {
            totals.clear();
        }
    }
}

/// Where a model run writes its intermediate activations.
pub trait TraceSink {
    fn record(&mut self, name: &str, value: ArrayViewD<f32>);

    /// Whether [`TraceSink::record`] is worth paying for on this name. The device
    /// path has to read a tensor back over PCIe, so it asks first.
    fn wants(&self, _name: &str) -> bool {
        true
    }
}

/// Discards every recorded tensor.
pub struct NoTrace;

impl TraceSink for NoTrace {
    fn record(&mut self, _name: &str, _value: ArrayViewD<f32>) {}

    fn wants(&self, _name: &str) -> bool {
        false
    }
}

/// Collects activations in memory, keyed by name.
#[derive(Default)]
pub struct VecTrace {
    pub entries: Vec<(String, Vec<usize>, Vec<f32>)>,
}

impl VecTrace {
    pub fn get(&self, name: &str) -> Option<&(String, Vec<usize>, Vec<f32>)> {
        self.entries.iter().find(|entry| entry.0 == name)
    }

    pub fn names(&self) -> Vec<&str> {
        self.entries.iter().map(|entry| entry.0.as_str()).collect()
    }
}

impl TraceSink for VecTrace {
    fn record(&mut self, name: &str, value: ArrayViewD<f32>) {
        self.entries.push((
            name.to_string(),
            value.shape().to_vec(),
            value.iter().copied().collect(),
        ));
    }
}

/// Per-batch normalisation constants the reference derives from the input.
#[derive(Debug, Clone)]
pub struct Normalisation {
    /// Frequency branch: mean / std over `(channels, bins, frames)`.
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    /// Waveform branch: mean / std over `(channels, samples)`.
    pub mean_t: Vec<f32>,
    pub std_t: Vec<f32>,
}

/// The model: configuration, derived layer table, weights and STFT pair.
pub struct Htdemucs {
    pub config: HtdemucsConfig,
    pub arch: HtdemucsArch,
    pub weights: HtdemucsWeights,
    pub spec: DemucsSpec,
}

impl Htdemucs {
    pub fn new(config: HtdemucsConfig, weights: HtdemucsWeights) -> Result<Self> {
        let arch = HtdemucsArch::from_config(&config)?;
        let spec = DemucsSpec::new(&config)?;
        Ok(Self {
            config,
            arch,
            weights,
            spec,
        })
    }

    /// `int(segment * samplerate)`: the length every input is padded to.
    pub fn training_length(&self) -> usize {
        self.config.training_length()
    }

    /// `HTDemucs.forward`. `mix` is `(batch, audio_channels, length)`; the result
    /// is `(batch, sources, audio_channels, length)`.
    pub fn forward(&self, mix: &Array3<f32>, trace: &mut dyn TraceSink) -> Result<Array4<f32>> {
        let (batch, audio_channels, length) = mix.dim();
        if audio_channels != self.config.audio_channels {
            return Err(Error::Shape(format!(
                "expected {} audio channels, got {audio_channels}",
                self.config.audio_channels
            )));
        }
        let training_length = self.training_length();
        if length > training_length {
            return Err(Error::Shape(format!(
                "input is {length} samples but the model segment is {training_length}; \
                 chunk the audio before calling forward"
            )));
        }
        // `use_train_segment` pads short inputs up to the training length and
        // trims the result back afterwards.
        let length_pre_pad = if length < training_length {
            Some(length)
        } else {
            None
        };
        let padded = if let Some(pre) = length_pre_pad {
            let mut wide = Array3::<f32>::zeros((batch, audio_channels, training_length));
            wide.slice_mut(s![.., .., ..pre]).assign(mix);
            wide
        } else {
            mix.clone()
        };
        if trace.wants("mix.in") {
            trace.record("mix.in", padded.view().into_dyn());
        }

        let z = {
            let _scope = profile::scope("_spec");
            self.spec.spec(&padded)?
        };
        if trace.wants("_spec") {
            // Same packing as `_magnitude`, but keeping the complex planes:
            // `view_as_real(z).permute(0, 1, 4, 2, 3).reshape(b, 2c, fr, t)`.
            let packed = pack_complex_as_channels(&z);
            trace.record("_spec", packed.view().into_dyn());
        }
        let mag = pack_complex_as_channels(&z);
        if trace.wants("_magnitude") {
            trace.record("_magnitude", mag.view().into_dyn());
        }

        // `(x - mean) / (1e-5 + std)`, per batch element.
        let norm = self.normalise(&mag, &padded);
        let mut x = mag;
        for bi in 0..batch {
            let (offset, scale) = (norm.mean[bi], 1.0 / (1e-5 + norm.std[bi]));
            affine_in_place(x.slice_mut(s![bi, .., .., ..]), offset, scale);
        }
        let mut xt = padded.clone();
        for bi in 0..batch {
            let (offset, scale) = (norm.mean_t[bi], 1.0 / (1e-5 + norm.std_t[bi]));
            affine_in_place(xt.slice_mut(s![bi, .., ..]), offset, scale);
        }
        if trace.wants("x_normalized") {
            trace.record("x_normalized", x.view().into_dyn());
        }
        if trace.wants("xt_normalized") {
            trace.record("xt_normalized", xt.view().into_dyn());
        }

        let depth = self.config.depth;
        let mut saved: Vec<Array4<f32>> = Vec::with_capacity(depth);
        let mut saved_t: Vec<Array3<f32>> = Vec::with_capacity(depth);
        let mut lengths: Vec<usize> = Vec::with_capacity(depth);
        let mut lengths_t: Vec<usize> = Vec::with_capacity(depth);

        for index in 0..depth {
            lengths.push(x.dim().3);
            if index < self.weights.tencoder.len() {
                lengths_t.push(xt.dim().2);
                xt = {
                    let _scope = profile::scope_index("encoder.time", index);
                    self.time_encoder_forward(
                        &self.weights.tencoder[index],
                        &xt,
                        &format!("tencoder.{index}"),
                        index,
                        trace,
                    )?
                };
                if trace.wants(&format!("tencoder.{index}")) {
                    trace.record(&format!("tencoder.{index}"), xt.view().into_dyn());
                }
                saved_t.push(xt.clone());
            }
            x = {
                let _scope = profile::scope_index("encoder.freq", index);
                self.freq_encoder_forward(
                    &self.weights.encoder[index],
                    &x,
                    &format!("encoder.{index}"),
                    index,
                    trace,
                )?
            };
            // The reference's module hook sees this value; the embedding below is
            // added by `HTDemucs.forward` itself, after the module returns.
            if trace.wants(&format!("encoder.{index}")) {
                trace.record(&format!("encoder.{index}"), x.view().into_dyn());
            }
            if index == 0 && self.config.freq_emb > 0.0 {
                self.add_frequency_embedding(&mut x)?;
                if trace.wants("encoder.0.post_emb") {
                    trace.record("encoder.0.post_emb", x.view().into_dyn());
                }
            }
            saved.push(x.clone());
        }

        let (_, _, freqs, frames) = x.dim();
        let pre_transformer = self.arch.pre_transformer_channels;
        if self.config.bottom_channels > 0 {
            let flat = x
                .view()
                .permuted_axes([0, 2, 3, 1])
                .as_standard_layout()
                .to_owned()
                .into_shape_with_order((batch, freqs * frames, pre_transformer))
                .expect("reshape")
                .to_owned();
            let _scope = profile::scope("channel_upsampler");
            let up = linear_3d(&self.weights.channel_upsampler, &flat);
            // The flat axis is `f * frames + t`, so split it back in that order.
            x = up
                .into_shape_with_order((batch, freqs, frames, self.config.bottom_channels))
                .expect("reshape")
                .permuted_axes([0, 3, 1, 2])
                .as_standard_layout()
                .to_owned();
            xt = channel_mix(&self.weights.channel_upsampler_t, &xt)?;
        }
        if trace.wants("freq_emb") {
            // `ScaledEmbedding.forward(arange(freqs))` times `emb_scale`.
            let rows = self.weights.freq_emb.dim().0;
            let columns = self.weights.freq_emb.dim().1;
            let mut table = Array2::<f32>::zeros((rows, columns));
            for f in 0..rows {
                for c in 0..columns {
                    table[[f, c]] = self.weights.freq_emb[[f, c]];
                }
            }
            trace.record("freq_emb", table.view().into_dyn());
        }
        if trace.wants("channel_upsampler") {
            let flat = x
                .view()
                .into_shape_with_order((batch, self.config.bottom_channels, freqs * frames))
                .expect("reshape")
                .to_owned();
            trace.record("channel_upsampler", flat.view().into_dyn());
        }
        if trace.wants("channel_upsampler_t") {
            trace.record("channel_upsampler_t", xt.view().into_dyn());
        }
        if trace.wants("crosstransformer#0") {
            trace.record("crosstransformer#0", x.view().into_dyn());
        }
        if trace.wants("crosstransformer#1") {
            trace.record("crosstransformer#1", xt.view().into_dyn());
        }

        let (x, xt) = {
            let _scope = profile::scope("transformer");
            self.transformer_forward(&x, &xt, trace)?
        };

        let mut x = x;
        let mut xt = xt;
        if self.config.bottom_channels > 0 {
            let flat = x
                .view()
                .permuted_axes([0, 2, 3, 1])
                .as_standard_layout()
                .to_owned()
                .into_shape_with_order((batch, freqs * frames, self.config.bottom_channels))
                .expect("reshape")
                .to_owned();
            let _scope = profile::scope("channel_downsampler");
            let down = linear_3d(&self.weights.channel_downsampler, &flat);
            x = down
                .into_shape_with_order((batch, freqs, frames, pre_transformer))
                .expect("reshape")
                .permuted_axes([0, 3, 1, 2])
                .as_standard_layout()
                .to_owned();
            xt = channel_mix(&self.weights.channel_downsampler_t, &xt)?;
        }
        if trace.wants("channel_downsampler") {
            let flat = x
                .view()
                .into_shape_with_order((batch, pre_transformer, freqs * frames))
                .expect("reshape")
                .to_owned();
            trace.record("channel_downsampler", flat.view().into_dyn());
        }
        if trace.wants("channel_downsampler_t") {
            trace.record("channel_downsampler_t", xt.view().into_dyn());
        }

        for index in 0..depth {
            let skip = saved.pop().expect("frequency skip connection");
            let length = lengths.pop().expect("frequency length");
            let last = index == depth - 1;
            let (out, pre) = ({
                let _scope = profile::scope_index("decoder.freq", index);
                self.freq_decoder_forward(
                &self.weights.decoder[index],
                &x,
                &skip,
                length,
                last,
                &format!("decoder.{index}"),
                trace,
            )?
            });
            if trace.wants(&format!("decoder.{index}")) {
                trace.record(&format!("decoder.{index}#0"), out.view().into_dyn());
                trace.record(&format!("decoder.{index}#1"), pre.view().into_dyn());
            }
            x = out;

            let skip_t = saved_t.pop().expect("waveform skip connection");
            let length_t = lengths_t.pop().expect("waveform length");
            let (out_t, pre_t) = ({
                let _scope = profile::scope_index("decoder.time", index);
                self.time_decoder_forward(
                &self.weights.tdecoder[index],
                &xt,
                &skip_t,
                length_t,
                last,
                &format!("tdecoder.{index}"),
                index,
                trace,
            )?
            });
            if trace.wants(&format!("tdecoder.{index}")) {
                trace.record(&format!("tdecoder.{index}#0"), out_t.view().into_dyn());
                trace.record(&format!("tdecoder.{index}#1"), pre_t.view().into_dyn());
            }
            xt = out_t;
        }

        if !saved.is_empty() || !saved_t.is_empty() {
            return Err(Error::Shape("skip connections left over".into()));
        }

        let sources = self.config.sources.len();
        let (_, channels, _, _) = x.dim();
        if channels != sources * 4 {
            return Err(Error::Shape(format!(
                "the frequency branch produced {channels} channels, expected {}",
                sources * 4
            )));
        }

        // Denormalise, then the CaC masking path.
        for bi in 0..batch {
            let (offset, scale) = (norm.mean[bi], norm.std[bi]);
            affine_in_place(x.slice_mut(s![bi, .., .., ..]), -offset / scale, scale);
        }
        if trace.wants("frequency_branch_out") {
            // The reference records `x.view(b, s, -1, fr, t)` — same memory, one
            // more axis, so the flat data can be reused verbatim.
            let flat: Vec<f32> = x.iter().copied().collect();
            let (b_x, _, fr_x, t_x) = x.dim();
            let view = ndarray::ArrayD::from_shape_vec(
                ndarray::IxDyn(&[b_x, sources, 4, fr_x, t_x]),
                flat,
            )
            .expect("the packed layout matches");
            trace.record("_mask.in", view.view().into_dyn());
        }
        let waveform = {
            let _scope = profile::scope("epilogue");
            let zout = {
                let _s = profile::scope("spec.unpack");
                unpack_channels_as_complex(&x, sources)?
            };
            let zout = {
                let _s = profile::scope("spec.pad");
                pad_for_ispec(&zout)
            };
            let _s = profile::scope("spec.ispec");
            self.spec.ispec(&zout, training_length, 0)?
        };
        if trace.wants("freq_ispec") {
            trace.record("_ispec.out", waveform.view().into_dyn());
        }

        let mut time = xt
            .view()
            .into_shape_with_order((batch, sources, audio_channels, training_length))
            .expect("contiguous view")
            .to_owned();
        if trace.wants("time_branch_out") {
            trace.record("time_branch_out", time.view().into_dyn());
        }
        for bi in 0..batch {
            let (offset, scale) = (norm.mean_t[bi], norm.std_t[bi]);
            affine_in_place(time.slice_mut(s![bi, .., .., ..]), -offset / scale, scale);
        }

        let mut out = time + waveform;
        if let Some(pre) = length_pre_pad {
            out = out.slice(s![.., .., .., ..pre]).to_owned();
        }
        if trace.wants("model_out") {
            trace.record("model_out", out.view().into_dyn());
        }
        Ok(out)
    }

    /// The `mean` / `std` pairs `forward` uses for both branches.
    pub fn normalise(&self, mag: &Array4<f32>, waveform: &Array3<f32>) -> Normalisation {
        let batch = mag.dim().0;
        let mut mean = Vec::with_capacity(batch);
        let mut std = Vec::with_capacity(batch);
        let mut mean_t = Vec::with_capacity(batch);
        let mut std_t = Vec::with_capacity(batch);
        for bi in 0..batch {
            let values: Vec<f32> = mag.slice(s![bi, .., .., ..]).iter().copied().collect();
            mean.push(crate::demucs::ops::mean(&values));
            std.push(crate::demucs::ops::std_unbiased(&values));
            let values: Vec<f32> = waveform.slice(s![bi, .., ..]).iter().copied().collect();
            mean_t.push(crate::demucs::ops::mean(&values));
            std_t.push(crate::demucs::ops::std_unbiased(&values));
        }
        Normalisation {
            mean,
            std,
            mean_t,
            std_t,
        }
    }

    /// `x = x + freq_emb_scale * freq_emb[:freqs].t()` broadcast over time.
    fn add_frequency_embedding(&self, x: &mut Array4<f32>) -> Result<()> {
        let (_, channels, freqs, _) = x.dim();
        let rows = self.weights.freq_emb.dim().0;
        let columns = self.weights.freq_emb.dim().1;
        if freqs != rows || channels != columns {
            return Err(Error::Shape(format!(
                "frequency embedding is {rows}x{columns} but the layer output is {freqs}x{channels}"
            )));
        }
        let scale = self.config.freq_emb as f32;
        // The tensor is `(batch, channels, freqs, time)` and the embedding is
        // constant along time, so one task per `(batch, channel, freq)` row; a
        // per-`(c, f)` slice would be a strided view.
        let (batch, _, _, frames) = x.dim();
        let _ = batch;
        x.as_slice_mut()
            .expect("standard layout")
            .par_chunks_mut(frames)
            .enumerate()
            .for_each(|(row, run)| {
                let f = row % freqs;
                let c = (row / freqs) % channels;
                let value = scale * self.weights.freq_emb[[f, c]];
                for slot in run.iter_mut() {
                    *slot += value;
                }
            });
        Ok(())
    }

    /// `HEncLayer` for the frequency branch: `(b, c, f, t)` in, same out.
    fn freq_encoder_forward(
        &self,
        layer: &EncLayerW,
        input: &Array4<f32>,
        name: &str,
        index: usize,
        trace: &mut dyn TraceSink,
    ) -> Result<Array4<f32>> {
        let pad = layer.conv.kernel()[0] / 4;
        let mut y = conv2d(
            input,
            &layer.conv,
            (self.config.stride, 1),
            (pad, 0),
            (1, 1),
        );
        if trace.wants(&format!("{name}.conv")) {
            trace.record(&format!("{name}.conv"), y.view().into_dyn());
        }
        gelu_in_place(y.as_slice_mut().expect("standard layout"));

        let (batch, channels, freqs, frames) = y.dim();
        let mut flat = uninit_array((batch * freqs, channels, frames));
        channels_to_leading_freq_into(&y, &mut flat)?;
        flat = {
            let _scope = profile::scope_index("dconv.encoder.freq", index);
            dconv_forward(&layer.dconv, &flat)?
        };
        if trace.wants(&format!("{name}.dconv")) {
            trace.record(&format!("{name}.dconv"), flat.view().into_dyn());
        }
        let mut y = uninit_array((batch, channels, freqs, frames));
        leading_freq_to_channels_into(&flat, batch, channels, freqs, &mut y)?;

        let rewritten = conv2d(
            &y,
            &layer.rewrite,
            (1, 1),
            (layer.rewrite.kernel()[0] / 2, layer.rewrite.kernel()[1] / 2),
            (1, 1),
        );
        if trace.wants(&format!("{name}.rewrite")) {
            trace.record(&format!("{name}.rewrite"), rewritten.view().into_dyn());
        }
        glu(&rewritten)
    }

    /// `HEncLayer(freq=False)`: `(b, c, t)` in, same out.
    fn time_encoder_forward(
        &self,
        layer: &EncLayerW,
        input: &Array3<f32>,
        name: &str,
        index: usize,
        trace: &mut dyn TraceSink,
    ) -> Result<Array3<f32>> {
        let stride = self.config.stride;
        let length = input.dim().2;
        let mut x = if length % stride != 0 {
            let _scope = profile::scope("t.pad");
            pad_right(input, stride - length % stride)
        } else {
            input.clone()
        };
        let pad = layer.conv.kernel()[0] / 4;
        let mut y = {
            let _scope = profile::scope("t.conv");
            conv1d(&x, &layer.conv, stride, pad, 1)
        };
        if trace.wants(&format!("{name}.conv")) {
            trace.record(&format!("{name}.conv"), y.view().into_dyn());
        }
        {
            let _scope = profile::scope("t.gelu");
            gelu_in_place(y.as_slice_mut().expect("standard layout"));
        }
        let y = {
            let _scope = profile::scope_index("dconv.encoder.time", index);
            dconv_forward(&layer.dconv, &y)?
        };
        if trace.wants(&format!("{name}.dconv")) {
            trace.record(&format!("{name}.dconv"), y.view().into_dyn());
        }
        let rewritten = {
            let _scope = profile::scope("t.rewrite");
            conv1d(
                &y,
                &layer.rewrite,
                1,
                layer.rewrite.kernel()[0] / 2,
                1,
            )
        };
        if trace.wants(&format!("{name}.rewrite")) {
            trace.record(&format!("{name}.rewrite"), rewritten.view().into_dyn());
        }
        x = {
            let _scope = profile::scope("t.glu");
            glu(&rewritten)?
        };
        Ok(x)
    }

    /// `HDecLayer` for the frequency branch.
    #[allow(clippy::too_many_arguments)]
    fn freq_decoder_forward(
        &self,
        layer: &DecLayerW,
        x: &Array4<f32>,
        skip: &Array4<f32>,
        length: usize,
        last: bool,
        name: &str,
        trace: &mut dyn TraceSink,
    ) -> Result<(Array4<f32>, Array4<f32>)> {
        let summed = {
            let _scope = profile::scope("dec.rewrite.add");
            add_nd(x, skip)
        };
        let rewritten = {
            let _scope = profile::scope("dec.rewrite.conv");
            conv2d(
                &summed,
                &layer.rewrite,
                (1, 1),
                (layer.rewrite.kernel()[0] / 2, layer.rewrite.kernel()[1] / 2),
                (1, 1),
            )
        };
        if trace.wants(&format!("{name}.rewrite")) {
            trace.record(&format!("{name}.rewrite"), rewritten.view().into_dyn());
        }
        let gated = {
            let _scope = profile::scope("dec.rewrite.glu");
            glu(&rewritten)?
        };

        let (batch, channels, freqs, frames) = gated.dim();
        let mut flat = uninit_array((batch * freqs, channels, frames));
        {
            let _scope = profile::scope("dec.permute.in");
            channels_to_leading_freq_into(&gated, &mut flat)?;
        }
        flat = {
            let _scope = profile::scope("dec.dconv");
            dconv_forward(&layer.dconv, &flat)?
        };
        if trace.wants(&format!("{name}.dconv")) {
            trace.record(&format!("{name}.dconv"), flat.view().into_dyn());
        }
        let mut pre = uninit_array((batch, channels, freqs, frames));
        {
            let _scope = profile::scope("dec.permute.out");
            leading_freq_to_channels_into(&flat, batch, channels, freqs, &mut pre)?;
        }

        let upsampled = {
            let _scope = profile::scope("dec.conv_tr");
            conv_transpose2d(&pre, &layer.conv_tr, (self.config.stride, 1))
        };
        if trace.wants(&format!("{name}.conv_tr")) {
            trace.record(&format!("{name}.conv_tr"), upsampled.view().into_dyn());
        }
        let pad = layer.conv_tr.kernel()[0] / 4;
        let bins = upsampled.dim().2;
        if 2 * pad > bins {
            return Err(Error::Shape(format!(
                "{name}: {bins} bins cannot lose {pad} on each side"
            )));
        }
        // The frequency crop is unconditional in the reference; `length` there is
        // the *frame* count, which only the waveform branch consumes.
        let _ = length;
        let mut z = {
            let _scope = profile::scope("dec.crop");
            upsampled.slice(s![.., .., pad..bins - pad, ..]).to_owned()
        };
        if !last {
            let _scope = profile::scope("dec.gelu");
            gelu_in_place(z.as_slice_mut().expect("standard layout"));
        }
        Ok((z, pre))
    }

    /// `HDecLayer(freq=False)`.
    #[allow(clippy::too_many_arguments)]
    fn time_decoder_forward(
        &self,
        layer: &DecLayerW,
        x: &Array3<f32>,
        skip: &Array3<f32>,
        length: usize,
        last: bool,
        name: &str,
        index: usize,
        trace: &mut dyn TraceSink,
    ) -> Result<(Array3<f32>, Array3<f32>)> {
        let summed = {
            let _scope = profile::scope("t.add");
            add_nd(x, skip)
        };
        let rewritten = {
            let _scope = profile::scope("t.rewrite");
            conv1d(
                &summed,
                &layer.rewrite,
                1,
                layer.rewrite.kernel()[0] / 2,
                1,
            )
        };
        if trace.wants(&format!("{name}.rewrite")) {
            trace.record(&format!("{name}.rewrite"), rewritten.view().into_dyn());
        }
        let gated = {
            let _scope = profile::scope("t.glu");
            glu(&rewritten)?
        };
        let pre = {
            let _scope = profile::scope_index("dconv.decoder.time", index);
            dconv_forward(&layer.dconv, &gated)?
        };
        if trace.wants(&format!("{name}.dconv")) {
            trace.record(&format!("{name}.dconv"), pre.view().into_dyn());
        }

        let upsampled = {
            let _scope = profile::scope("t.conv_tr");
            conv_transpose1d(&pre, &layer.conv_tr, self.config.stride)
        };
        if trace.wants(&format!("{name}.conv_tr")) {
            trace.record(&format!("{name}.conv_tr"), upsampled.view().into_dyn());
        }
        let pad = layer.conv_tr.kernel()[0] / 4;
        let produced = upsampled.dim().2;
        if pad + length > produced {
            return Err(Error::Shape(format!(
                "{name}: transposed convolution produced {produced} samples, need {}",
                pad + length
            )));
        }
        let mut z = {
            let _scope = profile::scope("t.crop");
            upsampled.slice(s![.., .., pad..pad + length]).to_owned()
        };
        if !last {
            let _scope = profile::scope("t.gelu");
            gelu_in_place(z.as_slice_mut().expect("standard layout"));
        }
        Ok((z, pre))
    }

    /// The cross-transformer stack, including both positional embeddings.
    pub fn transformer_forward(
        &self,
        x: &Array4<f32>,
        xt: &Array3<f32>,
        trace: &mut dyn TraceSink,
    ) -> Result<(Array4<f32>, Array3<f32>)> {
        let (batch, channels, freqs, frames) = x.dim();
        if channels != self.arch.transformer_channels {
            return Err(Error::Shape(format!(
                "transformer expects {} channels, got {channels}",
                self.arch.transformer_channels
            )));
        }
        let weights: &CrossTransformerW = &self.weights.transformer;

        // `rearrange(x, "b c fr t1 -> b (t1 fr) c")`
        let mut spec = {
            let _scope = profile::scope("xf.rearrange");
            x.view()
                .permuted_axes([0, 3, 2, 1])
                .as_standard_layout()
                .to_owned()
                .into_shape_with_order((batch, frames * freqs, channels))
                .expect("reshape")
                .to_owned()
        };
        let pos2d = {
            let _scope = profile::scope("xf.pos2d");
            two_dimensional_positions(
                channels,
                freqs,
                frames,
                self.config.t_max_period as f32,
            )
        };

        // `rearrange(xt, "b c t2 -> b t2 c")`
        let mut time = xt
            .view()
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .to_owned();
        let time_len = time.dim().1;
        let pos1d = {
            let _scope = profile::scope("xf.pos1d");
            one_dimensional_positions(time_len, channels, self.config.t_max_period as f32)
        };

        spec = {
            let _scope = profile::scope("xf.norm_in");
            normalise_rows(&spec, &weights.norm_in)?
        };
        if trace.wants("crosstransformer.norm_in") {
            trace.record("crosstransformer.norm_in", spec.view().into_dyn());
        }
        spec = &spec + &(pos2d.clone() * self.config.t_weight_pos_embed as f32);
        time = normalise_rows(&time, &weights.norm_in_t)?;
        if trace.wants("crosstransformer.norm_in_t") {
            trace.record("crosstransformer.norm_in_t", time.view().into_dyn());
        }
        time = &time + &(pos1d.clone() * self.config.t_weight_pos_embed as f32);

        if trace.wants("crosstransformer.in#0") {
            trace.record("crosstransformer.in#0", spec.view().into_dyn());
        }
        if trace.wants("crosstransformer.in#1") {
            trace.record("crosstransformer.in#1", time.view().into_dyn());
        }

        for index in 0..self.config.t_layers {
            let layer = &weights.layers[index];
            let layer_t = &weights.layers_t[index];
            if layer.is_cross {
                let old_spec = spec.clone();
                spec = cross_layer_forward(
                    layer,
                    &spec,
                    &time,
                    self.config.t_heads,
                    &format!("crosstransformer.layers.{index}"),
                    trace,
                )?;
                time = cross_layer_forward(
                    layer_t,
                    &time,
                    &old_spec,
                    self.config.t_heads,
                    &format!("crosstransformer.layers_t.{index}"),
                    trace,
                )?;
            } else {
                spec = classic_layer_forward(
                    layer,
                    &spec,
                    self.config.t_heads,
                    &format!("crosstransformer.layers.{index}"),
                    trace,
                )?;
                time = classic_layer_forward(
                    layer_t,
                    &time,
                    self.config.t_heads,
                    &format!("crosstransformer.layers_t.{index}"),
                    trace,
                )?;
            }
        }

        let spec = spec
            .into_shape_with_order((batch, frames, freqs, channels))
            .expect("reshape")
            .permuted_axes([0, 3, 2, 1])
            .as_standard_layout()
            .to_owned();
        let time = time.permuted_axes([0, 2, 1]).as_standard_layout().to_owned();
        Ok((spec, time))
    }
}

/// A 1x1 convolution over the channel axis of a `(b, c, t)` tensor: the
/// reference's `Conv1d(C_in, C_out, 1)`, which is a per-position linear map.
pub fn channel_mix(layer: &crate::ops::Linear, x: &Array3<f32>) -> Result<Array3<f32>> {
    let (b, c, t) = x.dim();
    if c != layer.in_features() {
        return Err(Error::Shape(format!(
            "channel mixer expects {} channels, got {c}",
            layer.in_features()
        )));
    }
    let flat = x
        .view()
        .permuted_axes([0, 2, 1])
        .as_standard_layout()
        .to_owned()
        .into_shape_with_order((b * t, c))
        .expect("reshape")
        .to_owned();
    let out = layer.forward(&flat);
    Ok(out
        .into_shape_with_order((b, t, layer.out_features()))
        .expect("reshape")
        .permuted_axes([0, 2, 1])
        .as_standard_layout()
        .to_owned())
}

/// `F.pad(x, (0, amount))` on a `(b, c, t)` tensor.
fn pad_right(x: &Array3<f32>, amount: usize) -> Array3<f32> {
    let (b, c, t) = x.dim();
    let mut out = Array3::<f32>::zeros((b, c, t + amount));
    out.slice_mut(s![.., .., ..t]).assign(x);
    out
}

/// The `DConv` residual branch on a `(rows, channels, time)` tensor.
pub fn dconv_forward(branch: &DConvW, x: &Array3<f32>) -> Result<Array3<f32>> {
    let mut out = x.clone();
    for (depth, layer) in branch.layers.iter().enumerate() {
        let dilation = 1usize << depth;
        let pad = dilation * (layer.conv1.kernel()[0] / 2);
        let mut y = {
            let _scope = profile::scope("dconv.conv1");
            conv1d(&out, &layer.conv1, 1, pad, dilation)
        };
        y = {
            let _scope = profile::scope("dconv.norm1");
            group_norm(&y, 1, &layer.norm1_weight, &layer.norm1_bias)?
        };
        gelu_in_place(y.as_slice_mut().expect("standard layout"));
        // The second half — `Conv1x1 -> GroupNorm(1) -> GLU -> LayerScale ->
        // residual add` — has one fused form: the group norm's statistics are
        // per row, the 1x1 already works one row at a time, and the intermediate
        // is 66 MB a call. `DEMUCS_FUSE_DCONV_TAIL=0` runs the four operators
        // separately, which is how the two were compared.
        let fuse_tail = layer.conv2.kernel().len() == 1
            && layer.conv2.kernel()[0] == 1
            && std::env::var("DEMUCS_FUSE_DCONV_TAIL").map_or(true, |v| v != "0");
        if fuse_tail {
            let _scope = profile::scope("dconv.tail");
            crate::demucs::ops::dconv_tail_into(
                &mut out,
                &y,
                &layer.conv2,
                &layer.norm2_weight,
                &layer.norm2_bias,
                &layer.gamma,
            )?;
            continue;
        }
        y = {
            let _scope = profile::scope("dconv.conv2");
            conv1d(&y, &layer.conv2, 1, 0, 1)
        };
        // `norm2 -> GLU` is one sweep when the two are fused: the norm's write
        // and the GLU's read-back are 132 MB a call at the DConv's sizes.
        // `DEMUCS_FUSE_NORM_GLU=0` keeps the pair, which is how they were
        // compared; the fused form is bit-identical (same statistics, same two
        // elementwise expressions).
        if std::env::var("DEMUCS_FUSE_NORM_GLU").is_ok_and(|v| v == "0") {
            y = {
                let _scope = profile::scope("dconv.norm2");
                group_norm(&y, 1, &layer.norm2_weight, &layer.norm2_bias)?
            };
            y = {
                let _scope = profile::scope("dconv.glu");
                glu(&y)?
            };
        } else {
            let _scope = profile::scope("dconv.norm2");
            y = group_norm_glu(&y, 1, &layer.norm2_weight, &layer.norm2_bias)?;
        }
        // `LayerScale`: `scale[:, None] * x`, one task per (row, channel) run.
        {
            let (_, channels, inner) = y.dim();
            let gamma = &layer.gamma;
            y.as_slice_mut()
                .expect("standard layout")
                .par_chunks_mut(inner)
                .enumerate()
                .for_each(|(row, dst)| {
                    let g = gamma[row % channels];
                    for value in dst.iter_mut() {
                        *value *= g;
                    }
                });
        }
        add_in_place_nd(&mut out, &y);
    }
    Ok(out)
}

/// `nn.LayerNorm` over the last axis of a `(b, t, c)` tensor.
fn normalise_rows(x: &Array3<f32>, norm: &LayerNormW) -> Result<Array3<f32>> {
    let (b, t, c) = x.dim();
    // The reshape is a view: the rows are already contiguous, and materialising
    // it was a full copy of the tensor per norm call.
    let flat = x
        .view()
        .into_shape_with_order((b * t, c))
        .expect("contiguous view");
    let out = layer_norm_rows(&flat, &norm.weight, &norm.bias)?;
    Ok(out.into_shape_with_order((b, t, c)).expect("reshape back"))
}

/// `create_2d_sin_embedding`, flattened to `(1, frame * freq + bin, dim)`.
///
/// Lanes `0..dim/2` carry the frame position and `dim/2..dim` the frequency
/// position, each with `sin` on the even and `cos` on the odd lanes.
pub fn two_dimensional_positions(
    dim: usize,
    height: usize,
    width: usize,
    max_period: f32,
) -> Array3<f32> {
    let half = dim / 2;
    let pairs = half / 2;
    let log_period = max_period.ln() / half as f32;
    let div_term: Vec<f32> = (0..pairs)
        .map(|j| (-((2 * j) as f32) * log_period).exp())
        .collect();

    let mut out = Array3::<f32>::zeros((1, width * height, dim));
    for t in 0..width {
        for f in 0..height {
            let token = t * height + f;
            for j in 0..pairs {
                let w = t as f32 * div_term[j];
                let h = f as f32 * div_term[j];
                out[[0, token, 2 * j]] = w.sin();
                out[[0, token, 2 * j + 1]] = w.cos();
                out[[0, token, half + 2 * j]] = h.sin();
                out[[0, token, half + 2 * j + 1]] = h.cos();
            }
        }
    }
    out
}

/// `create_sin_embedding`: `cos` on the first half of the channels, `sin` on the
/// second, both against `max_period ^ (i / (dim/2 - 1))`.
pub fn one_dimensional_positions(length: usize, dim: usize, max_period: f32) -> Array3<f32> {
    let half = dim / 2;
    let mut out = Array3::<f32>::zeros((1, length, dim));
    for t in 0..length {
        for i in 0..half {
            let exponent = i as f32 / (half as f32 - 1.0);
            let phase = t as f32 / max_period.powf(exponent);
            out[[0, t, i]] = phase.cos();
            out[[0, t, half + i]] = phase.sin();
        }
    }
    out
}

/// `MyGroupNorm(1, dim)` applied to a `(b, t, c)` tensor, in place.
///
/// The reference normalises the *transposed* tensor — `x.permute(0, 2, 1)` —
/// with a single group, so the statistics cover every element of every batch in
/// either layout and only the accumulation order changes. Computing them here
/// instead of after a transpose drops both transposed copies: with the round
/// trip this stage was 54.8 ms of a 982 ms segment (the ten transformer layers;
/// ~2 GB/s effective) for three sweeps over a 5.5 MB tensor and two
/// cache-hostile transposes.
///
/// `groups > 1` does split along the channel axis, which is the last one here,
/// so that case keeps the transposing path. 32 K element blocks combined in
/// order keep the f64 reduction from depending on how it was scheduled.
fn group_norm_positions(mut x: Array3<f32>, norm: &GroupNormW) -> Result<Array3<f32>> {
    if norm.groups != 1 || std::env::var("DEMUCS_GN_TRANSPOSE").is_ok_and(|v| v == "1") {
        let transposed = x
            .view()
            .permuted_axes([0, 2, 1])
            .as_standard_layout()
            .to_owned();
        let out = group_norm(&transposed, norm.groups, &norm.weight, &norm.bias)?;
        return Ok(out.permuted_axes([0, 2, 1]).as_standard_layout().to_owned());
    }
    let (_b, t, c) = x.dim();
    if norm.weight.len() != c || norm.bias.len() != c {
        return Err(Error::Shape(format!(
            "group_norm_positions: {c} channels, {} affine entries",
            norm.weight.len()
        )));
    }
    let plane = t * c;
    let flat = x
        .as_slice_mut()
        .ok_or_else(|| Error::Shape("group_norm_positions needs a standard-layout tensor".into()))?;
    flat.par_chunks_mut(plane).for_each(|batch| {
        let count = batch.len() as f64;
        let partials: Vec<(f64, f64)> = batch
            .par_chunks(32768)
            .map(|block| {
                let (mut sum, mut sum_sq) = (0.0f64, 0.0f64);
                for value in block {
                    let v = *value as f64;
                    sum += v;
                    sum_sq += v * v;
                }
                (sum, sum_sq)
            })
            .collect();
        let (mut sum, mut sum_sq) = (0.0f64, 0.0f64);
        for (block_sum, block_sq) in partials {
            sum += block_sum;
            sum_sq += block_sq;
        }
        let mean = sum / count;
        let variance = (sum_sq / count - mean * mean).max(0.0);
        let inv = 1.0 / ((variance + 1e-5).sqrt() as f32);
        let mean = mean as f32;
        // Same expressions as `group_norm`'s affine, so the rounding matches.
        let scale: Vec<f32> = norm.weight.iter().map(|weight| inv * weight).collect();
        let shift: Vec<f32> = norm
            .bias
            .iter()
            .zip(scale.iter())
            .map(|(bias, scale)| bias - mean * scale)
            .collect();
        for row in batch.chunks_mut(c) {
            for (value, (gain, offset)) in row.iter_mut().zip(scale.iter().zip(shift.iter())) {
                *value = *value * gain + offset;
            }
        }
    });
    Ok(x)
}

/// `nn.MultiheadAttention` on `(b, n, c)` tensors with `batch_first=True`.
fn multi_head_attention(
    layer: &TransformerLayerW,
    query: &Array3<f32>,
    key: &Array3<f32>,
    value: &Array3<f32>,
    heads: usize,
    name: Option<&str>,
    trace: &mut dyn TraceSink,
) -> Result<Array3<f32>> {
    let (batch, n_q, dim) = query.dim();
    let n_k = key.dim().1;
    let d_head = dim / heads;
    let scale = 1.0 / (d_head as f32).sqrt();

    // `weight` is PyTorch's `(out_features, in_features)`; `Linear::new`
    // transposes it into the `(in, out)` matrix the matmul wants.
    let project = |weight: &[f32], bias: &[f32], input: &Array3<f32>, n: usize| -> Result<Array2<f32>> {
        let flat = input
            .view()
            .into_shape_with_order((batch * n, dim))
            .expect("contiguous view")
            .to_owned();
        let w = ndarray::ArrayView2::from_shape((dim, dim), weight).expect("projection");
        let b = ndarray::ArrayView1::from_shape(dim, bias).expect("projection bias");
        let linear = crate::ops::Linear::new(w, Some(b));
        Ok(linear.forward(&flat))
    };
    // Keep the projections flat and gather the head rows out of the
    // (batch, n, heads, d_head)-logical layout directly: the extra permute pass
    // would add a full read and write of every projection.
    let mut qkv: Vec<Array2<f32>> = Vec::with_capacity(3);
    {
        let _s = profile::scope("transformer.attn.qkv");
        for (weight, bias, input, n) in [
            (&layer.q_weight, &layer.q_bias, query, n_q),
            (&layer.k_weight, &layer.k_bias, key, n_k),
            (&layer.v_weight, &layer.v_bias, value, n_k),
        ] {
            qkv.push(project(weight, bias, input, n)?);
        }
    }
    if let Some(nm) = name {
        if trace.wants(&format!("{nm}.attn.q")) {
            trace.record(&format!("{nm}.attn.q"), qkv[0].view().into_dyn());
            trace.record(&format!("{nm}.attn.k"), qkv[1].view().into_dyn());
            trace.record(&format!("{nm}.attn.v"), qkv[2].view().into_dyn());
        }
    }
    let (q_flat, k_flat, v_flat) = (
        qkv[0].as_slice().expect("standard layout"),
        qkv[1].as_slice().expect("standard layout"),
        qkv[2].as_slice().expect("standard layout"),
    );

    let _heads = profile::scope("transformer.attn.heads");
    let mut context = uninit_array((batch * heads, n_q, d_head));
    {
        let dst = context.as_slice_mut().expect("standard layout");
        let chunk = n_q * d_head;
        // Each (batch, head) owns one disjoint chunk of the output, so the
        // chunks can be written in parallel with no synchronisation.
        //
        // Throttling how many heads are in flight at once was tried here — every
        // `(batch, head)` pair on the 2688-token streams owns a 28.9 MB score
        // matrix that the softmax sweeps five times and the AV product then
        // reads, so eight at once is 231 MB of live data against a 36 MB L3 —
        // and it loses: with `DEMUCS_ATTN_HEADS=1` (run one head at a time) the
        // heads stage is 232.0 ms against 212.6 for eight, three runs each,
        // segments 1013.6 against 963.4. The parallelism the score block is
        // computed with is worth more than any residency the throttle buys.
        dst.par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(bh, slot)| {
                let (b, h) = (bh / heads, bh % heads);
                // The flat projection is (b*n, h*dh) row-major, so the head's
                // rows are d_head-wide runs h*d_head apart.
                let gather = |flat: &[f32], base: usize, n: usize| {
                    let mut head = uninit_array((n, d_head));
                    let out = head.as_slice_mut().expect("standard layout");
                    for (i, row) in out.chunks_mut(d_head).enumerate() {
                        let src = base + i * heads * d_head + h * d_head;
                        row.copy_from_slice(&flat[src..src + d_head]);
                    }
                    head
                };
                let q_head = gather(q_flat, b * n_q * dim, n_q);
                let k_head = gather(k_flat, b * n_k * dim, n_k);
                let v_head = gather(v_flat, b * n_k * dim, n_k);
                let out = crate::demucs::ops::attention_head(&q_head, &k_head, &v_head, scale, None);
                slot.copy_from_slice(out.as_slice().expect("standard layout"));
            });
    }

    let _project = profile::scope("transformer.attn.proj");
    // context is (batch, heads, n_q, d_head)-logical; merging the heads back is
    // a straight copy to (batch, n_q, dim).
    let mut merged = uninit_array((batch, n_q, dim));
    {
        let _merge = profile::scope("transformer.attn.merge");
        let src = context.as_slice().expect("standard layout");
        let dst = merged.as_slice_mut().expect("standard layout");
        dst.par_chunks_mut(n_q * dim)
            .enumerate()
            .for_each(|(b, batch_out)| {
                for (i, token) in batch_out.chunks_mut(dim).enumerate() {
                    for h in 0..heads {
                        let s = (b * heads + h) * n_q * d_head + i * d_head;
                        token[h * d_head..(h + 1) * d_head].copy_from_slice(&src[s..s + d_head]);
                    }
                }
            });
    }
    Ok(linear_3d(&layer.out_proj, &merged))
}

/// A `MyTransformerEncoderLayer` step (self attention).
fn classic_layer_forward(
    layer: &TransformerLayerW,
    x: &Array3<f32>,
    heads: usize,
    name: &str,
    trace: &mut dyn TraceSink,
) -> Result<Array3<f32>> {
    let normed = {
        let _s = profile::scope("transformer.norm1");
        normalise_rows(x, &layer.norm1)?
    };
    if trace.wants(&format!("{name}.norm1")) {
        trace.record(&format!("{name}.norm1"), normed.view().into_dyn());
    }
    let attended = {
        let _scope = profile::scope("transformer.attn");
        multi_head_attention(layer, &normed, &normed, &normed, heads, Some(name), &mut *trace)?
    };
    if trace.wants(&format!("{name}.self_attn")) {
        trace.record(&format!("{name}.self_attn#0"), attended.view().into_dyn());
    }
    let mut out = {
        let _s = profile::scope("transformer.resid1");
        x + &scaled_channels(&attended, &layer.gamma1)
    };

    let normed = {
        let _s = profile::scope("transformer.norm2");
        normalise_rows(&out, &layer.norm2)?
    };
    if trace.wants(&format!("{name}.norm2")) {
        trace.record(&format!("{name}.norm2"), normed.view().into_dyn());
    }
    let fed = feed_forward(layer, &normed, name, trace)?;
    out = {
        let _s = profile::scope("transformer.resid2");
        &out + &scaled_channels(&fed, &layer.gamma2)
    };

    let out = {
        let _s = profile::scope("transformer.norm_out");
        group_norm_positions(out, &layer.norm_out)?
    };
    if trace.wants(&format!("{name}.norm_out")) {
        trace.record(&format!("{name}.norm_out"), out.view().into_dyn());
    }
    if trace.wants(name) {
        trace.record(name, out.view().into_dyn());
    }
    Ok(out)
}

/// A `CrossTransformerEncoderLayer` step: `q` attends to `k`.
fn cross_layer_forward(
    layer: &TransformerLayerW,
    q: &Array3<f32>,
    k: &Array3<f32>,
    heads: usize,
    name: &str,
    trace: &mut dyn TraceSink,
) -> Result<Array3<f32>> {
    let normed_q = {
        let _s = profile::scope("transformer.norm1");
        normalise_rows(q, &layer.norm1)?
    };
    let normed_k = {
        let _s = profile::scope("transformer.norm2");
        normalise_rows(k, &layer.norm2)?
    };
    if trace.wants(&format!("{name}.norm1")) {
        trace.record(&format!("{name}.norm1"), normed_q.view().into_dyn());
    }
    if trace.wants(&format!("{name}.norm2")) {
        trace.record(&format!("{name}.norm2"), normed_k.view().into_dyn());
    }
    let attended = {
        let _scope = profile::scope("transformer.attn");
        multi_head_attention(layer, &normed_q, &normed_k, &normed_k, heads, Some(name), &mut *trace)?
    };
    if trace.wants(&format!("{name}.cross_attn")) {
        trace.record(&format!("{name}.cross_attn#0"), attended.view().into_dyn());
    }
    let mut out = {
        let _s = profile::scope("transformer.resid1");
        q + &scaled_channels(&attended, &layer.gamma1)
    };

    let norm3 = layer.norm3.as_ref().ok_or_else(|| {
        Error::Shape(format!("{name}: a cross layer must carry norm3"))
    })?;
    let normed = {
        let _s = profile::scope("transformer.norm3");
        normalise_rows(&out, norm3)?
    };
    if trace.wants(&format!("{name}.norm3")) {
        trace.record(&format!("{name}.norm3"), normed.view().into_dyn());
    }
    let fed = feed_forward(layer, &normed, name, trace)?;
    out = {
        let _s = profile::scope("transformer.resid2");
        &out + &scaled_channels(&fed, &layer.gamma2)
    };

    let out = {
        let _s = profile::scope("transformer.norm_out");
        group_norm_positions(out, &layer.norm_out)?
    };
    if trace.wants(&format!("{name}.norm_out")) {
        trace.record(&format!("{name}.norm_out"), out.view().into_dyn());
    }
    if trace.wants(name) {
        trace.record(name, out.view().into_dyn());
    }
    Ok(out)
}

/// `linear2(gelu(linear1(x)))`.
fn feed_forward(
    layer: &TransformerLayerW,
    x: &Array3<f32>,
    name: &str,
    trace: &mut dyn TraceSink,
) -> Result<Array3<f32>> {
    let mut hidden = {
        let _scope = profile::scope("transformer.ffn");
        linear_3d(&layer.linear1, x)
    };
    if trace.wants(&format!("{name}.linear1")) {
        trace.record(&format!("{name}.linear1"), hidden.view().into_dyn());
    }
    {
        let _s = profile::scope("transformer.ffn.gelu");
        gelu_in_place(hidden.as_slice_mut().expect("standard layout"));
    }
    let out = {
        let _scope = profile::scope("transformer.ffn");
        linear_3d(&layer.linear2, &hidden)
    };
    if trace.wants(&format!("{name}.linear2")) {
        trace.record(&format!("{name}.linear2"), out.view().into_dyn());
    }
    Ok(out)
}

/// `LayerScale(channel_last=True)`: a per-channel scale of a `(b, t, c)` tensor.
fn scaled_channels(x: &Array3<f32>, gamma: &[f32]) -> Array3<f32> {
    let mut out = x.clone();
    let (b, t, c) = out.dim();
    let _ = (b, t);
    // One task per token: the scale runs along the channel axis.
    out.as_slice_mut()
        .expect("standard layout")
        .par_chunks_mut(c)
        .for_each(|token| {
            for (value, g) in token.iter_mut().zip(gamma.iter()) {
                *value *= *g;
            }
        });
    out
}

/// `x = (x - offset) * scale` over a whole tensor, in parallel.
fn affine_in_place<S: ndarray::DataMut<Elem = f32> + Sync, D: ndarray::Dimension>(
    mut x: ndarray::ArrayBase<S, D>,
    offset: f32,
    scale: f32,
) {
    x.as_slice_mut()
        .expect("standard layout")
        .par_iter_mut()
        .for_each(|value| *value = (*value - offset) * scale);
}

/// The number of `f32` values the loaded weights hold.
pub fn parameter_count(weights: &HtdemucsWeights) -> usize {
    weights.parameter_count()
}
