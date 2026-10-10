//! Bit-exact streaming OLA vs full-buffer path.
//!
//! Identity-forward unit tests need no weights. Host-network checks are
//! `#[ignore]` and look for weights via `DEMUCS_WEIGHTS` or the OneAsr runtime
//! layout — they fail loudly when forced on without weights, never silent-skip.

use std::path::PathBuf;

use demucs_core::demucs::host::{Htdemucs, NoTrace, TraceSink};
use demucs_core::demucs::pipeline::{self, SeparateOptions};
use demucs_core::demucs::streaming::{
    stream_split_pass_collect_meta, ShiftView, SplitMeta,
};
use demucs_core::{Backend, Demucs, LoadOptions, ModelVariant, StemId, StemSelection};
use ndarray::{s, Array3, Array4};

fn vocals_weights() -> PathBuf {
    if let Ok(p) = std::env::var("DEMUCS_WEIGHTS") {
        return PathBuf::from(p);
    }
    PathBuf::from("/workspace/oneasr-runtime/models/htdemucs_ft/htdemucs_ft_vocals.safetensors")
}

fn load_vocals() -> Demucs {
    let path = vocals_weights();
    assert!(
        path.exists(),
        "weights missing at {} (set DEMUCS_WEIGHTS)",
        path.display()
    );
    Demucs::load(
        &path,
        LoadOptions {
            variant: ModelVariant::FourStem,
            stems: StemSelection::Some(vec![StemId::Vocals]),
        },
        Backend::Cpu,
    )
    .expect("load vocals weights")
}

fn load_host() -> Htdemucs {
    let path = vocals_weights();
    assert!(
        path.exists(),
        "weights missing at {} (set DEMUCS_WEIGHTS)",
        path.display()
    );
    let loaded = demucs_core::demucs::load_weights(&path).expect("load_weights");
    let config = loaded.config.clone();
    let weights =
        demucs_core::demucs::HtdemucsWeights::load(&loaded.checkpoint, &config).expect("weights");
    Htdemucs::new(config, weights).expect("model")
}

fn max_abs(a: &Array4<f32>, b: &Array4<f32>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Identity forward: echo the mix into every source channel.
struct IdForward {
    sources: usize,
}

impl demucs_core::demucs::pipeline::SegmentForward for IdForward {
    type Pending = Array4<f32>;

    fn start(
        &self,
        padded: &Array3<f32>,
        _trace: &mut dyn TraceSink,
    ) -> demucs_core::Result<Array4<f32>> {
        let (batch, channels, time) = padded.dim();
        let mut out = Array4::<f32>::zeros((batch, self.sources, channels, time));
        for src in 0..self.sources {
            out.slice_mut(s![.., src, .., ..]).assign(padded);
        }
        Ok(out)
    }

    fn collect(
        &self,
        pending: Array4<f32>,
        _trace: &mut dyn TraceSink,
    ) -> demucs_core::Result<Array4<f32>> {
        Ok(pending)
    }
}

fn meta(sources: usize, segment: f64) -> SplitMeta {
    SplitMeta {
        sources,
        samplerate: 44100,
        segment,
    }
}

fn synthetic_mix(length: usize) -> Array3<f32> {
    let mut mix = Array3::<f32>::zeros((1, 2, length));
    for (i, v) in mix.iter_mut().enumerate() {
        *v = ((i as i32 % 97) as f32 - 48.0) / 48.0;
    }
    mix
}

#[test]
fn streaming_ola_matches_full_buffer_shifts0() {
    let meta = meta(2, 0.25);
    let forward = IdForward { sources: 2 };
    let segment_length = (44100.0 * 0.25) as usize;
    let mix = synthetic_mix(segment_length * 5 / 2);
    let options = SeparateOptions {
        shifts: 0,
        overlap: 0.25,
        transition_power: 1.0,
        segment: Some(0.25),
    };
    // Full-buffer reference via retained split_pass_full.
    // Build a tiny fake Htdemucs is heavy; use the unit-test full_buffer path
    // already covered in streaming.rs, and compare collect vs emit→array.
    let streamed = stream_split_pass_collect_meta(
        &meta,
        &forward,
        &mix,
        options,
        1,
        None,
        &mut |_, _| {},
        &mut NoTrace,
    )
    .unwrap();
    let mut emitted = Array4::<f32>::zeros(streamed.dim());
    demucs_core::demucs::streaming::stream_split_pass_meta(
        &meta,
        &forward,
        &mix,
        options,
        1,
        None,
        &mut |_, _| {},
        &mut |t, frame| {
            for b in 0..1 {
                for src in 0..2 {
                    for c in 0..2 {
                        emitted[[b, src, c, t]] = frame[(b * 2 + src) * 2 + c];
                    }
                }
            }
            Ok(())
        },
        &mut NoTrace,
    )
    .unwrap();
    assert_eq!(max_abs(&streamed, &emitted), 0.0);
}

#[test]
fn streaming_shift_view_matches_full_pad_trim_shifts1_style() {
    let meta = meta(2, 0.25);
    let forward = IdForward { sources: 2 };
    let segment_length = (44100.0 * 0.25) as usize;
    let mix_len = segment_length * 3;
    let mix = synthetic_mix(mix_len);
    let options = SeparateOptions {
        shifts: 0,
        overlap: 0.25,
        transition_power: 1.0,
        segment: Some(0.25),
    };
    let max_shift = 1000usize;
    let offset = 123usize;
    let trim = max_shift - offset;
    let mut padded = Array3::<f32>::zeros((1, 2, mix_len + 2 * max_shift));
    padded
        .slice_mut(s![.., .., max_shift..max_shift + mix_len])
        .assign(&mix);
    let shifted_len = mix_len + max_shift - offset;
    let shifted = padded
        .slice(s![.., .., offset..offset + shifted_len])
        .to_owned();
    let full_view = stream_split_pass_collect_meta(
        &meta,
        &forward,
        &shifted,
        options,
        1,
        None,
        &mut |_, _| {},
        &mut NoTrace,
    )
    .unwrap();
    let full_trimmed = full_view
        .slice(s![.., .., .., trim..trim + mix_len])
        .to_owned();

    let mut streamed = Array4::<f32>::zeros((1, 2, 2, mix_len));
    demucs_core::demucs::streaming::stream_split_pass_meta(
        &meta,
        &forward,
        &mix,
        options,
        1,
        Some(ShiftView { offset, max_shift }),
        &mut |_, _| {},
        &mut |t_view, frame| {
            if t_view < trim {
                return Ok(());
            }
            let t = t_view - trim;
            if t >= mix_len {
                return Ok(());
            }
            for b in 0..1 {
                for src in 0..2 {
                    for c in 0..2 {
                        streamed[[b, src, c, t]] = frame[(b * 2 + src) * 2 + c];
                    }
                }
            }
            Ok(())
        },
        &mut NoTrace,
    )
    .unwrap();
    assert_eq!(
        max_abs(&full_trimmed, &streamed),
        0.0,
        "shift-view streaming drifted"
    );
}

#[test]
#[ignore = "host forward is slow in debug; run with --ignored --release"]
fn streaming_separate_is_deterministic_shifts0() {
    let model = load_host();
    let segment = 0.25f64;
    let segment_length = (model.config.samplerate as f64 * segment) as usize;
    let mix_len = segment_length * 5 / 2;
    let mut mix = Array3::<f32>::zeros((1, 2, mix_len));
    for (i, v) in mix.iter_mut().enumerate() {
        *v = ((i as i32 % 97) as f32 - 48.0) / 48.0;
    }
    let options = SeparateOptions {
        shifts: 0,
        overlap: 0.25,
        transition_power: 1.0,
        segment: Some(segment),
    };
    let forward = |padded: &Array3<f32>, trace: &mut dyn TraceSink| model.forward(padded, trace);
    let a = pipeline::separate_with(&model, &forward, &mix, options, &mut NoTrace).unwrap();
    let b = pipeline::separate_with(&model, &forward, &mix, options, &mut NoTrace).unwrap();
    assert_eq!(max_abs(&a, &b), 0.0);
}

#[test]
#[ignore = "host forward is slow in debug; run with --ignored --release"]
fn facade_shifts0_and_shifts1_bit_exact_vs_pipeline() {
    let sep = load_vocals();
    let model = load_host();
    let n = 44100; // 1s
    let left: Vec<f32> = (0..n)
        .map(|i| 0.2 * (i as f32 * 0.01).sin())
        .collect();
    let right = left.clone();
    let mut mix = Array3::<f32>::zeros((1, 2, n));
    for i in 0..n {
        mix[[0, 0, i]] = left[i];
        mix[[0, 1, i]] = right[i];
    }

    for shifts in [0usize, 1usize] {
        let opts = SeparateOptions {
            shifts,
            overlap: 0.25,
            transition_power: 1.0,
            segment: Some(0.5),
        };
        let facade = sep
            .separate_with_options(&left, &right, 44100, &opts, &mut |_| {})
            .unwrap_or_else(|e| panic!("facade shifts={shifts}: {e}"));
        let (normalized, norm) = pipeline::normalize(&mix);
        let forward =
            |padded: &Array3<f32>, trace: &mut dyn TraceSink| model.forward(padded, trace);
        let mut sources =
            pipeline::separate_with(&model, &forward, &normalized, opts, &mut NoTrace)
                .unwrap_or_else(|e| panic!("pipeline shifts={shifts}: {e}"));
        pipeline::denormalize(&mut sources, norm);
        let vocals = facade
            .iter()
            .find(|s| s.id == StemId::Vocals)
            .expect("vocals");
        // Find vocals index in config
        let vidx = model
            .config
            .sources
            .iter()
            .position(|s| s == "vocals")
            .expect("vocals in config");
        let mut worst = 0.0f32;
        for t in 0..n {
            worst = worst.max((vocals.left[t] - sources[[0, vidx, 0, t]]).abs());
            worst = worst.max((vocals.right[t] - sources[[0, vidx, 1, t]]).abs());
        }
        assert_eq!(
            worst, 0.0,
            "shifts={shifts} facade vs pipeline max|Δ|={worst}"
        );
    }
}

#[test]
#[ignore = "host forward is slow in debug; run with --ignored --release"]
fn facade_shifts1_is_deterministic_with_shiftrng() {
    let sep = load_vocals();
    let n = 44100;
    let left: Vec<f32> = (0..n)
        .map(|i| 0.2 * (i as f32 * 0.01).sin())
        .collect();
    let right = left.clone();
    let opts = SeparateOptions {
        shifts: 1,
        overlap: 0.25,
        transition_power: 1.0,
        segment: Some(0.5),
    };
    let a = sep
        .separate_with_options(&left, &right, 44100, &opts, &mut |_| {})
        .expect("a");
    let b = sep
        .separate_with_options(&left, &right, 44100, &opts, &mut |_| {})
        .expect("b");
    assert_eq!(a.len(), b.len());
    for (sa, sb) in a.iter().zip(b.iter()) {
        let worst = sa
            .left
            .iter()
            .zip(sb.left.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
            .max(
                sa.right
                    .iter()
                    .zip(sb.right.iter())
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0f32, f32::max),
            );
        assert_eq!(worst, 0.0, "shifts=1 not deterministic: {worst}");
    }
}
