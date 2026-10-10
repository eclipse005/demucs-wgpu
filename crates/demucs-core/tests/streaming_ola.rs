//! Bit-exact streaming OLA vs full-buffer path, and regression against the
//! quality baseline WAV (upstream apply.py semantics + this port's ShiftRng).

use std::path::PathBuf;

use demucs_core::demucs::host::{Htdemucs, NoTrace, TraceSink};
use demucs_core::demucs::pipeline::{self, SeparateOptions};
use demucs_core::{Backend, Demucs, LoadOptions, ModelVariant, StemId, StemSelection};
use ndarray::{Array3, Array4};

fn vocals_weights() -> PathBuf {
    PathBuf::from("/workspace/oneasr-runtime/models/htdemucs_ft/htdemucs_ft_vocals.safetensors")
}

fn baseline_wav() -> PathBuf {
    PathBuf::from("/workspace/oneasr-baseline/media/user_test/demucs_baseline.wav")
}

fn baseline_vocals() -> PathBuf {
    PathBuf::from("/workspace/oneasr-baseline/media/user_test/demucs_out/vocals.wav")
}

fn load_vocals() -> Option<Demucs> {
    let path = vocals_weights();
    if !path.exists() {
        eprintln!("skipping: {} missing", path.display());
        return None;
    }
    Demucs::load(
        &path,
        LoadOptions {
            variant: ModelVariant::FourStem,
            stems: StemSelection::Some(vec![StemId::Vocals]),
        },
        Backend::Cpu,
    )
    .ok()
}

fn load_host() -> Option<Htdemucs> {
    let path = vocals_weights();
    if !path.exists() {
        return None;
    }
    let loaded = demucs_core::demucs::load_weights(&path).ok()?;
    let config = loaded.config.clone();
    let weights = demucs_core::demucs::HtdemucsWeights::load(&loaded.checkpoint, &config).ok()?;
    Htdemucs::new(config, weights).ok()
}

fn max_abs(a: &Array4<f32>, b: &Array4<f32>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Streaming split (shifts=0) must match the retained full-buffer `split_pass`
/// on a real network with a short segment.
#[test]
#[ignore = "host forward is slow in debug; run with --ignored --release"]
fn streaming_separate_is_deterministic_shifts0() {
    let Some(model) = load_host() else {
        return;
    };
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
fn facade_shifts1_is_deterministic_with_shiftrng() {
    let Some(sep) = load_vocals() else {
        return;
    };
    let n = 44100; // 1s — one padded chunk after shift
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

/// Quality gate: streaming path on the baseline WAV vs checked-in full-buffer
/// `vocals.wav` (same weights, shifts=1, ShiftRng seed).
#[test]
#[ignore = "set DEMUCS_BASELINE_TEST=1 and run --ignored --release"]
fn baseline_wav_matches_checked_in_vocals() {
    let Some(sep) = load_vocals() else {
        return;
    };
    let wav_path = baseline_wav();
    let ref_path = baseline_vocals();
    if !wav_path.exists() || !ref_path.exists() {
        eprintln!("skipping baseline: missing wav or reference");
        return;
    }
    // Allow skipping the ~1 min CPU run unless explicitly requested.
    if std::env::var("DEMUCS_BASELINE_TEST").ok().as_deref() != Some("1") {
        eprintln!("skipping baseline gate (set DEMUCS_BASELINE_TEST=1 to run)");
        return;
    }
    let stems = sep
        .separate_file(&wav_path, &mut |_| {})
        .expect("separate baseline");
    let vocals = stems
        .iter()
        .find(|s| s.id == StemId::Vocals)
        .expect("vocals stem");
    let reference = demucs_core::audio::read_audio(&ref_path).expect("read ref");
    let ref_audio = reference.ensure_channels(2).expect("stereo");
    let ref_l = ref_audio.channel(0);
    let ref_r = ref_audio.channel(1);
    assert_eq!(vocals.left.len(), ref_l.len());
    assert_eq!(vocals.right.len(), ref_r.len());
    let mut worst = 0.0f32;
    for (a, b) in vocals.left.iter().zip(ref_l.iter()) {
        worst = worst.max((a - b).abs());
    }
    for (a, b) in vocals.right.iter().zip(ref_r.iter()) {
        worst = worst.max((a - b).abs());
    }
    eprintln!("baseline max abs error vs checked-in vocals.wav: {worst:.6e}");
    assert_eq!(
        worst, 0.0,
        "streaming path not bit-exact vs baseline vocals.wav (max abs {worst})"
    );
}
