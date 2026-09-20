//! Layer-by-layer alignment of the host `HTDemucs` forward against the reference
//! dump written by `tools/dump_demucs.py`.
//!
//! Every activation the reference records is matched by name against the Rust
//! trace, so a disagreement localises to the module that produced it. Run with
//! `--nocapture` to see the per-layer table.

use std::path::PathBuf;

use demucs_core::demucs::{Htdemucs, NoTrace, VecTrace};
use demucs_core::fixtures::compare;
use demucs_core::npy::read_npy;
use ndarray::Array3;

fn th_checkpoint() -> PathBuf {
    demucs_core::paths::htdemucs_checkpoint()
}

fn workspace() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn dump_dir(tag: &str) -> PathBuf {
    if let Ok(path) = std::env::var("DEMUCS_DUMP_DIR") {
        return PathBuf::from(path);
    }
    workspace().join("bench/ref_dump").join(tag)
}

fn load_model() -> Option<Htdemucs> {
    load_model_with_segment(None)
}

/// The reference dump records the segment it ran with; a short dump needs the
/// same value, otherwise `forward` pads the input back up to 7.8 s.
fn load_model_with_segment(segment_samples: Option<usize>) -> Option<Htdemucs> {
    let checkpoint = th_checkpoint();
    if !checkpoint.exists() {
        eprintln!("skipping: {} is not present", checkpoint.display());
        return None;
    }
    let loaded = demucs_core::demucs::load_weights(&checkpoint).ok()?;
    let mut config = loaded.config.clone();
    if let Some(samples) = segment_samples {
        config.segment = samples as f64 / config.samplerate as f64;
    }
    let weights = demucs_core::demucs::HtdemucsWeights::load(&loaded.checkpoint, &config).ok()?;
    Some(Htdemucs::new(config, weights).expect("model"))
}

/// Compares one traced activation against the reference and returns its SNR.
fn compare_entry(dump: &demucs_core::fixtures::ReferenceDump, trace: &VecTrace, name: &str) -> Option<f32> {
    let Some(entry) = dump.entry(name) else {
        eprintln!("  {name}: not in the dump");
        return None;
    };
    let reference = match dump.load_entry(entry) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("  {name}: cannot read {}: {error}", entry.file);
            return None;
        }
    };
    let Some((_, shape, values)) = trace.get(name) else {
        eprintln!("  {name}: not in the trace");
        return None;
    };
    if shape != &reference.shape {
        eprintln!(
            "  {name}: SHAPE MISMATCH rust {shape:?} vs reference {:?}",
            reference.shape
        );
        return Some(f32::NEG_INFINITY);
    }
    let result = compare(&reference.data, values).ok()?;
    Some(result.snr_db())
}

struct Report {
    worst: Vec<(String, f32)>,
    missing: Vec<String>,
    compared: usize,
}

fn run_alignment(tag: &str, min_snr: f32, expected_layers: usize) -> Option<Report> {
    let dir = dump_dir(tag);
    if !dir.join("manifest.json").exists() {
        eprintln!("skipping: no reference dump at {}", dir.display());
        return None;
    }
    let dump = demucs_core::fixtures::ReferenceDump::open(&dir).expect("open dump");
    let segment_length = dump.meta_usize("segment_length").expect("segment_length meta");
    let model = load_model_with_segment(Some(segment_length))?;

    let input = read_npy(dir.join("input.npy")).expect("input.npy");
    assert_eq!(input.shape.len(), 2, "input.npy is (channels, samples)");
    let input = Array3::from_shape_vec(
        (1, input.shape[0], input.shape[1]),
        input.data,
    )
    .expect("input shape");

    let mut trace = VecTrace::default();
    let started = std::time::Instant::now();
    let out = model.forward(&input, &mut trace).expect("forward");
    if let Ok(path) = std::env::var("DEMUCS_RUST_DUMP") {
        write_dump(std::path::Path::new(&path), &trace);
    }
    println!(
        "forward produced {:?} in {:.2}s with {} traced activations",
        out.shape(),
        started.elapsed().as_secs_f64(),
        trace.entries.len()
    );

    if std::env::var("DEMUCS_TRACE_NAMES").is_ok() {
        for name in trace.names() {
            println!("  trace: {name}");
        }
    }
    let mut report = Report {
        worst: Vec::new(),
        missing: Vec::new(),
        compared: 0,
    };
    let verbose = std::env::var("DEMUCS_TABLE").is_ok();
    for entry in &dump.manifest.tensors {
        match compare_entry(&dump, &trace, &entry.name) {
            Some(snr) => {
                report.compared += 1;
                if verbose {
                    println!("  {snr:9.2} dB  {}", entry.name);
                }
                if snr < min_snr {
                    report.worst.push((entry.name.clone(), snr));
                }
            }
            None => report.missing.push(entry.name.clone()),
        }
    }
    report.worst.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    println!(
        "{}: compared {} activations, {} below {min_snr} dB, {} not traced",
        tag,
        report.compared,
        report.worst.len(),
        report.missing.len()
    );
    let show = report.worst.len().min(40);
    let worst_sorted: Vec<(String, f32)> = report.worst[..show].to_vec();
    for (name, snr) in &worst_sorted {
        println!("  {snr:8.2} dB  {name}");
    }
    let _ = expected_layers;
    Some(report)
}

/// Writes the Rust trace as `.npy` files, so a mismatch can be dissected with
/// NumPy exactly like the reference dump.
fn write_dump(dir: &std::path::Path, trace: &VecTrace) {
    std::fs::create_dir_all(dir).expect("create dump dir");
    for (name, shape, values) in &trace.entries {
        let safe = name.replace('.', "_").replace('#', "_").replace('/', "_");
        demucs_core::npy::write_npy(dir.join(format!("{safe}.npy")), shape, values)
            .expect("write npy");
    }
    println!("wrote {} rust activations to {}", trace.entries.len(), dir.display());
}

#[test]
fn the_host_forward_matches_the_reference_layer_by_layer() {
    let Some(report) = run_alignment("short", 60.0, 0) else {
        return;
    };
    assert!(
        report.missing.is_empty(),
        "the trace is missing {} activations: {:?}",
        report.missing.len(),
        &report.missing[..report.missing.len().min(20)]
    );
    assert!(
        report.compared > 100,
        "only {} activations were compared",
        report.compared
    );
    assert!(
        report.worst.is_empty(),
        "{} activations are below 60 dB, worst: {:?}",
        report.worst.len(),
        &report.worst[..report.worst.len().min(10)]
    );
}

/// The same check on a full 7.8 s segment, which is what the pipeline actually
/// feeds the model. Slower, so it is behind an environment variable.
#[test]
fn the_host_forward_matches_the_reference_on_a_full_segment() {
    if std::env::var("DEMUCS_FULL_SEGMENT").is_err() {
        eprintln!("skipping: set DEMUCS_FULL_SEGMENT=1 to run the 7.8 s alignment");
        return;
    }
    let Some(report) = run_alignment("htdemucs", 60.0, 0) else {
        return;
    };
    assert!(
        report.worst.is_empty(),
        "{} activations are below 60 dB, worst: {:?}",
        report.worst.len(),
        &report.worst[..report.worst.len().min(10)]
    );
}

/// A cheap smoke test: the forward runs end to end and the output has the right
/// shape, without any reference dump.
#[test]
fn the_forward_runs_without_tracing() {
    let Some(model) = load_model() else {
        return;
    };
    let length = 44_100;
    let mut config = model.config.clone();
    let _ = &mut config;
    let input = Array3::<f32>::zeros((1, 2, length));
    let mut trace = NoTrace;
    let out = model.forward(&input, &mut trace).expect("forward");
    assert_eq!(out.shape(), &[1, 4, 2, length]);
}

/// Batch > 1 must work, and each element must match the single-element run: the
/// model's statistics are per batch element.
#[test]
fn batching_is_independent_per_sample() {
    let Some(model) = load_model() else {
        return;
    };
    let length = 44_100;
    let mut rng = 12345u32;
    let mut random = || {
        rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((rng >> 8) as f32 / 8_388_608.0) - 1.0
    };
    let mut single = Array3::<f32>::zeros((1, 2, length));
    for value in single.iter_mut() {
        *value = random() * 0.1;
    }
    let mut pair = Array3::<f32>::zeros((2, 2, length));
    pair.slice_mut(ndarray::s![0, .., ..]).assign(&single.slice(ndarray::s![0, .., ..]));
    for value in pair.slice_mut(ndarray::s![1, .., ..]).iter_mut() {
        *value = random() * 0.05;
    }

    let one = model.forward(&single, &mut NoTrace).expect("single");
    let two = model.forward(&pair, &mut NoTrace).expect("pair");
    assert_eq!(two.shape(), &[2, 4, 2, length]);
    let result = compare(
        one.as_slice().unwrap(),
        two.slice(ndarray::s![0, .., .., ..]).as_slice().unwrap(),
    )
    .unwrap();
    println!("batch-of-2 first element vs single run: {:.2} dB", result.snr_db());
    assert!(result.snr_db() > 100.0, "{:.2} dB", result.snr_db());
}
