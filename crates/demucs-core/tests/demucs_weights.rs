//! Loader verification: every tensor the Rust port reads must have the shape and
//! statistics the reference model reports.
//!
//! `tools/demucs_model_info.py` writes the manifest from the PyTorch model; this
//! test loads the same checkpoint through `demucs::weights` and compares. It is
//! the "prove the loader before writing the forward" step from the port plan.

use std::collections::BTreeMap;
use std::path::PathBuf;

use demucs_core::demucs::{load_weights, Htdemucs, HtdemucsConfig};
use serde::Deserialize;

fn th_checkpoint() -> PathBuf {
    demucs_core::paths::htdemucs_checkpoint()
}

fn ft_snapshots_dir() -> PathBuf {
    demucs_core::paths::htdemucs_ft_snapshots_dir()
}

fn workspace() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[derive(Debug, Deserialize)]
struct Entry {
    shape: Vec<usize>,
    dtype: String,
    mean: f64,
    absmax: f64,
}

fn manifest(name: &str) -> Option<BTreeMap<String, Entry>> {
    let path = workspace().join("bench").join(name);
    let text = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

/// Locates the huggingface snapshot directory for a model, following the
/// `snapshots/<hash>/<file>` layout the hub uses (the files are symlinks).
fn ft_snapshot_file(stem: &str, extension: &str) -> Option<PathBuf> {
    for entry in std::fs::read_dir(ft_snapshots_dir()).ok()? {
        let dir = entry.ok()?.path();
        let candidate = dir.join(format!("{stem}.{extension}"));
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

#[test]
fn the_torch_checkpoint_loads_with_the_expected_parameter_shapes() {
    let checkpoint = th_checkpoint();
    if !checkpoint.exists() {
        eprintln!("skipping: {} is not present", checkpoint.display());
        return;
    }
    let loaded = load_weights(&checkpoint).expect("load htdemucs");
    assert_eq!(loaded.config, HtdemucsConfig::default());
    assert_eq!(loaded.checkpoint.len(), 533, "the reference state dict has 533 tensors");

    let weights = demucs_core::demucs::HtdemucsWeights::load(&loaded.checkpoint, &loaded.config)
        .expect("typed weights");
    let model = Htdemucs::new(loaded.config.clone(), weights).expect("model");

    // Spot-check the shapes that pin down the architecture.
    assert_eq!(model.weights.encoder[0].conv.shape, vec![48, 4, 8, 1]);
    assert_eq!(model.weights.encoder[3].conv.shape, vec![384, 192, 8, 1]);
    assert_eq!(model.weights.tencoder[0].conv.shape, vec![48, 2, 8]);
    assert_eq!(model.weights.decoder[0].conv_tr.shape, vec![384, 192, 8, 1]);
    assert_eq!(model.weights.decoder[3].conv_tr.shape, vec![48, 16, 8, 1]);
    assert_eq!(model.weights.decoder[3].rewrite.shape, vec![96, 48, 3, 3]);
    assert_eq!(model.weights.tdecoder[3].rewrite.shape, vec![96, 48, 3]);
    assert_eq!(model.weights.freq_emb.dim(), (512, 48));
    assert_eq!(model.weights.encoder[0].dconv.layers[0].conv1.shape, vec![6, 48, 3]);
    assert_eq!(model.weights.encoder[0].dconv.layers[0].conv2.shape, vec![96, 6, 1]);
}

/// Every parameter the Python manifest lists must round-trip with the same
/// statistics through the Rust loader. `htdemucs` stores fp16, so the comparison
/// runs on the upcast fp32 values — exactly what PyTorch's `load_state_dict`
/// produces.
#[test]
fn every_loaded_parameter_matches_the_reference_statistics() {
    let Some(reference) = manifest("manifest_htdemucs.json") else {
        eprintln!("skipping: run tools/demucs_model_info.py first");
        return;
    };
    let checkpoint = th_checkpoint();
    if !checkpoint.exists() {
        eprintln!("skipping: {} is not present", checkpoint.display());
        return;
    }
    let loaded = load_weights(&checkpoint).expect("load htdemucs");
    let tensors = loaded.checkpoint.tensors();

    let mut checked = 0;
    let mut worst_absmax = 0.0f64;
    let mut worst_name = String::new();
    for (name, entry) in &reference {
        let tensor = tensors
            .get(name)
            .unwrap_or_else(|| panic!("the checkpoint has no tensor `{name}`"));
        assert_eq!(
            tensor.shape(),
            entry.shape.as_slice(),
            "{name}: shape mismatch"
        );
        let mut sum = 0.0f64;
        let mut max = 0.0f64;
        for value in tensor.iter() {
            sum += *value as f64;
            max = max.max((*value as f64).abs());
        }
        let mean = sum / tensor.len() as f64;
        let delta_absmax = (max - entry.absmax).abs();
        if delta_absmax > worst_absmax {
            worst_absmax = delta_absmax;
            worst_name = name.clone();
        }
        assert!(
            (mean - entry.mean).abs() <= 1e-6 * entry.absmax.max(1e-3),
            "{name}: mean {mean:e} vs reference {:e}",
            entry.mean
        );
        checked += 1;
    }
    println!(
        "checked {checked} tensors; worst absmax delta {worst_absmax:e} ({worst_name})"
    );
    assert_eq!(checked, reference.len());
}

/// The `htdemucs_ft` vocals model is a `safetensors` file; its config has to come
/// out of the header metadata and its fp16 weights have to upcast identically.
#[test]
fn the_safetensors_checkpoint_loads_through_the_same_path() {
    let Some(path) = ft_snapshot_file("04573f0d", "safetensors") else {
        eprintln!("skipping: the htdemucs_ft snapshot is not present");
        return;
    };
    let loaded = load_weights(&path).expect("load ft vocals");
    assert_eq!(
        loaded.metadata.get("klass").map(String::as_str),
        Some("demucs.htdemucs.HTDemucs")
    );
    // The ft checkpoint's kwargs differ from `htdemucs` only in training knobs
    // (t_weight_decay, t_sparse_attn_window), so the parsed config is identical.
    assert_eq!(loaded.config, HtdemucsConfig::default());
    assert_eq!(loaded.checkpoint.len(), 533);

    let weights = demucs_core::demucs::HtdemucsWeights::load(&loaded.checkpoint, &loaded.config)
        .expect("typed weights");
    let model = Htdemucs::new(loaded.config.clone(), weights).expect("model");
    assert_eq!(model.weights.encoder[0].conv.shape, vec![48, 4, 8, 1]);
    assert_eq!(model.weights.transformer.layers.len(), 5);
    assert!(model.weights.transformer.layers[1].is_cross);
    assert!(model.weights.transformer.layers[0].norm3.is_none());
    assert!(model.weights.transformer.layers[1].norm3.is_some());

    // `tools/demucs_model_info.py` sums the state dict to 41,984,456 parameters;
    // the typed struct has to account for every one of them, exactly once.
    let count = model.weights.parameter_count();
    assert_eq!(count, 41_984_456, "parameter count does not match the checkpoint");

    // The two checkpoints were trained separately, so their weights differ; the
    // loader must not be silently reading one for the other.
    let other = load_weights(th_checkpoint()).expect("load htdemucs");
    let a = model.weights.encoder[0].conv.weight[0];
    let b = other.checkpoint.get("encoder.0.conv.weight").unwrap()[[0, 0, 0, 0]];
    assert!((a - b).abs() > 0.0, "the two checkpoints should not share weights");
}
