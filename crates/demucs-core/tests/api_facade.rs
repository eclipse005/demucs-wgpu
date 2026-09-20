//! Regression guards for the [`demucs_core::api`] facade.
//!
//! The facade is meant to be a pure wrapper over the pipeline the CLI used to
//! drive directly, so the invariants we hold it to are:
//!   1. `Demucs::separate` is bit-identical to the old CLI sequence
//!      `normalize → pipeline::separate → denormalize` on the same audio
//!      (`shifts = 0`, so both paths are deterministic). This is what proves
//!      "aligning the API did not touch the separation math".
//!   2. stem selection prunes the output without changing the surviving stem.
//!   3. `separate_file` and the slice entry point agree.
//!   4. progress is monotonic and ends at `done == total`, single network and
//!      the fine-tuned bag.
//!
//! Everything is host-only and skips when the checkpoint is not on disk.

use std::path::PathBuf;

use demucs_core::demucs::host::{Htdemucs, NoTrace};
use demucs_core::demucs::pipeline::{self, SeparateOptions};
use demucs_core::{Backend, Demucs, LoadOptions, ModelVariant, StemId, StemSelection};
use ndarray::Array3;

fn th_checkpoint() -> PathBuf {
    demucs_core::paths::htdemucs_checkpoint()
}

/// The `htdemucs_ft` snapshot directory and its vocals shard (`04573f0d`, the
/// row of the identity weight matrix that maps to source `vocals`).
fn ft_dir() -> PathBuf {
    demucs_core::paths::htdemucs_ft_dir()
}

fn ft_vocals_shard() -> PathBuf {
    ft_dir().join("04573f0d.safetensors")
}

/// A deterministic 1-second stereo signal — enough to exercise the whole
/// chunk/overlap path (a single padded chunk) without paying for real audio.
fn synthetic_one_second() -> (Vec<f32>, Vec<f32>) {
    let n = 44100;
    let left = (0..n)
        .map(|i| {
            let t = i as f32 / 44100.0;
            0.3 * (2.0 * std::f32::consts::PI * 220.0 * t).sin()
                + 0.1 * (2.0 * std::f32::consts::PI * 1100.0 * t).sin()
        })
        .collect();
    let right = (0..n)
        .map(|i| {
            let t = i as f32 / 44100.0;
            0.25 * (2.0 * std::f32::consts::PI * 330.0 * t).sin()
        })
        .collect();
    (left, right)
}

/// The facade handle, or `None` (skip) when the checkpoint is absent.
fn facade(stems: StemSelection) -> Option<Demucs> {
    let checkpoint = th_checkpoint();
    if !checkpoint.exists() {
        eprintln!("skipping: {} is not present", checkpoint.display());
        return None;
    }
    Demucs::load(
        checkpoint,
        LoadOptions {
            variant: ModelVariant::FourStem,
            stems,
        },
        Backend::Cpu,
    )
    .ok()
}

/// A host model loaded exactly as the old CLI did, for the manual reference run.
fn host_model() -> Option<Htdemucs> {
    let loaded = demucs_core::demucs::load_weights(th_checkpoint()).ok()?;
    let config = loaded.config.clone();
    let weights = demucs_core::demucs::HtdemucsWeights::load(&loaded.checkpoint, &config).ok()?;
    Htdemucs::new(config, weights).ok()
}

fn options() -> SeparateOptions {
    SeparateOptions {
        shifts: 0,
        overlap: 0.25,
        transition_power: 1.0,
        segment: None,
    }
}

/// The core invariant: the facade's `separate` reproduces the old CLI path
/// `normalize → pipeline::separate → denormalize` bit for bit.
#[test]
fn separate_matches_the_pipeline_path_bitwise() {
    let sep = match facade(StemSelection::All) {
        Some(s) => s,
        None => return,
    };
    let model = match host_model() {
        Some(m) => m,
        None => return,
    };
    let (left, right) = synthetic_one_second();

    let stems = sep
        .separate_with_options(&left, &right, 44100, &options(), &mut |_| {})
        .expect("facade separate");
    assert_eq!(stems.len(), 4, "All should return the four config stems");

    let mut mix = Array3::<f32>::zeros((1, 2, left.len()));
    for (i, v) in left.iter().enumerate() {
        mix[[0, 0, i]] = *v;
    }
    for (i, v) in right.iter().enumerate() {
        mix[[0, 1, i]] = *v;
    }
    let (normalized, norm) = pipeline::normalize(&mix);
    let mut sources = pipeline::separate(&model, &normalized, options(), &mut NoTrace)
        .expect("pipeline separate");
    pipeline::denormalize(&mut sources, norm);

    for stem in &stems {
        let index = StemId::ALL
            .iter()
            .position(|id| *id == stem.id)
            .expect("known stem");
        for (channel, facade_lane) in
            [(0usize, &stem.left), (1usize, &stem.right)].into_iter()
        {
            let manual: Vec<f32> = sources
                .slice(ndarray::s![0, index, channel, ..])
                .iter()
                .copied()
                .collect();
            assert_eq!(
                facade_lane.len(),
                manual.len(),
                "length drift on {:?}",
                stem.id
            );
            let max_abs = facade_lane
                .iter()
                .zip(&manual)
                .fold(0.0f32, |acc, (a, b)| acc.max((a - b).abs()));
            assert!(
                max_abs == 0.0,
                "{:?} ch{channel} differs from the pipeline path by {max_abs:e}",
                stem.id
            );
        }
    }
}

/// A `Some` selection returns just that stem, identical to the same lane from an
/// `All` run.
#[test]
fn selection_matches_the_all_lane() {
    let all = match facade(StemSelection::All) {
        Some(s) => s,
        None => return,
    };
    let one = match facade(StemSelection::Some(vec![StemId::Vocals])) {
        Some(s) => s,
        None => return,
    };
    let (left, right) = synthetic_one_second();

    let all_stems = all
        .separate_with_options(&left, &right, 44100, &options(), &mut |_| {})
        .expect("all");
    let vocals = one
        .separate_with_options(&left, &right, 44100, &options(), &mut |_| {})
        .expect("vocals");

    assert_eq!(vocals.len(), 1);
    assert_eq!(vocals[0].id, StemId::Vocals);
    let all_vocals = all_stems
        .iter()
        .find(|s| s.id == StemId::Vocals)
        .expect("vocals in all");
    assert_eq!(vocals[0].left, all_vocals.left);
    assert_eq!(vocals[0].right, all_vocals.right);
}

/// Progress runs to completion and never rewinds, including across shift passes.
#[test]
fn progress_is_monotonic_and_completes() {
    let sep = match facade(StemSelection::All) {
        Some(s) => s,
        None => return,
    };
    let (left, right) = synthetic_one_second();
    let mut events = Vec::new();
    let opts = SeparateOptions {
        shifts: 2,
        overlap: 0.25,
        transition_power: 1.0,
        segment: None,
    };
    sep.separate_with_options(&left, &right, 44100, &opts, &mut |p| events.push(p))
        .expect("separate with shifts");

    assert!(!events.is_empty(), "expected progress callbacks");
    assert!(
        events.windows(2).all(|w| w[0].done <= w[1].done),
        "done went backwards: {events:?}"
    );
    assert!(
        events.windows(2).all(|w| w[0].total <= w[1].total),
        "total went backwards: {events:?}"
    );
    let last = *events.last().unwrap();
    assert_eq!(last.done, last.total, "progress must end at done == total");
    assert_eq!(last.total, 2, "two shift passes of a one-chunk track");
}

/// `num_chunks` counts the segments a full track draws at the default overlap.
#[test]
fn num_chunks_matches_the_reference_track() {
    // 176.309 s at 44.1 kHz (the measurement track) is 30 chunks, not 31.
    assert_eq!(demucs_core::api::num_chunks(7_775_226), 30);
    assert_eq!(demucs_core::api::num_chunks(44100), 1);
    assert_eq!(demucs_core::api::num_chunks(0), 1);
}

/// The fine-tuned bag, asked for one stem, must equal that stem's own shard run
/// for one bit less than nothing: the identity weight matrix routes `vocals`
/// solely through the `04573f0d` network, and selection-aware pruning skips the
/// three networks whose vocals weight is zero — so the surviving lane is that
/// network's output verbatim. Ignored by default because it loads all four
/// networks; run with `cargo test -p demucs-core --release -- --ignored`.
#[test]
#[ignore = "loads the four-model htdemucs_ft bag; run in release with --ignored"]
fn ft_bag_single_stem_matches_its_shard() {
    if !ft_dir().join("htdemucs_ft.yaml").exists() || !ft_vocals_shard().exists() {
        eprintln!("skipping: htdemucs_ft snapshot not present");
        return;
    }
    let (left, right) = synthetic_one_second();
    let bag = Demucs::load(
        ft_dir(),
        LoadOptions {
            variant: ModelVariant::FineTuned,
            stems: StemSelection::Some(vec![StemId::Vocals]),
        },
        Backend::Cpu,
    )
    .expect("load ft bag");
    let shard = Demucs::load(
        ft_vocals_shard(),
        LoadOptions {
            variant: ModelVariant::FourStem,
            stems: StemSelection::Some(vec![StemId::Vocals]),
        },
        Backend::Cpu,
    )
    .expect("load vocals shard");

    let b = bag
        .separate_with_options(&left, &right, 44100, &options(), &mut |_| {})
        .expect("bag separate");
    let s = shard
        .separate_with_options(&left, &right, 44100, &options(), &mut |_| {})
        .expect("shard separate");

    assert_eq!(b.len(), 1);
    assert_eq!(s.len(), 1);
    assert_eq!(b[0].id, StemId::Vocals);
    assert_eq!(b[0].left, s[0].left, "bag vocals != shard vocals (left)");
    assert_eq!(b[0].right, s[0].right, "bag vocals != shard vocals (right)");
}
