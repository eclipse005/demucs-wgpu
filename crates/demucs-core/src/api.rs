//! The library facade: load a model once, separate as many tracks as you like.
//!
//! This is the shape other Rust projects should code against — the CLI is a thin
//! wrapper over it. It mirrors the API of the sibling CUDA port
//! (`demucs-native-rs`):
//!
//! ```ignore
//! use demucs_core::{Backend, Demucs, LoadOptions, ModelVariant, StemSelection};
//!
//! let sep = Demucs::load(
//!     "htdemucs.safetensors",
//!     LoadOptions {
//!         variant: ModelVariant::FourStem,
//!         stems: StemSelection::All,
//!     },
//!     Backend::Auto,
//! )?;
//! let stems = sep.separate(&left, &right, 44100)?;
//! for stem in &stems {
//!     println!("{}: {} samples", stem.id.as_str(), stem.left.len());
//! }
//! ```
//!
//! `Demucs::load` accepts a checkpoint file (`.th` or `.safetensors`) for
//! [`ModelVariant::FourStem`], or a HuggingFace snapshot directory (the one with
//! `htdemucs_ft.yaml` in it) for [`ModelVariant::FineTuned`], which is the
//! four-model bag the reference runs as `htdemucs_ft`.

use std::path::Path;

use crate::demucs::host::{Htdemucs, TraceSink};
use crate::demucs::pipeline::{self, SeparateOptions};
use crate::demucs::weights::LoadedWeights;
use crate::demucs::host::NoTrace;
use crate::gpu::{DeviceSelector, Gpu};
use crate::gpu::htdemucs::GpuHtdemucsRunner;
use crate::{Error, Result};

/// Model hyperparameters of the HTDemucs family, reported for callers that want
/// to size buffers without walking the config.
pub const AUDIO_CHANNELS: usize = 2;
pub const SAMPLE_RATE: usize = 44100;
pub const N_FFT: usize = 4096;
pub const HOP_LENGTH: usize = 1024;
/// Training segment length in samples (= 39/5 * 44100).
pub const TRAINING_LENGTH: usize = 343980;

/// One of the four stems the HTDemucs family separates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StemId {
    Drums,
    Bass,
    Other,
    Vocals,
}

impl StemId {
    /// All four, in the order the model's config lists them.
    pub const ALL: [StemId; 4] = [StemId::Drums, StemId::Bass, StemId::Other, StemId::Vocals];

    pub fn as_str(&self) -> &'static str {
        match self {
            StemId::Drums => "drums",
            StemId::Bass => "bass",
            StemId::Other => "other",
            StemId::Vocals => "vocals",
        }
    }

    pub fn parse(name: &str) -> Option<StemId> {
        StemId::ALL.into_iter().find(|stem| stem.as_str() == name)
    }
}

impl std::fmt::Display for StemId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Which model variant to load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelVariant {
    /// The single `htdemucs` network (4 stems).
    FourStem,
    /// The `htdemucs_ft` bag: four fine-tuned networks, one per stem.
    FineTuned,
}

/// Static description of a variant.
#[derive(Debug, Clone, Copy)]
pub struct ModelInfo {
    /// The name the reference CLI uses (`-m`).
    pub id: &'static str,
    /// Stems it produces.
    pub sources: usize,
    /// Networks it runs per separation (`htdemucs_ft` runs four).
    pub bag: usize,
    /// What to pass to [`Demucs::load`]: a checkpoint file, or a directory for
    /// the bag.
    pub path_hint: &'static str,
}

impl ModelVariant {
    pub fn info(&self) -> &'static ModelInfo {
        match self {
            ModelVariant::FourStem => &ModelInfo {
                id: "htdemucs",
                sources: 4,
                bag: 1,
                path_hint: "checkpoint file (.th or .safetensors)",
            },
            ModelVariant::FineTuned => &ModelInfo {
                id: "htdemucs_ft",
                sources: 4,
                bag: 4,
                path_hint: "the HuggingFace snapshot directory (contains htdemucs_ft.yaml)",
            },
        }
    }

    /// The variant a model name maps to, like the reference's `-m`.
    pub fn from_name(name: &str) -> Option<ModelVariant> {
        match name {
            "htdemucs" => Some(ModelVariant::FourStem),
            "htdemucs_ft" => Some(ModelVariant::FineTuned),
            _ => None,
        }
    }
}

/// Compute backend selection.
///
/// `Auto` picks the best adapter wgpu can see and falls back to the host path
/// when no device is usable; `Cpu` forces the host path; `Gpu` takes an explicit
/// adapter selector (`demucs_core::gpu::DeviceSelector`, e.g. `vulkan:1` or
/// `Intel`) — `demucs adapters` lists what is available.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum Backend {
    Auto,
    Cpu,
    Gpu(DeviceSelector),
}

impl Backend {
    /// Short human label — useful for logs.
    pub fn tag(&self) -> &str {
        match self {
            Backend::Auto => "auto",
            Backend::Cpu => "cpu",
            Backend::Gpu(DeviceSelector::Auto) => "gpu:auto",
            Backend::Gpu(DeviceSelector::Runtime { api, .. }) => match api {
                wgpu::Backend::Vulkan => "gpu:vulkan",
                wgpu::Backend::Dx12 => "gpu:dx12",
                wgpu::Backend::Gl => "gpu:gl",
                wgpu::Backend::Metal => "gpu:metal",
                _ => "gpu",
            },
            Backend::Gpu(DeviceSelector::Name(_)) => "gpu:named",
        }
    }
}

/// Which stems to extract.
#[derive(Debug, Clone)]
pub enum StemSelection {
    /// All four.
    All,
    /// Only these — for the fine-tuned bag this also skips the networks that
    /// contribute nothing to the requested stems.
    Some(Vec<StemId>),
}

/// Options for [`Demucs::load`].
#[derive(Debug, Clone)]
pub struct LoadOptions {
    pub variant: ModelVariant,
    pub stems: StemSelection,
}

/// One separated stem, split into channels as the reference does.
#[derive(Debug, Clone)]
pub struct Stem {
    pub id: StemId,
    pub left: Vec<f32>,
    pub right: Vec<f32>,
}

/// Progress of one separation run.
///
/// One work unit is one `TRAINING_LENGTH` chunk of the (44.1 kHz) audio. For
/// bagged fine-tunes every network runs the whole track, so `total` counts the
/// chunks the handle actually processes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SeparationProgress {
    pub done: usize,
    pub total: usize,
}

impl SeparationProgress {
    /// Completion in `0.0..=1.0` (`1.0` when `total` is 0 to avoid div-by-zero).
    pub fn fraction(&self) -> f32 {
        if self.total == 0 {
            return 1.0;
        }
        (self.done as f32 / self.total as f32).clamp(0.0, 1.0)
    }

    /// Completion in percent, rounded down.
    pub fn percent(&self) -> u32 {
        (self.fraction() * 100.0) as u32
    }
}

/// Number of chunks for chunked inference over `n_samples` of audio at the
/// default overlap.
pub fn num_chunks(n_samples: usize) -> usize {
    let segment = TRAINING_LENGTH;
    let stride = segment * 3 / 4;
    if n_samples <= segment || stride == 0 {
        return 1;
    }
    (n_samples - segment).div_ceil(stride) + 1
}

/// One network plus its device runner, when a device backend was picked.
struct Engine {
    host: Htdemucs,
    runner: Option<GpuHtdemucsRunner>,
}

impl Engine {
    fn separate(
        &self,
        mix: &ndarray::Array3<f32>,
        options: SeparateOptions,
        on_progress: &mut dyn FnMut(usize, usize),
        trace: &mut dyn TraceSink,
    ) -> Result<ndarray::Array4<f32>> {
        match &self.runner {
            Some(runner) => pipeline::separate_gpu_progress(
                &self.host,
                runner,
                mix,
                options,
                on_progress,
                trace,
            ),
            None => pipeline::separate_with_progress(
                &self.host,
                &|padded, trace| self.host.forward(padded, trace),
                mix,
                options,
                on_progress,
                trace,
            ),
        }
    }
}

/// Top-level inference handle.
pub struct Demucs {
    inner: Inner,
    backend_tag: String,
    /// Stems the caller asked for; what [`Demucs::separate`] returns.
    selection: StemSelection,
}

enum Inner {
    Single(Engine),
    /// The bag: each network with its per-source weights, in config source
    /// order. The reference scales each network's output per source and divides
    /// by the sum of the weights; the shipped `htdemucs_ft.yaml` is the identity,
    /// so each source comes from exactly one network.
    Bag(Vec<(Engine, Vec<f32>)>),
}

impl Demucs {
    /// Loads a model from a checkpoint file (`.th` or `.safetensors`) for
    /// [`ModelVariant::FourStem`], or from a HuggingFace snapshot directory for
    /// [`ModelVariant::FineTuned`].
    pub fn load(path: impl AsRef<Path>, opts: LoadOptions, backend: Backend) -> Result<Self> {
        let path = path.as_ref();
        match opts.variant {
            ModelVariant::FourStem => {
                let host = load_host(path)?;
                let runner = make_runner(&host, &backend, path)?;
                let tag = describe(&backend, runner.as_ref());
                Ok(Self { inner: Inner::Single(Engine { host, runner }), backend_tag: tag, selection: opts.stems.clone() })
            }
            ModelVariant::FineTuned => {
                let bag = load_bag(path, &backend)?;
                let tag = describe(&backend, bag.first().and_then(|(engine, _)| engine.runner.as_ref()));
                Ok(Self { inner: Inner::Bag(bag), backend_tag: tag, selection: opts.stems.clone() })
            }
        }
    }

    /// Loads a single-network model straight from checkpoint bytes.
    pub fn from_bytes(bytes: &[u8], opts: LoadOptions, backend: Backend) -> Result<Self> {
        let loaded = crate::demucs::weights::load_weights_from_bytes(bytes, false)?;
        let host = host_from(loaded)?;
        let runner = make_runner(&host, &backend, Path::new("<bytes>"))?;
        let tag = describe(&backend, runner.as_ref());
        Ok(Self { inner: Inner::Single(Engine { host, runner }), backend_tag: tag, selection: opts.stems.clone() })
    }

    /// Separates a stereo track (`left`/`right` must have equal length) with the
    /// reference's defaults: one random shift, 25% overlap, the model's segment.
    pub fn separate(&self, left: &[f32], right: &[f32], sample_rate: u32) -> Result<Vec<Stem>> {
        self.separate_with_progress(left, right, sample_rate, &mut |_| {})
    }

    /// [`Self::separate`] while reporting chunk-level progress.
    pub fn separate_with_progress(
        &self,
        left: &[f32],
        right: &[f32],
        sample_rate: u32,
        on_progress: &mut dyn FnMut(SeparationProgress),
    ) -> Result<Vec<Stem>> {
        self.separate_with_options(left, right, sample_rate, &SeparateOptions::default(), on_progress)
    }

    /// [`Self::separate`] with explicit separation options.
    pub fn separate_with_options(
        &self,
        left: &[f32],
        right: &[f32],
        sample_rate: u32,
        options: &SeparateOptions,
        on_progress: &mut dyn FnMut(SeparationProgress),
    ) -> Result<Vec<Stem>> {
        self.run_channels(left, right, sample_rate, options, &mut NoTrace, on_progress)
    }

    /// [`Self::separate_with_options`] while recording activations into `trace`
    /// (the CLI's `--dump-trace`). The bag is not supported here: the four
    /// networks would write the same layer names into one sink and the dump
    /// would be ambiguous, so it returns an error instead.
    pub fn separate_with_trace(
        &self,
        left: &[f32],
        right: &[f32],
        sample_rate: u32,
        options: &SeparateOptions,
        trace: &mut dyn TraceSink,
        on_progress: &mut dyn FnMut(SeparationProgress),
    ) -> Result<Vec<Stem>> {
        if matches!(self.inner, Inner::Bag(_)) {
            return Err(Error::Shape(
                "--dump-trace is only supported for a single network (htdemucs), \
                 not the four-model htdemucs_ft bag"
                    .into(),
            ));
        }
        self.run_channels(left, right, sample_rate, options, trace, on_progress)
    }

    /// Shared body for the slice entry points: validate, build the mix, run the
    /// normalise → forward → denormalise round-trip, split into stems.
    fn run_channels(
        &self,
        left: &[f32],
        right: &[f32],
        sample_rate: u32,
        options: &SeparateOptions,
        trace: &mut dyn TraceSink,
        on_progress: &mut dyn FnMut(SeparationProgress),
    ) -> Result<Vec<Stem>> {
        if left.len() != right.len() {
            return Err(Error::Shape(format!(
                "left/right lengths differ ({} vs {})",
                left.len(),
                right.len()
            )));
        }
        let rate = match &self.inner {
            Inner::Single(engine) => engine.host.config.samplerate,
            Inner::Bag(bag) => bag[0].0.host.config.samplerate,
        };
        if sample_rate as usize != rate {
            return Err(Error::Shape(format!(
                "audio is at {sample_rate} Hz but the model wants {rate} Hz"
            )));
        }
        let samples = left.len();
        let mut mix = ndarray::Array3::<f32>::zeros((1, AUDIO_CHANNELS, samples));
        for (index, value) in left.iter().enumerate() {
            mix[[0, 0, index]] = *value;
        }
        for (index, value) in right.iter().enumerate() {
            mix[[0, 1, index]] = *value;
        }
        // `Htdemucs::forward` (host and device) works in the normalised domain,
        // exactly as the reference's `Separator.separate_tensor` does: normalise
        // the raw track, run the (bag of) networks, undo it on the way out. This
        // is the round-trip the CLI used to own; the facade does it so every
        // caller of `separate` gets real samples back, not normalised ones.
        let (normalized, norm) = pipeline::normalize(&mix);
        let mut sources = self.collect(&normalized, options, trace, on_progress)?;
        pipeline::denormalize(&mut sources, norm);
        Ok(self.stems(sources))
    }

    /// Separates a WAV file (mono is duplicated to stereo). The `--two-stems`-style
    /// companion tracks are the caller's business; this returns every requested
    /// stem.
    pub fn separate_file(
        &self,
        path: impl AsRef<Path>,
        on_progress: &mut dyn FnMut(SeparationProgress),
    ) -> Result<Vec<Stem>> {
        let path = path.as_ref();
        let rate = match &self.inner {
            Inner::Single(engine) => engine.host.config.samplerate,
            Inner::Bag(bag) => bag[0].0.host.config.samplerate,
        };
        let audio = crate::audio::read_audio(path)?;
        if audio.sample_rate as usize != rate {
            return Err(Error::Shape(format!(
                "{} is at {} Hz but the model wants {rate} Hz",
                path.display(),
                audio.sample_rate
            )));
        }
        let audio = audio.ensure_channels(AUDIO_CHANNELS)?;
        let left = audio.channel(0).to_vec();
        let right = audio.channel(1).to_vec();
        self.separate_with_options(
            &left,
            &right,
            audio.sample_rate,
            &SeparateOptions::default(),
            on_progress,
        )
    }

    /// Which backend this handle actually runs on.
    pub fn backend_tag(&self) -> &str {
        &self.backend_tag
    }

    /// The variant this handle was loaded as.
    pub fn variant(&self) -> ModelVariant {
        match &self.inner {
            Inner::Single(_) => ModelVariant::FourStem,
            Inner::Bag(_) => ModelVariant::FineTuned,
        }
    }

    /// The sample rate the handle expects (Hz). Feeding anything else to
    /// [`Self::separate`] is rejected.
    pub fn sample_rate(&self) -> usize {
        self.first_config().samplerate
    }

    /// The training segment length in seconds — what every chunk is padded to.
    pub fn segment(&self) -> f64 {
        self.first_config().segment
    }

    /// The source names in config order (drums, bass, other, vocals), which is
    /// the order [`Self::separate`] lays the returned stems' samples in.
    pub fn sources(&self) -> &[String] {
        &self.first_config().sources
    }

    /// Config of the first network; every network in a bag shares it.
    fn first_config(&self) -> &crate::demucs::config::HtdemucsConfig {
        match &self.inner {
            Inner::Single(engine) => &engine.host.config,
            Inner::Bag(bag) => &bag[0].0.host.config,
        }
    }

    /// Runs the (bag of) networks and returns the per-source output in config
    /// source order.
    fn collect(
        &self,
        mix: &ndarray::Array3<f32>,
        options: &SeparateOptions,
        trace: &mut dyn TraceSink,
        on_progress: &mut dyn FnMut(SeparationProgress),
    ) -> Result<ndarray::Array4<f32>> {
        match &self.inner {
            Inner::Single(engine) => engine.separate(mix, *options, &mut |done, total| {
                on_progress(SeparationProgress { done, total })
            }, trace),
            Inner::Bag(bag) => {
                let samples = mix.dim().2;
                let names = &bag[0].0.host.config.sources;
                let sources = names.len();
                // Which config sources the caller actually wants.
                let requested: Vec<bool> = names
                    .iter()
                    .map(|name| match &self.selection {
                        StemSelection::All => true,
                        StemSelection::Some(list) => {
                            StemId::parse(name).is_some_and(|id| list.contains(&id))
                        }
                    })
                    .collect();
                // A network earns a forward only if it carries nonzero weight on a
                // requested source; under the shipped identity weights a single-stem
                // request therefore runs exactly one network instead of four.
                let needed = |weights: &[f32]| {
                    weights
                        .iter()
                        .enumerate()
                        .any(|(s, w)| *w != 0.0 && requested.get(s).copied().unwrap_or(false))
                };
                // Progress spans every network that actually runs. All networks see
                // the same mix and options, so each contributes the same per-run
                // chunk count (`net_total`), learned from the first callback.
                let active = bag.iter().filter(|(_, w)| needed(w)).count().max(1);
                let mut estimates =
                    ndarray::Array4::<f32>::zeros((1, sources, AUDIO_CHANNELS, samples));
                let mut totals = vec![0.0f32; sources];
                let mut net_total = 0usize;
                let mut net_base = 0usize;
                for (engine, weights) in bag {
                    if !needed(weights) {
                        continue;
                    }
                    let out = engine.separate(mix, *options, &mut |done, total| {
                        if net_total == 0 {
                            net_total = total;
                        }
                        on_progress(SeparationProgress {
                            done: net_base + done,
                            total: net_total * active,
                        });
                    }, trace)?;
                    net_base += net_total;
                    for (source, weight) in weights.iter().enumerate() {
                        if *weight == 0.0
                            || !requested.get(source).copied().unwrap_or(false)
                        {
                            continue;
                        }
                        let mut lane = out.slice(ndarray::s![0, source, .., ..]).to_owned();
                        lane.mapv_inplace(|v| v * weight);
                        estimates
                            .slice_mut(ndarray::s![0, source, .., ..])
                            .zip_mut_with(&lane, |a, b| *a += b);
                        totals[source] += weight;
                    }
                }
                for (source, total) in totals.iter().enumerate() {
                    if !requested.get(source).copied().unwrap_or(false) {
                        continue;
                    }
                    if *total == 0.0 {
                        return Err(Error::Shape(format!(
                            "no network covers stem {}",
                            names[source]
                        )));
                    }
                    estimates
                        .slice_mut(ndarray::s![0, source, .., ..])
                        .mapv_inplace(|v| v / total);
                }
                Ok(estimates)
            }
        }
    }

    /// Picks the requested stems out of a per-source array.
    fn stems(&self, sources: ndarray::Array4<f32>) -> Vec<Stem> {
        let names = match &self.inner {
            Inner::Single(engine) => engine.host.config.sources.clone(),
            Inner::Bag(bag) => bag[0].0.host.config.sources.clone(),
        };
        let wanted = |id: StemId| match &self.selection {
            StemSelection::All => true,
            StemSelection::Some(list) => list.contains(&id),
        };
        names
            .iter()
            .enumerate()
            .filter_map(|(index, name)| {
                let id = StemId::parse(name)?;
                if !wanted(id) {
                    return None;
                }
                Some(Stem {
                    id,
                    left: sources.slice(ndarray::s![0, index, 0, ..]).to_vec(),
                    right: sources.slice(ndarray::s![0, index, 1, ..]).to_vec(),
                })
            })
            .collect()
    }
}

/// Loads a single-network host model from a checkpoint file.
fn load_host(path: &Path) -> Result<Htdemucs> {
    let loaded = crate::demucs::weights::load_weights(path)?;
    host_from(loaded)
}

fn host_from(loaded: crate::demucs::weights::LoadedWeights) -> Result<Htdemucs> {
    let weights = crate::demucs::HtdemucsWeights::load(&loaded.checkpoint, &loaded.config)?;
    Htdemucs::new(loaded.config, weights)
}

/// Builds the device runner for `host`, or `None` for the host path. `Auto`
/// falls back to the host path when no usable adapter exists.
fn make_runner(host: &Htdemucs, backend: &Backend, path: &Path) -> Result<Option<GpuHtdemucsRunner>> {
    let selector = match backend {
        Backend::Cpu => return Ok(None),
        Backend::Auto => {
            match GpuHtdemucsRunner::with_selector(host, DeviceSelector::Auto) {
                Ok(runner) => return Ok(Some(runner)),
                Err(e) => {
                    eprintln!("demucs: no usable wgpu adapter ({e}); using the host path");
                    return Ok(None);
                }
            }
        }
        Backend::Gpu(selector) => selector.clone(),
    };
    GpuHtdemucsRunner::with_selector(host, selector)
        .map(Some)
        .map_err(|e| {
            Error::Gpu(format!(
                "no usable wgpu adapter for `{}`: {e}",
                path.display()
            ))
        })
}

fn describe(backend: &Backend, runner: Option<&GpuHtdemucsRunner>) -> String {
    match runner {
        Some(runner) => format!("wgpu:{}", runner.device_name()),
        None => backend.tag().to_string(),
    }
}

/// The bag: parse `htdemucs_ft.yaml` from a HuggingFace snapshot directory and
/// load each network it names.
///
/// The accepted grammar is exactly the file as published: a `models:` line with
/// the shard stems and a `weights:` matrix with one row per network, in the
/// model's source order.
fn load_bag(dir: &Path, backend: &Backend) -> Result<Vec<(Engine, Vec<f32>)>> {
    if !dir.is_dir() {
        return Err(Error::Shape(format!(
            "htdemucs_ft is a directory (the HuggingFace snapshot), got {}",
            dir.display()
        )));
    }
    let yaml = dir.join("htdemucs_ft.yaml");
    let text = std::fs::read_to_string(&yaml)
        .map_err(|e| Error::Gpu(format!("reading {}: {e}", yaml.display())))?;
    let names = parse_yaml_list(&text, "models")?;
    let weights = parse_yaml_matrix(&text, "weights")?;
    if weights.len() != names.len() {
        return Err(Error::Shape(format!(
            "htdemucs_ft.yaml lists {} networks but {} weight rows",
            names.len(),
            weights.len()
        )));
    }
    let mut bag = Vec::with_capacity(names.len());
    for (name, source_weights) in names.iter().zip(weights.iter()) {
        let shard = dir.join(format!("{name}.safetensors"));
        let host = load_host(&shard)?;
        let runner = make_runner(&host, backend, &shard)?;
        bag.push((Engine { host, runner }, source_weights.clone()));
    }
    Ok(bag)
}

/// Extracts a `key: [a, b, c]` line's items as bare strings.
fn parse_yaml_list(text: &str, key: &str) -> Result<Vec<String>> {
    let line = text
        .lines()
        .map(str::trim)
        .find(|line| line.starts_with(&format!("{key}:")))
        .ok_or_else(|| Error::Shape(format!("htdemucs_ft.yaml has no `{key}:` line")))?;
    let open = line
        .find('[')
        .ok_or_else(|| Error::Shape(format!("`{key}:` is not a list")))?;
    let close = line
        .rfind(']')
        .ok_or_else(|| Error::Shape(format!("`{key}:` list is not closed")))?;
    Ok(line[open + 1..close]
        .split(',')
        .map(|item| item.trim().trim_matches(|c| c == '\'' || c == '"').to_string())
        .filter(|item| !item.is_empty())
        .collect())
}

/// Extracts a `key:` block of `[..]` rows into a matrix of floats.
fn parse_yaml_matrix(text: &str, key: &str) -> Result<Vec<Vec<f32>>> {
    let mut lines = text.lines().map(str::trim).peekable();
    while let Some(line) = lines.next() {
        if !line.starts_with(&format!("{key}:")) {
            continue;
        }
        let mut rows = Vec::new();
        loop {
            let Some(next) = lines.peek() else { break };
            if !next.starts_with('[') {
                break;
            }
            let next = lines.next().expect("peeked");
            let open = next
                .find('[')
                .ok_or_else(|| Error::Shape("malformed weights row".into()))?;
            let close = next
                .rfind(']')
                .ok_or_else(|| Error::Shape("weights row is not closed".into()))?;
            let row: Result<Vec<f32>> = next[open + 1..close]
                .split(',')
                .map(|item| {
                    item.trim()
                        .parse::<f32>()
                        .map_err(|e| Error::Shape(format!("weights row has `{item}`: {e}")))
                })
                .collect();
            rows.push(row?);
        }
        return Ok(rows);
    }
    Err(Error::Shape(format!("htdemucs_ft.yaml has no `{key}:` block")))
}
