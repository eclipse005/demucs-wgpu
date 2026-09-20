//! `demucs` — separate a track's stems with the htdemucs family.
//!
//! Scope is deliberately the ASR use case: read a WAV, run one model, write the
//! stems. The reference CLI's mp3 output, `--jobs` and the other model families
//! are not implemented; the stem selection surface (`--two-stems` with
//! `--other-method`) matches the reference's semantics.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use demucs_core::demucs::host::{Htdemucs, NoTrace, VecTrace};
/// The host forward allocates and frees multi-megabyte intermediates hundreds of
/// times per chunk; on Windows a fresh large allocation pays for its page faults
/// on first touch, which cost the attention's score matrix 4.5x (see
/// `alloc.rs`). Every measurement in the README is taken with this installed.
#[global_allocator]
static ALLOCATOR: demucs_core::alloc::CachingAllocator = demucs_core::alloc::CachingAllocator::new();

fn allocator_stats() -> (usize, usize) {
    ALLOCATOR.stats()
}
use demucs_core::demucs::pipeline::{self, SeparateOptions};
use demucs_core::demucs::{load_weights, HtdemucsWeights};
use demucs_core::{
    paths, Backend, Demucs, LoadOptions, ModelVariant, StemId, StemSelection, AUDIO_CHANNELS,
};
use ndarray::{s, Array1, Array3};

/// The bare `--model htdemucs` / `htdemucs_ft` names resolve to the standard
/// cache locations via `demucs_core::paths` — no developer machine path is
/// baked in. Pass `--model <path>` to use weights elsewhere.

#[derive(Parser)]
#[command(name = "demucs", about = "HTDemucs separation (Rust + wgpu port)")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Separate a track and write one WAV per stem.
    Separate(SeparateArgs),
    /// Print the model's parameter count, config and derived layer table.
    Info(InfoArgs),
    /// Compare two WAV files and report the SNR between them.
    Compare { reference: PathBuf, other: PathBuf },
    /// List the wgpu adapters `--device` accepts.
    Adapters,
    /// Run the network on one segment and print the timings.
    Bench(BenchArgs),
    /// Per-stage timings of one segment (`DEMUCS_PROFILE` is set internally).
    Profile(ProfileArgs),
    /// Per-kernel timings on the shapes the model actually uses.
    Kernels(KernelArgs),
}

/// Which execution path a `--device` value selects.
enum DeviceChoice {
    Cpu,
    Gpu(demucs_core::gpu::DeviceSelector),
}

/// `cpu`, or anything `demucs_core::gpu::DeviceSelector` parses: `auto`, a
/// runtime with an optional index (`vulkan`, `vulkan:1`), or a substring of the
/// adapter name (`nvidia`, `intel`). `demucs adapters` lists what is available.
fn parse_device(spec: &str) -> Result<DeviceChoice> {
    if spec.eq_ignore_ascii_case("cpu") {
        return Ok(DeviceChoice::Cpu);
    }
    Ok(DeviceChoice::Gpu(demucs_core::gpu::DeviceSelector::parse(spec)?))
}

#[derive(Parser)]
struct SeparateArgs {
    /// Input WAV.
    input: PathBuf,
    /// Output directory (one `<stem>.wav` per source) or a `.wav` path.
    #[arg(short, long)]
    output: PathBuf,
    /// Model name or checkpoint path.
    #[arg(short, long, default_value = "htdemucs")]
    model: String,
    /// Only write this stem. Shorthand for the reference's
    /// `--two-stems <STEM> --other-method none`.
    #[arg(long, conflicts_with = "two_stems")]
    stem: Option<String>,
    /// Separate audio into STEM and no_STEM, like the reference CLI.
    #[arg(long, conflicts_with = "stem")]
    two_stems: Option<String>,
    /// How to derive "no_STEM" in --two-stems mode (the reference's default is
    /// `add`): add the other stems, subtract from the original track, or skip.
    #[arg(long, value_enum, default_value_t = OtherMethod::Add)]
    other_method: OtherMethod,
    /// Random time shifts averaged together; 0 disables the shift trick.
    #[arg(long, default_value_t = 1)]
    shifts: usize,
    /// Chunk overlap fraction.
    #[arg(long, default_value_t = 0.25)]
    overlap: f64,
    /// Override the model's segment length in seconds.
    #[arg(long)]
    segment: Option<f64>,
    /// Which execution path to use: `cpu`, `auto`, `vulkan[:index]`, or an
    /// adapter name substring (see `demucs adapters`).
    #[arg(long, default_value = "cpu")]
    device: String,
    /// Print per-stage timings.
    #[arg(long)]
    timings: bool,
    /// Record the activations of the first chunk (a reference dump in Rust form).
    #[arg(long)]
    dump_trace: Option<PathBuf>,
}

#[derive(Parser)]
struct InfoArgs {
    #[arg(short, long, default_value = "htdemucs")]
    model: String,
}

/// How `--two-stems` derives the companion track, matching the reference CLI.
#[derive(Copy, Clone, PartialEq, Eq, ValueEnum)]
enum OtherMethod {
    /// Sum the other stems ("no_STEM.wav").
    Add,
    /// The original track minus the stem ("minus_STEM.wav").
    Minus,
    /// Do not write a companion.
    None,
}

#[derive(Parser)]
struct ProfileArgs {
    /// Audio file; the first segment is used.
    input: PathBuf,
    #[arg(short, long, default_value = "htdemucs")]
    model: String,
    /// Forward passes to average over (after one warm-up).
    #[arg(short, long, default_value_t = 3)]
    repeats: usize,
}

#[derive(Parser)]
struct KernelArgs {
    #[arg(short, long, default_value_t = 20)]
    repeats: usize,
}

#[derive(Parser)]
struct BenchArgs {
    #[arg(short, long, default_value = "htdemucs")]
    model: String,
    #[arg(short, long, default_value_t = 3)]
    repeats: usize,
    /// `cpu`, `auto`, `vulkan[:index]`, or an adapter name substring.
    #[arg(long, default_value = "cpu")]
    device: String,
}

fn resolve_model(name: &str) -> PathBuf {
    let path = Path::new(name);
    if path.exists() {
        return path.to_path_buf();
    }
    if name == "htdemucs" {
        return paths::htdemucs_checkpoint();
    }
    path.to_path_buf()
}

fn load_model(name: &str) -> Result<(Htdemucs, f64)> {
    let path = resolve_model(name);
    let started = Instant::now();
    let loaded = load_weights(&path)
        .with_context(|| format!("loading {}", path.display()))?;
    let weights = HtdemucsWeights::load(&loaded.checkpoint, &loaded.config)?;
    let model = Htdemucs::new(loaded.config, weights)?;
    Ok((model, started.elapsed().as_secs_f64()))
}

/// Map `--model` to the variant and path the facade should load. A directory
/// holding `htdemucs_ft.yaml` is the fine-tuned bag; a bare name resolves to the
/// on-disk defaults; anything else is treated as a single checkpoint file.
fn model_spec(name: &str) -> Result<(ModelVariant, PathBuf)> {
    let path = Path::new(name);
    if path.is_dir() {
        if path.join("htdemucs_ft.yaml").exists() {
            return Ok((ModelVariant::FineTuned, path.to_path_buf()));
        }
        bail!("{} is a directory but has no htdemucs_ft.yaml", path.display());
    }
    if path.exists() {
        return Ok((ModelVariant::FourStem, path.to_path_buf()));
    }
    match name {
        "htdemucs" => Ok((ModelVariant::FourStem, paths::htdemucs_checkpoint())),
        "htdemucs_ft" => Ok((ModelVariant::FineTuned, paths::htdemucs_ft_dir())),
        other => Ok((ModelVariant::FourStem, Path::new(other).to_path_buf())),
    }
}

/// Turn a `--device` choice into a facade `Backend`.
fn to_backend(choice: &DeviceChoice) -> Backend {
    match choice {
        DeviceChoice::Cpu => Backend::Cpu,
        DeviceChoice::Gpu(selector) => Backend::Gpu(selector.clone()),
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Separate(args) => separate(args),
        Command::Info(args) => info(args),
        Command::Compare { reference, other } => compare(&reference, &other),
        Command::Adapters => adapters(),
        Command::Bench(args) => bench(args),
        Command::Profile(args) => profile(args),
        Command::Kernels(args) => kernels(args),
    }
}

fn info(args: InfoArgs) -> Result<()> {
    let (model, seconds) = load_model(&args.model)?;
    println!(
        "loaded in {seconds:.2}s | {} parameters | {} sources {:?} | segment {:.3}s",
        model.weights.parameter_count(),
        model.config.sources.len(),
        model.config.sources,
        model.config.segment
    );
    println!(
        "nfft {} hop {} depth {} channels {} transformer {} layers {} heads {}",
        model.config.nfft,
        model.config.hop_length(),
        model.config.depth,
        model.config.channels,
        model.arch.transformer_channels,
        model.config.t_layers,
        model.config.t_heads
    );
    println!("frequency branch:");
    for spec in &model.arch.encoder {
        println!(
            "  encoder.{} {:>5} bins, {:>3} -> {:>3} channels, k={} s={} p={}",
            spec.index, spec.freqs_in, spec.chin, spec.chout, spec.kernel_size, spec.stride, spec.pad
        );
    }
    println!("  bottleneck: {} bins, {} channels", model.arch.bottleneck_freqs, model.arch.bottleneck_channels);
    println!("waveform branch:");
    for spec in &model.arch.tencoder {
        println!(
            "  tencoder.{} {:>3} -> {:>3} channels, k={} s={} p={}",
            spec.index, spec.chin, spec.chout, spec.kernel_size, spec.stride, spec.pad
        );
    }
    println!("decoder (execution order):");
    for spec in &model.arch.decoder {
        println!(
            "  decoder.{} {:>3} -> {:>3} channels",
            spec.index, spec.chin, spec.chout
        );
    }
    Ok(())
}

fn compare(reference: &Path, other: &Path) -> Result<()> {
    let a = demucs_core::audio::read_audio(reference)?;
    let b = demucs_core::audio::read_audio(other)?;
    if a.samples != b.samples || a.channels() != b.channels() {
        bail!(
            "shape mismatch: {} vs {}",
            format!("{}x{}", a.channels(), a.samples),
            format!("{}x{}", b.channels(), b.samples)
        );
    }
    let result = demucs_core::fixtures::compare(&a.data, &b.data)?;
    println!(
        "{} vs {}: {:.2} dB SNR (max abs {:.3e}, reference peak {:.6})",
        reference.display(),
        other.display(),
        result.snr_db(),
        result.max_abs,
        result.reference_max_abs
    );
    Ok(())
}

/// Reads a WAV into the `(batch, channels, samples)` layout the model wants.
fn read_mix(path: &Path, wanted_channels: usize) -> Result<Array3<f32>> {
    let audio = demucs_core::audio::read_audio(path)
        .with_context(|| format!("reading {}", path.display()))?;
    let audio = audio.ensure_channels(wanted_channels)?;
    let (channels, samples) = (audio.channels(), audio.samples);
    Ok(Array3::from_shape_vec((1, channels, samples), audio.data)?)
}

fn separate(args: SeparateArgs) -> Result<()> {
    let device = parse_device(&args.device)?;
    let (variant, path) = model_spec(&args.model)?;

    // Fail fast on an unknown stem, like the reference does before it separates.
    let requested = args.stem.as_ref().or(args.two_stems.as_ref());
    let selected_id = match requested {
        Some(stem) => Some(StemId::parse(stem).ok_or_else(|| {
            anyhow::anyhow!("--stem/--two-stems must be one of {:?}, got {stem:?}", StemId::ALL)
        })?),
        None => None,
    };
    // Only a request that writes just the named stem can prune the fine-tuned
    // bag to the one network that produces it; `--two-stems` with an add/minus
    // companion needs the other stems, so it loads all four.
    let prune = match (&args.stem, &args.two_stems, args.other_method) {
        (Some(_), _, _) => true,
        (None, Some(_), OtherMethod::None) => true,
        _ => false,
    };
    let selection = match selected_id {
        Some(id) if prune => StemSelection::Some(vec![id]),
        _ => StemSelection::All,
    };

    let load_started = Instant::now();
    let sep = Demucs::load(&path, LoadOptions { variant, stems: selection }, to_backend(&device))
        .with_context(|| format!("loading {}", path.display()))?;
    let load_seconds = load_started.elapsed().as_secs_f64();
    let sample_rate = sep.sample_rate() as u32;

    // `read_mix` keeps the raw track around for `--other-method minus`; the
    // facade normalises internally so the CLI no longer touches that step.
    let mix = read_mix(&args.input, AUDIO_CHANNELS)?;
    let audio_seconds = mix.dim().2 as f64 / sep.sample_rate() as f64;
    let left = mix.slice(s![0, 0, ..]).to_vec();
    let right = mix.slice(s![0, 1, ..]).to_vec();

    let options = SeparateOptions {
        shifts: args.shifts,
        overlap: args.overlap,
        transition_power: 1.0,
        segment: args.segment,
    };

    let mut tracing = VecTrace::default();
    let started = Instant::now();
    let stems = if let Some(dir) = &args.dump_trace {
        let out = sep.separate_with_trace(
            &left,
            &right,
            sample_rate,
            &options,
            &mut tracing,
            &mut |_| {},
        )?;
        write_trace(dir, &tracing)?;
        out
    } else {
        sep.separate_with_options(&left, &right, sample_rate, &options, &mut |_| {})?
    };
    let elapsed = started.elapsed().as_secs_f64();

    // Lay the returned stems back down in config-source order for the writers,
    // leaving unselected lanes zero (they are only read for a companion, and
    // those runs load all four stems).
    let names = sep.sources().to_vec();
    let samples = left.len();
    let mut sources = ndarray::Array4::<f32>::zeros((1, names.len(), AUDIO_CHANNELS, samples));
    for stem in &stems {
        let index = names
            .iter()
            .position(|name| name == stem.id.as_str())
            .expect("the facade returned a stem the config does not name");
        sources
            .slice_mut(s![0, index, 0, ..])
            .assign(&Array1::from(stem.left.clone()));
        sources
            .slice_mut(s![0, index, 1, ..])
            .assign(&Array1::from(stem.right.clone()));
    }

    report(
        audio_seconds,
        elapsed,
        load_seconds,
        args.timings,
        sep.sample_rate(),
        sep.segment(),
    );
    write_stems(&sources, &mix, &names, sep.sample_rate(), &args)
}

fn report(
    audio_seconds: f64,
    elapsed: f64,
    load_seconds: f64,
    timings: bool,
    samplerate: usize,
    segment: f64,
) {
    let segment_length = (samplerate as f64 * segment) as usize;
    let stride = ((1.0 - 0.25) * segment_length as f64) as usize;
    let chunks = audio_seconds * samplerate as f64 / stride as f64;
    println!(
        "separation {elapsed:.2}s for {audio_seconds:.3}s -> RTFx {:.2}x | + {load_seconds:.2}s load = \
         end-to-end RTFx {:.2}x",
        audio_seconds / elapsed,
        audio_seconds / (elapsed + load_seconds)
    );
    if timings {
        println!(
            "per chunk: {:.0} ms of model input, {:.1}x raw ({chunks:.0} chunks)",
            elapsed / chunks * 1000.0,
            segment / (elapsed / chunks)
        );
    }
}

/// Writes the requested stems. `mix` is the track as read (only `--other-method
/// minus` needs it), `sources` the model output in config order, `source_names`
/// the config's source list, `samplerate` the model's rate.
fn write_stems(
    sources: &ndarray::Array4<f32>,
    mix: &Array3<f32>,
    source_names: &[String],
    samplerate: usize,
    args: &SeparateArgs,
) -> Result<()> {
    let (_, count, channels, samples) = sources.dim();
    // The stem actually written, plus the companion the reference's two-stem
    // mode derives from the others. `--stem` is the ASR shorthand: it writes
    // only the named stem, which is the reference's `--other-method none`.
    let (selected, companion): (Vec<(usize, String)>, Option<(String, OtherMethod)>) =
        match (&args.stem, &args.two_stems) {
            (Some(stem), _) | (None, Some(stem)) => {
                let index = source_names.iter().position(|name| name == stem).ok_or_else(|| {
                    anyhow::anyhow!(
                        "--two-stems must be one of {:?}, got {stem:?}",
                        source_names
                    )
                })?;
                let method = if args.two_stems.is_some() {
                    args.other_method
                } else {
                    OtherMethod::None
                };
                (
                    vec![(index, stem.clone())],
                    (method != OtherMethod::None).then_some((stem.clone(), method)),
                )
            }
            (None, None) => (
                source_names.iter().enumerate().map(|(index, name)| (index, name.clone())).collect(),
                None,
            ),
        };
    if selected.iter().any(|(index, _)| *index >= count) {
        bail!("the model produced {count} stems but the config names {} sources", source_names.len());
    }

    // A single output may go straight to a `.wav` path; anything with a
    // companion lands in a directory next to it, as the reference does.
    let single_file = companion.is_none() && args.output.extension().is_some();
    let mut written = Vec::new();
    let mut save = |name: &str, data: &[f32]| -> Result<()> {
        let path = if single_file {
            args.output.clone()
        } else {
            std::fs::create_dir_all(&args.output).ok();
            args.output.join(format!("{name}.wav"))
        };
        demucs_core::audio::write_wav_f32(&path, samplerate as u32, channels, data)?;
        let peak = data.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        written.push(format!("{} (peak {peak:.4})", path.display()));
        Ok(())
    };

    for (index, name) in &selected {
        let data: Vec<f32> = sources.slice(ndarray::s![0, *index, .., ..]).iter().copied().collect();
        save(name, &data)?;
    }
    if let Some((stem, method)) = companion {
        let index = selected[0].0;
        match method {
            OtherMethod::None => {}
            OtherMethod::Add => {
                // Sum every other stem: the reference's `no_<stem>`.
                let mut summed = ndarray::Array2::<f32>::zeros((channels, samples));
                for (other, _) in source_names.iter().enumerate() {
                    if other != index {
                        summed += &sources.slice(ndarray::s![0, other, .., ..]);
                    }
                }
                let data: Vec<f32> = summed.iter().copied().collect();
                save(&format!("no_{stem}"), &data)?;
            }
            OtherMethod::Minus => {
                // The original track minus the stem: the reference's `minus_<stem>`.
                if mix.dim() != (1, channels, samples) {
                    bail!(
                        "the track is {:?} but the model output is (1, {channels}, {samples}); \
                         --other-method minus needs the original track's shape",
                        mix.dim()
                    );
                }
                let mut data: Vec<f32> = sources.slice(ndarray::s![0, index, .., ..]).iter().copied().collect();
                let origin: Vec<f32> = mix.iter().copied().collect();
                for (value, base) in data.iter_mut().zip(origin.iter()) {
                    *value = base - *value;
                }
                save(&format!("minus_{stem}"), &data)?;
            }
        }
    }
    println!("wrote {}", written.join(", "));
    Ok(())
}

fn write_trace(dir: &Path, trace: &VecTrace) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    for (name, shape, values) in &trace.entries {
        let safe = name.replace('.', "_").replace('#', "_").replace('/', "_");
        demucs_core::npy::write_npy(dir.join(format!("{safe}.npy")), shape, values)?;
    }
    println!("wrote {} traced activations to {}", trace.entries.len(), dir.display());
    Ok(())
}

/// Runs `repeats` forwards of one real segment with the stage profiler on, and
/// prints the table sorted by total time. The point is to compare this against
/// torch's own per-module numbers for the same segment, not to guess.
fn profile(args: ProfileArgs) -> Result<()> {
    std::env::set_var("DEMUCS_PROFILE", "1");
    let (model, load_seconds) = load_model(&args.model)?;
    let mix = read_mix(&args.input, model.config.audio_channels)?;
    let (normalized, _norm) = pipeline::normalize(&mix);
    let segment = model.training_length();
    let length = normalized.dim().2.min(segment);
    let mut padded = Array3::<f32>::zeros((1, model.config.audio_channels, segment));
    padded
        .slice_mut(ndarray::s![.., .., ..length])
        .assign(&normalized.slice(ndarray::s![.., .., ..length]));

    println!(
        "model loaded in {load_seconds:.2}s | segment {} samples ({:.2}s) | {} repeats",
        segment,
        segment as f64 / model.config.samplerate as f64,
        args.repeats
    );
    let _ = model.forward(&padded, &mut NoTrace)?; // warm-up
    demucs_core::demucs::host::profile::reset();
    let started = Instant::now();
    for _ in 0..args.repeats {
        let out = model.forward(&padded, &mut NoTrace)?;
        std::hint::black_box(&out);
    }
    let wall = started.elapsed().as_secs_f64() / args.repeats as f64;
    let entries = demucs_core::demucs::host::profile::take();
    println!("one segment: {:.1} ms wall per forward", wall * 1000.0);
    println!("{:<28} {:>10} {:>9} {:>9}", "stage", "total ms", "per call", "calls");
    for (name, seconds, count) in entries {
        println!(
            "{:<28} {:>10.1} {:>9.2} {:>9}",
            name,
            seconds * 1000.0 / args.repeats as f64,
            seconds * 1000.0 / args.repeats as f64 / count as f64,
            count / args.repeats
        );
    }
    Ok(())
}

/// Times the individual kernels on the exact shapes the 7.8 s segment uses, so
/// each one can be compared against torch's equivalent on the same machine.
fn kernels(args: KernelArgs) -> Result<()> {
    use demucs_core::demucs::ops::{
        conv1d, conv2d, conv_transpose2d, group_norm, matmul, matmul_bt, softmax_rows,
    };
    use demucs_core::demucs::weights::ConvW;
    use ndarray::{Array2, Array3, Array4};

    /// Runs the expression once as a warm-up, then `repeats` times, and returns
    /// the mean seconds. A macro rather than a closure so the operands can stay
    /// borrowed by the surrounding scope.
    macro_rules! timed {
        ($repeats:expr, $body:expr) => {{
            std::hint::black_box(&$body);
            let started = Instant::now();
            for _ in 0..$repeats {
                std::hint::black_box(&$body);
            }
            started.elapsed().as_secs_f64() / $repeats as f64
        }};
    }

    let mut seed = 12345u32;
    let mut fill = move |n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((seed >> 8) as f32 / 8_388_608.0 - 1.0) * 0.2
            })
            .collect()
    };
    let repeats = args.repeats;
    let (hits, misses) = allocator_stats();
    println!("allocator: {hits} hits / {misses} misses so far");
    println!(
        "cpu features: avx2={} fma={} avx512f={} avx512vnni={} | rayon threads={} | gemm backend={}",
        is_x86_feature_detected!("avx2"),
        is_x86_feature_detected!("fma"),
        is_x86_feature_detected!("avx512f"),
        is_x86_feature_detected!("avx512vnni"),
        rayon::current_num_threads(),
        if demucs_core::demucs::ops::use_ndarray_gemm() { "ndarray" } else { "gemm" },
    );
    println!("{:<34} {:>9}   {:>11}   {:>10}", "kernel", "ms", "GFLOP/s", "GB/s");

    let report = |name: &str, flops: f64, bytes: f64, seconds: f64| {
        println!(
            "{name:<34} {:>9.2}   {:>11.1}   {:>10.1}",
            seconds * 1000.0,
            flops / seconds / 1e9,
            bytes / seconds / 1e9,
        );
    };

    // Scaling probe: the same shapes with and without the crate's own rayon
    // split, to tell a slow microkernel apart from a missing parallel path.
    if std::env::var("DEMUCS_GEMM_SCALING").is_ok() {
        for (m, k, n) in [
            (2688usize, 512usize, 2048usize),
            (768, 384, 2688),
            (21504, 64, 2688),
            (21504, 2688, 64),
            (96, 432, 73920),
        ] {
            let a = Array2::from_shape_vec((m, k), fill(m * k)).unwrap();
            let b = Array2::from_shape_vec((k, n), fill(k * n)).unwrap();
            let mut out = Array2::<f32>::zeros((m, n));
            for (label, parallelism) in [
                ("none  ", gemm::Parallelism::None),
                ("rayon ", gemm::Parallelism::Rayon(0)),
            ] {
                // `fresh` re-allocates the output every call, which is what the
                // wrapper in `demucs::ops` does; `reused` keeps one buffer.
                let mut run_fresh = || {
                    let mut fresh = Array2::<f32>::zeros((m, n));
                    unsafe {
                        gemm::gemm(
                            m, n, k,
                            fresh.as_mut_ptr(), 1, n as isize, false,
                            a.as_ptr(), a.strides()[1], a.strides()[0],
                            b.as_ptr(), b.strides()[1], b.strides()[0],
                            1.0, 0.0, false, false, false, parallelism,
                        );
                    }
                };
                let mut run = || {
                    unsafe {
                        gemm::gemm(
                            m, n, k,
                            out.as_mut_ptr(), 1, n as isize, false,
                            a.as_ptr(), a.strides()[1], a.strides()[0],
                            b.as_ptr(), b.strides()[1], b.strides()[0],
                            1.0, 0.0, false, false, false, parallelism,
                        );
                    }
                };
                run();
                let started = Instant::now();
                for _ in 0..5 {
                    run();
                }
                let reuse = started.elapsed().as_secs_f64() / 5.0;
                run_fresh();
                let started = Instant::now();
                for _ in 0..5 {
                    run_fresh();
                }
                let fresh = started.elapsed().as_secs_f64() / 5.0;
                println!(
                    "  scaling {label} m={m:<6} k={k:<5} n={n:<6} reuse {:7.2} ms {:7.1} GFLOP/s |                      fresh alloc {:7.2} ms {:7.1} GFLOP/s",
                    reuse * 1000.0,
                    2.0 * (m * k * n) as f64 / reuse / 1e9,
                    fresh * 1000.0,
                    2.0 * (m * k * n) as f64 / fresh / 1e9,
                );
            }
        }
    }

    // linear / FFN shape
    {
        let (n, din, dout) = (2688usize, 512usize, 2048usize);
        let a = Array2::from_shape_vec((n, din), fill(n * din)).unwrap();
        let b = Array2::from_shape_vec((din, dout), fill(din * dout)).unwrap();
        let seconds = timed!(repeats, matmul(&a, &b));
        report(
            "matmul 2688x512x2048",
            2.0 * (n * din * dout) as f64,
            ((n * din + din * dout + n * dout) * 4) as f64,
            seconds,
        );
    }
    // attention scores + AV
    {
        let (rows, dh, nk) = (2688usize * 8, 64usize, 2688usize);
        let q = Array2::from_shape_vec((rows, dh), fill(rows * dh)).unwrap();
        let k = Array2::from_shape_vec((nk, dh), fill(nk * dh)).unwrap();
        let seconds = timed!(repeats, matmul_bt(&q, &k));
        report(
            "matmul_bt scores 21504x64x2688",
            2.0 * (rows * dh * nk) as f64,
            ((rows * nk * 3 + rows * dh + nk * dh) * 4) as f64,
            seconds,
        );
        let scores = matmul_bt(&q, &k);
        let v = Array2::from_shape_vec((nk, dh), fill(nk * dh)).unwrap();
        let seconds = timed!(repeats, matmul(&scores, &v));
        report(
            "matmul AV 21504x2688x64",
            2.0 * (rows * dh * nk) as f64,
            ((rows * nk + rows * dh) * 4) as f64,
            seconds,
        );
    }
    {
        let (rows, cols) = (21504usize, 2688usize);
        let mut scores = Array2::from_shape_vec((rows, cols), fill(rows * cols)).unwrap();
        let seconds = timed!(repeats, softmax_rows(&mut scores));
        report("softmax 21504x2688", (rows * cols) as f64 * 8.0, (rows * cols) as f64 * 8.0, seconds);
    }
    // the biggest convolution: the last decoder's 3x3 rewrite
    {
        let x = Array4::from_shape_vec((1, 48, 512, 336), fill(48 * 512 * 336)).unwrap();
        let w = ConvW {
            weight: fill(96 * 48 * 9),
            bias: fill(96),
            shape: vec![96, 48, 3, 3],
        };
        let seconds = timed!(repeats, conv2d(&x, &w, (1, 1), (1, 1), (1, 1)));
        report(
            "conv2d 3x3 48->96 @512x336",
            2.0 * (96.0 * 48.0 * 9.0 * 512.0 * 336.0),
            ((48.0 * 512.0 * 336.0 * 2.0 + 96.0 * 512.0 * 336.0) * 4.0),
            seconds,
        );
    }
    // 1x1 rewrite at the bottleneck
    {
        let x = Array4::from_shape_vec((1, 384, 8, 336), fill(384 * 8 * 336)).unwrap();
        let w = ConvW {
            weight: fill(768 * 384),
            bias: fill(768),
            shape: vec![768, 384, 1, 1],
        };
        let seconds = timed!(repeats, conv2d(&x, &w, (1, 1), (0, 0), (1, 1)));
        report(
            "conv2d 1x1 384->768 @8x336",
            2.0 * (768.0 * 384.0 * 8.0 * 336.0),
            0.0,
            seconds,
        );
    }
    // waveform encoder 0
    {
        let x = Array3::from_shape_vec((1, 2, 343980), fill(2 * 343980)).unwrap();
        let w = ConvW {
            weight: fill(48 * 2 * 8),
            bias: fill(48),
            shape: vec![48, 2, 8],
        };
        let seconds = timed!(repeats, conv1d(&x, &w, 4, 2, 1));
        report(
            "conv1d k8 s4 2->48 @343980",
            2.0 * (48.0 * 2.0 * 8.0 * 85995.0),
            0.0,
            seconds,
        );
    }
    // the waveform DConv's 1x1 (48 -> 96 over 85995 samples)
    {
        let x = Array3::from_shape_vec((1, 48, 85995), fill(48 * 85995)).unwrap();
        let w = ConvW {
            weight: fill(96 * 48),
            bias: fill(96),
            shape: vec![96, 48, 1],
        };
        let seconds = timed!(repeats, conv1d(&x, &w, 1, 0, 1));
        report(
            "conv1d 1x1 48->96 @85995",
            2.0 * (96.0 * 48.0 * 85995.0),
            0.0,
            seconds,
        );
    }
    {
        let x = Array3::from_shape_vec((512, 48, 336), fill(512 * 48 * 336)).unwrap();
        let gamma = fill(48);
        let beta = fill(48);
        let seconds = timed!(repeats, group_norm(&x, 1, &gamma, &beta).unwrap());
        report(
            "group_norm 512x48x336",
            0.0,
            (512.0 * 48.0 * 336.0 * 3.0 * 4.0),
            seconds,
        );
    }
    // The DConv's four shapes, which is where the segment's time now goes.
    {
        // conv1d(48 -> 6, k=3) over the frequency branch's permuted tensor.
        let x = Array3::from_shape_vec((512, 48, 336), fill(512 * 48 * 336)).unwrap();
        let w = ConvW {
            weight: fill(6 * 48 * 3),
            bias: fill(6),
            shape: vec![6, 48, 3],
        };
        let seconds = timed!(repeats, conv1d(&x, &w, 1, 1, 1));
        report(
            "dconv conv1 k3 48->6 @512x336",
            2.0 * (6.0 * 48.0 * 3.0 * 512.0 * 336.0),
            ((48.0 + 6.0) * 512.0 * 336.0 * 4.0),
            seconds,
        );
    }
    {
        // conv2: the 1x1 that expands back to 2C.
        let x = Array3::from_shape_vec((512, 6, 336), fill(512 * 6 * 336)).unwrap();
        let w = ConvW {
            weight: fill(96 * 6),
            bias: fill(96),
            shape: vec![96, 6, 1],
        };
        // Same FLOPs as one row-block of the frequency branch vs the whole
        // thing: is the cost per FLOP or per call?
        for (label, rows) in [("rows=32of512", 32usize), ("rows=512", 512)] {
            let xx = if rows == 512 { x.clone() } else { x.slice(ndarray::s![..rows, .., ..]).to_owned() };
            let seconds = timed!(repeats, conv1d(&xx, &w, 1, 0, 1));
            report(
                &format!("dconv conv2 {label}"),
                2.0 * (96.0 * 6.0 * rows as f64 * 336.0),
                ((6.0 + 2.0 * 96.0) * rows as f64 * 336.0 * 4.0),
                seconds,
            );
        }
        for scale in [1usize, 4, 8, 12, 20] {
            let xx = x.slice(ndarray::s![..scale.min(512), .., ..]).to_owned();
            let seconds = timed!(repeats, conv1d(&xx, &w, 1, 0, 1));
            report(
                &format!("dconv conv2 rows={}", xx.dim().0),
                2.0 * (96.0 * 6.0 * xx.dim().0 as f64 * 336.0),
                ((6.0 + 2.0 * 96.0) * xx.dim().0 as f64 * 336.0 * 4.0),
                seconds,
            );
        }
        let seconds = timed!(repeats, conv1d(&x, &w, 1, 0, 1));
        report(
            "dconv conv2 1x1 6->96 @512x336",
            2.0 * (96.0 * 6.0 * 512.0 * 336.0),
            ((6.0 + 2.0 * 96.0) * 512.0 * 336.0 * 4.0),
            seconds,
        );
    }
    {
        // norm2: GroupNorm(1, 96) over the same tensor.
        let x = Array3::from_shape_vec((512, 96, 336), fill(512 * 96 * 336)).unwrap();
        let gamma = fill(96);
        let beta = fill(96);
        let seconds = timed!(repeats, group_norm(&x, 1, &gamma, &beta).unwrap());
        report(
            "dconv norm2 g1 96ch @512x336",
            0.0,
            (512.0 * 96.0 * 336.0 * 3.0 * 4.0),
            seconds,
        );
    }
    {
        // glu on (96 -> 48)
        let x = Array3::from_shape_vec((512, 96, 336), fill(512 * 96 * 336)).unwrap();
        let seconds = timed!(repeats, demucs_core::demucs::ops::glu(&x).unwrap());
        report(
            "dconv glu 96->48 @512x336",
            0.0,
            (512.0 * 96.0 * 336.0 * 1.5 * 4.0),
            seconds,
        );
    }
    {
        let x = Array4::from_shape_vec((1, 48, 512, 336), fill(48 * 512 * 336)).unwrap();
        let w = ConvW {
            weight: fill(48 * 16 * 8),
            bias: fill(16),
            shape: vec![48, 16, 8, 1],
        };
        let seconds = timed!(repeats, conv_transpose2d(&x, &w, (4, 1)));
        report(
            "convT2d k8 s4 48->16 @512x336",
            2.0 * (16.0 * 48.0 * 8.0 * 512.0 * 336.0),
            ((48.0 * 512.0 * 336.0 + 16.0 * 2052.0 * 336.0) * 4.0),
            seconds,
        );
    }
    Ok(())
}

fn bench(args: BenchArgs) -> Result<()> {
    let device = parse_device(&args.device)?;
    let (model, load_seconds) = load_model(&args.model)?;
    println!("load {load_seconds:.2}s, {} parameters", model.weights.parameter_count());
    let length = model.training_length();
    let mut rng = 12345u32;
    let mut mix = Array3::<f32>::zeros((1, model.config.audio_channels, length));
    for value in mix.iter_mut() {
        rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *value = ((rng >> 8) as f32 / 8_388_608.0 - 1.0) * 0.1;
    }
    let seconds = length as f64 / model.config.samplerate as f64;
    let elapsed = if let DeviceChoice::Gpu(selector) = &device {
        // The GPU path goes through the same device forward the separation uses,
        // including its readback, so the number is comparable to `separate`'s.
        let runner = demucs_core::gpu::htdemucs::GpuHtdemucsRunner::with_selector(&model, selector.clone())
            .map_err(|e| anyhow::anyhow!("no usable wgpu adapter for `{}`: {e}", args.device))?;
        let _ = demucs_core::gpu::htdemucs::separate_gpu(&model, &runner, &mix, &mut NoTrace)?;
        let started = Instant::now();
        for _ in 0..args.repeats {
            let out = demucs_core::gpu::htdemucs::separate_gpu(&model, &runner, &mix, &mut NoTrace)?;
            std::hint::black_box(&out);
        }
        started.elapsed().as_secs_f64() / args.repeats as f64
    } else {
        // One warm-up so the measurement is of steady-state work.
        let _ = model.forward(&mix, &mut NoTrace)?;
        let started = Instant::now();
        for _ in 0..args.repeats {
            let out = model.forward(&mix, &mut NoTrace)?;
            std::hint::black_box(&out);
        }
        started.elapsed().as_secs_f64() / args.repeats as f64
    };
    println!(
        "chunk forward {:.1} ms for {:.3}s of audio -> {:.2}x raw ({:.1} GB/s of input audio)",
        elapsed * 1000.0,
        seconds,
        seconds / elapsed,
        seconds * model.config.audio_channels as f64 * 4.0 / elapsed / 1e9
    );
    Ok(())
}

/// Lists the wgpu adapters with the index a `--device vulkan:<n>` would use.
fn adapters() -> Result<()> {
    let all = demucs_core::gpu::Gpu::list_adapters();
    if all.is_empty() {
        println!("no wgpu adapters visible");
        return Ok(());
    }
    println!("{:<10} {:<4} {:<40} {}", "runtime", "idx", "name", "type");
    for entry in &all {
        println!(
            "{:<10} {:<4} {:<40} {}",
            entry.runtime, entry.index, entry.name, entry.kind
        );
    }
    println!("accepts: cpu | auto | <runtime>[:<idx>] | <name substring>, e.g. `--device intel`");
    Ok(())
}
