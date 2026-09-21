# demucs-wgpu

Hand-written **CPU + wgpu** inference for **HTDemucs v4** (`htdemucs`, `htdemucs_ft`) — pure Rust,
no Python/torch, no CUDA toolkit. Runs on any GPU (or CPU) that `wgpu` can drive: **Vulkan, DX12,
Metal, OpenGL**. Built to feed vocal isolation into an ASR pipeline.

This is a sibling of the CUDA port `demucs-native-rs`: same crate shape and
same public API (`Demucs::load` / `separate` / `separate_with_progress` / `Stem` / `StemSelection` /
`LoadOptions` / `Backend` / `ModelVariant`), so switching a consumer between the two is a matter of
the `Backend` enum and where the `htdemucs_ft` weights live (see below).

| Path | Status |
|------|--------|
| CPU (host forward, rayon + `gemm`) | ✅ |
| GPU via wgpu — **Vulkan** | ✅ |
| GPU via wgpu — Metal / OpenGL | compiles & runs via wgpu; validated on Vulkan only |
| GPU via wgpu — DX12 | driver-dependent (stalled on the NVIDIA dev card; use Vulkan) |

Package: `demucs-core` (library) · CLI binary: `demucs`.

## Scope

Inference only, and only the `htdemucs` family: `htdemucs` and `htdemucs_ft`. Not implemented (by
design): `mdx_*`, `hdemucs_mmi`, the v3 `Demucs`, `htdemucs_6s` (6 stems, a different architecture),
training, mp3/flac I/O (float WAV only), and the upstream CLI's `--jobs`/segment-model ensemble.

## Model weights (not in this repo)

Weights are not distributed here. The loader reads a checkpoint by extension: `.th`/`.pth`/`.pt` are
torch-zip archives, everything else is treated as **`safetensors`** (config comes from the file's
metadata, falling back to the built-in `htdemucs` defaults). `htdemucs_ft` is stored as fp16 and
upcast to fp32 on load.

| Model (`--model`) | What to point it at | Size |
|-------------------|---------------------|------|
| `htdemucs` | one file: `955717e8-8726e21a.th` **or** a `htdemucs.safetensors` | ~84 MB |
| `htdemucs_ft` | a **directory** holding `htdemucs_ft.yaml` + the four shard `.safetensors` | ~4 × 84 MB |

**`htdemucs_ft` is a directory, not one merged file.** The fine-tuned variant is a *bag* of four
single-stem networks plus a weight matrix (`htdemucs_ft.yaml`). The shipped matrix is the identity,
so each stem is produced by exactly one specialist network:

| Shard | Stem |
|-------|------|
| `f7e0c4bc.safetensors` | drums |
| `d12395a8.safetensors` | bass |
| `92cfc3b6.safetensors` | other |
| `04573f0d.safetensors` | vocals |

Two consequences:

- **Load the bag from its directory.** `Demucs::load(dir, { variant: FineTuned, .. }, backend)`
  expects the folder containing `htdemucs_ft.yaml`; it is *not* a single `.safetensors`. (A single
  333 MB merged `htdemucs_ft.safetensors`, as some CUDA builds ship, is a different layout this port
  does not read.)
- **For a single stem you can skip the bag entirely.** Because of the identity matrix, the bag's
  vocals lane is bit-for-bit the `04573f0d` shard run on its own (locked by a test). So a vocals-only
  ASR consumer can ship just that one shard and load it as `ModelVariant::FourStem` — same output,
  ~4× smaller. When you do load the bag, a `StemSelection::Some([...])` request runs only the
  networks that feed the requested stems.

Grab the official weights from the [adefossez/HTDemucs](https://huggingface.co/adefossez/HTDemucs)
Hugging Face snapshot (or any mirror of the `htdemucs` v4 checkpoints).

## CLI

```text
demucs separate <input.wav> -o <out-dir|out.wav> [options]

  -m, --model <name|path>     htdemucs | htdemucs_ft | a checkpoint file | an ft directory
  -o, --output <path>         output directory, or a single .wav for one stem
      --device <spec>         cpu | auto | vulkan | vulkan:<idx> | <adapter-name substring>
      --stem <STEM>           write only this stem (ASR shorthand for --two-stems --other-method none)
      --two-stems <STEM>      stem + companion, like the reference CLI
      --other-method <m>      add | minus | none   (companion for --two-stems; default add)
      --shifts <N>            random time shifts averaged; 0 is deterministic (default 1)
      --overlap <F>           chunk overlap fraction (default 0.25)
      --segment <SEC>         override the model's segment length
      --timings               print per-chunk timing
      --dump-trace <dir>      record first-chunk activations (single network only)
```

```powershell
# pick the GPU adapter, isolate vocals deterministically
demucs separate song.wav -o out --device vulkan --shifts 0 --stem vocals

# the full four-stem fine-tuned bag
demucs separate song.wav -o out --model htdemucs_ft --device auto
```

Other subcommands: `demucs adapters` (list wgpu adapters `--device` accepts), `info` (model config
and derived layer table), `compare` (SNR between two WAVs), and the dev tools `bench`, `profile`,
`kernels`. The built-in default checkpoint paths are specific to the author's machine — pass
`--model <path>` to use your own.

## Library

```rust
use demucs_core::{Backend, Demucs, LoadOptions, ModelVariant, StemSelection};

// htdemucs: a single checkpoint file.
let sep = Demucs::load(
    "models/htdemucs.safetensors",
    LoadOptions { variant: ModelVariant::FourStem, stems: StemSelection::All },
    Backend::Auto, // or Backend::Cpu, or Backend::Gpu(DeviceSelector::parse("vulkan")?)
)?;

let stems = sep.separate(&left, &right, 44100)?;
for s in &stems {
    println!("{}: {} samples", s.id, s.left.len()); // s.id -> "drums" | "bass" | ...
}
```

For progress (a GUI task list, say), pass a callback — it runs once per chunk and always ends at
`done == total`, including across shift passes and the fine-tuned bag:

```rust
let stems = sep.separate_with_progress(&left, &right, 44100, &mut |p| {
    eprintln!("{}%", p.percent());
})?;
```

`htdemucs_ft` loads the same way but from its **directory** (`ModelVariant::FineTuned`).
`Demucs::separate_file(path, &mut on_progress)` reads a WAV directly (mono is duplicated to stereo).
`from_bytes` is available for in-memory checkpoints. See `cargo doc -p demucs-core`.

## Building

Requires Rust 1.82+ (edition 2021). No CUDA, no Python, no external toolkit — `wgpu` links against
the system graphics driver (Vulkan/DX12/GL/Metal) at runtime.

```bash
cargo build --release          # library + target/release/demucs
cargo test --workspace         # unit + integration tests
```

The alignment and device tests compare against reference dumps and the real `htdemucs` checkpoint;
they skip cleanly when those assets are absent. The heavyweight `htdemucs_ft` bag-vs-shard test is
`#[ignore]`d — run it with `cargo test --release -p demucs-core --test api_facade -- --ignored`.

## Performance (measured)

Numerical alignment against the Python reference (this port's acceptance gate ①, `shifts=0`):

| Check | Result |
|-------|--------|
| Device vs Python, 1 s segment (vocals) | **124.15 dB** SNR |
| Device vs Python, full 176.3 s track (vocals) | **128.79 dB** SNR / 121.36 dB SI-SDR |
| Host (CPU) full track, vocals | 125.69 dB SNR |

Speed (acceptance gate ② is to beat the CUDA reference; measured on a 10 GB NVIDIA P104-100, no
fp16, serial):

| Path | Input | Wall clock | RTFx |
|------|-------|-----------|------|
| torch + CUDA (reference) | whole 176.3 s track | 6.75 s | **26.1x** (target to beat) |
| torch + CUDA (reference) | 20 s clip | — | 26.2x (218 ms/chunk) |
| **this port, wgpu / Vulkan (dGPU)** | 20 s clip | 1.29 s | **15.6x** (265 ms/chunk) |
| this port, wgpu / Vulkan (Intel iGPU) | whole 176.3 s track | 72.1 s | 2.45x |
| this port, CPU (host) | whole 176.3 s track | 48.7 s | 3.62x |

The device path's per-chunk cost is now 265 ms against the reference's 218 ms, i.e. **~1.2x off
cuBLAS** rather than the 2.8x the previous revision of this table recorded: the steady-state chunk
went 299 ms -> 265 ms over the last two commits (the softmax now writes only the per-row
`(max, 1/Σexp)` and the AV product folds the exponential into its own `A` staging, so the
`(heads·tokens, tokens)` probability matrix is never materialised). The host (CPU) path, by
contrast, is complete and the default; the whole-track device figure is remeasured per commit and
is not quoted here yet.

## Project layout

```text
crates/demucs-core/src/
  api.rs        library facade: Demucs, Backend, LoadOptions, Stem/StemId, SeparationProgress
  paths.rs      runtime-resolved default checkpoint locations (no baked-in machine path)
  demucs/
    config.rs   HtdemucsConfig (+ safetensors kwargs) and the derived layer table
    weights.rs  .th / safetensors loaders -> typed weights
    host.rs     CPU forward (per-layer trace matches the reference names)
    pipeline.rs apply_model: chunking, triangular overlap-add, shifts, global normalise
    ops.rs      host operators (conv, group_norm, attention, ...)
  dsp/          torch-aligned STFT / iSTFT
  gpu/          wgpu device layer: arena/recorder, GEMM, conv, group_norm, attention, ...
  audio.rs      WAV I/O;  alloc.rs  caching allocator (installed by the CLI)
crates/demucs-cli/   the `demucs` binary
```

## License

MIT (see workspace metadata). HTDemucs weights remain under their original Meta Research / Demucs
terms and are **not** included in this repository.
