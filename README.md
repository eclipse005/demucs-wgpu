# Demucs wgpu

**HTDemucs v4 music source separation in Rust with wgpu.**

**English** · [简体中文](README.zh-CN.md)

A lightweight, cross-platform Rust implementation of Meta's [HTDemucs v4](https://huggingface.co/adefossez/HTDemucs) (`htdemucs`, `htdemucs_ft`), using [wgpu](https://github.com/gfx-rs/wgpu) for GPU acceleration. Hand-written CPU + wgpu inference — no Python, no torch, no CUDA toolkit.

The goal is simple: separate a stereo mix into **drums / bass / other / vocals** locally and natively, on any GPU that wgpu can drive (Vulkan, DX12, Metal, OpenGL) or on the CPU. Built to feed vocal isolation into an ASR pipeline.

### Features

* 🦀 Pure Rust
* 🎮 GPU acceleration with wgpu
* 🌍 Vulkan / DX12 / Metal / OpenGL
* 🖥️ Windows / macOS / Linux
* ⚡ CPU fallback
* 📦 Offline local inference
* 🎵 HTDemucs v4 (`htdemucs` / `htdemucs_ft`)
* 🎤 Single-stem vocals mode for ASR pipelines
* 📊 Per-chunk progress callbacks
* 🧩 CLI + Rust library

### Install

As a Cargo dependency:

```toml
[dependencies]
demucs-core = { git = "https://github.com/eclipse005/demucs-wgpu.git" }
```

Or build the CLI from source:

```bash
git clone https://github.com/eclipse005/demucs-wgpu.git
cd demucs-wgpu
cargo build --release        # target/release/demucs
cargo test --workspace       # unit + integration tests
```

Requires Rust 1.82+ (edition 2021). No CUDA, no Python, no external toolkit — `wgpu` links against the system graphics driver (Vulkan/DX12/GL/Metal) at runtime. Tests compare against reference dumps and the real `htdemucs` checkpoint and skip cleanly when those assets are absent; the heavyweight `htdemucs_ft` bag-vs-shard test is `#[ignore]`d — run it with `cargo test --release -p demucs-core --test api_facade -- --ignored`.

### Model weights

Weights are **not** included in this repository. Grab the official checkpoints from the [adefossez/HTDemucs](https://huggingface.co/adefossez/HTDemucs) Hugging Face snapshot (rights remain with the original authors). The loader picks the format by extension: `.th` / `.pth` / `.pt` are torch-zip archives, anything else is treated as `safetensors` (config from the file's metadata, falling back to built-in `htdemucs` defaults). `htdemucs_ft` is stored as fp16 and upcast to fp32 on load.

| Model (`--model`) | What to point it at | Size |
|-------------------|---------------------|------|
| `htdemucs` | one file: `955717e8-8726e21a.th` **or** a `htdemucs.safetensors` | ~84 MB |
| `htdemucs_ft` | a **directory** holding `htdemucs_ft.yaml` + the four shard `.safetensors` | ~4 × 84 MB |

The fine-tuned variant is a *bag* of four single-stem networks plus a weight matrix (`htdemucs_ft.yaml`). The shipped matrix is the identity, so each shard produces exactly one stem — load the bag from its directory, not as a single merged file:

| Shard | Stem |
|-------|------|
| `f7e0c4bc.safetensors` | drums |
| `d12395a8.safetensors` | bass |
| `92cfc3b6.safetensors` | other |
| `04573f0d.safetensors` | vocals |

For a vocals-only ASR consumer, the bag can be skipped entirely: the bag's vocals lane is bit-for-bit identical to the `04573f0d` shard run on its own (locked by a test), so shipping just that one shard as `ModelVariant::FourStem` gives the same output at roughly a quarter of the size. When the bag is loaded, `StemSelection::Some([...])` runs only the networks that feed the requested stems.

### Quick Start

```powershell
# pick the GPU adapter, isolate vocals deterministically
demucs separate song.wav -o out --device vulkan --shifts 0 --stem vocals

# the full four-stem fine-tuned bag
demucs separate song.wav -o out --model htdemucs_ft --device auto
```

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

Other subcommands: `demucs adapters` (list wgpu adapters), `info` (model config and derived layer table), `compare` (SNR between two WAVs), and the dev tools `bench`, `profile`, `kernels`. The built-in default checkpoint paths are machine-specific — pass `--model <path>` to use your own.

### Library

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

For progress (a GUI task list, say), pass a callback — it runs once per chunk and always ends at `done == total`, including across shift passes and the fine-tuned bag:

```rust
let stems = sep.separate_with_progress(&left, &right, 44100, &mut |p| {
    eprintln!("{}%", p.percent());
})?;
```

`htdemucs_ft` loads the same way but from its **directory** (`ModelVariant::FineTuned`). `Demucs::separate_file(path, &mut on_progress)` reads a WAV directly (mono is duplicated to stereo); `from_bytes` accepts in-memory checkpoints. See `cargo doc -p demucs-core` for the full API.

### Backend support

| Backend | Status |
|---------|--------|
| CPU (host forward, rayon + `gemm`) | supported |
| wgpu — **Vulkan** | supported; the validated path |
| wgpu — Metal / OpenGL | compile and run via wgpu; validated on Vulkan only |
| wgpu — DX12 | driver-dependent; Vulkan is recommended |

### Scope

Inference only, and only the `htdemucs` family: `htdemucs` and `htdemucs_ft`. Not implemented (by design): `mdx_*`, `hdemucs_mmi`, the v3 `Demucs`, `htdemucs_6s` (6 stems, a different architecture), training, mp3/flac I/O (float WAV only), and the upstream CLI's `--jobs` / segment-model ensemble.

### Performance (measured)

Numerical alignment against the Python reference (`shifts=0`, the port's acceptance gate):

| Check | Result |
|-------|--------|
| Device vs Python, 1 s segment (vocals) | **124.15 dB** SNR |
| Device vs Python, full 176.3 s track (vocals) | **129.13 dB** SNR |
| Device vs Python, 20 s clip (vocals) | **125.78 dB** SNR |
| Host (CPU) full track, vocals | 125.69 dB SNR |

Speed, measured on a 10 GB NVIDIA P104-100 (no fp16, serial). The two 176.3 s rows were measured back to back on the same input in the same session:

| Path | Input | Wall clock | RTFx |
|------|-------|-----------|------|
| torch + CUDA (reference), same session | 176.3 s track | 8.86 s | **19.90x** |
| **this port, wgpu / Vulkan (dGPU)** | 176.3 s track | 9.82 s | **17.96x** |
| this port, wgpu / Vulkan (Intel iGPU) | 176.3 s track | 72.1 s | 2.45x |
| this port, CPU (host) | 176.3 s track | 48.7 s | 3.62x |
| this port, wgpu / Vulkan (dGPU) | 20 s clip | 1.28 s | 15.6x |

Acceptance numbers for the `htdemucs_ft` vocals specialist — the model an ASR consumer ships — against the reference run through `apply_model` with the port's settings (`shifts=0`, `overlap=0.25`, `split=True`):

| Input | Reference (CUDA) | This port (Vulkan) | SNR vs reference |
|-------|------------------|--------------------|------------------|
| 20 s clip | 1.87 s / 10.7x | 0.92 s / **21.7x** | 125.34 dB |
| 176.3 s track | 7.99 s / **22.06x** | 6.51 s / **27.08x** | 127.17 dB |

Measurement sessions drift ±4% on this machine, so paired back-to-back runs are the meaningful comparison; the reference methodology is documented in `tools/ft_vocals_reference.py`.

### Why wgpu?

Instead of relying on CUDA, ROCm, or other vendor-specific runtimes, this project uses **wgpu** as a unified GPU abstraction.

This makes it possible to build a single Rust-based separation runtime for different platforms and GPU vendors.

### Project Status

🚧 **Active development**

Performance and hardware compatibility are still being actively optimized and tested across different GPUs.

### Related

* [HTDemucs](https://huggingface.co/adefossez/HTDemucs) — the original model and reference implementation
* [wgpu](https://github.com/gfx-rs/wgpu)
* [demucs-native-rs](https://github.com/eclipse005/demucs-native-rs) — the CUDA port sibling; same crate shape and public API (`Demucs::load` / `separate` / `separate_with_progress` / `Stem` / `StemSelection` / `LoadOptions` / `Backend` / `ModelVariant`), so switching a consumer between the two is a matter of the `Backend` enum and where the weights live

### License

MIT. HTDemucs weights remain under their original Meta Research / Demucs terms and are **not** included in this repository.
