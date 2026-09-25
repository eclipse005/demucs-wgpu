# Demucs wgpu

**基于 wgpu 的 HTDemucs v4 音乐源分离（Rust 实现）。**

[English](README.md) · **简体中文**

Meta [HTDemucs v4](https://huggingface.co/adefossez/HTDemucs)（`htdemucs`、`htdemucs_ft`）的轻量级跨平台 Rust 实现，使用 [wgpu](https://github.com/gfx-rs/wgpu) 进行 GPU 加速。手写 CPU + wgpu 推理——不依赖 Python、torch，也不需要 CUDA 工具链。

目标很简单：把立体声混音分离为**鼓 / 贝斯 / 其他 / 人声**四轨，在 wgpu 能驱动的任何显卡（Vulkan、DX12、Metal、OpenGL）或 CPU 上本地原生运行。为 ASR 流水线提供人声隔离而构建。

### 特性

* 🦀 纯 Rust
* 🎮 wgpu GPU 加速
* 🌍 Vulkan / DX12 / Metal / OpenGL
* 🖥️ Windows / macOS / Linux
* ⚡ CPU 回退
* 📦 离线本地推理
* 🎵 HTDemucs v4（`htdemucs` / `htdemucs_ft`）
* 🎤 面向 ASR 流水线的单轨人声模式
* 📊 分块进度回调
* 🧩 CLI + Rust 库

### 安装

作为 Cargo 依赖：

```toml
[dependencies]
demucs-core = { git = "https://github.com/eclipse005/demucs-wgpu.git" }
```

或从源码构建 CLI：

```bash
git clone https://github.com/eclipse005/demucs-wgpu.git
cd demucs-wgpu
cargo build --release        # target/release/demucs
cargo test --workspace       # 单元 + 集成测试
```

需要 Rust 1.82+（edition 2021）。不依赖 CUDA、Python 或任何外部工具链——`wgpu` 在运行时链接系统显卡驱动（Vulkan/DX12/GL/Metal）。测试会与参考转储及真实 `htdemucs` 检查点比对，缺失这些资源时自动跳过；重量级的 `htdemucs_ft` 集合与分片对比测试标记为 `#[ignore]`，可用 `cargo test --release -p demucs-core --test api_facade -- --ignored` 运行。

### 模型权重

权重**不在**本仓库内。请从 [adefossez/HTDemucs](https://huggingface.co/adefossez/HTDemucs) 的 Hugging Face 快照获取官方检查点（版权归原作者）。加载器按扩展名识别格式：`.th` / `.pth` / `.pt` 为 torch-zip 归档，其余一律按 `safetensors` 处理（配置来自文件元数据，缺省回退到内置 `htdemucs` 默认值）。`htdemucs_ft` 以 fp16 存储，加载时提升为 fp32。

| 模型（`--model`） | 指向内容 | 体积 |
|-------------------|----------|------|
| `htdemucs` | 单个文件：`955717e8-8726e21a.th` 或 `htdemucs.safetensors` | ~84 MB |
| `htdemucs_ft` | 一个**目录**，包含 `htdemucs_ft.yaml` + 四个分片 `.safetensors` | ~4 × 84 MB |

微调版本是四个单轨网络加上一个权重矩阵组成的**集合**（`htdemucs_ft.yaml`）。出厂矩阵是单位矩阵，因此每个分片恰好产生一个音轨——请以目录形式加载，而不是单个合并文件：

| 分片 | 音轨 |
|------|------|
| `f7e0c4bc.safetensors` | drums |
| `d12395a8.safetensors` | bass |
| `92cfc3b6.safetensors` | other |
| `04573f0d.safetensors` | vocals |

仅需要人声的 ASR 消费者可以完全跳过集合：人声通道与单独运行 `04573f0d` 分片逐比特一致（有测试锁定），因此只随包分发这一个分片并以 `ModelVariant::FourStem` 加载，输出相同而体积约为四分之一。加载完整集合时，`StemSelection::Some([...])` 只运行所请求音轨对应的网络。

### 快速上手

```powershell
# 指定 GPU 适配器，确定性输出人声
demucs separate song.wav -o out --device vulkan --shifts 0 --stem vocals

# 完整的四轨微调集合
demucs separate song.wav -o out --model htdemucs_ft --device auto
```

```text
demucs separate <input.wav> -o <out-dir|out.wav> [options]

  -m, --model <name|path>     htdemucs | htdemucs_ft | 检查点文件 | ft 目录
  -o, --output <path>         输出目录，或单音轨时的单个 .wav
      --device <spec>         cpu | auto | vulkan | vulkan:<idx> | <适配器名子串>
      --stem <STEM>           只输出该音轨（ASR 速记，等价 --two-stems --other-method none）
      --two-stems <STEM>      音轨 + 伴生轨，与参考 CLI 一致
      --other-method <m>      add | minus | none   （--two-stems 的伴生方式；默认 add）
      --shifts <N>            随机时移取平均；0 为确定性输出（默认 1）
      --overlap <F>           分块重叠比例（默认 0.25）
      --segment <SEC>         覆盖模型的分段长度
      --timings               打印每块耗时
      --dump-trace <dir>      记录首块激活（仅单网络）
```

其他子命令：`demucs adapters`（列出 wgpu 适配器）、`info`（模型配置与派生层表）、`compare`（两个 WAV 的 SNR），以及开发工具 `bench`、`profile`、`kernels`。内置的默认检查点路径与作者机器绑定——请通过 `--model <path>` 指定自己的权重。

### 作为库使用

```rust
use demucs_core::{Backend, Demucs, LoadOptions, ModelVariant, StemSelection};

// htdemucs：单个检查点文件。
let sep = Demucs::load(
    "models/htdemucs.safetensors",
    LoadOptions { variant: ModelVariant::FourStem, stems: StemSelection::All },
    Backend::Auto, // 或 Backend::Cpu，或 Backend::Gpu(DeviceSelector::parse("vulkan")?)
)?;

let stems = sep.separate(&left, &right, 44100)?;
for s in &stems {
    println!("{}: {} samples", s.id, s.left.len()); // s.id -> "drums" | "bass" | ...
}
```

需要进度（例如 GUI 任务列表）时传入回调——每个分块触发一次，且最终必然到达 `done == total`，跨时移轮次与微调集合同样成立：

```rust
let stems = sep.separate_with_progress(&left, &right, 44100, &mut |p| {
    eprintln!("{}%", p.percent());
})?;
```

`htdemucs_ft` 以相同方式从其**目录**加载（`ModelVariant::FineTuned`）。`Demucs::separate_file(path, &mut on_progress)` 直接读取 WAV（单声道会复制为立体声）；`from_bytes` 接受内存中的检查点。完整 API 见 `cargo doc -p demucs-core`。

### 后端支持

| 后端 | 状态 |
|------|------|
| CPU（宿主前向，rayon + `gemm`） | 支持 |
| wgpu —— **Vulkan** | 支持；经过验证的路径 |
| wgpu —— Metal / OpenGL | 可经 wgpu 编译运行；仅在 Vulkan 上做过验证 |
| wgpu —— DX12 | 依赖驱动；建议使用 Vulkan |

### 范围

仅推理，且仅覆盖 `htdemucs` 家族：`htdemucs` 与 `htdemucs_ft`。有意不支持：`mdx_*`、`hdemucs_mmi`、v3 `Demucs`、`htdemucs_6s`（六轨，不同架构）、训练、mp3/flac 读写（仅 float WAV）、上游 CLI 的 `--jobs` / 分段模型集成。

### 性能（实测）

与 Python 参考实现的数值对齐（`shifts=0`，本移植的验收门槛）：

| 检查项 | 结果 |
|--------|------|
| 设备 vs Python，1 s 片段（人声） | **124.15 dB** SNR |
| 设备 vs Python，完整 176.3 s 曲目（人声） | **129.13 dB** SNR |
| 设备 vs Python，20 s 片段（人声） | **125.78 dB** SNR |
| 宿主（CPU）完整曲目，人声 | 125.69 dB SNR |

速度，测量于 10 GB NVIDIA P104-100（无 fp16，串行）。两行 176.3 s 在同一会话中对同一输入背靠背测得：

| 路径 | 输入 | 墙钟时间 | RTFx |
|------|------|----------|------|
| torch + CUDA（参考），同会话 | 176.3 s 曲目 | 8.86 s | **19.90x** |
| **本移植，wgpu / Vulkan（独立显卡）** | 176.3 s 曲目 | 9.82 s | **17.96x** |
| 本移植，wgpu / Vulkan（Intel 核显） | 176.3 s 曲目 | 72.1 s | 2.45x |
| 本移植，CPU（宿主） | 176.3 s 曲目 | 48.7 s | 3.62x |
| 本移植，wgpu / Vulkan（独立显卡） | 20 s 片段 | 1.28 s | 15.6x |

`htdemucs_ft` 人声专项——ASR 消费者实际随包分发的模型——对照参考实现（经 `apply_model`、采用本移植的设置：`shifts=0`、`overlap=0.25`、`split=True`）的验收数字：

| 输入 | 参考（CUDA） | 本移植（Vulkan） | 对参考 SNR |
|------|--------------|------------------|------------|
| 20 s 片段 | 1.87 s / 10.7x | 0.92 s / **21.7x** | 125.34 dB |
| 176.3 s 曲目 | 7.99 s / **22.06x** | 6.51 s / **27.08x** | 127.17 dB |

本机的测量会话存在 ±4% 漂移，因此背靠背的成对运行才是有意义的比较；参考方法学见 `tools/ft_vocals_reference.py`。

### 为什么选择 wgpu？

本项目不依赖 CUDA、ROCm 或其他特定厂商的运行时，而是以 **wgpu** 作为统一的 GPU 抽象层。

这使得同一套 Rust 分离运行时可以覆盖不同平台与不同显卡厂商。

### 项目状态

🚧 **积极开发中**

性能与硬件兼容性仍在不同显卡上持续优化与测试。

### 相关项目

* [HTDemucs](https://huggingface.co/adefossez/HTDemucs) —— 原始模型与参考实现
* [wgpu](https://github.com/gfx-rs/wgpu)
* [demucs-native-rs](https://github.com/eclipse005/demucs-native-rs) —— CUDA 版姊妹项目；crate 结构与公开 API 相同（`Demucs::load` / `separate` / `separate_with_progress` / `Stem` / `StemSelection` / `LoadOptions` / `Backend` / `ModelVariant`），在两者之间切换消费端只需改 `Backend` 枚举与权重路径

### 许可证

MIT。HTDemucs 权重仍遵循其原始的 Meta Research / Demucs 条款，**不**包含在本仓库内。
