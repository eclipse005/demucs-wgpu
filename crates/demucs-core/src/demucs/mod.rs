//! `HTDemucs` (demucs v4) — the hybrid spectrogram/waveform separator.
//!
//! Scope: inference for the `htdemucs` and `htdemucs_ft` checkpoints only. The
//! `mdx*`, `hdemucs_mmi` and legacy `Demucs` families (BLSTM / `LocalState`
//! attention / Wiener filtering) are deliberately absent, as is training.

pub mod config;
pub mod host;
pub mod ops;
pub mod pipeline;
pub mod mix;
pub mod streaming;
pub mod spec;
pub mod weights;

pub use config::{HtdemucsArch, HtdemucsConfig, LayerSpec};
pub use host::{Htdemucs, NoTrace, Normalisation, TraceSink, VecTrace};
pub use pipeline::{normalize, separate, SeparateOptions};
pub use mix::{scan_norm_file, window_cap_for, MixSamples, WindowedMix};
pub use streaming::{apply_norm_inplace, scan_norm, separate_streaming_normalized, ShiftView, SplitMeta};
pub use weights::{load_weights, HtdemucsWeights, LoadedWeights};
