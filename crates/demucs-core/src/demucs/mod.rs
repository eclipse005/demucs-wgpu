//! `HTDemucs` (demucs v4) — the hybrid spectrogram/waveform separator.
//!
//! Scope: inference for the `htdemucs` and `htdemucs_ft` checkpoints only. The
//! `mdx*`, `hdemucs_mmi` and legacy `Demucs` families (BLSTM / `LocalState`
//! attention / Wiener filtering) are deliberately absent, as is training.

pub mod config;
pub mod host;
pub mod ops;
pub mod pipeline;
pub mod spec;
pub mod weights;

pub use config::{HtdemucsArch, HtdemucsConfig, LayerSpec};
pub use host::{Htdemucs, NoTrace, Normalisation, TraceSink, VecTrace};
pub use pipeline::{normalize, separate, SeparateOptions};
pub use weights::{load_weights, HtdemucsWeights, LoadedWeights};
