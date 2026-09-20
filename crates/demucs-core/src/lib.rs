//! Pure-Rust port of the demucs v4 inference stack. The device layer, the STFT and
//! the checkpoint readers came from the previous port in this series.
//!
//! Scope is inference only: config loading, audio I/O, torch-compatible
//! STFT/iSTFT, chunked overlap-add demixing, and model forwards. Training,
//! datasets, augmentation, lightning and wandb are deliberately out of scope.
//!
//! # Two ways in
//!
//! The [`api`] module is what other Rust projects should code against — load a
//! model once, separate as many tracks as you like:
//!
//! ```ignore
//! use demucs_core::{Backend, Demucs, LoadOptions, ModelVariant, StemSelection};
//!
//! let sep = Demucs::load(
//!     "htdemucs.safetensors",
//!     LoadOptions { variant: ModelVariant::FourStem, stems: StemSelection::All },
//!     Backend::Auto,
//! )?;
//! let stems = sep.separate(&left, &right, 44100)?;
//! ```
//!
//! The `demucs` binary (crate `demucs-cli`) is a thin wrapper over it, so the CLI
//! and any library user run exactly the same code.

pub mod alloc;
pub mod api;
pub mod audio;
pub mod ckpt;
pub mod conv;
pub mod demucs;
pub mod dsp;
pub mod error;
pub mod fixtures;
pub mod gpu;
pub mod npy;
pub mod ops;
pub mod paths;
pub mod safetensors;

pub use api::{
    Backend, Demucs, LoadOptions, ModelVariant, SeparationProgress, Stem, StemId, StemSelection,
    AUDIO_CHANNELS, N_FFT, SAMPLE_RATE,
};
pub use error::{Error, Result};
