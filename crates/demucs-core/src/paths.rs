//! Default on-disk locations for the bundled checkpoints, resolved from the
//! user's home / cache directories at runtime.
//!
//! The CLI's bare `--model htdemucs` / `htdemucs_ft` names and the tests'
//! asset lookups point here, so no developer's absolute path is baked into the
//! crate. Every accessor honours the conventional env override (`TORCH_HOME`,
//! `HF_HOME`) and otherwise falls back to the standard cache layout; if a home
//! directory cannot be determined it returns a path that simply will not
//! exist, which callers already treat as "not present".

use std::path::PathBuf;

/// The htdemucs v4 checkpoint filename (torch hub cache name).
pub const HTDEMUCS_CHECKPOINT: &str = "955717e8-8726e21a.th";
/// The `htdemucs_ft` Hugging Face snapshot revision that carries
/// `htdemucs_ft.yaml` and the four per-stem shards.
pub const HTDEMUCS_FT_REVISION: &str = "d74ac89c3a1e874fc78f152555cf4d8533f06cd4";

/// Best-effort home directory (POSIX `$HOME`, Windows `%USERPROFILE%`).
pub fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("USERPROFILE").filter(|s| !s.is_empty()).map(PathBuf::from))
}

/// The torch-hub checkpoint directory: `$TORCH_HOME/hub/checkpoints`, else
/// `$XDG_CACHE_HOME/torch/hub/checkpoints`, else `~/.cache/torch/hub/checkpoints`.
fn torch_checkpoints_dir() -> PathBuf {
    let base = std::env::var_os("TORCH_HOME")
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("XDG_CACHE_HOME")
                .filter(|s| !s.is_empty())
                .map(|c| PathBuf::from(c).join("torch"))
        })
        .or_else(|| home_dir().map(|h| h.join(".cache").join("torch")));
    base.unwrap_or_else(|| PathBuf::from("torch")).join("hub").join("checkpoints")
}

/// `~/.cache/huggingface/hub` (or `$HF_HOME/hub` / `$HF_HUB_CACHE`).
fn hf_hub_dir() -> PathBuf {
    std::env::var_os("HF_HUB_CACHE")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HF_HOME").filter(|s| !s.is_empty()).map(|h| PathBuf::from(h).join("hub")))
        .or_else(|| home_dir().map(|h| h.join(".cache").join("huggingface").join("hub")))
        .unwrap_or_else(|| PathBuf::from("huggingface").join("hub"))
}

/// The default local path to the `htdemucs` checkpoint.
pub fn htdemucs_checkpoint() -> PathBuf {
    torch_checkpoints_dir().join(HTDEMUCS_CHECKPOINT)
}

/// The `htdemucs_ft` snapshot **parent** directory (`…/snapshots`), which holds
/// one subdirectory per revision.
pub fn htdemucs_ft_snapshots_dir() -> PathBuf {
    hf_hub_dir()
        .join("models--adefossez--HTDemucs-ft")
        .join("snapshots")
}

/// The default `htdemucs_ft` snapshot directory — the folder containing
/// `htdemucs_ft.yaml` and the four shard `.safetensors`.
pub fn htdemucs_ft_dir() -> PathBuf {
    htdemucs_ft_snapshots_dir().join(HTDEMUCS_FT_REVISION)
}
