//! Host RAM admit check before separation.
//!
//! After P0 streaming OLA, peak RSS still scales with the input mix
//! (~56.3 MiB/min ≈ 5.58 full-track f32 channel planes at 44.1 kHz). Call
//! [`admit_separation`] before allocating the mix / running separate so a
//! long track returns a clear error instead of being killed by the OS.

use crate::error::{Error, Result};

/// Measured post-P0 host RSS slope (MiB wall-clock minute of mix @ 44.1 kHz).
pub const SLOPE_MIB_PER_MIN: f64 = 56.3;

/// Default fraction of available RAM the estimate may consume.
pub const DEFAULT_RAM_FRACTION: f64 = 0.75;

/// Activation / arena / OLA ring budget that does not scale with duration.
/// Covers a 7.8 s segment forward plus host caches; weights are counted
/// separately via [`model_working_set_bytes`].
const ACTIVATION_BUDGET_BYTES: u64 = 768 << 20;

/// Bytes for model weights (`parameter_count` f32s) plus a fixed activation budget.
pub fn model_working_set_bytes(parameter_count: usize) -> u64 {
    (parameter_count as u64)
        .saturating_mul(4)
        .saturating_add(ACTIVATION_BUDGET_BYTES)
}

/// Estimated peak host bytes for a track of `duration_secs` once the model is loaded.
pub fn estimate_peak_bytes(duration_secs: f64, model_ws_bytes: u64) -> u64 {
    let minutes = duration_secs.max(0.0) / 60.0;
    let slope = (SLOPE_MIB_PER_MIN * minutes * 1024.0 * 1024.0).ceil() as u64;
    slope.saturating_add(model_ws_bytes)
}

/// Fraction of available RAM to allow; override with `DEMUCS_MEM_FRACTION` (0, 1].
pub fn ram_fraction() -> f64 {
    match std::env::var("DEMUCS_MEM_FRACTION") {
        Ok(raw) => raw
            .parse::<f64>()
            .ok()
            .filter(|v| *v > 0.0 && *v <= 1.0)
            .unwrap_or(DEFAULT_RAM_FRACTION),
        Err(_) => DEFAULT_RAM_FRACTION,
    }
}

/// Best-effort available RAM. `None` when the platform cannot report it.
pub fn available_ram_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        return available_ram_linux();
    }
    #[cfg(windows)]
    {
        return available_ram_windows();
    }
    #[cfg(not(any(target_os = "linux", windows)))]
    {
        None
    }
}

#[cfg(target_os = "linux")]
fn available_ram_linux() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in text.lines() {
        let Some(rest) = line.strip_prefix("MemAvailable:") else {
            continue;
        };
        let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
        return Some(kb.saturating_mul(1024));
    }
    None
}

#[cfg(windows)]
fn available_ram_windows() -> Option<u64> {
    #[repr(C)]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page_file: u64,
        avail_page_file: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended_virtual: u64,
    }
    extern "system" {
        fn GlobalMemoryStatusEx(status: *mut MemoryStatusEx) -> i32;
    }
    unsafe {
        let mut status = std::mem::zeroed::<MemoryStatusEx>();
        status.length = std::mem::size_of::<MemoryStatusEx>() as u32;
        if GlobalMemoryStatusEx(&mut status) == 0 {
            return None;
        }
        // Prefer commit headroom when larger; else physical available.
        Some(status.avail_page_file.max(status.avail_phys))
    }
}

/// Reject the run when the P0 slope estimate exceeds `ram_fraction()` of available RAM.
///
/// When available RAM cannot be queried, the check is skipped (no false refusal).
pub fn admit_separation(duration_secs: f64, model_ws_bytes: u64) -> Result<()> {
    admit_separation_with_fraction(duration_secs, model_ws_bytes, ram_fraction())
}

/// Same as [`admit_separation`] with an explicit fraction (tests / callers).
pub fn admit_separation_with_fraction(
    duration_secs: f64,
    model_ws_bytes: u64,
    fraction: f64,
) -> Result<()> {
    let Some(available) = available_ram_bytes() else {
        return Ok(());
    };
    let fraction = fraction.clamp(0.0, 1.0);
    let budget = ((available as f64) * fraction).floor() as u64;
    let estimate = estimate_peak_bytes(duration_secs, model_ws_bytes);
    if estimate <= budget {
        return Ok(());
    }
    let est_mib = estimate as f64 / (1024.0 * 1024.0);
    let avail_mib = available as f64 / (1024.0 * 1024.0);
    let budget_mib = budget as f64 / (1024.0 * 1024.0);
    let mins = duration_secs / 60.0;
    let pct = fraction * 100.0;
    Err(Error::Memory(format!(
        "audio is {mins:.1} min; estimated peak ~{est_mib:.0} MiB \
         (slope {SLOPE_MIB_PER_MIN} MiB/min + model WS) exceeds \
         {pct:.0}% of available RAM ({avail_mib:.0} MiB available, \
         budget {budget_mib:.0} MiB). Split the track or free memory."
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slope_matches_p0_two_hour_ballpark() {
        // 120 min * 56.3 ≈ 6756 MiB of slope alone.
        let est = estimate_peak_bytes(7200.0, 0);
        let mib = est as f64 / (1024.0 * 1024.0);
        assert!((mib - 6756.0).abs() < 2.0, "got {mib}");
    }

    #[test]
    fn admit_allows_short_clip_when_ram_known() {
        if available_ram_bytes().is_none() {
            return;
        }
        admit_separation_with_fraction(
            2.0,
            model_working_set_bytes(42_000_000),
            DEFAULT_RAM_FRACTION,
        )
        .unwrap();
    }

    #[test]
    fn admit_rejects_absurd_duration() {
        let Some(available) = available_ram_bytes() else {
            return;
        };
        let result = admit_separation_with_fraction(
            7200.0,
            model_working_set_bytes(42_000_000),
            0.01,
        );
        assert!(result.is_err(), "available was {available}");
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("estimated peak") || msg.contains("MiB"), "{msg}");
    }
}
