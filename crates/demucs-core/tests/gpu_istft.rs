//! Device `_ispec` vs the host STFT inverse on HTDemucs shapes.

use demucs_core::demucs::config::HtdemucsConfig;
use demucs_core::demucs::spec::{pad_for_ispec, unpack_channels_as_complex, DemucsSpec};
use demucs_core::dsp::stft::hann_window;
use demucs_core::fixtures::compare;
use demucs_core::gpu::arena::{Arena, Recorder};
use demucs_core::gpu::kernels::Kernels;
use demucs_core::gpu::Gpu;
use ndarray::Array4;
fn gpu_or_skip() -> Option<Gpu> {
    match Gpu::new() {
        Ok(gpu) => Some(gpu),
        Err(e) => {
            eprintln!("skipping: no usable wgpu adapter ({e})");
            None
        }
    }
}

fn fill(n: usize, seed: u32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 0.05
        })
        .collect()
}

#[test]
fn device_ispec_matches_the_host_on_a_short_segment() {
    let Some(gpu) = gpu_or_skip() else {
        return;
    };
    let config = HtdemucsConfig::default();
    let spec = DemucsSpec::new(&config).unwrap();
    let nfft = config.nfft;
    let bins = nfft / 2;
    let hop = config.hop_length();
    let length = hop * 8;
    let frames = spec.frames_for(length);
    let batch = 1usize;
    let sources = 4usize;
    let packed = sources * 4;

    let data = fill(batch * packed * bins * frames, 7);
    let packed_spec =
        Array4::from_shape_vec((batch, packed, bins, frames), data).unwrap();

    let zout = unpack_channels_as_complex(&packed_spec, sources).unwrap();
    let zout = pad_for_ispec(&zout);
    let host = spec.ispec(&zout, length, 0).unwrap();

    let kernels = Kernels::new(&gpu).unwrap();
    let mut arena = Arena::new(&gpu, 256 << 20);
    let mut recorder = Recorder::new(&gpu);
    let window = hann_window(nfft);
    let win = arena.upload(&gpu, &[nfft], &window, "win").unwrap();
    let spec_dev = arena
        .upload(
            &gpu,
            packed_spec.shape(),
            packed_spec.as_slice().unwrap(),
            "spec",
        )
        .unwrap();
    let out = arena
        .tensor(&gpu, &[batch, sources * 2, length], "wave")
        .unwrap();
    kernels
        .ispec_cac_into(
            &gpu,
            &mut arena,
            &mut recorder,
            &spec_dev,
            &win,
            &out,
            batch,
            sources,
            frames,
            bins,
            length,
        )
        .unwrap();
    recorder.submit(&gpu).unwrap();
    let bytes = gpu
        .readback(&out.buffer, (out.len() * 4) as u64)
        .unwrap();
    let values: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();
    let device = Array4::from_shape_vec((batch, sources, 2, length), values).unwrap();

    let snr = compare(host.as_slice().unwrap(), device.as_slice().unwrap())
        .unwrap()
        .snr_db();
    println!("device ispec vs host: {snr:.2} dB");
    assert!(
        snr > 100.0,
        "device ispec {snr:.2} dB, want > 100 against the host inverse"
    );
}
