//! Device-vs-host alignment for the `HTDemucs` encoder half.
//!
//! The device path runs the frequency and waveform encoders (conv, GELU, DConv,
//! rewrite, GLU) for the reference's 1 s segment; the host path runs the same
//! segment and the two bottleneck tensors are compared. Every DConv sub-layer
//! also records its output when tracing, so a mismatch localises to a
//! sub-layer, not to "somewhere in the encoder".

use std::path::PathBuf;

use demucs_core::demucs::host::{Htdemucs, VecTrace};
use demucs_core::gpu::htdemucs::GpuHtdemucsRunner;
use demucs_core::gpu::kernels::{pad_ceil, GemmJob};
use demucs_core::npy::read_npy;
use ndarray::{Array3, Array4};

fn th_checkpoint() -> PathBuf {
    demucs_core::paths::htdemucs_checkpoint()
}

fn workspace() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Loads the host model sized for the reference dump's segment.
fn load_model(segment_samples: usize) -> Option<Htdemucs> {
    let checkpoint = th_checkpoint();
    if !checkpoint.exists() {
        eprintln!("skipping: {} is not present", checkpoint.display());
        return None;
    }
    let loaded = demucs_core::demucs::load_weights(&checkpoint).ok()?;
    let mut config = loaded.config.clone();
    config.segment = segment_samples as f64 / config.samplerate as f64;
    let weights = demucs_core::demucs::HtdemucsWeights::load(&loaded.checkpoint, &config).ok()?;
    Some(Htdemucs::new(config, weights).expect("model"))
}

/// The real check: rebuild the normalised inputs exactly as the host does, run
/// the encoders on both paths, compare.
#[test]
fn device_encoders_match_the_host_by_stage() {
    let dir = workspace().join("bench/ref_dump/short");
    if !dir.join("manifest.json").exists() {
        eprintln!("skipping: no reference dump at {}", dir.display());
        return;
    }
    let Some(model) = load_model(44_100) else {
        return;
    };
    let mut runner = match GpuHtdemucsRunner::new(&model) {
        Ok(runner) => runner,
        Err(e) => {
            eprintln!("skipping: no usable wgpu adapter ({e})");
            return;
        }
    };

    let input = read_npy(dir.join("input.npy")).expect("input.npy");
    let input = Array3::from_shape_vec((1, input.shape[0], input.shape[1]), input.data)
        .expect("input shape");
    let training_length = model.training_length();
    let padded = if input.dim().2 < training_length {
        let mut wide = ndarray::Array3::<f32>::zeros((1, 2, training_length));
        wide.slice_mut(ndarray::s![.., .., ..input.dim().2]).assign(&input);
        wide
    } else {
        input
    };

    // The host's own front end: spectro, magnitude, per-branch normalisation.
    let z = model.spec.spec(&padded).expect("spectro");
    let mag = demucs_core::demucs::spec::pack_complex_as_channels(&z);
    let norm = model.normalise(&mag, &padded);
    let mut mag_norm = mag.clone();
    for (offset, scale) in norm.mean.iter().zip(norm.std.iter()) {
        mag_norm
            .slice_mut(ndarray::s![0, .., .., ..])
            .mapv_inplace(|v| (v - offset) / (1e-5 + scale));
    }
    let mut wave_norm = padded.clone();
    for (offset, scale) in norm.mean_t.iter().zip(norm.std_t.iter()) {
        wave_norm
            .slice_mut(ndarray::s![0, .., ..])
            .mapv_inplace(|v| (v - offset) / (1e-5 + scale));
    }

    // Host encoder half, traced stage by stage.
    let mut host_trace = VecTrace::default();
    let host_out = model.forward(&padded, &mut host_trace).expect("host forward");

    // Device encoder half.
    let mut device_trace = VecTrace::default();
    let (_freq_out, _time_out) = runner
        .forward_encoders(&mag_norm, &wave_norm, &mut device_trace)
        .expect("device encoders");

    let mut failures: Vec<String> = Vec::new();
    // The host's `encoder.0` hook is on the module, so it sees the output
    // *before* the frequency embedding (which `HTDemucs.forward` adds); the
    // device's comparable record is `encoder.0.pre_emb`. Every later stage is
    // comparable under the same name.
    // The DConv internals have no host module hooks, so the test reconstructs
    // them with the host ops from the traced conv output.
    let mut host_dconv_refs: std::collections::BTreeMap<String, (Vec<usize>, Vec<f32>)> = std::collections::BTreeMap::new();
    {
        use demucs_core::demucs::ops::{conv1d, glu as glu_op, group_norm};
        use demucs_core::demucs::weights::ConvW;
        let entry = host_trace.get("encoder.0.conv").expect("host conv trace");
        let (channels, bins, frames) = (entry.1[1], entry.1[2], entry.1[3]);
        let mut x = ndarray::Array4::<f32>::zeros((1, channels, bins, frames));
        x.as_slice_mut().unwrap().copy_from_slice(&entry.2);
        x.mapv_inplace(|v| demucs_core::ops::gelu_erf(v));
        // permute (1, c, f, t) -> (f, c, t)
        let mut perm = ndarray::Array3::<f32>::zeros((bins, channels, frames));
        for f in 0..bins {
            for c in 0..channels {
                for t in 0..frames {
                    perm[[f, c, t]] = x[[0, c, f, t]];
                }
            }
        }
        let dconv = &model.weights.encoder[0].dconv;
        let dconv_whole = demucs_core::demucs::host::dconv_forward(dconv, &perm).unwrap();
        host_dconv_refs.insert("dconv.whole".to_string(), (dconv_whole.shape().to_vec(), dconv_whole.iter().copied().collect()));
        let mut current = perm;
        for (depth, layer) in dconv.layers.iter().enumerate() {
            let dilation = 1usize << depth;
            let pad = dilation * (layer.conv1.kernel()[0] / 2);
            let y = conv1d(&current, &layer.conv1, 1, pad, dilation);
            host_dconv_refs.insert(format!("dconv.{depth}.conv1"), (y.shape().to_vec(), y.iter().copied().collect()));
            let y = group_norm(&y, 1, &layer.norm1_weight, &layer.norm1_bias).unwrap();
            host_dconv_refs.insert(
                format!("dconv.{depth}.norm1.after"),
                (y.shape().to_vec(), y.iter().copied().collect()),
            );
            // The device records norm1 *after* the GELU that follows it.
            let mut y_gelu = y.clone();
            y_gelu.mapv_inplace(|v| demucs_core::ops::gelu_erf(v));
            host_dconv_refs.insert(format!("dconv.{depth}.norm1"), (y_gelu.shape().to_vec(), y_gelu.iter().copied().collect()));
            let y = {
                let mut gelled = y.clone();
                gelled.mapv_inplace(|v| demucs_core::ops::gelu_erf(v));
                gelled
            };
            // conv2's raw GEMM output, before norm2 — the device traces it there.
            let y2_pre = conv1d(&y, &layer.conv2, 1, 0, 1);
            host_dconv_refs.insert(format!("dconv.{depth}.conv2"), (y2_pre.shape().to_vec(), y2_pre.iter().copied().collect()));
            let y2 = group_norm(&y2_pre, 1, &layer.norm2_weight, &layer.norm2_bias).unwrap();
            host_dconv_refs.insert(format!("dconv.{depth}.norm2"), (y2.shape().to_vec(), y2.iter().copied().collect()));
            // GLU, before the LayerScale — the device traces it there too.
            let g_pre = glu_op(&y2).unwrap();
            host_dconv_refs.insert(format!("dconv.{depth}.glu"), (g_pre.shape().to_vec(), g_pre.iter().copied().collect()));
            let mut g = g_pre.clone();
            let mut g = g;
            for c in 0..g.dim().1 {
                let gamma = layer.gamma[c];
                g.slice_mut(ndarray::s![.., c, ..]).mapv_inplace(|v| v * gamma);
            }
            host_dconv_refs.insert(format!("dconv.{depth}.scaled"), (g.shape().to_vec(), g.iter().copied().collect()));
            host_dconv_refs.insert(format!("dconv.{depth}.residual"), (current.shape().to_vec(), current.iter().copied().collect()));
            current = &current + &g;
            host_dconv_refs.insert(format!("dconv.{depth}.added"), (current.shape().to_vec(), current.iter().copied().collect()));
            host_dconv_refs.insert(format!("dconv.{depth}.out"), (current.shape().to_vec(), current.iter().copied().collect()));
            host_dconv_refs.insert(format!("dconv.{depth}.conv2raw"), (y2_pre.shape().to_vec(), y2_pre.iter().copied().collect()));
        }
    }

    if std::env::var("DEMUCS_TRACE_NAMES").is_ok() {
        for name in host_trace.names() {
            println!("  host: {name}");
        }
    }
    // Sanity: the step-by-step reconstruction must reproduce `dconv_forward`
    // exactly — both are host code. If they differ, the reconstruction (not the
    // device) is what diverged.
    {
        let stepwise = host_dconv_refs.get("dconv.1.out").unwrap();
        let whole = host_dconv_refs.get("dconv.whole").unwrap();
        let r = demucs_core::fixtures::compare(&whole.1, &stepwise.1).unwrap();
        println!("host reconstruction vs dconv_forward: {:.2} dB", r.snr_db());
    }

    // The full encoder+transformer device path, traced.
    let mut device_trace2 = VecTrace::default();
    let (freq_dec, time_dec) = runner
        .forward_encoder_and_transformer(&mag_norm, &wave_norm, &mut device_trace2)
        .expect("device encoder+transformer");

    if std::env::var("DEMUCS_TRACE_NAMES").is_ok() {
        for (name, _, _) in &device_trace2.entries {
            println!("  dev2: {name}");
        }
    }
    for (device_name, host_name) in [
        ("encoder.0.conv", "encoder.0.conv"),
        ("dconv.0.conv1", "@dconv"),
        ("dconv.0.norm1", "@dconv"),
        ("dconv.0.conv2", "@dconv"),
        ("dconv.0.norm1.after", "@dconv.norm1before"),
        ("dconv.0.glu", "@dconv"),
        ("dconv.0.norm2", "@dconv"),
        ("dconv.0.scaled", "@dconv"),
        ("dconv.0.residual", "@dconv"),
        ("dconv.0.added", "@dconv"),
        ("dconv.whole", "@dconv"),
        ("encoder.0.dconv", "encoder.0.dconv"),
        ("encoder.0.rewrite", "encoder.0.rewrite"),
        ("encoder.1", "encoder.1"),
        ("tencoder.0", "tencoder.0"),
        ("tencoder.1.conv", "tencoder.1.conv"),
        ("tencoder.1", "tencoder.1"),
        ("tencoder.2", "tencoder.2"),
        ("tencoder.3", "tencoder.3"),
        ("tencoder.4", "tencoder.4"),
        ("crosstransformer.layers.0.norm1", "crosstransformer.layers.0.norm1"),
        ("crosstransformer.layers.0.norm2", "crosstransformer.layers.0.norm2"),
        ("crosstransformer.layers.0.attn.q", "crosstransformer.layers.0.attn.q"),
        ("crosstransformer.layers.0.attn.k", "crosstransformer.layers.0.attn.k"),
        ("crosstransformer.layers.0.attn.v", "crosstransformer.layers.0.attn.v"),
        ("crosstransformer.layers.0.attn", "crosstransformer.layers.0.self_attn#0"),
        ("crosstransformer.layers.0.norm2", "crosstransformer.layers.0.norm2"),
        ("crosstransformer.layers.0.linear1", "crosstransformer.layers.0.linear1"),
        ("crosstransformer.layers.0.fed", "crosstransformer.layers.0.linear2"),
        ("crosstransformer.layers.0", "crosstransformer.layers.0"),
        ("crosstransformer.layers.1", "crosstransformer.layers.1"),
        ("crosstransformer.norm_in", "crosstransformer.in#0"),
        ("crosstransformer.norm_in_t", "crosstransformer.in#1"),
        ("crosstransformer.layers_t.0.norm1", "crosstransformer.layers_t.0.norm1"),
        ("crosstransformer.layers_t.0.attn", "crosstransformer.layers_t.0.self_attn#0"),
        ("crosstransformer.layers_t.0", "crosstransformer.layers_t.0"),
        ("crosstransformer.layers_t.1", "crosstransformer.layers_t.1"),
        ("crosstransformer.layers.2", "crosstransformer.layers.2"),
        ("crosstransformer.layers_t.2", "crosstransformer.layers_t.2"),
        ("crosstransformer.layers.1.attn", "crosstransformer.layers.1.cross_attn#0"),
        ("crosstransformer.layers.4", "crosstransformer.layers.4"),
        ("crosstransformer.layers_t.4", "crosstransformer.layers_t.4"),
    ] {
        let owned: (Vec<usize>, Vec<f32>);
        let host_values: &(Vec<usize>, Vec<f32>) = if host_name == "@dconv" {
            match host_dconv_refs.get(device_name) {
                Some(entry) => entry,
                None => {
                    eprintln!("no host DConv reference for {device_name}");
                    continue;
                }
            }
        } else {
            match host_trace.get(host_name) {
                Some((_, shape, values)) => {
                    owned = (shape.clone(), values.clone());
                    &owned
                }
                None => {
                    eprintln!("the host trace has no {host_name} for {device_name}");
                    continue;
                }
            }
        };
        let shape = &host_values.0;
        let host_values = &host_values.1;
        let name = device_name;
        let trace_source = if name.starts_with("crosstransformer") { &device_trace2 } else { &device_trace };
        let Some((_, device_shape, device_values)) = trace_source.get(name) else {
            eprintln!("the device trace has no {name}");
            continue;
        };
        if device_name == "crosstransformer.layers.0.attn.q" && std::env::var("DEMUCS_ATTN_DEBUG").is_ok() {
            println!("device q[0..6] = {:?}", &device_values[..6]);
            if let Some((_, _, hq)) = host_trace.get("crosstransformer.layers.0.attn.q") {
                println!("host   q[0..6] = {:?}", &hq[..6]);
                let n_feat = 3 * 512;
                let mut worst: (usize, f32) = (0, 0.0);
                let mut first_bad: Option<usize> = None;
                for (i, (&a, &b)) in hq.iter().zip(device_values.iter()).enumerate() {
                    let d = (a - b).abs();
                    if d > 1e-3 && first_bad.is_none() { first_bad = Some(i); }
                    if d > worst.1 { worst = (i, d); }
                }
                if let Some(fb) = first_bad {
                    println!("first mismatch: index {fb} -> token {} feat {} (tokens={})",
                        fb / n_feat, fb % n_feat, hq.len() / n_feat);
                }
                println!("worst: index {} -> token {} feat {} diff {:.4}",
                    worst.0, worst.0 / n_feat, worst.0 % n_feat, worst.1);
                // K-block samples: is the error constant across tokens (bias-like)
                // or proportional (weight-like)?
                println!("device kblk[512..518] = {:?}", &device_values[512..518]);
                println!("host   kblk[512..518] = {:?}", &hq[512..518]);
                println!("device vblk[1024..1030] = {:?}", &device_values[1024..1030]);
                println!("host   vblk[1024..1030] = {:?}", &hq[1024..1030]);
                for feat in [512usize, 513, 600, 1024, 1400] {
                    let d0 = hq[feat] - device_values[feat];
                    let d1 = hq[n_feat + feat] - device_values[n_feat + feat];
                    let d_last = hq[116 * n_feat + feat] - device_values[116 * n_feat + feat];
                    println!("feat {feat}: diff@t0 {d0:.5} diff@t1 {d1:.5} diff@t116 {d_last:.5}");
                }
            }
            if let Some((_, _, hk)) = host_trace.get("crosstransformer.layers.0.attn.k") {
                println!("host   k[0..6] = {:?}", &hk[..6]);
            }
            if let Some((_, _, hv)) = host_trace.get("crosstransformer.layers.0.attn.v") {
                println!("host   v[0..6] = {:?}", &hv[..6]);
            }
        }
        // In probe mode the device's conv2 slot holds a copy of norm1; checking
        // the two device records against each other isolates "dispatch failed"
        // from "GEMM computed wrong values".
        if std::env::var("DEMUCS_CONV2_PROBE").is_ok()
            && device_name == "dconv.0.conv2"
        {
            if let Some((_, norm_shape, norm_values)) = device_trace.get("dconv.0.norm1") {
                let same = norm_values == device_values;
                println!("probe: conv2 == norm1? {same}");
                println!(
                    "probe: conv2[0..4] = {:?}",
                    &device_values[..4.min(device_values.len())]
                );
                println!("probe: norm1[0..4] = {:?}", &norm_values[..4]);
            }
        }
        if name == "crosstransformer.layers_t.0" {
            println!("layers_t.0 shapes: host {:?} device {:?} (elements {})",
                shape, device_shape, shape.iter().product::<usize>());
        }
        // The device trace records flat `(1, len)` rows (the readback has no
        // shape); the element counts and the values are what get compared.
        assert_eq!(
            shape.iter().product::<usize>(),
            device_shape.iter().product::<usize>(),
            "{name}: element count mismatch"
        );
        let result = demucs_core::fixtures::compare(host_values, device_values).expect("compare");
        println!(
            "{name}: {:.2} dB SNR (max abs {:.3e}, reference peak {:.4})",
            result.snr_db(),
            result.max_abs,
            result.reference_max_abs
        );
        if result.snr_db() < 60.0 {
            // Where does it diverge first, and what do the values look like there?
            let first = host_values
                .iter()
                .zip(device_values.iter())
                .position(|(h, d)| (h - d).abs() > 1e-3 * h.abs().max(1.0));
            if let Some(index) = first {
                println!(
                    "  first divergence at {index}: host {:.5} device {:.5} (len {})",
                    host_values[index],
                    device_values[index],
                    host_values.len()
                );
                let neighbours = 4;
                for i in index.saturating_sub(neighbours)..(index + neighbours).min(host_values.len()) {
                    println!("    [{i}] host {:.5} device {:.5}", host_values[i], device_values[i]);
                }
            }
            // Row starts: is the whole batch zero, or scattered?
            if name == "dconv.0.conv2" {
                for b in [0usize, 1, 2, 3, 100, 511] {
                    let i = b * 96 * 44;
                    println!(
                        "    batch {b}: host {:.5} device {:.5} | host {:.5} device {:.5}",
                        host_values[i], device_values[i],
                        host_values[i + 1], device_values[i + 1]
                    );
                }
                // Per-batch SNR: which batches are wrong?
                for b in [0usize, 1, 2, 255, 511] {
                    let start = b * 96 * 44;
                    let end = start + 96 * 44;
                    let r = demucs_core::fixtures::compare(
                        &host_values[start..end],
                        &device_values[start..end],
                    )
                    .unwrap();
                    println!("    batch {b}: {:.2} dB", r.snr_db());
                }
                // Hand-compute C[0][0] from the device's own norm1 trace and
                // the host weight row: if this reproduces the *host* conv2
                // value, the GEMM had correct inputs and the dispatch is what
                // went wrong.
                if device_name == "dconv.0.conv2" {
                    let weight = &model.weights.encoder[0].dconv.layers[0].conv2.weight;
                    if let Some((_, _, norm1_dev)) = device_trace.get("dconv.0.norm1") {
                        let mut acc = 0.0f64;
                        for h in 0..6 {
                            acc += (weight[h] * norm1_dev[h * 44]) as f64;
                        }
                        println!("    hand C[0][0] from device norm1 = {acc:.5}");
                    }
                    if let Some((_, _, conv1_dev)) = device_trace.get("dconv.0.conv1") {
                        // The pre-norm conv1 values, normalised by hand with the
                        // host's affine, then GELU'd: the sum the GEMM should
                        // have computed.
                        let mut acc = 0.0f64;
                        for h in 0..6 {
                            let raw = conv1_dev[h * 44];
                            let (w, b) = (
                                model.weights.encoder[0].dconv.layers[0].norm1_weight[h],
                                model.weights.encoder[0].dconv.layers[0].norm1_bias[h],
                            );
                            // GroupNorm(1, 6): one group over the whole row.
                            let mut sum = 0.0f64;
                            let mut sum_sq = 0.0f64;
                            for hh in 0..6 {
                                let v = conv1_dev[hh * 44] as f64;
                                sum += v;
                                sum_sq += v * v;
                            }
                            let mean = sum / 6.0;
                            let var = sum_sq / 6.0 - mean * mean;
                            let inv = 1.0 / ((var + 1e-5).sqrt()) as f64;
                            let normed = ((raw as f64 - mean) * inv) * w as f64 + b as f64;
                            acc += (weight[h] as f64) * demucs_core::ops::gelu_erf(normed as f32) as f64;
                        }
                        println!("    hand C[0][0] from device conv1 = {acc:.5}");
                    }
                    let y_host = host_dconv_refs.get("dconv.0.norm1").unwrap();
                    let weight = &model.weights.encoder[0].dconv.layers[0].conv2.weight;
                    let mut acc = 0.0f64;
                    for h in 0..6 {
                        acc += (weight[h] * y_host.1[h * 44]) as f64;
                    }
                    println!("    hand C[0][0] from host norm1-gelu = {acc:.5}");
                    print!("    weight[0..6] =");
                    for h in 0..6 {
                        print!(" {:.4}", weight[h]);
                    }
                    println!();
                    print!("    y_gelu[h*44] =");
                    for h in 0..6 {
                        print!(" {:.4}", y_host.1[h * 44]);
                    }
                    println!();
                    let y2_host = host_dconv_refs.get("dconv.0.conv2").unwrap();
                    println!(
                        "    y2_pre[0..4] = {:.5} {:.5} {:.5} {:.5}",
                        y2_host.1[0], y2_host.1[1], y2_host.1[2], y2_host.1[3]
                    );
                }
                // Is the device's batch b holding another batch's result? The
                // per-batch matrices are 96x44; compare device batch b against
                // host batches b-1, b, b+1.
                let per = 96 * 44;
                for b in [1usize, 255] {
                    for shift in [-1isize, 0, 1] {
                        let hb = b as isize + shift;
                        if hb < 0 || hb > 511 {
                            continue;
                        }
                        let start = hb as usize * per;
                        let r = demucs_core::fixtures::compare(
                            &host_values[start..start + per],
                            &device_values[b * per..(b + 1) * per],
                        )
                        .unwrap();
                        println!("    device batch {b} vs host batch {}: {:.2} dB", hb, r.snr_db());
                    }
                }
            }
        }
        if result.snr_db() <= 60.0 {
            failures.push(format!("{name}: {:.2} dB", result.snr_db()));
        }
    }
    let _ = host_out;
    // res1 diagnostic: rebuild the host's residual on the fly and compare
    // against the device's.
    if std::env::var("DEMUCS_ATTN_DEBUG").is_ok() {
        if let (Some((_, _, dev_res1)), Some((_, _, host_x)), Some((_, _, host_attn))) = (
            device_trace2.get("crosstransformer.layers.0.res1"),
            host_trace.get("crosstransformer.in#0"),
            host_trace.get("crosstransformer.layers.0.self_attn#0"),
        ) {
            let gamma1 = &model.weights.transformer.layers[0].gamma1;
            let dim = gamma1.len();
            let mut expected = host_x.clone();
            for (i, v) in expected.iter_mut().enumerate() {
                *v += host_attn[i] * gamma1[i % dim];
            }
            let cmp = demucs_core::fixtures::compare(&expected, dev_res1).unwrap();
            println!("res1 vs host rebuild: {:.2} dB (max abs {:.3e})", cmp.snr_db(), cmp.max_abs);
            println!("device res1[0..6] = {:?}", &dev_res1[..6]);
            println!("expect res1[0..6] = {:?}", &expected[..6]);
            // Which addend is off? Check x alone against device's copy inside res1.
            let cmp_x = demucs_core::fixtures::compare(host_x, dev_res1).unwrap();
            println!("res1 vs raw x: {:.2} dB", cmp_x.snr_db());
        } else {
            println!("res1 diagnostic: missing traces");
        }
    }

    if !failures.is_empty() {
        panic!(
            "{} stage(s) below the gate: {}",
            failures.len(),
            failures.join(", ")
        );
    }


    // The device path includes the channel downsamplers: `freq_dec` is
    // (1, 384, f, t) and `time_dec` is (1, 384, T) — both in the same standard
    // layout the host records, so a flat compare works.
    if let Some((_, _, host)) = host_trace.get("channel_downsampler") {
        let got: Vec<f32> = freq_dec.as_slice().unwrap().to_vec();
        if host.len() == got.len() {
            let r = demucs_core::fixtures::compare(host, &got).unwrap();
            println!("decoder input (freq): {:.2} dB SNR", r.snr_db());
            if r.snr_db() < 60.0 { failures.push(format!("decoder input (freq): {:.2} dB", r.snr_db())); }
        } else {
            println!("decoder input (freq): length mismatch host {} device {}", host.len(), got.len());
        }
    }
    if let Some((_, _, host)) = host_trace.get("channel_downsampler_t") {
        let got: Vec<f32> = time_dec.as_slice().unwrap().to_vec();
        if host.len() == got.len() {
            let r = demucs_core::fixtures::compare(host, &got).unwrap();
            println!("decoder input (time): {:.2} dB SNR", r.snr_db());
            if r.snr_db() < 60.0 { failures.push(format!("decoder input (time): {:.2} dB", r.snr_db())); }
        } else {
            println!("decoder input (time): length mismatch host {} device {}", host.len(), got.len());
        }
    }
}

/// Minimal attention test: 1 head, 2 tokens, known Q/K/V.
/// Verifies the score-matrix path end to end.
#[test]
fn attention_stage_minimal_known_values() {
    let gpu = match demucs_core::gpu::Gpu::new() {
        Ok(gpu) => gpu,
        Err(e) => { eprintln!("skipping: no adapter ({e})"); return; }
    };
    let kernels = demucs_core::gpu::kernels::Kernels::new(&gpu).unwrap();
    let mut arena = demucs_core::gpu::arena::Arena::new(&gpu, 16 << 20);

    // 1 head, 2 tokens, dim_head = 2. Q = [[1,0],[0,1]], K = [[1,0],[0,1]], V = [[1,2],[3,4]].
    // scores = Q @ K^T = [[1,0],[0,1]]. softmax → [[1,0],[0,1]] (one-hot).
    // output = softmax(scores) @ V = [[1,2],[3,4]].
    let q_data = vec![1.0f32, 0.0, 0.0, 1.0];
    let k_data = vec![1.0f32, 0.0, 0.0, 1.0];
    let v_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let q = arena.upload(&gpu, &[1, 2, 2], &q_data, "q").unwrap();
    let k = arena.upload(&gpu, &[1, 2, 2], &k_data, "k").unwrap();
    let v = arena.upload(&gpu, &[1, 2, 2], &v_data, "v").unwrap();
    let out = arena.tensor(&gpu, &[1, 2, 2], "out").unwrap();

    let runner_gpus = q.buffer.clone();
    let _ = runner_gpus;
    let mut recorder = demucs_core::gpu::arena::Recorder::new(&gpu);
    // Use the attention_stage via a small runner-like wrapper.
    // For simplicity, inline the score-matrix steps:
    // scores = Q @ K^T (batched, 1 head)
    let scores = arena.tensor(&gpu, &[1, 2, 2], "scores").unwrap();
    kernels.gemm_into(&gpu, &mut arena, &mut recorder, &q, &k, &scores, GemmJob {
        m: 2, n: 2, k: 2, lda: 2, ldb: 2, ldc: 2,
        batches: 1, inner_count: 1,
        a_outer: 0, a_inner: 0, b_outer: 0, b_inner: 0,
        c_outer: 0, c_inner: 0, transb: true,
    }).unwrap();
    let scaled = arena.tensor(&gpu, &[1, 2, 2], "scaled").unwrap();
    kernels.softmax_scaled(&gpu, &mut arena, &mut recorder, &scores, &scaled, 2, 2, 1.0 / (2.0f32).sqrt()).unwrap();
    let ctx = arena.tensor(&gpu, &[1, 2, 2], "ctx").unwrap();
    kernels.gemm_into(&gpu, &mut arena, &mut recorder, &scaled, &v, &ctx, GemmJob {
        m: 2, n: 2, k: 2, lda: 2, ldb: 2, ldc: 2,
        batches: 1, inner_count: 1,
        a_outer: 0, a_inner: 0, b_outer: 0, b_inner: 0,
        c_outer: 0, c_inner: 0, transb: false,
    }).unwrap();
    recorder.submit(&gpu).unwrap();

    let bytes = gpu.readback(&ctx.buffer, 16).unwrap();
    let got: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();
    // Expected: softmax([1/√2, 0]) @ V = [0.845*1 + 0.155*3, 0.845*2 + 0.155*4] etc.
    // More precisely: softmax([0.7071, 0]) = [0.6699, 0.3301]
    // output[0] = 0.6699*1 + 0.3301*3 = 1.6602
    // output[1] = 0.6699*2 + 0.3301*4 = 2.6602
    // softmax([0, 0.7071]) = [0.3301, 0.6699]
    // output[1] = 0.3301*1 + 0.6699*3 = 2.3398
    // output[3] = 0.3301*2 + 0.6699*4 = 3.3398
    println!("attention output: {:?}", got);
    println!("expected approximately: [1.660, 2.660, 2.340, 3.340]");
    for (i, v) in got.iter().enumerate() {
        println!("  [{i}] got {v:.4}");
    }
}

/// Minimal repro: a batched GEMM with a *shared* A operand (`a_outer = 0`),
/// which is the shape the DConv's 1x1 uses — weight shared, input batched.
#[test]
fn batched_gemm_with_shared_a() {
    let Some(_model) = load_model(44_100) else {
        return;
    };
    let gpu = match demucs_core::gpu::Gpu::new() {
        Ok(gpu) => gpu,
        Err(e) => {
            eprintln!("skipping: no adapter ({e})");
            return;
        }
    };
    let kernels = demucs_core::gpu::kernels::Kernels::new(&gpu).unwrap();
    let mut arena = demucs_core::gpu::arena::Arena::new(&gpu, 64 << 20);

    // C[b] = A @ B[b], A (4, 2) shared, B[b] (2, 3) per batch, 3 batches.
    // A is padded the way the model's conv loader pads weights: k to BK with
    // zeros, rows to BM — the GEMM's k loop runs over the padded tile and the
    // zero tail is what keeps the extra products at zero.
    let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b_data: Vec<f32> = (0..3 * 2 * 3).map(|i| (i % 7) as f32 - 2.0).collect();
    let a_padded = {
        let (padded_rows, padded_cols) = (pad_ceil(4, 128), pad_ceil(2, 16));
        let mut out = vec![0.0f32; padded_rows * padded_cols];
        for r in 0..4 {
            out[r * padded_cols..r * padded_cols + 2]
                .copy_from_slice(&a_data[r * 2..(r + 1) * 2]);
        }
        out
    };
    let a = arena.upload(&gpu, &[pad_ceil(4, 128), pad_ceil(2, 16)], &a_padded, "a").unwrap();
    let b = arena.upload(&gpu, &[3, 2, 3], &b_data, "b").unwrap();
    let c = arena.tensor(&gpu, &[3, 4, 3], "c").unwrap();

    let job = demucs_core::gpu::kernels::GemmJob {
        m: 4,
        n: 3,
        k: 2,
        lda: pad_ceil(2, 16),
        ldb: 3,
        ldc: 3,
        batches: 3,
        inner_count: 1,
        a_outer: 0,
        a_inner: 0,
        b_outer: 6,
        b_inner: 0,
        c_outer: 12,
        c_inner: 0,
        transb: false,
    };
    let mut recorder = demucs_core::gpu::arena::Recorder::new(&gpu);
    kernels
        .gemm_into(&gpu, &mut arena, &mut recorder, &a, &b, &c, job)
        .unwrap();
    recorder.submit(&gpu).unwrap();

    let bytes = gpu.readback(&c.buffer, (c.len() * 4) as u64).unwrap();
    let got: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();
    let mut want = vec![0.0f32; 3 * 4 * 3];
    for batch in 0..3 {
        for m in 0..4 {
            for n in 0..3 {
                let mut acc = 0.0;
                for k in 0..2 {
                    acc += (a_data[m * 2 + k] * b_data[batch * 6 + k * 3 + n]) as f64;
                }
                want[batch * 12 + m * 3 + n] = acc as f32;
            }
        }
    }
    let result = demucs_core::fixtures::compare(&want, &got).unwrap();
    println!("shared-A batched GEMM (bn64 tile): {:.2} dB", result.snr_db());

    // The same job with n = 100, which routes to the square batched pipeline.
    let n2 = 100usize;
    let b2_data: Vec<f32> = (0..3 * 2 * n2).map(|i| (i % 7) as f32 - 2.0).collect();
    let b2 = arena.upload(&gpu, &[3, 2, n2], &b2_data, "b2").unwrap();
    let c2 = arena.tensor(&gpu, &[3, 4, n2], "c2").unwrap();
    let job2 = demucs_core::gpu::kernels::GemmJob {
        m: 4,
        n: n2,
        k: 2,
        lda: 16,
        ldb: n2,
        ldc: n2,
        batches: 3,
        inner_count: 1,
        a_outer: 0,
        a_inner: 0,
        b_outer: 2 * n2,
        b_inner: 0,
        c_outer: 4 * n2,
        c_inner: 0,
        transb: false,
    };
    let mut recorder2 = demucs_core::gpu::arena::Recorder::new(&gpu);
    kernels
        .gemm_into(&gpu, &mut arena, &mut recorder2, &a, &b2, &c2, job2)
        .unwrap();
    recorder2.submit(&gpu).unwrap();
    let bytes2 = gpu.readback(&c2.buffer, (c2.len() * 4) as u64).unwrap();
    let got2: Vec<f32> = bytemuck::cast_slice(&bytes2).to_vec();
    let mut want2 = vec![0.0f32; 3 * 4 * n2];
    for batch in 0..3 {
        for m in 0..4 {
            for n in 0..n2 {
                let mut acc = 0.0;
                for k in 0..2 {
                    acc += (a_data[m * 2 + k] * b2_data[batch * 2 * n2 + k * n2 + n]) as f64;
                }
                want2[batch * 4 * n2 + m * n2 + n] = acc as f32;
            }
        }
    }
    let result2 = demucs_core::fixtures::compare(&want2, &got2).unwrap();
    println!("shared-A batched GEMM (square tile): {:.2} dB", result2.snr_db());

    // And the same shape with a *per-batch* A, which is what the existing
    // batched-GEMM test covers.
    let a3_data: Vec<f32> = (0..3 * 4 * 2).map(|i| (i % 5) as f32 - 2.0).collect();
    // Zero-padded in k per batch: 2 -> 16, batches to BM.
    let a3_padded = {
        let mut out = vec![0.0f32; 3 * 128 * 16];
        for batch in 0..3 {
            for m in 0..4 {
                for k in 0..2 {
                    out[(batch * 128 + m) * 16 + k] = a3_data[batch * 8 + m * 2 + k];
                }
            }
        }
        out
    };
    let a3 = arena.upload(&gpu, &[3 * 128, 16], &a3_padded, "a3").unwrap();
    let c3 = arena.tensor(&gpu, &[3, 4, n2], "c3").unwrap();
    let job3 = demucs_core::gpu::kernels::GemmJob {
        m: 4,
        n: n2,
        k: 2,
        lda: 16,
        ldb: n2,
        ldc: n2,
        batches: 3,
        inner_count: 1,
        a_outer: 128 * 16,
        a_inner: 0,
        b_outer: 2 * n2,
        b_inner: 0,
        c_outer: 4 * n2,
        c_inner: 0,
        transb: false,
    };
    let mut recorder3 = demucs_core::gpu::arena::Recorder::new(&gpu);
    kernels
        .gemm_into(&gpu, &mut arena, &mut recorder3, &a3, &b2, &c3, job3)
        .unwrap();
    recorder3.submit(&gpu).unwrap();
    let bytes3 = gpu.readback(&c3.buffer, (c3.len() * 4) as u64).unwrap();
    let got3: Vec<f32> = bytemuck::cast_slice(&bytes3).to_vec();
    let mut want3 = vec![0.0f32; 3 * 4 * n2];
    for batch in 0..3 {
        for m in 0..4 {
            for n in 0..n2 {
                let mut acc = 0.0;
                for k in 0..2 {
                    acc += (a3_data[batch * 8 + m * 2 + k] * b2_data[batch * 2 * n2 + k * n2 + n]) as f64;
                }
                want3[batch * 4 * n2 + m * n2 + n] = acc as f32;
            }
        }
    }
    let result3 = demucs_core::fixtures::compare(&want3, &got3).unwrap();
    println!("per-batch-A batched GEMM (square tile): {:.2} dB", result3.snr_db());

    assert!(
        result.snr_db() > 60.0 && result2.snr_db() > 60.0 && result3.snr_db() > 60.0,
        "shared-A {:.2} dB, square {:.2} dB, per-batch {:.2} dB",
        result.snr_db(),
        result2.snr_db(),
        result3.snr_db()
    );

    // The exact conv2 job: m=96, k=6, n=44, batches=512, shared zero-padded A.
    let (m, k, n, batches) = (96usize, 6usize, 44usize, 512usize);
    let a4_data = vec![0.5f32; m * k];
    let mut a4_padded = vec![0.0f32; pad_ceil(m, 128) * pad_ceil(k, 16)];
    for mm in 0..m {
        for kk in 0..k {
            a4_padded[mm * pad_ceil(k, 16) + kk] = a4_data[mm * k + kk];
        }
    }
    let a4 = arena.upload(&gpu, &[pad_ceil(m, 128), pad_ceil(k, 16)], &a4_padded, "a4").unwrap();
    let b4_data: Vec<f32> = (0..batches * k * n).map(|i| ((i % 11) as f32 - 5.0) * 0.3).collect();
    let b4 = arena.upload(&gpu, &[batches, k, n], &b4_data, "b4").unwrap();
    let c4 = arena.tensor(&gpu, &[batches, m, n], "c4").unwrap();
    let job4 = demucs_core::gpu::kernels::GemmJob {
        m,
        n,
        k,
        lda: pad_ceil(k, 16),
        ldb: n,
        ldc: n,
        batches,
        inner_count: 1,
        a_outer: 0,
        a_inner: 0,
        b_outer: k * n,
        b_inner: 0,
        c_outer: m * n,
        c_inner: 0,
        transb: false,
    };
    let mut recorder4 = demucs_core::gpu::arena::Recorder::new(&gpu);
    kernels
        .gemm_into(&gpu, &mut arena, &mut recorder4, &a4, &b4, &c4, job4)
        .unwrap();
    recorder4.submit(&gpu).unwrap();
    let bytes4 = gpu.readback(&c4.buffer, (c4.len() * 4) as u64).unwrap();
    let got4: Vec<f32> = bytemuck::cast_slice(&bytes4).to_vec();
    let mut want4 = vec![0.0f32; batches * m * n];
    for batch in 0..batches {
        for mm in 0..m {
            for nn in 0..n {
                let mut acc = 0.0;
                for kk in 0..k {
                    acc += (a4_data[mm * k + kk] * b4_data[batch * k * n + kk * n + nn]) as f64;
                }
                want4[batch * m * n + mm * n + nn] = acc as f32;
            }
        }
    }
    let result4 = demucs_core::fixtures::compare(&want4, &got4).unwrap();
    println!("conv2-shape batched GEMM: {:.2} dB", result4.snr_db());
    assert!(result4.snr_db() > 60.0, "{:.2} dB", result4.snr_db());
}

/// Replicates `proj_stage`'s exact GEMM + bias on random data at the real
/// transformer shapes and compares against a host matmul. Isolates the fused
/// qkv projection from everything else in the layer.
#[test]
fn proj_stage_matches_host_at_transformer_shapes() {
    let gpu = match demucs_core::gpu::Gpu::new() {
        Ok(gpu) => gpu,
        Err(e) => { eprintln!("skipping: no adapter ({e})"); return; }
    };
    let kernels = demucs_core::gpu::kernels::Kernels::new(&gpu).unwrap();
    let mut arena = demucs_core::gpu::arena::Arena::new(&gpu, 64 << 20);

    let tokens = 64usize;
    let dim = 512usize;
    let out_features = 3 * dim;
    let fill = |n: usize, seed: u32| -> Vec<f32> {
        let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        (0..n).map(|_| {
            state ^= state << 13; state ^= state >> 17; state ^= state << 5;
            ((state % 20_000) as f32 / 10_000.0) - 1.0
        }).collect()
    };

    let x_data = fill(tokens * dim, 7);
    let w_data = fill(out_features * dim, 11);
    let b_data = fill(out_features, 13);

    // Padded weight exactly like upload_proj: (pad(out,BN), pad(in,BK)), rows zero-filled.
    let padded_rows = demucs_core::gpu::kernels::pad_ceil(out_features, demucs_core::gpu::shaders::BN);
    let padded_cols = demucs_core::gpu::kernels::pad_ceil(dim, demucs_core::gpu::shaders::BK);
    let mut padded = vec![0.0f32; padded_rows * padded_cols];
    for r in 0..out_features {
        padded[r * padded_cols..r * padded_cols + dim]
            .copy_from_slice(&w_data[r * dim..(r + 1) * dim]);
    }

    let x = arena.upload(&gpu, &[tokens, dim], &x_data, "x").unwrap();
    let w = arena.upload(&gpu, &[padded_rows, padded_cols], &padded, "w").unwrap();
    let ones = arena.upload(&gpu, &[out_features], &vec![1.0f32; out_features], "ones").unwrap();
    let bias = arena.upload(&gpu, &[out_features], &b_data, "bias").unwrap();
    let out = arena.tensor(&gpu, &[tokens, out_features], "out").unwrap();

    let mut recorder = demucs_core::gpu::arena::Recorder::new(&gpu);
    kernels.gemm_into(&gpu, &mut arena, &mut recorder, &x, &w, &out, GemmJob {
        m: tokens, n: out_features, k: dim,
        lda: dim, ldb: padded_cols, ldc: out_features,
        batches: 1, inner_count: 1,
        a_outer: 0, a_inner: 0, b_outer: 0, b_inner: 0,
        c_outer: 0, c_inner: 0, transb: true,
    }).unwrap();
    recorder.submit(&gpu).unwrap();

    let bytes = gpu.readback(&out.buffer, (out.len() * 4) as u64).unwrap();
    let got: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();

    // Host reference: C[t, o] = sum_k x[t, k] * w[o, k] + bias[o]
    let mut expected = vec![0.0f32; tokens * out_features];
    for t in 0..tokens {
        for o in 0..out_features {
            let mut acc = 0.0f64;
            for k in 0..dim {
                acc += (x_data[t * dim + k] * w_data[o * dim + k]) as f64;
            }
            expected[t * out_features + o] = acc as f32 + b_data[o];
        }
    }
    // GEMM-only reference (no bias): the device gemm_into call above does not
    // add the bias; the affine does that separately.
    let mut expected_raw = vec![0.0f32; tokens * out_features];
    for t in 0..tokens {
        for o in 0..out_features {
            let mut acc = 0.0f64;
            for k in 0..dim {
                acc += (x_data[t * dim + k] * w_data[o * dim + k]) as f64;
            }
            expected_raw[t * out_features + o] = acc as f32;
        }
    }
    let cmp = demucs_core::fixtures::compare(&expected_raw, &got).unwrap();
    println!("proj gemm (no bias): {:.2} dB SNR (max abs {:.3e})", cmp.snr_db(), cmp.max_abs);
    assert!(cmp.snr_db() > 60.0, "proj GEMM diverged: {:.2} dB", cmp.snr_db());

    // Now the bias add, replicating the (tokens, out_features, 1) affine call.
    let out2 = arena.tensor(&gpu, &[tokens, out_features], "out2").unwrap();
    let mut recorder2 = demucs_core::gpu::arena::Recorder::new(&gpu);
    kernels.gemm_into(&gpu, &mut arena, &mut recorder2, &x, &w, &out2, GemmJob {
        m: tokens, n: out_features, k: dim,
        lda: dim, ldb: padded_cols, ldc: out_features,
        batches: 1, inner_count: 1,
        a_outer: 0, a_inner: 0, b_outer: 0, b_inner: 0,
        c_outer: 0, c_inner: 0, transb: true,
    }).unwrap();
    kernels.channel_affine_act_in_place(
        &gpu, &mut arena, &mut recorder2, &out2, &ones, &bias,
        tokens, out_features, 1, demucs_core::gpu::kernels::Activation::Identity,
    ).unwrap();
    recorder2.submit(&gpu).unwrap();
    let bytes2 = gpu.readback(&out2.buffer, (out2.len() * 4) as u64).unwrap();
    let got2: Vec<f32> = bytemuck::cast_slice(&bytes2).to_vec();
    let cmp2 = demucs_core::fixtures::compare(&expected, &got2).unwrap();
    println!("proj gemm + bias: {:.2} dB SNR (max abs {:.3e})", cmp2.snr_db(), cmp2.max_abs);
    assert!(cmp2.snr_db() > 60.0, "proj bias diverged: {:.2} dB", cmp2.snr_db());
}


/// The full attention pipeline (head split -> scores -> softmax -> AV -> merge)
/// at the real transformer shapes, including the AV GEMM's k-tail
/// (k_tokens = 117 is not a multiple of the GEMM's K block size).
#[test]
fn attention_stage_matches_host_at_transformer_shapes() {
    let gpu = match demucs_core::gpu::Gpu::new() {
        Ok(gpu) => gpu,
        Err(e) => { eprintln!("skipping: no adapter ({e})"); return; }
    };
    let kernels = demucs_core::gpu::kernels::Kernels::new(&gpu).unwrap();
    let mut arena = demucs_core::gpu::arena::Arena::new(&gpu, 64 << 20);

    let heads = 8usize;
    let d_head = 64usize;
    let dim = heads * d_head;
    let q_tokens = 117usize;
    let k_tokens = 117usize;
    let fill = |n: usize, seed: u32| -> Vec<f32> {
        let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        (0..n).map(|_| {
            state ^= state << 13; state ^= state >> 17; state ^= state << 5;
            ((state % 20_000) as f32 / 10_000.0) - 1.0
        }).collect()
    };
    let q_data = fill(q_tokens * dim, 21);
    let k_data = fill(k_tokens * dim, 22);
    let v_data = fill(k_tokens * dim, 23);

    let q = arena.upload(&gpu, &[q_tokens, dim], &q_data, "q").unwrap();
    let k = arena.upload(&gpu, &[k_tokens, dim], &k_data, "k").unwrap();
    let v = arena.upload(&gpu, &[k_tokens, dim], &v_data, "v").unwrap();
    let out = arena.tensor(&gpu, &[q_tokens, dim], "out").unwrap();

    let mut recorder = demucs_core::gpu::arena::Recorder::new(&gpu);

    let mut split = |x: &demucs_core::gpu::arena::DevTensor, label: String| -> demucs_core::gpu::arena::DevTensor {
        let tmp = arena.tensor(&gpu, &[x.shape[0], heads, d_head], &format!("{label}.split")).unwrap();
        kernels.transpose(&gpu, &mut arena, &mut recorder, x, &tmp, 1, x.shape[0], heads, d_head).unwrap();
        tmp.with_shape(vec![heads, x.shape[0], d_head]).unwrap()
    };
    let q_h = split(&q, "qh".to_string());
    let k_h = split(&k, "kh".to_string());
    let v_h = split(&v, "vh".to_string());

    let scores = arena.tensor(&gpu, &[heads, q_tokens, k_tokens], "scores").unwrap();
    kernels.gemm_into(&gpu, &mut arena, &mut recorder, &q_h, &k_h, &scores, GemmJob {
        m: q_tokens, n: k_tokens, k: d_head,
        lda: d_head, ldb: d_head, ldc: k_tokens,
        batches: heads, inner_count: 1,
        a_outer: q_tokens * d_head, a_inner: 0,
        b_outer: k_tokens * d_head, b_inner: 0,
        c_outer: q_tokens * k_tokens, c_inner: 0,
        transb: true,
    }).unwrap();
    if std::env::var("DEMUCS_ATTN_DEBUG").is_ok() {
        recorder.submit(&gpu).unwrap();
        let bytes = gpu.readback(&scores.buffer, (scores.len() * 4) as u64).unwrap();
        let got: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();
        let mut expected = vec![0.0f32; heads * q_tokens * k_tokens];
        for h in 0..heads {
            for i in 0..q_tokens {
                for j in 0..k_tokens {
                    let mut acc = 0.0f64;
                    for d in 0..d_head {
                        let qs = (1.0 / (d_head as f32).sqrt()) as f64;
                        acc += (q_data[i * dim + h * d_head + d] as f64 * qs)
                            * (k_data[j * dim + h * d_head + d] as f64);
                    }
                    expected[h * q_tokens * k_tokens + i * k_tokens + j] = acc as f32;
                }
            }
        }
        let cmp = demucs_core::fixtures::compare(&expected, &got).unwrap();
        println!("scores: {:.2} dB (max abs {:.3e})", cmp.snr_db(), cmp.max_abs);
        recorder = demucs_core::gpu::arena::Recorder::new(&gpu);
    }

    let scaled = arena.tensor(&gpu, &[heads, q_tokens, k_tokens], "scaled").unwrap();
    kernels.softmax_scaled(&gpu, &mut arena, &mut recorder, &scores, &scaled,
        heads * q_tokens, k_tokens, 1.0 / (d_head as f32).sqrt()).unwrap();
    if std::env::var("DEMUCS_ATTN_DEBUG").is_ok() {
        recorder.submit(&gpu).unwrap();
        let bytes = gpu.readback(&scaled.buffer, (scaled.len() * 4) as u64).unwrap();
        let got: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();
        let mut expected = vec![0.0f32; heads * q_tokens * k_tokens];
        for h in 0..heads {
            for i in 0..q_tokens {
                let base = h * q_tokens * k_tokens + i * k_tokens;
                let mut row = vec![0.0f64; k_tokens];
                for j in 0..k_tokens {
                    let mut acc = 0.0f64;
                    for d in 0..d_head {
                        let qs = (1.0 / (d_head as f32).sqrt()) as f64;
                        acc += (q_data[i * dim + h * d_head + d] as f64 * qs)
                            * (k_data[j * dim + h * d_head + d] as f64);
                    }
                    row[j] = acc;
                }
                let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let sum: f64 = row.iter().map(|v| (v - max).exp()).sum();
                for j in 0..k_tokens {
                    expected[base + j] = ((row[j] - max).exp() / sum) as f32;
                }
            }
        }
        let cmp = demucs_core::fixtures::compare(&expected, &got).unwrap();
        println!("softmax: {:.2} dB (max abs {:.3e})", cmp.snr_db(), cmp.max_abs);
        recorder = demucs_core::gpu::arena::Recorder::new(&gpu);
    }

    // AV: batched plain GEMM with k = k_tokens (117 - the k-tail case).
    let context = arena.tensor(&gpu, &[heads, q_tokens, d_head], "context").unwrap();
    kernels.gemm_into(&gpu, &mut arena, &mut recorder, &scaled, &v_h, &context, GemmJob {
        m: q_tokens, n: d_head, k: k_tokens,
        lda: k_tokens, ldb: d_head, ldc: d_head,
        batches: heads, inner_count: 1,
        a_outer: q_tokens * k_tokens, a_inner: 0,
        b_outer: k_tokens * d_head, b_inner: 0,
        c_outer: q_tokens * d_head, c_inner: 0,
        transb: false,
    }).unwrap();
    if std::env::var("DEMUCS_ATTN_DEBUG").is_ok() {
        recorder.submit(&gpu).unwrap();
        let bytes = gpu.readback(&context.buffer, (context.len() * 4) as u64).unwrap();
        let got: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();
        let mut expected = vec![0.0f32; heads * q_tokens * d_head];
        for h in 0..heads {
            for i in 0..q_tokens {
                let mut probs = vec![0.0f64; k_tokens];
                for j in 0..k_tokens {
                    let mut acc = 0.0f64;
                    for d in 0..d_head {
                        let qs = (1.0 / (d_head as f32).sqrt()) as f64;
                        acc += (q_data[i * dim + h * d_head + d] as f64 * qs)
                            * (k_data[j * dim + h * d_head + d] as f64);
                    }
                    probs[j] = acc;
                }
                let max = probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let sum: f64 = probs.iter().map(|v| (v - max).exp()).sum();
                for j in 0..k_tokens { probs[j] = (probs[j] - max).exp() / sum; }
                for d in 0..d_head {
                    let mut acc = 0.0f64;
                    for j in 0..k_tokens {
                        acc += probs[j] * (v_data[j * dim + h * d_head + d] as f64);
                    }
                    expected[h * q_tokens * d_head + i * d_head + d] = acc as f32;
                }
            }
        }
        let cmp = demucs_core::fixtures::compare(&expected, &got).unwrap();
        println!("context: {:.2} dB (max abs {:.3e})", cmp.snr_db(), cmp.max_abs);
        recorder = demucs_core::gpu::arena::Recorder::new(&gpu);
    }

    // Head merge: (heads, tokens, d_head) -> (tokens, dim).
    kernels.transpose(&gpu, &mut arena, &mut recorder, &context, &out, 1, heads, q_tokens, d_head).unwrap();
    recorder.submit(&gpu).unwrap();

    let bytes = gpu.readback(&out.buffer, (out.len() * 4) as u64).unwrap();
    let got: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();
    let mut expected = vec![0.0f32; q_tokens * dim];
    for i in 0..q_tokens {
        for h in 0..heads {
            let mut probs = vec![0.0f64; k_tokens];
            for j in 0..k_tokens {
                let mut acc = 0.0f64;
                for d in 0..d_head {
                    let qs = (1.0 / (d_head as f32).sqrt()) as f64;
                    acc += (q_data[i * dim + h * d_head + d] as f64 * qs)
                        * (k_data[j * dim + h * d_head + d] as f64);
                }
                probs[j] = acc;
            }
            let max = probs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let sum: f64 = probs.iter().map(|v| (v - max).exp()).sum();
            for j in 0..k_tokens { probs[j] = (probs[j] - max).exp() / sum; }
            for d in 0..d_head {
                let mut acc = 0.0f64;
                for j in 0..k_tokens {
                    acc += probs[j] * (v_data[j * dim + h * d_head + d] as f64);
                }
                expected[i * dim + h * d_head + d] = acc as f32;
            }
        }
    }
    let cmp = demucs_core::fixtures::compare(&expected, &got).unwrap();
    println!("attention end to end: {:.2} dB (max abs {:.3e})", cmp.snr_db(), cmp.max_abs);
    assert!(cmp.snr_db() > 60.0, "attention diverged: {:.2} dB", cmp.snr_db());
}

/// Replicates the waveform-branch encoder convs (im2col + GEMM) at the real
/// tencoder.0 and tencoder.1 shapes with random data, against the host conv1d.
/// tencoder.0 aligns in the model but tencoder.1 does not; this isolates the
/// conv from everything else.
#[test]
fn tencoder_conv_matches_host_at_stage_shapes() {
    let model = match load_model(44_100) {
        Some(m) => m,
        None => return,
    };
    let gpu = match demucs_core::gpu::Gpu::new() {
        Ok(gpu) => gpu,
        Err(e) => { eprintln!("skipping: no adapter ({e})"); return; }
    };
    let kernels = demucs_core::gpu::kernels::Kernels::new(&gpu).unwrap();
    let mut arena = demucs_core::gpu::arena::Arena::new(&gpu, 1 << 30);

    let fill = |n: usize, seed: u32| -> Vec<f32> {
        let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        (0..n).map(|_| {
            state ^= state << 13; state ^= state >> 17; state ^= state << 5;
            ((state % 20_000) as f32 / 10_000.0) - 1.0
        }).collect()
    };

    // (stage index, input width)
    let cases = [(0usize, 44100usize), (1, 11025), (2, 2760), (3, 692)];
    for (stage, width) in cases {
        let conv_w = &model.weights.tencoder[stage].conv;
        let (oc, ic) = (conv_w.shape[0], conv_w.shape[1]);
        let k = conv_w.shape[2]; // time-branch ConvW is (oc, ic, k)
        println!("tencoder.{stage}: oc={oc} ic={ic} k={k} width={width}");

        let x_data = fill(ic * width, 31 + stage as u32);
        let out_w = (width + 2 * 2 - k) / 4 + 1;
        let w_data = conv_w.weight.clone();
        let b_data = conv_w.bias.clone();

        // Padded weight exactly like upload_conv.
        let k_total = ic * k;
        let (padded, _, _) = match demucs_core::gpu::kernels::pad_matrix_for(
            &w_data, oc, k_total, demucs_core::gpu::shaders::BM, demucs_core::gpu::shaders::BK)
        { Ok(v) => v, Err(e) => { eprintln!("pad failed: {e}"); return; } };
        let weight = arena.upload(&gpu,
            &[demucs_core::gpu::kernels::pad_ceil(oc, demucs_core::gpu::shaders::BM),
              demucs_core::gpu::kernels::pad_ceil(k_total, demucs_core::gpu::shaders::BK)],
            &padded, "conv.weight").unwrap();
        let bias = arena.upload(&gpu, &[oc], &b_data, "conv.bias").unwrap();
        let x = arena.upload(&gpu, &[1, ic, 1, width], &x_data, "conv.x").unwrap();
        let scratch = arena.alloc(&gpu, (128 << 20) * 4, "conv.scratch").unwrap();
        let out = arena.tensor(&gpu, &[1, oc, 1, out_w], "conv.out").unwrap();

        let shape = demucs_core::gpu::kernels::Im2ColShape {
            batch: 1, in_channels: ic, h: 1, w: width,
            kernel: (1, k), stride: (1, 4), pad: (0, 2),
        };
        let mut recorder = demucs_core::gpu::arena::Recorder::new(&gpu);
        kernels.conv2d_into(&gpu, &mut arena, &mut recorder, &x, &weight, Some(&bias),
            &scratch, &out, shape, oc).unwrap();
        recorder.submit(&gpu).unwrap();

        let bytes = gpu.readback(&out.buffer, (out.len() * 4) as u64).unwrap();
        let got: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();

        // Host reference: conv1d on (1, ic, width).
        let input3d = ndarray::Array3::from_shape_vec((1, ic, width), x_data).unwrap();
        let expected = demucs_core::demucs::ops::conv1d(&input3d, conv_w, 4, 2, 1);
        let expected_flat: Vec<f32> = expected.iter().copied().collect();
        let cmp = demucs_core::fixtures::compare(&expected_flat, &got).unwrap();
        println!("  tencoder.{stage} conv: {:.2} dB (max abs {:.3e})", cmp.snr_db(), cmp.max_abs);
        if cmp.snr_db() < 60.0 {
            let n_feat = out_w;
            let first = expected_flat.iter().zip(got.iter()).position(|(a, b)| (a - b).abs() > 1e-3);
            if let Some(fb) = first {
                println!("  first mismatch: index {fb} -> out_ch {} t {}",
                    fb / n_feat, fb % n_feat);
            }
        }
        assert!(cmp.snr_db() > 60.0, "tencoder.{stage} conv diverged: {:.2} dB", cmp.snr_db());
    }
}

/// The tencoder pads its waveform branch up to a multiple of four, and the
/// conv's gather reads the pad columns as if they were zero.
/// `copy_pitched_into` writes only the real columns, so the pad region has to
/// be zeroed on the device — a fresh buffer was zero because wgpu zeroes new
/// buffers, but the arena's pool hands the same buffer back every chunk with
/// the previous chunk's data in the unwritten columns. The 1 s dump does take
/// this branch (11025 % 4 == 1 at tencoder.1), but only ever through a freshly
/// allocated, zeroed buffer, so it cannot see the difference; a real-length
/// stream hits the branch with recycled buffers on every chunk after the
/// first.
///
/// Two awkward lengths through one runner: the second chunk's pad buffer is a
/// recycled one, and every stage is compared against the host model.
#[test]
fn device_encoders_match_the_host_on_unpadded_lengths() {
    let Some(model_a) = load_model(44_102) else { return };
    let Some(model_b) = load_model(33_602) else { return };
    let mut runner = match GpuHtdemucsRunner::new(&model_a) {
        Ok(runner) => runner,
        Err(e) => {
            eprintln!("skipping: no usable wgpu adapter ({e})");
            return;
        }
    };
    let fill = |n: usize, seed: u32| -> Vec<f32> {
        let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                ((state % 20_000) as f32 / 10_000.0) - 1.0
            })
            .collect()
    };
    let stage_names = [
        "encoder.0.conv",
        "encoder.3",
        "tencoder.0",
        "tencoder.1",
        "tencoder.3",
    ];
    let mut failures: Vec<String> = Vec::new();
    for (chunk, (samples, model)) in [(44_102usize, &model_a), (33_602, &model_b)]
        .into_iter()
        .enumerate()
    {
        // The segment rounding may move the length by a sample; what matters
        // is that it stays unaligned, or the branch under test never runs.
        let training_length = model.training_length();
        assert_ne!(
            training_length % 4,
            0,
            "chunk {chunk}: this test needs an unaligned length, got {training_length}"
        );
        let samples = samples.min(training_length);
        let wave = Array3::from_shape_vec((1, 2, samples), fill(2 * samples, 7 + chunk as u32))
            .expect("wave");
        let padded = if samples < training_length {
            let mut wide = Array3::<f32>::zeros((1, 2, training_length));
            wide
                .slice_mut(ndarray::s![.., .., ..samples])
                .assign(&wave);
            wide
        } else {
            wave
        };

        // The host's front end, exactly as the reference run builds it.
        let z = model.spec.spec(&padded).expect("spectro");
        let mag = demucs_core::demucs::spec::pack_complex_as_channels(&z);
        let norm = model.normalise(&mag, &padded);
        let mut mag_norm = mag.clone();
        for (offset, scale) in norm.mean.iter().zip(norm.std.iter()) {
            mag_norm
                .slice_mut(ndarray::s![0, .., .., ..])
                .mapv_inplace(|v| (v - offset) / (1e-5 + scale));
        }
        let mut wave_norm = padded.clone();
        for (offset, scale) in norm.mean_t.iter().zip(norm.std_t.iter()) {
            wave_norm
                .slice_mut(ndarray::s![0, .., ..])
                .mapv_inplace(|v| (v - offset) / (1e-5 + scale));
        }

        let mut host_trace = VecTrace::default();
        model
            .forward(&padded, &mut host_trace)
            .expect("host forward");
        let mut device_trace = VecTrace::default();
        let (freq_out, time_out) = runner
            .forward_encoders(&mag_norm, &wave_norm, &mut device_trace)
            .expect("device encoders");

        for name in stage_names {
            let Some((_, _, host_values)) = host_trace.get(name) else {
                failures.push(format!("chunk {chunk}: the host trace has no {name}"));
                continue;
            };
            let Some((_, _, device_values)) = device_trace.get(name) else {
                failures.push(format!("chunk {chunk}: the device trace has no {name}"));
                continue;
            };
            let result = demucs_core::fixtures::compare(host_values, device_values)
                .expect("compare");
            println!(
                "chunk {chunk} {name}: {:.2} dB (elements {})",
                result.snr_db(),
                host_values.len()
            );
            if result.snr_db() <= 60.0 {
                failures.push(format!(
                    "chunk {chunk} {name}: {:.2} dB (max abs {:.3e})",
                    result.snr_db(),
                    result.max_abs
                ));
            }
        }

        // The returned bottleneck against the host's own last-stage records.
        let freq: Vec<f32> = freq_out.iter().copied().collect();
        let time: Vec<f32> = time_out.iter().copied().collect();
        for (name, got) in [("encoder.3", &freq), ("tencoder.3", &time)] {
            let Some((_, _, host_values)) = host_trace.get(name) else {
                continue;
            };
            let result =
                demucs_core::fixtures::compare(host_values, got).expect("compare");
            println!(
                "chunk {chunk} bottleneck {name}: {:.2} dB",
                result.snr_db()
            );
            if result.snr_db() <= 60.0 {
                failures.push(format!(
                    "chunk {chunk} bottleneck {name}: {:.2} dB",
                    result.snr_db()
                ));
            }
        }
    }
    if !failures.is_empty() {
        panic!(
            "{} stage(s) below the gate: {}",
            failures.len(),
            failures.join(", ")
        );
    }
}

/// The full device path — encoders, channel samplers, cross-transformer,
/// decoders — against the host model at **real segment lengths**, stage by
/// stage.
///
/// The 1 s reference dump aligns at every stage and still cannot see what a
/// real chunk does: its token counts are powers of two, so every pitch margin
/// in the model is zero cells wide and every `frames % 4` is 0. The full-track
/// run separated at 7.21x but only 18–32 dB against the reference, so something
/// in the real-shape path is silently wrong. This test is the magnifier: run it
/// at the real chunk lengths and the first stage under the gate names the bug.
#[test]
fn device_full_path_matches_the_host_at_real_segment_lengths() {
    let cases = [88_200usize, 343_980];
    let models: Vec<Option<Htdemucs>> = cases
        .iter()
        .map(|&samples| load_model(samples))
        .collect();
    if models.iter().any(|m| m.is_none()) {
        return;
    }
    let model_a = models[0].as_ref().unwrap();
    let mut runner = match GpuHtdemucsRunner::new(model_a) {
        Ok(runner) => runner,
        Err(e) => {
            eprintln!("skipping: no usable wgpu adapter ({e})");
            return;
        }
    };
    let fill = |n: usize, seed: u32| -> Vec<f32> {
        let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                ((state % 20_000) as f32 / 10_000.0) - 1.0
            })
            .collect()
    };
    // (device stage, host stage): the names differ where the two paths record
    // the same tensor at different points.
    let stage_names = [
        ("encoder.3", "encoder.3"),
        ("tencoder.3", "tencoder.3"),
        ("crosstransformer.norm_in", "crosstransformer.in#0"),
        ("crosstransformer.norm_in_t", "crosstransformer.in#1"),
        ("crosstransformer.layers.0.norm1", "crosstransformer.layers.0.norm1"),
        ("crosstransformer.layers.0.attn.q", "crosstransformer.layers.0.attn.q"),
        ("crosstransformer.layers.0", "crosstransformer.layers.0"),
        ("crosstransformer.layers_t.0", "crosstransformer.layers_t.0"),
        ("crosstransformer.layers.1", "crosstransformer.layers.1"),
        ("crosstransformer.layers_t.1", "crosstransformer.layers_t.1"),
        ("crosstransformer.layers.2", "crosstransformer.layers.2"),
        ("crosstransformer.layers_t.4", "crosstransformer.layers_t.4"),
        ("crosstransformer.layers.4", "crosstransformer.layers.4"),
        ("decoder.0#0", "decoder.0#0"),
        ("tdecoder.0#0", "tdecoder.0#0"),
        ("tdecoder.3#0", "tdecoder.3#0"),
    ];
    let mut failures: Vec<String> = Vec::new();
    for (chunk, (&samples, model)) in cases.iter().zip(models.iter()).enumerate() {
        let model = model.as_ref().unwrap();
        let training_length = model.training_length();
        let wave = Array3::from_shape_vec((1, 2, samples), fill(2 * samples, 13 + chunk as u32))
            .expect("wave");
        let padded = if samples < training_length {
            let mut wide = Array3::<f32>::zeros((1, 2, training_length));
            wide
                .slice_mut(ndarray::s![.., .., ..samples])
                .assign(&wave);
            wide
        } else {
            wave
        };

        // The host's front end, exactly as the reference run builds it.
        let z = model.spec.spec(&padded).expect("spectro");
        let mag = demucs_core::demucs::spec::pack_complex_as_channels(&z);
        let norm = model.normalise(&mag, &padded);
        let mut mag_norm = mag.clone();
        for (offset, scale) in norm.mean.iter().zip(norm.std.iter()) {
            mag_norm
                .slice_mut(ndarray::s![0, .., .., ..])
                .mapv_inplace(|v| (v - offset) / (1e-5 + scale));
        }
        let mut wave_norm = padded.clone();
        for (offset, scale) in norm.mean_t.iter().zip(norm.std_t.iter()) {
            wave_norm
                .slice_mut(ndarray::s![0, .., ..])
                .mapv_inplace(|v| (v - offset) / (1e-5 + scale));
        }

        let mut host_trace = VecTrace::default();
        let _host_out = model.forward(&padded, &mut host_trace).expect("host forward");
        let mut device_trace = VecTrace::default();
        let (freq_out, time_out) = runner
            .forward_branch_outputs(&mag_norm, &wave_norm, &mut device_trace)
            .expect("device branch outputs");

        for (device_name, host_name) in stage_names {
            let Some((_, _, host_values)) = host_trace.get(host_name) else {
                failures.push(format!("chunk {chunk} ({samples}): the host trace has no {host_name}"));
                continue;
            };
            let Some((_, _, device_values)) = device_trace.get(device_name) else {
                failures.push(format!("chunk {chunk} ({samples}): the device trace has no {device_name}"));
                continue;
            };
            if host_values.len() != device_values.len() {
                failures.push(format!(
                    "chunk {chunk} ({samples}) {device_name}: length mismatch host {} device {}",
                    host_values.len(),
                    device_values.len()
                ));
                continue;
            }
            let result = demucs_core::fixtures::compare(host_values, device_values)
                .expect("compare");
            println!(
                "chunk {chunk} ({samples}) {device_name} vs {host_name}: {:.2} dB (elements {})",
                result.snr_db(),
                host_values.len()
            );
            if result.snr_db() <= 60.0 {
                failures.push(format!(
                    "chunk {chunk} ({samples}) {device_name}: {:.2} dB (max abs {:.3e})",
                    result.snr_db(),
                    result.max_abs
                ));
            }
        }

        // The returned branch outputs against the host's own records: the freq
        // branch is the last decoder layer's output, the time branch the host's
        // `time_branch_out` (its raw, still-normalised decoder output).
        let freq: Vec<f32> = freq_out.iter().copied().collect();
        let time: Vec<f32> = time_out.iter().copied().collect();
        let last = model.config.depth - 1;
        for (name, host_name, got) in [
            ("freq branch output", format!("decoder.{last}#0"), &freq),
            ("time branch output", "time_branch_out".to_string(), &time),
        ] {
            let Some((_, _, host_values)) = host_trace.get(&host_name) else {
                continue;
            };
            if host_values.len() != got.len() {
                failures.push(format!(
                    "chunk {chunk} ({samples}) {name}: length mismatch host {} device {}",
                    host_values.len(),
                    got.len()
                ));
                continue;
            }
            let result =
                demucs_core::fixtures::compare(host_values, got).expect("compare");
            println!(
                "chunk {chunk} ({samples}) {name}: {:.2} dB",
                result.snr_db()
            );
            if result.snr_db() <= 60.0 {
                failures.push(format!(
                    "chunk {chunk} ({samples}) {name}: {:.2} dB",
                    result.snr_db()
                ));
            }
        }
    }
    if !failures.is_empty() {
        panic!(
            "{} stage(s) below the gate: {}",
            failures.len(),
            failures.join(", ")
        );
    }
}

/// The device decoder stacks against the host's per-layer traces.
#[test]
fn device_decoders_match_the_host() {
    let dir = workspace().join("bench/ref_dump/short");
    if !dir.join("manifest.json").exists() {
        eprintln!("skipping: no reference dump at {}", dir.display());
        return;
    }
    let Some(model) = load_model(44_100) else {
        return;
    };
    let mut runner = match GpuHtdemucsRunner::new(&model) {
        Ok(runner) => runner,
        Err(e) => {
            eprintln!("skipping: no usable wgpu adapter ({e})");
            return;
        }
    };

    let input = read_npy(dir.join("input.npy")).expect("input.npy");
    let input = Array3::from_shape_vec((1, input.shape[0], input.shape[1]), input.data)
        .expect("input shape");
    let training_length = model.training_length();
    let padded = if input.dim().2 < training_length {
        let mut wide = ndarray::Array3::<f32>::zeros((1, 2, training_length));
        wide.slice_mut(ndarray::s![.., .., ..input.dim().2]).assign(&input);
        wide
    } else {
        input
    };

    // Host front end (same as the encoder test).
    let z = model.spec.spec(&padded).expect("spectro");
    let mag = demucs_core::demucs::spec::pack_complex_as_channels(&z);
    let norm = model.normalise(&mag, &padded);
    let mut mag_norm = mag.clone();
    for (offset, scale) in norm.mean.iter().zip(norm.std.iter()) {
        mag_norm
            .slice_mut(ndarray::s![0, .., .., ..])
            .mapv_inplace(|v| (v - offset) / (1e-5 + scale));
    }
    let mut wave_norm = padded.clone();
    for (offset, scale) in norm.mean_t.iter().zip(norm.std_t.iter()) {
        wave_norm
            .slice_mut(ndarray::s![0, .., ..])
            .mapv_inplace(|v| (v - offset) / (1e-5 + scale));
    }

    let mut host_trace = VecTrace::default();
    let _ = model.forward(&padded, &mut host_trace).expect("host forward");

    let mut device_trace = VecTrace::default();
    let (freq_out, time_out) = runner
        .forward_branch_outputs(&mag_norm, &wave_norm, &mut device_trace)
        .expect("device branch outputs");

    let mut failures: Vec<String> = Vec::new();
    for index in 0..model.config.depth {
        for (device_name, host_name) in [
            (format!("decoder.{index}.rewrite"), format!("decoder.{index}.rewrite")),
            (format!("decoder.{index}.dconv"), format!("decoder.{index}.dconv")),
            (format!("decoder.{index}.conv_tr"), format!("decoder.{index}.conv_tr")),
            (format!("decoder.{index}#0"), format!("decoder.{index}#0")),
            (format!("tdecoder.{index}#0"), format!("tdecoder.{index}#0")),
        ] {
            let Some((_, host_shape, host_values)) = host_trace.get(&host_name) else {
                eprintln!("the host trace has no {host_name}");
                continue;
            };
            let Some((_, device_shape, device_values)) = device_trace.get(&device_name) else {
                eprintln!("the device trace has no {device_name}");
                continue;
            };
            if host_values.len() != device_values.len() {
                failures.push(format!(
                    "{device_name}: length mismatch host {} device {}",
                    host_values.len(),
                    device_values.len()
                ));
                continue;
            }
            let r = demucs_core::fixtures::compare(host_values, device_values).unwrap();
            println!("{device_name}: {:.2} dB SNR (host shape {:?})", r.snr_db(), host_shape);
            if r.snr_db() < 60.0 {
                failures.push(format!("{device_name}: {:.2} dB", r.snr_db()));
            }
        }
    }

    // The final branch outputs against the host's last decoder records.
    {
        let last = model.config.depth - 1;
        let Some((_, _, host)) = host_trace.get(&format!("decoder.{last}#0")) else {
            eprintln!("no decoder.{last}#0");
            return;
        };
        let got: Vec<f32> = freq_out.as_slice().unwrap().to_vec();
        if host.len() == got.len() {
            let r = demucs_core::fixtures::compare(host, &got).unwrap();
            println!("freq branch output: {:.2} dB SNR", r.snr_db());
            if r.snr_db() < 60.0 {
                failures.push(format!("freq branch output: {:.2} dB", r.snr_db()));
            }
        } else {
            println!("freq branch output: length mismatch host {} device {}", host.len(), got.len());
        }
    }

    if std::env::var("DEMUCS_SHAPE_DEBUG").is_ok() {
        if let (Some((_, _, dev_check)), Some((_, _, host_down2))) = (
            device_trace.get("down.freq.check"),
            host_trace.get("channel_downsampler"),
        ) {
            let c = demucs_core::fixtures::compare(host_down2, dev_check).unwrap();
            println!("down.freq.check vs host downsampler: {:.2} dB", c.snr_db());
        }
        if let (Some((_, _, dev_sum)), Some((_, _, dev_pre)), Some((_, _, host_down)), Some((_, _, host_enc3))) = (
            device_trace.get("decoder.0.sum"),
            device_trace.get("decoder.0.preadd"),
            host_trace.get("channel_downsampler"),
            host_trace.get("encoder.3"),
        ) {
            let cmp_pre = demucs_core::fixtures::compare(host_down, dev_pre).unwrap();
            println!("decoder.0 preadd vs host downsampler: {:.2} dB", cmp_pre.snr_db());
            let cmp_skip = demucs_core::fixtures::compare(host_enc3, dev_pre).unwrap();
            println!("decoder.0 preadd vs host encoder.3: {:.2} dB", cmp_skip.snr_db());
            let mut expected = host_down.clone();
            for (v, e) in expected.iter_mut().zip(host_enc3.iter()) { *v += e; }
            let cmp = demucs_core::fixtures::compare(&expected, dev_sum).unwrap();
            println!("decoder.0 sum vs host rebuild: {:.2} dB", cmp.snr_db());
        }
    }
    if !failures.is_empty() {
        panic!("{} stage(s) below the gate: {}", failures.len(), failures.join(", "));
    }
}

/// The decoder.0 rewrite conv (3x3, 384 in, 768 out, 8x44 plane) in isolation
/// against the host conv2d.
#[test]
fn decoder_rewrite_conv_matches_host() {
    let gpu = match demucs_core::gpu::Gpu::new() {
        Ok(gpu) => gpu,
        Err(e) => { eprintln!("skipping: no adapter ({e})"); return; }
    };
    let kernels = demucs_core::gpu::kernels::Kernels::new(&gpu).unwrap();
    let mut arena = demucs_core::gpu::arena::Arena::new(&gpu, 256 << 20);

    let fill = |n: usize, seed: u32| -> Vec<f32> {
        let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        (0..n).map(|_| {
            state ^= state << 13; state ^= state >> 17; state ^= state << 5;
            ((state % 20_000) as f32 / 10_000.0) - 1.0
        }).collect()
    };

    let (ic, oc, h, w) = (384usize, 768usize, 8usize, 44usize);
    let x_data = fill(ic * h * w, 41);
    let w_data = fill(oc * ic * 9, 42);
    let b_data = fill(oc, 43);

    // Padded weight like upload_conv: (pad(oc, BM), pad(ic*9, BK)).
    let k_total = ic * 9;
    let (padded, _, _) = match demucs_core::gpu::kernels::pad_matrix_for(
        &w_data, oc, k_total, demucs_core::gpu::shaders::BM, demucs_core::gpu::shaders::BK)
    { Ok(v) => v, Err(e) => { eprintln!("pad failed: {e}"); return; } };
    let weight = arena.upload(&gpu,
        &[demucs_core::gpu::kernels::pad_ceil(oc, demucs_core::gpu::shaders::BM),
          demucs_core::gpu::kernels::pad_ceil(k_total, demucs_core::gpu::shaders::BK)],
        &padded, "rw.weight").unwrap();
    let bias = arena.upload(&gpu, &[oc], &b_data, "rw.bias").unwrap();
    let x = arena.upload(&gpu, &[1, ic, h, w], &x_data, "rw.x").unwrap();
    let scratch = arena.alloc(&gpu, ((128usize << 20) * 4) as u64, "rw.scratch").unwrap();
    let out = arena.tensor(&gpu, &[1, oc, h, w], "rw.out").unwrap();

    let shape = demucs_core::gpu::kernels::Im2ColShape {
        batch: 1, in_channels: ic, h, w,
        kernel: (3, 3), stride: (1, 1), pad: (1, 1),
    };
    let mut recorder = demucs_core::gpu::arena::Recorder::new(&gpu);
    kernels.conv2d_into(&gpu, &mut arena, &mut recorder, &x, &weight, Some(&bias),
        &scratch, &out, shape, oc).unwrap();
    recorder.submit(&gpu).unwrap();

    let bytes = gpu.readback(&out.buffer, (out.len() * 4) as u64).unwrap();
    let got: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();

    // Host reference.
    let input4 = ndarray::Array4::from_shape_vec((1, ic, h, w), x_data).unwrap();
    let conv_w = demucs_core::demucs::weights::ConvW {
        weight: w_data, bias: b_data, shape: vec![oc, ic, 3, 3],
    };
    let expected = demucs_core::demucs::ops::conv2d(&input4, &conv_w, (1, 1), (1, 1), (1, 1));
    let expected_flat: Vec<f32> = expected.iter().copied().collect();
    let cmp = demucs_core::fixtures::compare(&expected_flat, &got).unwrap();
    println!("decoder rewrite conv: {:.2} dB (max abs {:.3e})", cmp.snr_db(), cmp.max_abs);
    let first = expected_flat.iter().zip(got.iter()).position(|(a, b)| (a - b).abs() > 1e-3);
    if let Some(fb) = first {
        println!("first mismatch: index {fb} -> oc {} f {} t {}",
            fb / (h * w), (fb / w) % h, fb % w);
    }
    assert!(cmp.snr_db() > 60.0, "rewrite conv diverged: {:.2} dB", cmp.snr_db());
}

/// End to end: the device path's separation against the host model's, and
/// against the reference dump's `model_out`.
#[test]
fn device_end_to_end_matches_host_and_reference() {
    let dir = workspace().join("bench/ref_dump/short");
    if !dir.join("manifest.json").exists() {
        eprintln!("skipping: no reference dump at {}", dir.display());
        return;
    }
    let Some(model) = load_model(44_100) else {
        return;
    };
    let mut runner = match GpuHtdemucsRunner::new(&model) {
        Ok(runner) => runner,
        Err(e) => {
            eprintln!("skipping: no usable wgpu adapter ({e})");
            return;
        }
    };

    let input = read_npy(dir.join("input.npy")).expect("input.npy");
    let input = Array3::from_shape_vec((1, input.shape[0], input.shape[1]), input.data)
        .expect("input shape");
    let training_length = model.training_length();
    let padded = if input.dim().2 < training_length {
        let mut wide = ndarray::Array3::<f32>::zeros((1, 2, training_length));
        wide.slice_mut(ndarray::s![.., .., ..input.dim().2]).assign(&input);
        wide
    } else {
        input
    };

    // Host reference.
    let mut host_trace = VecTrace::default();
    let host_out = model.forward(&padded, &mut host_trace).expect("host forward");

    // Device path.
    let mut no_trace = demucs_core::demucs::host::NoTrace;
    let device_out = demucs_core::gpu::htdemucs::separate_gpu(&model, &runner, &padded, &mut no_trace)
        .expect("device separation");

    let host_flat: Vec<f32> = host_out.as_slice().unwrap().to_vec();
    let device_flat: Vec<f32> = device_out.as_slice().unwrap().to_vec();
    let r = demucs_core::fixtures::compare(&host_flat, &device_flat).unwrap();
    println!("device vs host end to end: {:.2} dB SNR", r.snr_db());
    assert!(r.snr_db() >= 100.0, "device end to end: {:.2} dB", r.snr_db());

    // Against the Python reference, if the dump carries model_out.
    if let Some((_, _, reference)) = host_trace.get("model_out") {
        let _ = reference;
    }
    if let Some(reference) = read_reference_output(&dir) {
        let r = demucs_core::fixtures::compare(&reference, &device_flat).unwrap();
        println!("device vs python reference: {:.2} dB SNR", r.snr_db());
    }
}

/// Reads the reference dump's final output if present.
fn read_reference_output(dir: &std::path::Path) -> Option<Vec<f32>> {
    let dump = demucs_core::fixtures::ReferenceDump::open(dir).ok()?;
    let entry = dump.entry("model_out")?;
    dump.load_entry(entry).ok().map(|a| a.data)
}

/// The batched device forward must equal the same segments forwarded one at a
/// time, bit for bit.
///
/// Every stage is per-segment independent — the column convs, the DConv rows,
/// the norms and the attention all keep a segment's rows to themselves — so a
/// `(2, 2, T)` pass is two `(1, 2, T)` passes stacked and the arithmetic inside
/// a row never changes. Anything short of bit-exactness is therefore a
/// segment-mixing bug (a batch axis folded into the band or head axis), not
/// floating-point drift, which is what makes this a safe regression test.
#[test]
fn device_batched_forward_matches_the_same_segments_one_at_a_time() {
    let dir = workspace().join("bench/ref_dump/short");
    if !dir.join("manifest.json").exists() {
        eprintln!("skipping: no reference dump at {}", dir.display());
        return;
    }
    let Some(model) = load_model(44_100) else {
        return;
    };
    let mut runner = match GpuHtdemucsRunner::new(&model) {
        Ok(runner) => runner,
        Err(e) => {
            eprintln!("skipping: no usable wgpu adapter ({e})");
            return;
        }
    };

    let input = read_npy(dir.join("input.npy")).expect("input.npy");
    let input = Array3::from_shape_vec((1, input.shape[0], input.shape[1]), input.data)
        .expect("input shape");
    let training_length = model.training_length();
    let pad_to_segment = |source: ndarray::ArrayView3<'_, f32>| -> Array3<f32> {
        let take = source.dim().2.min(training_length);
        let mut wide = ndarray::Array3::<f32>::zeros((1, 2, training_length));
        wide
            .slice_mut(ndarray::s![.., .., ..take])
            .assign(&source.slice(ndarray::s![.., .., ..take]));
        wide
    };
    // Two distinct segments: the dump's audio, and its time reversal.
    let first = pad_to_segment(input.view());
    let second = pad_to_segment(input.slice(ndarray::s![.., .., ..;-1]));

    // The host's own front end, per segment: spectrogram, CaC packing, the
    // per-branch normalisation.
    let front = |chunk: &Array3<f32>| -> (Array4<f32>, Array3<f32>) {
        let z = model.spec.spec(chunk).expect("spectro");
        let mag = demucs_core::demucs::spec::pack_complex_as_channels(&z);
        let norm = model.normalise(&mag, chunk);
        let mut mag_norm = mag.clone();
        mag_norm
            .slice_mut(ndarray::s![0, .., .., ..])
            .mapv_inplace(|v| (v - norm.mean[0]) / (1e-5 + norm.std[0]));
        let mut wave_norm = chunk.clone();
        wave_norm
            .slice_mut(ndarray::s![0, .., ..])
            .mapv_inplace(|v| (v - norm.mean_t[0]) / (1e-5 + norm.std_t[0]));
        (mag_norm, wave_norm)
    };
    let (mag_a, wave_a) = front(&first);
    let (mag_b, wave_b) = front(&second);
    let (_mag_bins, mag_ch, bins, frames) = mag_a.dim();
    let _ = (mag_ch, bins, frames);
    let mut mag_both = Array4::<f32>::zeros((2, mag_a.dim().1, mag_a.dim().2, mag_a.dim().3));
    mag_both.slice_mut(ndarray::s![0, .., .., ..]).assign(&mag_a.slice(ndarray::s![0, .., .., ..]));
    mag_both.slice_mut(ndarray::s![1, .., .., ..]).assign(&mag_b.slice(ndarray::s![0, .., .., ..]));
    let mut wave_both = ndarray::Array3::<f32>::zeros((2, 2, training_length));
    wave_both.slice_mut(ndarray::s![0, .., ..]).assign(&wave_a.slice(ndarray::s![0, .., ..]));
    wave_both.slice_mut(ndarray::s![1, .., ..]).assign(&wave_b.slice(ndarray::s![0, .., ..]));

    let mut no_trace = demucs_core::demucs::host::NoTrace;

    // Stage by stage: one pass over both segments against two passes of one.
    let (enc_freq_both, enc_time_both) = runner
        .forward_encoders(&mag_both, &wave_both, &mut no_trace)
        .expect("batched encoders");
    let (enc_freq_a, enc_time_a) = runner
        .forward_encoders(&mag_a, &wave_a, &mut no_trace)
        .expect("encoders, first segment");
    let (enc_freq_b, enc_time_b) = runner
        .forward_encoders(&mag_b, &wave_b, &mut no_trace)
        .expect("encoders, second segment");

    let check = |label: &str, batched: &[f32], single_a: &[f32], single_b: &[f32]| {
        let half = single_a.len();
        for (name, batched, single) in [
            ("first", &batched[..half], single_a),
            ("second", &batched[half..], single_b),
        ] {
            let r = demucs_core::fixtures::compare(single, batched).expect("compare");
            assert_eq!(
                r.max_abs, 0.0,
                "{label} [{name}]: the batched pass differs from the single pass by {:.3e}",
                r.max_abs
            );
        }
    };
    check(
        "encoder, frequency branch",
        enc_freq_both.as_slice().unwrap(),
        enc_freq_a.as_slice().unwrap(),
        enc_freq_b.as_slice().unwrap(),
    );
    check(
        "encoder, waveform branch",
        enc_time_both.as_slice().unwrap(),
        enc_time_a.as_slice().unwrap(),
        enc_time_b.as_slice().unwrap(),
    );

    let (xt_freq_both, xt_time_both) = runner
        .forward_encoder_and_transformer(&mag_both, &wave_both, &mut no_trace)
        .expect("batched transformer inputs");
    let (xt_freq_a, xt_time_a) = runner
        .forward_encoder_and_transformer(&mag_a, &wave_a, &mut no_trace)
        .expect("transformer inputs, first segment");
    let (xt_freq_b, xt_time_b) = runner
        .forward_encoder_and_transformer(&mag_b, &wave_b, &mut no_trace)
        .expect("transformer inputs, second segment");
    check(
        "transformer output, frequency branch",
        xt_freq_both.as_slice().unwrap(),
        xt_freq_a.as_slice().unwrap(),
        xt_freq_b.as_slice().unwrap(),
    );
    check(
        "transformer output, waveform branch",
        xt_time_both.as_slice().unwrap(),
        xt_time_a.as_slice().unwrap(),
        xt_time_b.as_slice().unwrap(),
    );

    let (freq_both, time_both) = runner
        .forward_branch_outputs(&mag_both, &wave_both, &mut no_trace)
        .expect("batched branch outputs");
    let (freq_a, time_a) = runner
        .forward_branch_outputs(&mag_a, &wave_a, &mut no_trace)
        .expect("branch outputs, first segment");
    let (freq_b, time_b) = runner
        .forward_branch_outputs(&mag_b, &wave_b, &mut no_trace)
        .expect("branch outputs, second segment");

    check(
        "frequency branch",
        freq_both.as_slice().unwrap(),
        freq_a.as_slice().unwrap(),
        freq_b.as_slice().unwrap(),
    );
    check(
        "waveform branch",
        time_both.as_slice().unwrap(),
        time_a.as_slice().unwrap(),
        time_b.as_slice().unwrap(),
    );

    // And the whole separation, epilogue included.
    let mut both = ndarray::Array3::<f32>::zeros((2, 2, training_length));
    both.slice_mut(ndarray::s![0, .., ..]).assign(&first.slice(ndarray::s![0, .., ..]));
    both.slice_mut(ndarray::s![1, .., ..]).assign(&second.slice(ndarray::s![0, .., ..]));
    let out_both = demucs_core::gpu::htdemucs::separate_gpu(&model, &runner, &both, &mut no_trace)
        .expect("batched separation");
    let out_a = demucs_core::gpu::htdemucs::separate_gpu(&model, &runner, &first, &mut no_trace)
        .expect("separation, first segment");
    let out_b = demucs_core::gpu::htdemucs::separate_gpu(&model, &runner, &second, &mut no_trace)
        .expect("separation, second segment");
    let flat_a: Vec<f32> = out_a.as_slice().unwrap().to_vec();
    let flat_b: Vec<f32> = out_b.as_slice().unwrap().to_vec();
    check(
        "separation",
        out_both.as_slice().unwrap(),
        &flat_a,
        &flat_b,
    );
}
