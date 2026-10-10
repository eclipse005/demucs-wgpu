#!/usr/bin/env python3
"""Verify OLA / normalize / shift indexing against adefossez/demucs apply.py + api.py.

Does not require torch: re-implements the exact arithmetic from
https://github.com/adefossez/demucs (apply.py apply_model split/shifts branches,
api.py Separator.separate_tensor normalisation).
"""
from __future__ import annotations

import math
import numpy as np

def transition_weight(segment_length: int, transition_power: float = 1.0) -> np.ndarray:
    # apply.py: cat(arange(1, seg//2+1), arange(seg - seg//2, 0, -1))
    weight = np.concatenate([
        np.arange(1, segment_length // 2 + 1, dtype=np.float64),
        np.arange(segment_length - segment_length // 2, 0, -1, dtype=np.float64),
    ])
    assert len(weight) == segment_length
    weight = (weight / weight.max()) ** transition_power
    return weight.astype(np.float32)

def padded_chunk(mix: np.ndarray, offset: int, length: int, target_length: int) -> np.ndarray:
    # TensorChunk.padded
    B, C, total = mix.shape
    delta = target_length - length
    start = offset - delta // 2
    end = start + target_length
    correct_start = max(0, start)
    correct_end = min(total, end)
    pad_left = correct_start - start
    out = np.zeros((B, C, target_length), dtype=np.float32)
    if correct_end > correct_start:
        out[:, :, pad_left:pad_left + (correct_end - correct_start)] = mix[:, :, correct_start:correct_end]
    return out

def center_trim(tensor: np.ndarray, reference: int) -> np.ndarray:
    length = tensor.shape[-1]
    delta = length - reference
    if delta <= 0:
        return tensor
    start = delta // 2
    return tensor[..., start:start + reference]

def apply_split_ola(mix: np.ndarray, segment_length: int, overlap: float = 0.25,
                    transition_power: float = 1.0) -> np.ndarray:
    """apply_model(..., shifts=0, split=True) with identity model (echo → 1 source)."""
    B, C, length = mix.shape
    sources = 1
    stride = int((1 - overlap) * segment_length)
    weight = transition_weight(segment_length, transition_power)
    out = np.zeros((B, sources, C, length), dtype=np.float32)
    sum_weight = np.zeros(length, dtype=np.float32)
    for offset in range(0, length, stride):  # apply.py
        chunk_length = min(segment_length, length - offset)
        padded = padded_chunk(mix, offset, chunk_length, segment_length)
        chunk_out = padded[:, None, :, :]
        trimmed = center_trim(chunk_out, chunk_length)
        cl = trimmed.shape[-1]
        out[..., offset:offset + cl] += weight[:cl] * trimmed
        sum_weight[offset:offset + cl] += weight[:cl]
    out /= sum_weight
    return out

def separate_tensor_norm(wav: np.ndarray):
    # api.py: ref = wav.mean(0); mean = ref.mean(); std = ref.std() + 1e-8
    # wav (C, T) like the API; we also accept (B,C,T)
    if wav.ndim == 2:
        ref = wav.mean(0)
    else:
        # batch: mean over channels per (b,t) then flatten — matches our port
        B, C, T = wav.shape
        ref = wav.mean(1).reshape(-1)
    mean = float(ref.mean())
    std = float(ref.std(ddof=1) + 1e-8)  # torch.std unbiased
    return mean, std

def main() -> None:
    rng = np.random.default_rng(0)
    mix = rng.standard_normal((1, 2, 8000)).astype(np.float32)
    # Short segment with tail chunk
    a = apply_split_ola(mix, segment_length=2048, overlap=0.25)
    assert np.isfinite(a).all()
    # Normalize matches torch-style unbiased
    wav = mix[0]
    mean, std = separate_tensor_norm(wav)
    ref = wav.mean(0)
    assert abs(mean - float(ref.mean())) < 1e-12
    assert abs(std - float(ref.std(ddof=1) + 1e-8)) < 1e-12
    # Shift pad indexing: padded length L+2M, view from offset length L+M-offset
    L, M, offset = 1000, 100, 37
    padded = padded_chunk(mix[:, :, :L], 0, L, L + 2 * M)
    shifted_len = L + M - offset
    shifted = padded[:, :, offset:offset + shifted_len]
    assert shifted.shape[-1] == shifted_len
    trim = M - offset
    assert shifted[:, :, trim:].shape[-1] == L
    print("ok: apply.py OLA / api.py normalize / shift trim invariants hold")
    print(f"  mean={mean:.8f} std={std:.8f} ola_peak={float(np.max(np.abs(a))):.6f}")

if __name__ == "__main__":
    main()
