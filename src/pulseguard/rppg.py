"""Pulse extraction from mean-RGB traces: CHROM and POS.

Both algorithms take a (T, 3) array of spatially averaged R, G, B values over
skin pixels and return a 1-D pulse signal of length T. They are designed so
that illumination changes that scale all channels equally (shadows, specular
highlights, brightness drift) cancel out. That is why they are preferred over
the plain green channel for lighting robustness.

References: see docs/REFERENCES.md ([R2] CHROM, [R3] POS).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

HR_LOW_HZ = 0.7   # 42 bpm
HR_HIGH_HZ = 4.0  # 240 bpm


def bandpass(x: np.ndarray, fs: float, low: float = HR_LOW_HZ, high: float = HR_HIGH_HZ,
             order: int = 3) -> np.ndarray:
    """Zero-phase Butterworth band-pass over the plausible heart-rate band."""
    nyq = fs / 2.0
    high = min(high, 0.95 * nyq)
    sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    # sosfiltfilt needs enough samples for its padding; fall back gracefully.
    padlen = 3 * (2 * len(sos) + 1)
    if x.shape[-1] <= padlen:
        return x - x.mean(axis=-1, keepdims=True)
    return sosfiltfilt(sos, x, axis=-1)


def _overlap_add(segments: list[tuple[int, np.ndarray]], length: int) -> np.ndarray:
    out = np.zeros(length)
    for start, seg in segments:
        out[start:start + len(seg)] += seg
    return out


def _std_norm(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / s if s > 1e-12 else x - x.mean()


def chrom(rgb: np.ndarray, fs: float, win_sec: float = 1.6) -> np.ndarray:
    """CHROM (de Haan & Jeanne, 2013), windowed with Hann overlap-add."""
    rgb = np.asarray(rgb, dtype=float)
    n = len(rgb)
    win = max(int(round(win_sec * fs)), 8)
    if n < win:
        win = n
    step = max(win // 2, 1)
    hann = np.hanning(win) if win > 2 else np.ones(win)
    segs = []
    for start in range(0, n - win + 1, step):
        c = rgb[start:start + win]
        mean = c.mean(axis=0)
        if np.any(mean <= 0):
            continue
        cn = c / mean
        r, g, b = cn[:, 0], cn[:, 1], cn[:, 2]
        xs = 3.0 * r - 2.0 * g
        ys = 1.5 * r + g - 1.5 * b
        xf = bandpass(xs, fs)
        yf = bandpass(ys, fs)
        alpha = xf.std() / (yf.std() + 1e-12)
        s = xf - alpha * yf
        segs.append((start, _std_norm(s) * hann))
    return _overlap_add(segs, n)


_POS_PROJ = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])


def pos(rgb: np.ndarray, fs: float, win_sec: float = 1.6) -> np.ndarray:
    """POS, plane-orthogonal-to-skin (Wang et al., 2017), windowed overlap-add."""
    rgb = np.asarray(rgb, dtype=float)
    n = len(rgb)
    win = max(int(round(win_sec * fs)), 8)
    if n < win:
        win = n
    step = max(win // 2, 1)
    hann = np.hanning(win) if win > 2 else np.ones(win)
    segs = []
    for start in range(0, n - win + 1, step):
        c = rgb[start:start + win]
        mean = c.mean(axis=0)
        if np.any(mean <= 0):
            continue
        cn = (c / mean).T            # (3, win)
        s = _POS_PROJ @ cn           # (2, win)
        h = s[0] + (s[0].std() / (s[1].std() + 1e-12)) * s[1]
        segs.append((start, _std_norm(h) * hann))
    return bandpass(_overlap_add(segs, n), fs)


def green(rgb: np.ndarray, fs: float) -> np.ndarray:
    """Plain green-channel baseline (Verkruysse et al., 2008). Kept for comparison."""
    g = np.asarray(rgb, dtype=float)[:, 1]
    return bandpass(g / (g.mean() + 1e-12) - 1.0, fs)


METHODS = {"chrom": chrom, "pos": pos, "green": green}
