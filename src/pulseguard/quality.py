"""Signal-quality index (SQI) for pulse signals.

The most important reliability feature in PulseGuard: when the pulse signal is
too weak (dark skin under dim light, heavy compression, motion), we report that
instead of treating "no clear pulse" as "fake".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import welch

from .rppg import HR_LOW_HZ

# Plausible resting/active heart-rate band used for peak search (42-180 bpm).
PEAK_LOW_HZ = HR_LOW_HZ
PEAK_HIGH_HZ = 3.0


@dataclass
class QualityReport:
    hr_bpm: float
    snr_db: float
    peak_ratio: float
    usable: bool


def _spectrum(x: np.ndarray, fs: float, nfft: int = 8192):
    nperseg = len(x)
    f, p = welch(x - x.mean(), fs=fs, window="hann", nperseg=nperseg,
                 noverlap=0, nfft=max(nfft, nperseg), detrend=False)
    return f, p


def estimate_hr(x: np.ndarray, fs: float) -> float:
    """Dominant spectral peak in the heart-rate band, in beats per minute."""
    f, p = _spectrum(x, fs)
    band = (f >= PEAK_LOW_HZ) & (f <= PEAK_HIGH_HZ)
    if not band.any() or p[band].max() <= 0:
        return float("nan")
    return float(f[band][np.argmax(p[band])] * 60.0)


def snr_db(x: np.ndarray, fs: float, half_width_hz: float = 0.1) -> tuple[float, float]:
    """Pulse SNR in the style of de Haan & Jeanne (2013).

    Signal power = power within +/- half_width around the dominant peak and its
    first harmonic. Noise = the remaining power in the heart-rate band.
    Returns (snr_db, peak_ratio) where peak_ratio is signal/total band power.
    """
    f, p = _spectrum(x, fs)
    band = (f >= HR_LOW_HZ) & (f <= 4.0)
    peak_band = (f >= PEAK_LOW_HZ) & (f <= PEAK_HIGH_HZ)
    if not peak_band.any() or p[peak_band].max() <= 0:
        return -30.0, 0.0
    f0 = f[peak_band][np.argmax(p[peak_band])]
    sig_mask = (np.abs(f - f0) <= half_width_hz) | (np.abs(f - 2 * f0) <= half_width_hz)
    sig_mask &= band
    sig = p[sig_mask].sum()
    noise = p[band & ~sig_mask].sum()
    total = sig + noise
    if total <= 0:
        return -30.0, 0.0
    snr = 10.0 * np.log10((sig + 1e-18) / (noise + 1e-18))
    return float(np.clip(snr, -30.0, 30.0)), float(sig / total)


def assess(x: np.ndarray, fs: float, min_snr_db: float = 0.0) -> QualityReport:
    """Compute HR, SNR and a usable flag (SNR >= min_snr_db)."""
    snr, ratio = snr_db(x, fs)
    return QualityReport(hr_bpm=estimate_hr(x, fs), snr_db=snr, peak_ratio=ratio,
                         usable=bool(snr >= min_snr_db))


@dataclass
class CaptureConditions:
    """How trustworthy is the *capture*, independent of what the pulse looks like?

    This separation matters. A low pulse SNR can mean "fake video" or "bad
    capture" (dim light, motion, compression, dark skin). SNR alone cannot tell
    these apart, so reliability is computed from capture conditions only.
    All thresholds are heuristic defaults, to be calibrated on labelled data.
    """

    duration_s: float
    face_fraction: float      # share of frames where a face was found (0-1)
    brightness: float         # mean skin luma, 0-255
    saturation_fraction: float  # share of clipped skin pixels (0-1)
    motion: float             # 90th pct frame-to-frame face shift / face width
    ita_deg: float | None = None  # Individual Typology Angle skin-tone proxy

    def factors(self, dark_skin_scale: float = 0.75) -> dict[str, float]:
        def ramp(v, lo, hi):
            return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))

        brightness = min(ramp(self.brightness, 40, 80), 1.0 - ramp(self.brightness, 210, 250))
        f = {
            "duration": ramp(self.duration_s, 4.0, 10.0),
            "face": ramp(self.face_fraction, 0.5, 0.95),
            "brightness": brightness,
            "saturation": 1.0 - ramp(self.saturation_fraction, 0.02, 0.2),
            "motion": 1.0 - ramp(self.motion, 0.01, 0.05),
            # Dark skin lowers rPPG amplitude, so a missing pulse is less
            # informative. Soft penalty, not a hard rule. Calibrate per dataset.
            "skin_tone": dark_skin_scale if (self.ita_deg is not None and self.ita_deg < 10.0) else 1.0,
        }
        return f

    def reliability(self, dark_skin_scale: float = 0.75) -> float:
        """Product of per-factor scores in [0, 1]."""
        r = 1.0
        for v in self.factors(dark_skin_scale).values():
            r *= v
        return float(r)
