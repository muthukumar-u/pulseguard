"""From RGB traces to a pulse result: extract, combine regions, score quality."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .quality import CaptureConditions, QualityReport, assess
from .rppg import METHODS, bandpass


@dataclass
class PulseResult:
    method: str
    hr_bpm: float
    snr_db: float
    usable: bool                     # pulse signal strong enough to analyse
    coherence: float                 # cross-region pulse agreement, 0-1
    per_roi: dict[str, QualityReport]
    conditions: CaptureConditions | None = None
    signal: np.ndarray = field(default_factory=lambda: np.zeros(0), repr=False)


def illumination_correct(trace: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Divide skin trace by the normalised non-skin reference (per channel).

    Removes scene-wide brightness changes (auto-exposure, flicker, screen glow)
    that hit skin and background alike. Heuristic: fails if the reference region
    contains its own motion or the lighting is not spatially uniform.
    """
    ref = np.asarray(reference, dtype=float)
    ref = ref / (ref.mean(axis=0, keepdims=True) + 1e-12)
    return np.asarray(trace, dtype=float) / np.maximum(ref, 1e-3)


def _coherence(signals: list[np.ndarray]) -> float:
    """Mean pairwise correlation of per-region pulses, clipped to [0, 1].

    Real faces share one cardiac pulse across forehead and cheeks. The review
    paper notes fakes often show inconsistent or uniform rPPG across regions.
    """
    if len(signals) < 2:
        return 0.0
    cs = []
    for i in range(len(signals)):
        for j in range(i + 1, len(signals)):
            a, b = signals[i], signals[j]
            if a.std() < 1e-12 or b.std() < 1e-12:
                cs.append(0.0)
            else:
                cs.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.clip(np.mean(cs), 0.0, 1.0))


def analyze_traces(traces: dict[str, np.ndarray], fs: float, method: str = "pos",
                   reference: np.ndarray | None = None,
                   conditions: CaptureConditions | None = None,
                   min_snr_db: float = 0.0) -> PulseResult:
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(METHODS)}")
    algo = METHODS[method]

    signals: dict[str, np.ndarray] = {}
    reports: dict[str, QualityReport] = {}
    for name, rgb in traces.items():
        rgb = np.asarray(rgb, dtype=float)
        if reference is not None:
            rgb = illumination_correct(rgb, reference)
        s = algo(rgb, fs)
        signals[name] = s
        reports[name] = assess(s, fs, min_snr_db)

    # Combine regions, weighting each by its linear SNR (usable regions only).
    usable = [n for n, r in reports.items() if r.usable]
    pool = usable if usable else [max(reports, key=lambda n: reports[n].snr_db)]
    w = np.array([10 ** (reports[n].snr_db / 10.0) for n in pool])
    w = w / w.sum()
    combined = sum(wi * bandpass(signals[n] / (signals[n].std() + 1e-12), fs)
                   for wi, n in zip(w, pool))

    overall = assess(combined, fs, min_snr_db)
    return PulseResult(
        method=method,
        hr_bpm=overall.hr_bpm,
        snr_db=overall.snr_db,
        usable=overall.usable,
        coherence=_coherence([signals[n] for n in pool]) if len(pool) > 1 else 0.0,
        per_roi=reports,
        conditions=conditions,
        signal=combined,
    )
