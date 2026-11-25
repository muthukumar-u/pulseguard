"""Quality-aware fusion of modality scores.

Each modality reports (fake_prob, reliability). Fusion is a reliability-weighted
average. If total evidence is too low, or the result sits in the uncertain
band, the verdict is INCONCLUSIVE rather than a forced guess.

Only the rPPG modality is implemented in this prototype. Visual, lip-sync and
audio detectors plug in by returning a ModalityScore (see README roadmap).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from .pipeline import PulseResult


class Verdict(str, Enum):
    REAL = "real"
    FAKE = "fake"
    INCONCLUSIVE = "inconclusive"


@dataclass
class ModalityScore:
    name: str
    fake_prob: float    # 0 = looks real, 1 = looks fake
    reliability: float  # 0 = ignore, 1 = fully trustworthy in these conditions


@dataclass
class FusionConfig:
    real_below: float = 0.35
    fake_above: float = 0.65
    min_total_reliability: float = 0.5
    snr_for_full_presence_db: float = 8.0
    dark_skin_scale: float = 0.75


def rppg_modality(p: PulseResult, cfg: FusionConfig | None = None) -> ModalityScore:
    """Turn a PulseResult into a modality score.

    fake_prob comes from pulse *presence* (SNR) and cross-region *coherence*.
    reliability comes from *capture conditions*, deliberately not from SNR, so
    a weak pulse in good conditions counts against the video, while a weak pulse
    in poor conditions is simply down-weighted.

    The mapping is a hand-set heuristic, NOT calibrated. Replace with a
    classifier trained on labelled data before trusting any verdict.
    """
    cfg = cfg or FusionConfig()
    presence = float(np.clip(p.snr_db / cfg.snr_for_full_presence_db, 0.0, 1.0))
    evidence = 0.6 * presence + 0.4 * p.coherence
    fake_prob = 1.0 - evidence
    if p.conditions is None:
        reliability = 0.0  # unknown conditions: do not trust
    else:
        reliability = p.conditions.reliability(cfg.dark_skin_scale)
    return ModalityScore("rppg", float(fake_prob), float(reliability))


def fuse(scores: list[ModalityScore], cfg: FusionConfig | None = None) -> tuple[Verdict, float, float]:
    """Return (verdict, fused_fake_prob, total_reliability)."""
    cfg = cfg or FusionConfig()
    total = float(sum(s.reliability for s in scores))
    if total < cfg.min_total_reliability:
        return Verdict.INCONCLUSIVE, float("nan"), total
    p = float(sum(s.fake_prob * s.reliability for s in scores) / total)
    if p <= cfg.real_below:
        return Verdict.REAL, p, total
    if p >= cfg.fake_above:
        return Verdict.FAKE, p, total
    return Verdict.INCONCLUSIVE, p, total
