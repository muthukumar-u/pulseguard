"""Synthetic-signal tests. They check the maths, NOT real-world accuracy."""

import numpy as np
import pytest

from pulseguard.fusion import FusionConfig, ModalityScore, Verdict, fuse, rppg_modality
from pulseguard.pipeline import analyze_traces
from pulseguard.quality import CaptureConditions
from pulseguard.rppg import chrom, green, pos

FS = 30.0
PULSE_VEC = np.array([0.4, 1.0, 0.6])  # relative pulse amplitude per channel (R, G, B)


def synth(hr_bpm=72.0, amp=0.005, seconds=20, illum=0.05, noise=0.001, seed=0,
          base=(180.0, 130.0, 110.0)):
    """Mean-RGB trace: multiplicative illumination drift * (1 + pulse) + sensor noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * FS)) / FS
    pulse = np.sin(2 * np.pi * hr_bpm / 60.0 * t)
    drift = 1 + illum * np.sin(2 * np.pi * 0.5 * t) + illum * np.sin(2 * np.pi * 1.9 * t + 1.0)
    base = np.asarray(base)
    rgb = base * drift[:, None] * (1 + amp * PULSE_VEC * pulse[:, None])
    return rgb + rng.normal(0, noise * base, rgb.shape)


def good_conditions(ita=40.0):
    return CaptureConditions(duration_s=20, face_fraction=1.0, brightness=140,
                             saturation_fraction=0.0, motion=0.0, ita_deg=ita)


@pytest.mark.parametrize("method", [chrom, pos])
def test_recovers_heart_rate_under_lighting_drift(method):
    rgb = synth(hr_bpm=72)
    res = analyze_traces({"a": rgb, "b": synth(seed=1), "c": synth(seed=2)}, FS,
                         method=method.__name__)
    assert res.usable
    assert abs(res.hr_bpm - 72) < 3


def test_weak_pulse_dark_skin_still_found_with_pos():
    # 4x smaller pulse amplitude simulates strong melanin absorption.
    res = analyze_traces({"a": synth(amp=0.0012, seed=3), "b": synth(amp=0.0012, seed=4)},
                         FS, method="pos")
    assert abs(res.hr_bpm - 72) < 4


def test_pos_beats_green_under_strong_illumination_drift():
    # A periodic lighting artefact (1.9 Hz = 114 bpm) larger than the pulse.
    # Green locks onto it; POS cancels it. NOTE: green still reports a high SNR
    # because SNR only measures peakiness, not whether the peak is cardiac.
    rgb = synth(illum=0.15, amp=0.004, seed=5)
    err_pos = abs(analyze_traces({"a": rgb}, FS, "pos").hr_bpm - 72)
    err_green = abs(analyze_traces({"a": rgb}, FS, "green").hr_bpm - 72)
    assert err_pos < 3
    assert err_green > 20


def test_no_pulse_is_unusable():
    rng = np.random.default_rng(7)
    flat = np.array([180.0, 130.0, 110.0]) * (1 + rng.normal(0, 0.002, (600, 3)))
    res = analyze_traces({"a": flat, "b": flat * 1.0}, FS, "pos", min_snr_db=3.0)
    assert not res.usable


def test_quality_gate_blocks_verdict_in_bad_conditions():
    # Weak pulse + dim, moving capture must give INCONCLUSIVE, never FAKE.
    bad = CaptureConditions(duration_s=20, face_fraction=0.6, brightness=35,
                            saturation_fraction=0.0, motion=0.06, ita_deg=-20.0)
    rng = np.random.default_rng(8)
    flat = np.array([60.0, 45.0, 40.0]) * (1 + rng.normal(0, 0.003, (600, 3)))
    res = analyze_traces({"a": flat, "b": flat * 1.0}, FS, "pos", conditions=bad)
    verdict, _, total = fuse([rppg_modality(res)])
    assert verdict == Verdict.INCONCLUSIVE
    assert total < FusionConfig().min_total_reliability


def test_missing_pulse_in_good_conditions_is_flagged_fake():
    rng = np.random.default_rng(9)
    flat = np.array([180.0, 130.0, 110.0]) * (1 + rng.normal(0, 0.002, (600, 3)))
    res = analyze_traces({"a": flat, "b": flat * 1.0, "c": flat * 1.0}, FS, "pos",
                         conditions=good_conditions())
    verdict, p, _ = fuse([rppg_modality(res)])
    assert verdict == Verdict.FAKE and p > 0.65


def test_real_pulse_in_good_conditions_is_real():
    res = analyze_traces({"a": synth(seed=1), "b": synth(seed=2), "c": synth(seed=3)},
                         FS, "pos", conditions=good_conditions())
    verdict, p, _ = fuse([rppg_modality(res)])
    assert verdict == Verdict.REAL and p < 0.35


def test_fusion_downweights_unreliable_modality():
    scores = [ModalityScore("rppg", 0.9, 0.05), ModalityScore("visual", 0.1, 0.95)]
    verdict, p, _ = fuse(scores)
    assert verdict == Verdict.REAL and p < 0.2


def test_video_extraction_end_to_end(tmp_path, monkeypatch):
    """Skin-coloured patch pulsing at 72 bpm, with the face detector stubbed."""
    import cv2

    from pulseguard import roi

    path = str(tmp_path / "synthetic.avi")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), FS, (320, 240))
    for i in range(int(12 * FS)):
        p = 1 + 0.03 * PULSE_VEC * np.sin(2 * np.pi * 1.2 * i / FS)
        rgb = np.array([180.0, 130.0, 110.0]) * p
        frame = np.zeros((240, 320, 3), np.uint8)
        frame[:] = np.clip(rgb[::-1], 0, 255).astype(np.uint8)
        w.write(frame)
    w.release()

    class FakeCascade:
        def __init__(self, *_):
            pass

        def detectMultiScale(self, *_, **__):
            return np.array([[60, 40, 200, 160]])

    monkeypatch.setattr(roi.cv2, "CascadeClassifier", FakeCascade)
    ex = roi.extract(path)
    assert set(ex.traces) == {"forehead", "left_cheek", "right_cheek"}
    assert ex.conditions.face_fraction > 0.95
    res = analyze_traces(ex.traces, ex.fs, "pos", conditions=ex.conditions)
    assert abs(res.hr_bpm - 72) < 4


def test_unknown_conditions_give_zero_reliability():
    res = analyze_traces({"a": synth()}, FS, "pos")
    assert rppg_modality(res).reliability == 0.0
