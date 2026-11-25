"""Face tracking, skin-pixel ROIs and mean-RGB trace extraction from video."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .quality import CaptureConditions

# ROI boxes as fractions of the detected face box: (x0, x1, y0, y1).
ROI_FRACTIONS = {
    "forehead": (0.28, 0.72, 0.06, 0.22),
    "left_cheek": (0.14, 0.36, 0.50, 0.72),
    "right_cheek": (0.64, 0.86, 0.50, 0.72),
}


@dataclass
class Extraction:
    fs: float
    traces: dict[str, np.ndarray]            # roi name -> (T, 3) mean RGB
    reference: np.ndarray | None              # (T, 3) non-face region, or None
    conditions: CaptureConditions
    notes: list[str] = field(default_factory=list)


def skin_mask(bgr: np.ndarray) -> np.ndarray:
    """Boolean skin mask via YCrCb thresholds. Drops hair, glasses, shadows.

    Fixed thresholds are a known weak point across skin tones and lighting; a
    learned segmenter is a planned replacement (see README roadmap).
    """
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    return cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127)) > 0


def _fill_nan(x: np.ndarray) -> np.ndarray:
    out = x.copy()
    idx = np.arange(len(x))
    for c in range(x.shape[1]):
        bad = np.isnan(out[:, c])
        if bad.all():
            raise ValueError("no valid samples to interpolate")
        if bad.any():
            out[bad, c] = np.interp(idx[bad], idx[~bad], out[~bad, c])
    return out


def _ita_deg(mean_rgb: np.ndarray) -> float:
    px = np.uint8([[np.clip(mean_rgb[::-1], 0, 255)]])  # RGB -> BGR
    lab = cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0, 0].astype(float)
    L = lab[0] * 100.0 / 255.0
    b = lab[2] - 128.0
    return float(np.degrees(np.arctan2(L - 50.0, b if abs(b) > 1e-6 else 1e-6)))


def extract(video_path: str | Path, max_seconds: float = 30.0, detect_every: int = 5,
            use_reference: bool = False) -> Extraction:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    fs = cap.get(cv2.CAP_PROP_FPS) or 30.0
    max_frames = int(max_seconds * fs)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    rows: dict[str, list] = {k: [] for k in ROI_FRACTIONS}
    ref_rows: list = []
    luma, sat, centers, widths = [], [], [], []
    box = None
    total = 0
    last_seen = -10**9

    while total < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if total % detect_every == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
            if len(faces):
                box = max(faces, key=lambda b: b[2] * b[3])
                last_seen = total
        # a box older than ~1 s is considered stale (face lost)
        have = box is not None and (total - last_seen) <= fs
        if not have:
            for k in rows:
                rows[k].append([np.nan] * 3)
            ref_rows.append([np.nan] * 3)
            total += 1
            continue

        x, y, w, h = [int(v) for v in box]
        centers.append((x + w / 2.0, y + h / 2.0))
        widths.append(w)
        for name, (x0, x1, y0, y1) in ROI_FRACTIONS.items():
            crop = frame[y + int(y0 * h): y + int(y1 * h), x + int(x0 * w): x + int(x1 * w)]
            if crop.size == 0:
                rows[name].append([np.nan] * 3)
                continue
            m = skin_mask(crop)
            if m.mean() < 0.2:
                rows[name].append([np.nan] * 3)
                continue
            px = crop[m]                         # (N, 3) BGR
            rows[name].append(px.mean(axis=0)[::-1].tolist())  # -> RGB
            luma.append(float(cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)[..., 0][m].mean()))
            sat.append(float((px.max(axis=1) >= 250).mean()))

        if use_reference:
            fh, fw = frame.shape[:2]
            if x >= 0.5 * w:
                patch = frame[y: y + h // 2, max(0, x - w // 2): x]
            elif fw - (x + w) >= 0.5 * w:
                patch = frame[y: y + h // 2, x + w: x + w + w // 2]
            else:
                patch = None
            ref_rows.append(patch.reshape(-1, 3).mean(axis=0)[::-1].tolist()
                            if patch is not None and patch.size else [np.nan] * 3)
        total += 1
    cap.release()

    if total == 0:
        raise ValueError("video contained no frames")

    arrays = {k: np.asarray(v, dtype=float) for k, v in rows.items()}
    valid = {k: v for k, v in arrays.items() if np.isfinite(v).any()}
    notes: list[str] = []
    if not valid:
        raise ValueError("no face/skin found in video")
    face_fraction = float(np.mean([np.isfinite(v[:, 0]).mean() for v in valid.values()]))
    traces = {k: _fill_nan(v) for k, v in valid.items()}

    reference = None
    if use_reference:
        r = np.asarray(ref_rows, dtype=float)
        if np.isfinite(r[:, 0]).mean() > 0.5:
            reference = _fill_nan(r)
        else:
            notes.append("reference region unavailable; illumination correction skipped")

    if len(centers) > 1:
        c = np.asarray(centers)
        step = np.linalg.norm(np.diff(c, axis=0), axis=1) / float(np.median(widths))
        motion = float(np.percentile(step, 90))
    else:
        motion = 0.0

    mean_rgb = np.mean([v.mean(axis=0) for v in traces.values()], axis=0)
    cond = CaptureConditions(
        duration_s=total / fs,
        face_fraction=face_fraction,
        brightness=float(np.mean(luma)) if luma else 0.0,
        saturation_fraction=float(np.mean(sat)) if sat else 0.0,
        motion=motion,
        ita_deg=_ita_deg(mean_rgb),
    )
    return Extraction(fs=fs, traces=traces, reference=reference, conditions=cond, notes=notes)
