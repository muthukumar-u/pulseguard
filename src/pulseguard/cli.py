"""Command line: pulseguard analyze VIDEO [--method pos|chrom|green]."""

from __future__ import annotations

import argparse
import json
import sys

from .fusion import fuse, rppg_modality
from .pipeline import analyze_traces
from .roi import extract


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pulseguard")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="analyze a video file")
    a.add_argument("video")
    a.add_argument("--method", default="pos", choices=["pos", "chrom", "green"])
    a.add_argument("--max-seconds", type=float, default=30.0)
    a.add_argument("--ref-correct", action="store_true",
                   help="illumination correction using a non-face reference region")
    args = ap.parse_args(argv)

    try:
        ex = extract(args.video, args.max_seconds, use_reference=args.ref_correct)
    except (FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    pulse = analyze_traces(ex.traces, ex.fs, args.method, ex.reference, ex.conditions)
    score = rppg_modality(pulse)
    verdict, p, total = fuse([score])

    print(json.dumps({
        "verdict": verdict.value,
        "fused_fake_prob": None if p != p else round(p, 3),
        "total_reliability": round(total, 3),
        "pulse": {
            "method": pulse.method,
            "hr_bpm": round(pulse.hr_bpm, 1),
            "snr_db": round(pulse.snr_db, 2),
            "coherence": round(pulse.coherence, 3),
            "per_roi_snr_db": {k: round(v.snr_db, 2) for k, v in pulse.per_roi.items()},
        },
        "capture_conditions": {
            **{k: round(v, 3) if isinstance(v, float) else v
               for k, v in vars(ex.conditions).items()},
            "factors": {k: round(v, 3) for k, v in ex.conditions.factors().items()},
        },
        "notes": ex.notes + ["Uncalibrated prototype. Only rPPG is active; see README."],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
