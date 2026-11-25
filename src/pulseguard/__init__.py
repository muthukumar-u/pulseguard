"""PulseGuard: quality-aware rPPG deepfake detection prototype."""

from .fusion import ModalityScore, Verdict, FusionConfig, fuse
from .pipeline import PulseResult, analyze_traces

__all__ = [
    "ModalityScore",
    "Verdict",
    "FusionConfig",
    "fuse",
    "PulseResult",
    "analyze_traces",
]
__version__ = "0.1.0"
