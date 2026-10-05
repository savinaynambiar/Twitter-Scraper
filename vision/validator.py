"""Pure checks on a window proposed by the LLM. No I/O, fully unit-tested."""
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import config


@dataclass
class Window:
    start: float
    end: float
    factor: float                      # multiply the confidence by this
    notes: List[str] = field(default_factory=list)


def _snap(value: float, points: List[float]) -> float:
    if not points:
        return value
    nearest = min(points, key=lambda p: abs(p - value))
    return nearest if abs(nearest - value) <= config.SNAP_S else value


def validate_window(start: float, end: float, video_duration: float, target: float,
                    segments: List[Dict], tolerance: float = config.DURATION_TOLERANCE_S) -> Optional[Window]:
    """Clamp to the video, snap to transcript boundaries, and fit the target length. None if unusable."""
    notes = []
    limit = video_duration if video_duration > 0 else float("inf")
    start, end = max(0.0, start), min(end, limit)
    if end <= start:
        return None

    start = _snap(start, [s["start"] for s in segments])
    end = _snap(end, [s["end"] for s in segments])
    if end <= start:
        return None

    factor = 1.0
    if abs((end - start) - target) > tolerance:
        factor = 0.9
        if end - start > target:
            end = start + target
            notes.append("window trimmed to the target length")
        else:
            end = min(limit, start + target)
            start = max(0.0, end - target)
            notes.append("window extended to the target length")
        if abs((end - start) - target) > tolerance:
            factor = 0.5
            notes.append(f"video too short for a {target:g}s clip")

    return Window(round(start, 2), round(end, 2), factor, notes)


def window_text(segments: List[Dict], start: float, end: float) -> str:
    """Transcript text that overlaps the window."""
    return " ".join(s["text"] for s in segments if s["end"] > start and s["start"] < end)


def quote_supported(quote: str, text: str, threshold: float = 0.6) -> bool:
    """True if most words of the LLM's quote really occur in the transcript text."""
    quote_words = re.findall(r"[a-z0-9']+", quote.lower())
    if not quote_words:
        return False
    text_words = set(re.findall(r"[a-z0-9']+", text.lower()))
    return sum(w in text_words for w in quote_words) / len(quote_words) >= threshold
