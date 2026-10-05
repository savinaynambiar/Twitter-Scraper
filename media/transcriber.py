"""Whisper transcription with segment timestamps, cached by tweet id."""
import json
from typing import Dict, List

import config

_model = None


def transcribe(video_path: str, tweet_id: str) -> List[Dict]:
    """Returns [{"start": float, "end": float, "text": str}, ...]; empty if there is no speech."""
    cache_path = config.CACHE_DIR / f"{tweet_id}.transcript.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    global _model
    if _model is None:
        import whisper
        _model = whisper.load_model(config.WHISPER_MODEL)

    result = _model.transcribe(str(video_path), fp16=False)
    segments = [
        {"start": round(s["start"], 2), "end": round(s["end"], 2), "text": s["text"].strip()}
        for s in result.get("segments", []) if s["text"].strip()
    ]
    cache_path.write_text(json.dumps(segments, indent=2), encoding="utf-8")
    return segments


def format_transcript(segments: List[Dict]) -> str:
    return "\n".join(f"[{s['start']:.1f}-{s['end']:.1f}] {s['text']}" for s in segments)
