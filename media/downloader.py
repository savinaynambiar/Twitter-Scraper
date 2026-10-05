"""Download a tweet's video with yt-dlp and cut clips with ffmpeg. Everything is cached by tweet id."""
import json
import subprocess
from pathlib import Path

import config


def download_video(tweet_url: str, tweet_id: str) -> dict:
    """
    Returns {"path", "video_url", "duration", "text", "author_handle", "author_name", "likes", "retweets"}.
    A second call for the same tweet reads the cache and does no network work.
    """
    config.CACHE_DIR.mkdir(exist_ok=True)
    video_path = config.CACHE_DIR / f"{tweet_id}.mp4"
    meta_path = config.CACHE_DIR / f"{tweet_id}.meta.json"
    if video_path.exists() and meta_path.exists():
        cached = json.loads(meta_path.read_text(encoding="utf-8"))
        if "video_url" in cached:      # older cache entries lack it; refresh those
            return cached

    import yt_dlp

    options = {
        "outtmpl": str(config.CACHE_DIR / f"{tweet_id}.%(ext)s"),
        "format": "best[ext=mp4][protocol^=http]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "playlist_items": "1",      # a tweet can hold several videos; take the first
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(tweet_url, download=True)
    if info.get("entries"):
        info = info["entries"][0]
    if not video_path.exists():
        raise RuntimeError(f"yt-dlp finished but {video_path.name} is missing")

    meta = {
        "path": str(video_path),
        "video_url": info.get("url") or "",
        "duration": float(info.get("duration") or 0) or probe_duration(video_path),
        "text": info.get("description") or info.get("title") or "",
        "author_handle": info.get("uploader_id") or "",
        "author_name": info.get("uploader") or "",
        "likes": info.get("like_count") or 0,
        "retweets": info.get("repost_count") or 0,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def probe_duration(video_path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def cut_clip(video_path: str, tweet_id: str, start: float, end: float) -> Path:
    """Cut [start, end] into a small 480p file. Re-encodes so the cut is frame-accurate."""
    out_path = config.CACHE_DIR / f"{tweet_id}_{start:.1f}_{end:.1f}.mp4"
    if not out_path.exists():
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.3f}", "-i", str(video_path),
             "-t", f"{end - start:.3f}", "-vf", "scale=-2:480",
             "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", str(out_path)],
            check=True,
        )
    return out_path
