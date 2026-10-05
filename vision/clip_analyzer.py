"""
Analyze one candidate for real:
  download -> transcribe -> LLM picks a window -> validate -> vision model checks the cut clip.
Returns None when the video does not match; never invents a result.
"""
import base64
from pathlib import Path
from typing import Dict, List, Optional

import config
from media import downloader, transcriber
from models import ClipPick, ClipSegment, TweetCandidate, VisualCheck
from vision.validator import quote_supported, validate_window, window_text


def _load_prompt(name: str) -> str:
    return (config.PROMPT_DIR / name).read_text(encoding="utf-8")


class ClipAnalyzer:
    def __init__(self, llm=None):
        self._llm = llm  # created on first use, so importing this module needs no API key

    # ------------------------------------------------------------------ LLM plumbing
    @property
    def llm(self):
        if self._llm is None:
            from langchain_google_genai import ChatGoogleGenerativeAI
            self._llm = ChatGoogleGenerativeAI(
                model=config.GEMINI_MODEL, temperature=0.0, timeout=120, max_retries=1)
        return self._llm

    def _ask(self, schema, prompt: str, video_path: Optional[Path] = None):
        from langchain_core.messages import HumanMessage
        model = self.llm.with_structured_output(schema)
        if video_path is None:
            return model.invoke([HumanMessage(content=prompt)])

        data = base64.b64encode(Path(video_path).read_bytes()).decode("utf-8")
        text = {"type": "text", "text": prompt}
        try:  # current langchain content-block format
            video = {"type": "video", "base64": data, "mime_type": "video/mp4"}
            return model.invoke([HumanMessage(content=[text, video])])
        except ValueError:  # older langchain-google-genai releases use this format
            video = {"type": "media", "data": data, "mime_type": "video/mp4"}
            return model.invoke([HumanMessage(content=[text, video])])

    def _pick(self, prompt: str, video_path: Optional[Path] = None) -> ClipPick:
        return self._ask(ClipPick, prompt, video_path)

    def _visual_check(self, description: str, clip_path: Path) -> VisualCheck:
        prompt = _load_prompt("visual_check.txt").format(description=description)
        return self._ask(VisualCheck, prompt, clip_path)

    # ------------------------------------------------------------------ public API
    def analyze(self, candidate: TweetCandidate, description: str, target_duration: float) -> Optional[ClipSegment]:
        def skip(why: str) -> None:
            print(f"   [SKIP] {candidate.tweet_url}: {why}")

        print("   downloading video...", flush=True)
        try:
            meta = downloader.download_video(candidate.tweet_url, candidate.tweet_id)
        except Exception as exc:
            if "no video" in str(exc).lower():   # tweet exists but has no video; not an error
                return skip("tweet has no video")
            raise
        candidate.local_video_path = meta["path"]
        candidate.video_url = candidate.video_url or meta.get("video_url", "")
        candidate.duration = meta["duration"] or candidate.duration
        candidate.tweet_text = candidate.tweet_text or meta.get("text", "")
        duration = candidate.duration

        if duration and duration < target_duration - config.DURATION_TOLERANCE_S:
            return skip(f"video is {duration:.0f}s, shorter than the {target_duration:g}s target")
        if duration > config.MAX_VIDEO_S:
            return skip(f"video is {duration:.0f}s, over the {config.MAX_VIDEO_S}s limit")

        print(f"   transcribing {duration:.0f}s of video (silent while it works)...", flush=True)
        segments: List[Dict] = transcriber.transcribe(meta["path"], candidate.tweet_id)
        prompt = _load_prompt("vision_analysis.txt").format(
            description=description,
            duration=f"{target_duration:g}",
            video_length=f"{duration:.1f}",
            tweet_text=candidate.tweet_text,
            transcript=transcriber.format_transcript(segments)
            or "(no speech detected - judge from the attached video and leave `quote` empty)",
        )

        print(f"   {len(segments)} transcript lines; asking Gemini to pick a window...", flush=True)
        if segments:
            pick = self._pick(prompt)
        elif duration <= config.MAX_SILENT_VIDEO_S:
            whole = downloader.cut_clip(meta["path"], candidate.tweet_id, 0.0, duration)
            pick = self._pick(prompt, whole)
        else:
            return skip("no speech, and too long to judge visually")

        if not pick.matched or pick.confidence <= 0:
            return skip(f"no match - {pick.reason}")

        window = validate_window(pick.start_time_s, pick.end_time_s, duration, target_duration, segments)
        if window is None:
            return skip(f"model returned an unusable window ({pick.start_time_s}-{pick.end_time_s})")

        confidence = pick.confidence * window.factor
        notes = list(window.notes)

        if segments and not quote_supported(pick.quote, window_text(segments, window.start, window.end)):
            confidence *= 0.5
            notes.append("quoted words not found in the transcript for this window")

        print(f"   window {window.start}-{window.end}s; cutting clip and running visual check...", flush=True)
        try:
            clip = downloader.cut_clip(meta["path"], candidate.tweet_id, window.start, window.end)
            seen = self._visual_check(description, clip)
            confidence *= seen.score
            notes.append(f"visual check {seen.score:.2f}: {seen.observed}")
        except Exception as exc:
            confidence *= 0.8
            notes.append(f"visual check unavailable ({exc!r})")

        return ClipSegment(
            tweet_url=candidate.tweet_url,
            video_url=candidate.video_url,
            start_time_s=window.start,
            end_time_s=window.end,
            confidence=round(confidence, 3),
            reason=pick.reason + (" [" + "; ".join(notes) + "]" if notes else ""),
        )
