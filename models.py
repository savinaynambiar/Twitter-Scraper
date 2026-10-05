"""Data models shared by every stage. Pydantic throughout; use .model_dump() for output."""
import re
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class TweetCandidate(BaseModel):
    tweet_url: str
    video_url: str = ""
    tweet_text: str = ""
    author_handle: str = ""
    author_name: str = ""
    created_at: str = ""
    likes: int = 0
    retweets: int = 0
    followers: int = 0
    duration: float = 0.0            # seconds; 0 means "not known yet"
    local_video_path: Optional[str] = None
    tweet_id: str = ""

    @model_validator(mode="after")
    def _set_tweet_id(self):
        match = re.search(r"/status/(\d+)", self.tweet_url)
        if not match:
            raise ValueError(f"Cannot extract tweet_id from {self.tweet_url}")
        self.tweet_id = match.group(1)
        return self


class ClipPick(BaseModel):
    """What the LLM returns when asked to choose a window."""
    matched: bool = Field(..., description="True only if the video really contains the requested content.")
    start_time_s: float = Field(..., description="Start of the chosen window in seconds.")
    end_time_s: float = Field(..., description="End of the chosen window in seconds.")
    quote: str = Field("", description="Words spoken inside the window, copied exactly from the transcript. Empty if there is no speech.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="0-1 confidence that the window matches the request.")
    reason: str = Field(..., description="One or two sentences on why this window was chosen, or why nothing matched.")


class VisualCheck(BaseModel):
    """What the vision model returns when shown the cut clip."""
    score: float = Field(..., ge=0.0, le=1.0, description="0-1: how well what is seen and heard matches the request.")
    observed: str = Field(..., description="Who or what is actually on screen.")


class ClipSegment(BaseModel):
    tweet_url: str
    video_url: str
    start_time_s: float
    end_time_s: float
    confidence: float
    reason: str


class Trace(BaseModel):
    candidates_considered: int = 0
    filtered_by_text: int = 0
    vision_calls: int = 0
    final_choice_rank: int = 0       # winner's position in the text-filter ranking (1 = top)
    errors: List[str] = Field(default_factory=list)


class ClipSelectionResult(BaseModel):
    tweet_url: Optional[str] = None
    video_url: Optional[str] = None
    start_time_s: Optional[float] = None
    end_time_s: Optional[float] = None
    confidence: float = 0.0
    reason: str = ""
    trace: Trace = Field(default_factory=Trace)
    alternates: List[ClipSegment] = Field(default_factory=list)
