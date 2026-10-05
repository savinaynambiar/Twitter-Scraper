"""All tunable settings in one place. Values can be overridden in .env."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # dotenv is optional for the unit tests
    pass

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache"
PROMPT_DIR = ROOT / "prompt"
COOKIES_FILE = ROOT / "cookies.json"

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

MAX_TWEETS = 20              # candidates pulled from search
TOP_N = 6                    # candidates sent to analysis
MAX_VIDEO_S = 600            # skip videos longer than this (cost control)
MAX_SILENT_VIDEO_S = 180     # longest speech-free video sent whole to the vision model
DURATION_TOLERANCE_S = 2.0   # allowed deviation from the target clip length
SNAP_S = 1.0                 # snap window edges to transcript boundaries within this distance
MIN_SEGMENT_CONFIDENCE = 0.1
MIN_FINAL_CONFIDENCE = 0.3
MAX_ALTERNATES = 2
