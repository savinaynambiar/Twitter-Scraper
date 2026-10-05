"""Cheap text stage: build the search query, then drop and rank candidates before any model call."""
import math
import re
from typing import List, Set

import config
from models import TweetCandidate

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "at", "for", "with", "by", "from",
    "about", "is", "are", "was", "were", "be", "being", "his", "her", "their", "its", "this",
    "that", "it", "he", "she", "they", "as", "talking", "talks", "talk", "speaking", "speaks",
    "saying", "says", "said", "discussing", "discusses", "video", "clip", "footage", "news",
}


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def keywords(description: str) -> List[str]:
    """Meaningful words of the description, in order, without duplicates."""
    seen, out = set(), []
    for word in tokenize(description):
        if word not in STOPWORDS and len(word) > 1 and word not in seen:
            seen.add(word)
            out.append(word)
    return out


def build_query(description: str) -> str:
    """Search query for X: keywords only, restricted to tweets with native video."""
    words = keywords(description) or tokenize(description)
    return " ".join(words) + " filter:native_video"


class TextFilter:
    def score(self, tweet: TweetCandidate, query_words: Set[str]) -> float:
        tweet_words = set(tokenize(tweet.tweet_text))
        coverage = len(query_words & tweet_words) / (len(query_words) or 1)
        engagement = math.log10(1 + tweet.likes + 2 * tweet.retweets)
        return coverage * (1 + 0.1 * engagement)

    def filter_and_rank(self, candidates: List[TweetCandidate], description: str,
                        target_duration: float = 0, top_n: int = config.TOP_N) -> List[TweetCandidate]:
        query_words = set(keywords(description))
        seen_ids, seen_videos, seen_texts = set(), set(), set()
        kept = []
        for tweet in candidates:
            text_key = " ".join(tokenize(tweet.tweet_text))
            if tweet.tweet_id in seen_ids or (tweet.video_url and tweet.video_url in seen_videos) \
                    or (text_key and text_key in seen_texts):
                continue  # repost of something already seen
            if not query_words & set(tokenize(tweet.tweet_text)):
                continue  # no keyword overlap at all
            if tweet.duration and target_duration and \
                    tweet.duration < target_duration - config.DURATION_TOLERANCE_S:
                continue  # video is shorter than the clip we need
            if tweet.duration > config.MAX_VIDEO_S:
                continue  # too long to analyze cheaply
            seen_ids.add(tweet.tweet_id)
            seen_videos.add(tweet.video_url)
            seen_texts.add(text_key)
            kept.append(tweet)

        kept.sort(key=lambda t: self.score(t, query_words), reverse=True)
        return kept[:top_n]
