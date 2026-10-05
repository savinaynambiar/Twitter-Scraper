"""
Find tweet links through ordinary web search (no X account needed).

Trade-offs compared with searching X directly: results are whatever the search
engines have indexed, so recent tweets are often missing, there are no like or
retweet counts, and some results have no video (those are skipped at download).
"""
import re
from typing import Dict, List

import config
from filters.text_filter import keywords, tokenize
from models import TweetCandidate

STATUS_URL = re.compile(r"https?://(?:www\.|mobile\.)?(?:x|twitter)\.com/(\w{1,15})/status/(\d+)", re.I)


def results_to_candidates(results: List[Dict], max_tweets: int) -> List[TweetCandidate]:
    """Keep results that link to a single tweet; dedupe by tweet id."""
    found, seen = [], set()
    for item in results:
        match = STATUS_URL.search(item.get("href") or "")
        if not match or match.group(1).lower() == "i" or match.group(2) in seen:
            continue
        handle, tweet_id = match.groups()
        seen.add(tweet_id)
        found.append(TweetCandidate(
            tweet_url=f"https://x.com/{handle}/status/{tweet_id}",
            tweet_text=f"{item.get('title') or ''} {item.get('body') or ''}".strip(),
            author_handle=handle,
        ))
        if len(found) >= max_tweets:
            break
    return found


class WebSearchScraper:
    def search_video_tweets(self, description: str, max_tweets: int = config.MAX_TWEETS) -> List[TweetCandidate]:
        try:
            from ddgs import DDGS
        except ImportError as exc:
            raise RuntimeError("ddgs is not installed. Run: pip install -r requirements.txt") from exc

        words = " ".join(keywords(description) or tokenize(description))
        queries = [f"site:x.com {words} video", f"site:twitter.com {words} video", f"site:x.com {words}"]
        results, errors = [], []
        for query in queries:
            try:
                results += DDGS().text(query, max_results=30) or []
            except Exception as exc:  # one engine or query failing should not stop the others
                errors.append(f"{query!r}: {exc!r}")

        candidates = results_to_candidates(results, max_tweets)
        if not candidates and errors:
            raise RuntimeError("Web search failed for every query: " + " | ".join(errors))
        return candidates
