"""
Live X search through twikit (X's internal web API, driven by a logged-in account).

Use a throwaway account: this is against X's terms and accounts do get locked.
There is no mock fallback. If search fails, the pipeline stops with a clear error
instead of returning invented tweets. To test without search, use `main.py --url`.
"""
import asyncio
import os
from typing import List, Optional

import config
from filters.text_filter import build_query
from models import TweetCandidate


class ScraperError(RuntimeError):
    pass


def tweet_to_candidate(tweet) -> Optional[TweetCandidate]:
    """Map a twikit Tweet to a TweetCandidate. Returns None if it has no native video."""
    for media in getattr(tweet, "media", None) or []:
        if getattr(media, "type", "") != "video":
            continue
        streams = [s for s in getattr(media, "streams", None) or []
                   if getattr(s, "content_type", "") == "video/mp4"]
        if not streams:
            continue
        best = max(streams, key=lambda s: getattr(s, "bitrate", 0) or 0)
        user = tweet.user
        return TweetCandidate(
            tweet_url=f"https://x.com/{user.screen_name}/status/{tweet.id}",
            video_url=best.url,
            tweet_text=getattr(tweet, "full_text", None) or tweet.text or "",
            author_handle=user.screen_name,
            author_name=user.name or user.screen_name,
            created_at=str(tweet.created_at or ""),
            likes=tweet.favorite_count or 0,
            retweets=tweet.retweet_count or 0,
            followers=user.followers_count or 0,
            duration=(getattr(media, "duration_millis", 0) or 0) / 1000.0,
        )
    return None


class TwitterScraper:
    def __init__(self, cookies_file=config.COOKIES_FILE):
        self.cookies_file = str(cookies_file)

    def search_video_tweets(self, description: str, max_tweets: int = config.MAX_TWEETS) -> List[TweetCandidate]:
        return asyncio.run(self._search(build_query(description), max_tweets))

    async def _client(self):
        try:
            from twikit import Client
        except ImportError as exc:
            raise ScraperError("twikit is not installed. Run: pip install -r requirements.txt") from exc

        client = Client("en-US")
        if os.path.exists(self.cookies_file):
            client.load_cookies(self.cookies_file)
            return client

        username, email, password = (os.getenv(k) for k in ("X_USERNAME", "X_EMAIL", "X_PASSWORD"))
        if not (username and password):
            raise ScraperError("No cookies.json and no X_USERNAME / X_PASSWORD in .env. See .env.example.")
        try:
            await client.login(auth_info_1=username, auth_info_2=email, password=password)
        except Exception as exc:
            raise ScraperError(f"X login failed: {exc!r}") from exc
        client.save_cookies(self.cookies_file)
        return client

    async def _search(self, query: str, max_tweets: int) -> List[TweetCandidate]:
        client = await self._client()
        found, seen = [], set()
        try:
            page = await client.search_tweet(query, "Top", count=20)
            for _ in range(5):  # at most 5 pages
                tweets = list(page or [])
                if not tweets:
                    break
                for tweet in tweets:
                    tweet = getattr(tweet, "retweeted_tweet", None) or tweet
                    candidate = tweet_to_candidate(tweet)
                    if candidate and candidate.tweet_id not in seen:
                        seen.add(candidate.tweet_id)
                        found.append(candidate)
                if len(found) >= max_tweets:
                    break
                page = await page.next()
        except Exception as exc:
            if not found:
                raise ScraperError(
                    f"X search failed: {exc!r}. If this mentions auth or 403/404, delete cookies.json "
                    "and retry; if it persists, twikit is out of date with X (pip install -U twikit)."
                ) from exc
            print(f"[WARN] X search stopped early ({exc!r}); continuing with {len(found)} tweets.")
        return found[:max_tweets]
