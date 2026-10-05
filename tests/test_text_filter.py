from filters.text_filter import TextFilter, build_query, keywords
from models import TweetCandidate

DESC = "Trump talking about Charlie Kirk"


def tweet(i, text, likes=0, retweets=0, duration=60, video=None):
    return TweetCandidate(tweet_url=f"https://x.com/u/status/{i}", video_url=video or f"v{i}",
                          tweet_text=text, likes=likes, retweets=retweets, duration=duration)


def test_keywords_drop_stopwords():
    assert keywords(DESC) == ["trump", "charlie", "kirk"]
    assert build_query(DESC) == "trump charlie kirk filter:native_video"


def test_irrelevant_and_short_and_duplicate_are_dropped():
    tweets = [
        tweet(1, "Trump on Charlie Kirk", likes=100),
        tweet(2, "Elon Musk unveils Starship"),                 # no overlap
        tweet(3, "Trump on Charlie Kirk today", duration=5),    # shorter than target
        tweet(4, "Trump on Charlie Kirk"),                      # same text as 1
        tweet(5, "Charlie Kirk speech", video="v1"),            # same video as 1
        tweet(6, "trumpet solo"),                               # substring only, not a word match
    ]
    kept = TextFilter().filter_and_rank(tweets, DESC, target_duration=12)
    assert [t.tweet_id for t in kept] == ["1"]


def test_ranking_prefers_coverage_then_engagement():
    tweets = [
        tweet(1, "Kirk speaks", likes=900000),
        tweet(2, "Trump praises Charlie Kirk", likes=10),
        tweet(3, "Trump praises Charlie Kirk again", likes=50000),
    ]
    kept = TextFilter().filter_and_rank(tweets, DESC, target_duration=12)
    assert [t.tweet_id for t in kept] == ["3", "2", "1"]


def test_unknown_duration_is_kept():
    kept = TextFilter().filter_and_rank([tweet(1, "Trump and Kirk", duration=0)], DESC, target_duration=12)
    assert len(kept) == 1
