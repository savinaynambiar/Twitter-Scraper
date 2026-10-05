from types import SimpleNamespace

import pipeline
from models import ClipPick, ClipSegment, Trace, TweetCandidate, VisualCheck
from scraper.twitter_scraper import tweet_to_candidate
from vision import clip_analyzer
from vision.clip_analyzer import ClipAnalyzer


def cand(i):
    return TweetCandidate(tweet_url=f"https://x.com/u/status/{i}", video_url=f"v{i}", tweet_text="t")


def seg(i, conf):
    return ClipSegment(tweet_url=f"https://x.com/u/status/{i}", video_url=f"v{i}",
                       start_time_s=0, end_time_s=12, confidence=conf, reason="r")


def state(results, n=3):
    return {"filtered_candidates": [cand(i) for i in range(1, n + 1)], "analysis_results": results,
            "trace": Trace(candidates_considered=n, filtered_by_text=n)}


def test_select_reports_real_rank_and_full_alternates():
    result = pipeline.select_node(state([seg(1, 0.4), seg(3, 0.9), seg(2, 0.6)]))["final_result"]
    assert result.tweet_url.endswith("/3") and result.trace.final_choice_rank == 3
    assert [a.tweet_url[-1] for a in result.alternates] == ["2", "1"]
    assert result.alternates[0].reason == "r"


def test_select_says_no_match_instead_of_inventing_one():
    result = pipeline.select_node(state([seg(1, 0.2)]))["final_result"]
    assert result.tweet_url is None and result.confidence == 0 and result.reason.startswith("No match")
    assert pipeline.select_node(state([]))["final_result"].tweet_url is None


def test_one_failing_candidate_does_not_end_the_run():
    class Flaky:
        def analyze(self, candidate, description, duration):
            if candidate.tweet_id == "1":
                raise RuntimeError("download failed")
            return seg(candidate.tweet_id, 0.8)

    s = state([])
    s.update(description="d", duration=12)
    out = pipeline.analyze_node(s, analyzer=Flaky())
    assert len(out["analysis_results"]) == 2
    assert out["trace"].vision_calls == 3 and len(out["trace"].errors) == 1


def test_twikit_tweet_mapping_picks_best_mp4():
    stream = lambda url, rate, kind="video/mp4": SimpleNamespace(url=url, bitrate=rate, content_type=kind)
    user = SimpleNamespace(screen_name="nick", name="Nick", followers_count=5)
    video = SimpleNamespace(type="video", duration_millis=56000,
                            streams=[stream("low", 100), stream("high", 900), stream("hls", 9999, "application/x-mpegURL")])
    tweet = SimpleNamespace(id="42", text="hi", full_text="hello", user=user, created_at="now",
                            favorite_count=7, retweet_count=None, media=[SimpleNamespace(type="photo"), video])
    c = tweet_to_candidate(tweet)
    assert (c.tweet_id, c.video_url, c.duration, c.tweet_text, c.retweets) == ("42", "high", 56.0, "hello", 0)
    tweet.media = [SimpleNamespace(type="photo")]
    assert tweet_to_candidate(tweet) is None


SEGS = [{"start": 0.0, "end": 8.0, "text": "Thank you very much."},
        {"start": 8.0, "end": 20.0, "text": "Charlie Kirk is doing an incredible job with young people."}]


class FakeAnalyzer(ClipAnalyzer):
    def __init__(self, pick, visual=0.9):
        super().__init__()
        self.pick, self.visual = pick, visual

    def _pick(self, prompt, video_path=None):
        assert "Request: Trump on Kirk" in prompt and "[8.0-20.0] Charlie Kirk" in prompt
        return self.pick

    def _visual_check(self, description, clip_path):
        return VisualCheck(score=self.visual, observed="Trump at a podium")


def run_analyzer(monkey, pick, visual=0.9, duration=56.0):
    monkey(clip_analyzer.downloader, "download_video", lambda url, tid: {"path": "x.mp4", "duration": duration, "text": "tweet"})
    monkey(clip_analyzer.downloader, "cut_clip", lambda path, tid, s, e: "clip.mp4")
    monkey(clip_analyzer.transcriber, "transcribe", lambda path, tid: SEGS)
    return FakeAnalyzer(pick, visual).analyze(cand(1), "Trump on Kirk", 12)


def test_analyzer_combines_text_and_visual_confidence(monkey):
    pick = ClipPick(matched=True, start_time_s=8.2, end_time_s=19.7, confidence=0.9, reason="He praises Kirk.",
                    quote="Charlie Kirk is doing an incredible job")
    s = run_analyzer(monkey, pick)
    assert (s.start_time_s, s.end_time_s, s.confidence) == (8.0, 20.0, 0.81)
    assert "Trump at a podium" in s.reason


def test_analyzer_penalizes_invented_quote_and_rejects_non_match(monkey):
    made_up = ClipPick(matched=True, start_time_s=8.0, end_time_s=20.0, confidence=0.9, reason="r",
                       quote="I love his free speech advocacy")
    assert run_analyzer(monkey, made_up, visual=1.0).confidence == 0.45
    no = ClipPick(matched=False, start_time_s=0, end_time_s=0, confidence=0, reason="about rockets")
    assert run_analyzer(monkey, no) is None
    assert run_analyzer(monkey, made_up, duration=5.0) is None      # shorter than target


def test_web_results_become_tweet_candidates():
    from scraper.web_search import results_to_candidates
    results = [
        {"href": "https://twitter.com/nick/status/111?s=20", "title": "Nick on X", "body": "Trump on Kirk"},
        {"href": "https://x.com/nick/status/111", "title": "dup", "body": ""},
        {"href": "https://x.com/nick", "title": "profile page", "body": ""},
        {"href": "https://x.com/i/status/333", "title": "no handle", "body": ""},
        {"href": "https://mobile.x.com/Other_1/status/222/video/1", "title": "t", "body": "b"},
        {"href": "https://example.com/x.com/a/status/9", "title": "t", "body": ""},
    ]
    found = results_to_candidates(results, 10)
    assert [c.tweet_url for c in found] == ["https://x.com/nick/status/111", "https://x.com/Other_1/status/222"]
    assert found[0].tweet_text == "Nick on X Trump on Kirk"
    assert len(results_to_candidates(results, 1)) == 1
