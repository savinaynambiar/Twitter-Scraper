"""scrape -> filter -> analyze -> select, wired with LangGraph. Nodes are plain functions and testable alone."""
from typing import List, Optional, TypedDict

import config
from filters.text_filter import TextFilter
from models import ClipSegment, ClipSelectionResult, Trace, TweetCandidate


class PipelineState(TypedDict, total=False):
    description: str
    duration: float
    urls: List[str]                       # if given, skip search and analyze exactly these tweets
    search: str                           # "web" (no account) or "x" (twikit, needs an X account)
    candidates: List[TweetCandidate]
    filtered_candidates: List[TweetCandidate]
    analysis_results: List[ClipSegment]
    trace: Trace
    final_result: ClipSelectionResult


def scrape_node(state: PipelineState) -> dict:
    trace = state["trace"]
    if state.get("urls"):
        candidates = [TweetCandidate(tweet_url=url) for url in state["urls"]]
    elif state.get("search") == "x":
        from scraper.twitter_scraper import TwitterScraper
        candidates = TwitterScraper().search_video_tweets(state["description"], config.MAX_TWEETS)
    else:
        from scraper.web_search import WebSearchScraper
        candidates = WebSearchScraper().search_video_tweets(state["description"], config.MAX_TWEETS)
    trace.candidates_considered = len(candidates)
    print(f"Scrape: {len(candidates)} candidates.")
    return {"candidates": candidates, "trace": trace}


def filter_node(state: PipelineState) -> dict:
    trace = state["trace"]
    if state.get("urls"):                 # hand-picked tweets have no text yet; keep them all
        kept = state["candidates"]
    else:
        kept = TextFilter().filter_and_rank(state["candidates"], state["description"], state["duration"])
    trace.filtered_by_text = len(kept)
    print(f"Filter: {len(kept)} of {len(state['candidates'])} kept.")
    return {"filtered_candidates": kept, "trace": trace}


def analyze_node(state: PipelineState, analyzer=None) -> dict:
    trace = state["trace"]
    if analyzer is None:
        from vision.clip_analyzer import ClipAnalyzer
        analyzer = ClipAnalyzer()

    results = []
    for candidate in state["filtered_candidates"]:
        trace.vision_calls += 1
        print(f"Analyze {trace.vision_calls}: {candidate.tweet_url}")
        try:
            segment = analyzer.analyze(candidate, state["description"], state["duration"])
        except Exception as exc:          # one bad video must not end the run
            trace.errors.append(f"{candidate.tweet_url}: {exc!r}")
            print(f"   [ERROR] {exc!r}")
            continue
        if segment and segment.confidence > config.MIN_SEGMENT_CONFIDENCE:
            results.append(segment)
    return {"analysis_results": results, "trace": trace}


def select_node(state: PipelineState) -> dict:
    trace = state["trace"]
    ranked = sorted(state["analysis_results"], key=lambda s: s.confidence, reverse=True)

    if not ranked or ranked[0].confidence < config.MIN_FINAL_CONFIDENCE:
        best_seen = f"best confidence was {ranked[0].confidence:.2f}" if ranked else "no video matched"
        reason = (f"No match: {trace.filtered_by_text} of {trace.candidates_considered} candidates "
                  f"analyzed, {best_seen}.")
        if trace.errors:
            reason += f" {len(trace.errors)} candidate(s) failed; see trace.errors."
        return {"final_result": ClipSelectionResult(reason=reason, trace=trace, alternates=ranked[:config.MAX_ALTERNATES])}

    best = ranked[0]
    order = [c.tweet_url for c in state["filtered_candidates"]]
    trace.final_choice_rank = order.index(best.tweet_url) + 1
    return {"final_result": ClipSelectionResult(
        **best.model_dump(), trace=trace, alternates=ranked[1:1 + config.MAX_ALTERNATES],
    )}


def build_pipeline():
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(PipelineState)
    graph.add_node("scrape", scrape_node)
    graph.add_node("filter", filter_node)
    graph.add_node("analyze", analyze_node)
    graph.add_node("select", select_node)
    graph.add_edge(START, "scrape")
    graph.add_edge("scrape", "filter")
    graph.add_edge("filter", "analyze")
    graph.add_edge("analyze", "select")
    graph.add_edge("select", END)
    return graph.compile()


def run_pipeline(description: str, duration: float, urls: Optional[List[str]] = None,
                 search: str = "web") -> ClipSelectionResult:
    state = build_pipeline().invoke({
        "description": description, "duration": duration, "urls": urls or [], "search": search, "trace": Trace(),
    })
    return state["final_result"]
