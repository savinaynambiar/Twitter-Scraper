"""
Usage:
  python main.py --description "Trump talking about Charlie Kirk" --duration 12
  python main.py --description "..." --duration 12 --search x      (search X itself; needs an account)
  python main.py --description "..." --duration 12 --url https://x.com/user/status/123   (skip search)
"""
import argparse
import re

import config  # noqa: F401  (loads .env before anything else)
from pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Find the best matching clip in X/Twitter videos.")
    parser.add_argument("--description", required=True, help='What the clip should show, e.g. "Trump talking about Charlie Kirk"')
    parser.add_argument("--duration", type=float, required=True, help="Target clip length in seconds")
    parser.add_argument("--url", action="append", default=[],
                        help="Analyze this tweet URL instead of searching. Repeat for several.")
    parser.add_argument("--search", choices=["web", "x"], default="web",
                        help="web: find tweets through web search, no account (default). x: search X with twikit.")
    parser.add_argument("--out", help="Output JSON path (default: result_<description>.json)")
    args = parser.parse_args()

    result = run_pipeline(args.description, args.duration, args.url, args.search)
    output = result.model_dump_json(indent=2)
    print("\n" + output)

    slug = re.sub(r"[^a-z0-9]+", "_", args.description.lower()).strip("_")[:60]
    out_path = args.out or f"result_{slug}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
