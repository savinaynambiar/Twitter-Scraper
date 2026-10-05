# Setup and first run

1. Install ffmpeg and the packages:
   winget install ffmpeg
   pip install -r requirements.txt
2. Copy `.env.example` to `.env` and fill in `GOOGLE_API_KEY` (and the X account for search).
3. Run the unit tests (no network or API key needed):
   pytest
4. Test the analysis path on one real tweet you picked by hand (no X login needed):
   python main.py --description "Trump talking about Charlie Kirk" --duration 12 --url https://x.com/<user>/status/<id>
5. Run with search. The default finds tweets through web search and needs no X account:
   python main.py --description "Trump talking about Charlie Kirk" --duration 12
   To search X itself instead (needs the X_ lines in .env or a cookies.json):
   python main.py --description "Trump talking about Charlie Kirk" --duration 12 --search x

Files that replace your old ones: main.py, models.py, pipeline.py, requirements.txt,
scraper/twitter_scraper.py, filters/text_filter.py, vision/clip_analyzer.py, prompt/vision_analysis.txt.
New: config.py, media/, vision/validator.py, prompt/visual_check.txt, tests/, .gitignore, .env.example.

Downloads, transcripts and cut clips are cached in `cache/` by tweet id, so reruns are fast.
