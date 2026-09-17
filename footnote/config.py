import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

MODEL = os.environ.get("FOOTNOTE_MODEL", "claude-opus-5")
JUDGE_MODEL = os.environ.get("FOOTNOTE_JUDGE_MODEL", "claude-haiku-4-5")
EFFORT = os.environ.get("FOOTNOTE_EFFORT", "medium")
# Demo mode swaps the model for scripted replies. Search, tools and citations still run for real.
DEMO = os.environ.get("FOOTNOTE_DEMO") == "1" or not os.environ.get("ANTHROPIC_API_KEY")
EMBED_MODEL = os.environ.get("FOOTNOTE_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
INDEX_PATH = Path(os.environ.get("FOOTNOTE_INDEX", ROOT / "data" / "index.db"))
SOURCE_URL_BASE = os.environ.get(
    "FOOTNOTE_SOURCE_URL_BASE", "https://github.com/18F/handbook/blob/main/pages/"
)

TOP_K = 8
SEARCH_K = 5
PAGE_LIMIT = 12
MAX_STEPS = 6
CANDIDATES = 50
# Tuned on evals/golden.jsonl: keyword search is the weaker signal on this corpus, so it gets less say.
RRF_K = 10
DENSE_WEIGHT = 0.7

# USD per million tokens (input, output)
PRICES = {
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    if model not in PRICES:
        return None
    price_in, price_out = PRICES[model]
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000
