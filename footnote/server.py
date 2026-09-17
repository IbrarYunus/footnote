import json
from functools import lru_cache

import anthropic
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import config, demo, index
from .answer import stream_answer

app = FastAPI(title="footnote")
WEB = config.ROOT / "web"

EXAMPLES = [
    "How much annual leave do I earn, and can I carry it over?",
    "Can I work from another country for a few weeks?",
    "How do I get reimbursed for a conference?",
    "What is the dress code for the office Christmas party?",
]


@lru_cache(maxsize=1)
def get_index() -> index.Index:
    return index.load()


class Ask(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    mode: str = Field(default="hybrid", pattern="^(hybrid|bm25|dense)$")


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


def _api_message(error: anthropic.APIStatusError) -> str:
    body = error.body if isinstance(error.body, dict) else {}
    return body.get("error", {}).get("message") or error.message


@app.get("/")
def home():
    return FileResponse(WEB / "index.html")


@app.get("/api/stats")
def stats():
    idx = get_index()
    return {
        "chunks": len(idx.chunks),
        "documents": len({c.doc_path for c in idx.chunks}),
        "embed_model": idx.embed_model,
        "model": demo.MODEL_NAME if config.DEMO else config.MODEL,
        "demo": config.DEMO,
        "examples": demo.EXAMPLES if config.DEMO else EXAMPLES,
    }


@app.post("/api/ask")
def ask(body: Ask):
    def events():
        try:
            for event in stream_answer(body.question, get_index(), body.mode):
                yield _sse(event)
        except anthropic.AuthenticationError:
            yield _sse({"type": "error", "message": "ANTHROPIC_API_KEY is missing or invalid."})
        except anthropic.RateLimitError:
            yield _sse({"type": "error", "message": "Rate limited by the Claude API. Try again shortly."})
        except anthropic.APIStatusError as error:
            yield _sse({"type": "error", "message": f"Claude API error {error.status_code}: {_api_message(error)}"})
        except anthropic.APIConnectionError:
            yield _sse({"type": "error", "message": "Could not reach the Claude API."})

    return StreamingResponse(events(), media_type="text/event-stream")
