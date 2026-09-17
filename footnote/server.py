import json
import time
from functools import lru_cache

import anthropic
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import config, index
from .answer import stream_answer

app = FastAPI(title="footnote")
WEB = config.ROOT / "web"


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
        "model": config.MODEL,
    }


@app.post("/api/ask")
def ask(body: Ask):
    def events():
        started = time.perf_counter()
        hits = get_index().search(body.question, mode=body.mode)
        yield _sse({
            "type": "retrieval",
            "ms": round((time.perf_counter() - started) * 1000),
            "hits": [hit.to_dict() for hit in hits],
        })
        try:
            for event in stream_answer(body.question, hits):
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
