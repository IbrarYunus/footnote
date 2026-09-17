"""Turn retrieved chunks into a cited answer with Claude's native citations."""

import time
from collections.abc import Iterator

import anthropic

from . import config
from .index import Hit

SYSTEM = """You answer staff questions using only the handbook excerpts supplied as documents.

Ground every factual statement in the excerpts and cite it. The excerpts come from a search step and some will be irrelevant; ignore those. If the excerpts do not contain the answer, say so plainly and name what is missing. Never fill a gap from general knowledge, because readers act on these answers and cannot tell a guess from policy.

Answer first, then the details that matter. Plain prose or a short list. No headings, no preamble."""

_client: anthropic.Anthropic | None = None


def client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def build_messages(question: str, hits: list[Hit]) -> list[dict]:
    documents = [
        {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": hit.chunk.text},
            "title": hit.chunk.header,
            "citations": {"enabled": True},
        }
        for hit in hits
    ]
    return [{"role": "user", "content": [*documents, {"type": "text", "text": question}]}]


def _citation_event(block: int, citation) -> dict:
    return {
        "type": "cite",
        "block": block,
        "doc": citation.document_index,
        "start": getattr(citation, "start_char_index", None),
        "end": getattr(citation, "end_char_index", None),
        "cited_text": citation.cited_text,
    }


def stream_answer(question: str, hits: list[Hit], model: str | None = None) -> Iterator[dict]:
    """Yield text, cite and done events. Block numbers group text with its citations."""
    model = model or config.MODEL
    started = time.perf_counter()
    first_token_ms = None
    with client().messages.stream(
        model=model,
        max_tokens=4000,
        system=SYSTEM,
        messages=build_messages(question, hits),
        output_config={"effort": config.EFFORT},
        extra_headers={"anthropic-beta": "server-side-fallback-2026-07-01"},
        extra_body={"fallbacks": "default"},
    ) as stream:
        for event in stream:
            if event.type != "content_block_delta":
                continue
            if event.delta.type == "text_delta":
                if first_token_ms is None:
                    first_token_ms = round((time.perf_counter() - started) * 1000)
                yield {"type": "text", "block": event.index, "text": event.delta.text}
            elif event.delta.type == "citations_delta":
                yield _citation_event(event.index, event.delta.citation)
        final = stream.get_final_message()

    usage = final.usage
    yield {
        "type": "done",
        "model": final.model,
        "stop_reason": final.stop_reason,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cost_usd": config.cost_usd(final.model, usage.input_tokens, usage.output_tokens),
        "first_token_ms": first_token_ms,
        "total_ms": round((time.perf_counter() - started) * 1000),
    }


def answer(question: str, hits: list[Hit], model: str | None = None) -> dict:
    """Blocking variant used by the CLI and the evals."""
    blocks: dict[int, dict] = {}
    done: dict = {}
    for event in stream_answer(question, hits, model):
        if event["type"] == "done":
            done = event
            continue
        block = blocks.setdefault(event["block"], {"text": "", "citations": []})
        if event["type"] == "text":
            block["text"] += event["text"]
        else:
            block["citations"].append(event)
    ordered = [blocks[i] for i in sorted(blocks)]
    return {"text": "".join(b["text"] for b in ordered), "blocks": ordered, **done}
