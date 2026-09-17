"""The research agent. Claude decides what to search for, reads the results, searches again if they are
thin, and answers with citations that point into the passages its own tool calls returned."""

import re
import time
from collections.abc import Iterator

import anthropic

from . import config
from .index import Hit, Index

SYSTEM = """You answer staff questions from the organisation's handbook. You cannot see the handbook; you reach it through your tools, and you may use only what they return.

Work like a careful researcher. Search with the words the handbook would use, not only the asker's words. A question with several parts usually needs a search per part. If results look thin or off-topic, rephrase and search again; if a passage is clearly cut off mid-topic, read its page. Stop searching once you have the evidence. If two or three honest attempts find nothing, the handbook does not cover it: say so plainly and name what is missing. Never fill a gap from general knowledge, because readers act on these answers and cannot tell a guess from policy.

Your final message is the answer: lead with it, then the details that matter, in plain prose or a short list with no headings. Do not describe your searching in it."""

TOOLS = [
    {
        "name": "search_handbook",
        "description": "Search the handbook. Returns the best-matching passages that you have not already been shown. Call it several times with different wording or for different parts of a question.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "A short search query in the handbook's likely wording"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "read_page",
        "description": "Read a whole handbook page when a passage you found is cut off or you need what surrounds it. Use the page path shown as the passage's source.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"page": {"type": "string", "description": "Page path, for example travel-and-leave/leave.md"}},
            "required": ["page"],
            "additionalProperties": False,
        },
    },
]

# Sentence ends, blank lines and list items. A single newline inside a paragraph is only a line wrap.
_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}|\n(?=\s*(?:[-*|]|\d+\.)\s)")
_client: anthropic.Anthropic | None = None


def client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        if config.DEMO:
            from .demo import DemoClient

            _client = DemoClient()
        else:
            _client = anthropic.Anthropic()
    return _client


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Character ranges of each sentence. Each becomes one citable block, so citations are sentence-level."""
    spans, cursor = [], 0
    for piece in _SPLIT.split(text):
        if piece.strip():
            start = text.index(piece, cursor)
            cursor = start + len(piece)
            spans.append((start, cursor))
    return spans


class Research:
    """One question's working state: every passage shown to the model, in the order the API will index them."""

    def __init__(self, index: Index, mode: str = "hybrid"):
        self.index, self.mode = index, mode
        self.passages: list[Hit] = []
        self.spans: list[list[tuple[int, int]]] = []
        self.seen: set[int] = set()

    def _blocks(self, hits: list[Hit]) -> list[dict]:
        blocks = []
        for hit in hits:
            if hit.id in self.seen:
                continue
            self.seen.add(hit.id)
            spans = sentence_spans(hit.chunk.text)
            self.passages.append(hit)
            self.spans.append(spans)
            blocks.append({
                "type": "search_result",
                "source": hit.chunk.doc_path,
                "title": hit.chunk.header,
                "content": [{"type": "text", "text": hit.chunk.text[a:b]} for a, b in spans],
                "citations": {"enabled": True},
            })
        return blocks

    def run(self, name: str, tool_input: dict) -> tuple[list[dict] | str, list[Hit]]:
        if name == "search_handbook":
            hits = self.index.search(tool_input["query"], k=config.SEARCH_K, mode=self.mode)
        elif name == "read_page":
            page = tool_input["page"].strip().lstrip("/")
            hits = [Hit(i, c, 0.0) for i, c in enumerate(self.index.chunks) if c.doc_path == page][: config.PAGE_LIMIT]
            if not hits:
                return f"No page at '{page}'. Use the source path of a passage you were shown.", []
        else:
            return f"Unknown tool {name}.", []
        new = [hit for hit in hits if hit.id not in self.seen]
        blocks = self._blocks(hits)
        return (blocks or "Nothing new: every match for that was already shown to you."), new

    def citation_event(self, block: int, citation) -> dict:
        passage = citation.search_result_index
        spans = self.spans[passage]
        return {
            "type": "cite",
            "block": block,
            "doc": passage,
            "start": spans[citation.start_block_index][0],
            "end": spans[citation.end_block_index - 1][1],
            "cited_text": citation.cited_text,
        }


def stream_answer(question: str, index: Index, mode: str = "hybrid", model: str | None = None) -> Iterator[dict]:
    """Yield lookup, text, cite, interim and done events. A step's text is 'interim' if the step went on to call a tool."""
    model = model or config.MODEL
    research = Research(index, mode)
    messages: list[dict] = [{"role": "user", "content": question}]
    started = time.perf_counter()
    first_token_ms = None
    usage = {"input_tokens": 0, "output_tokens": 0}
    final, step = None, 0

    for step in range(config.MAX_STEPS):
        with client().messages.stream(
            model=model,
            max_tokens=4000,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
            output_config={"effort": config.EFFORT},
            extra_headers={"anthropic-beta": "server-side-fallback-2026-07-01"},
            extra_body={"fallbacks": "default"},
        ) as stream:
            for event in stream:
                if event.type != "content_block_delta":
                    continue
                block = step * 1000 + event.index
                if event.delta.type == "text_delta":
                    if first_token_ms is None:
                        first_token_ms = round((time.perf_counter() - started) * 1000)
                    yield {"type": "text", "block": block, "step": step, "text": event.delta.text}
                elif event.delta.type == "citations_delta" and event.delta.citation.type == "search_result_location":
                    yield research.citation_event(block, event.delta.citation)
            final = stream.get_final_message()

        usage["input_tokens"] += final.usage.input_tokens
        usage["output_tokens"] += final.usage.output_tokens
        if final.stop_reason != "tool_use":
            break

        yield {"type": "interim", "step": step}
        messages.append({"role": "assistant", "content": final.to_dict()["content"]})
        results = []
        for block in final.content:
            if block.type != "tool_use":
                continue
            lookup_started = time.perf_counter()
            content, new_hits = research.run(block.name, block.input)
            first = len(research.passages) - len(new_hits)
            yield {
                "type": "lookup",
                "step": step,
                "tool": block.name,
                "input": block.input,
                "ms": round((time.perf_counter() - lookup_started) * 1000),
                "hits": [{**hit.to_dict(), "doc": first + i} for i, hit in enumerate(new_hits)],
            }
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": content})
        messages.append({"role": "user", "content": results})

    yield {
        "type": "done",
        "model": final.model,
        "stop_reason": final.stop_reason,
        "steps": step + 1,
        **usage,
        "cost_usd": config.cost_usd(final.model, usage["input_tokens"], usage["output_tokens"]),
        "first_token_ms": first_token_ms,
        "total_ms": round((time.perf_counter() - started) * 1000),
    }


def answer(question: str, index: Index, mode: str = "hybrid", model: str | None = None) -> dict:
    """Blocking variant used by the CLI and the evals."""
    blocks: dict[int, dict] = {}
    lookups, passages, done = [], [], {}
    for event in stream_answer(question, index, mode, model):
        if event["type"] == "done":
            done = event
        elif event["type"] == "lookup":
            lookups.append({"tool": event["tool"], "input": event["input"], "new": len(event["hits"])})
            passages.extend(event["hits"])
        elif event["type"] == "interim":
            blocks = {key: value for key, value in blocks.items() if key // 1000 != event["step"]}
        else:
            block = blocks.setdefault(event["block"], {"text": "", "citations": []})
            if event["type"] == "text":
                block["text"] += event["text"]
            else:
                block["citations"].append(event)
    ordered = [blocks[i] for i in sorted(blocks)]
    return {"text": "".join(b["text"] for b in ordered), "blocks": ordered, "lookups": lookups, "passages": passages, **done}
