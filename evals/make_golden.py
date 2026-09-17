"""Build a synthetic question set: sample passages, have Claude write the question a colleague would ask.

Usage: uv run python evals/make_golden.py --n 60
"""

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel

from footnote import config, index
from footnote.answer import client

OUT = Path(__file__).parent / "golden.jsonl"

PROMPT = """Below is one passage from an internal staff handbook.

Write the question a staff member would type into a search box if this passage were what they needed. They have never seen the passage, so use the words a person with the problem would use, not the passage's own phrasing, and do not mention "the passage" or "the handbook". The question must have one clear answer that the passage alone supplies. Then write that answer in one or two sentences.

Set usable to false if the passage is mostly links, a table of contents, a form template, or too dependent on surrounding pages to stand alone.

<passage title="{header}">
{text}
</passage>"""


class Item(BaseModel):
    usable: bool
    question: str
    reference_answer: str


def generate(chunk) -> dict | None:
    response = client().messages.parse(
        model=config.MODEL,
        max_tokens=2000,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": PROMPT.format(header=chunk.header, text=chunk.text)}],
        output_format=Item,
    )
    item = response.parsed_output
    if not item or not item.usable:
        return None
    return {
        "question": item.question,
        "reference_answer": item.reference_answer,
        "doc_path": chunk.doc_path,
        "heading": chunk.heading,
        "text_prefix": chunk.text[:120],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    chunks = [c for c in index.load().chunks if len(c.text.split()) >= 80]
    random.Random(args.seed).shuffle(chunks)
    sample, seen = [], set()
    for chunk in chunks:
        if chunk.doc_path not in seen:
            seen.add(chunk.doc_path)
            sample.append(chunk)
        if len(sample) == args.n:
            break

    with ThreadPoolExecutor(max_workers=6) as pool:
        items = [item for item in pool.map(generate, sample) if item]
    OUT.write_text("".join(json.dumps(item) + "\n" for item in items))
    print(f"{len(items)} usable questions from {len(sample)} sampled passages -> {OUT}")


if __name__ == "__main__":
    main()
