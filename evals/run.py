"""Score retrieval (free, local) and answers (uses the API).

Usage:
  uv run python evals/run.py              # retrieval only
  uv run python evals/run.py --answers    # also generate and grade answers
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from footnote import config, index
from footnote.answer import answer, client

HERE = Path(__file__).parent
KS = (1, 3, 8)

JUDGE = """Grade an answer produced by a document question-answering system.

Question: {question}
Reference answer (written from the source passage): {reference}
System answer: {answer}

Pick the first verdict that fits:
abstained: it says the documents do not contain the answer to this question. It may still mention related material it did find.
correct: it states the same facts as the reference. Extra correct detail is fine.
partial: it gets part of the reference right and misses or blurs the rest.
wrong: it contradicts the reference, or confidently answers something else."""


class Verdict(BaseModel):
    verdict: Literal["correct", "partial", "wrong", "abstained"]
    reason: str


def read_jsonl(name: str) -> list[dict]:
    return [json.loads(line) for line in (HERE / name).read_text().splitlines() if line.strip()]


def gold_id(idx: index.Index, item: dict) -> int:
    for i, chunk in enumerate(idx.chunks):
        if chunk.doc_path == item["doc_path"] and chunk.text.startswith(item["text_prefix"]):
            return i
    raise LookupError(f"Gold passage not in index: {item['doc_path']}. Rebuild golden.jsonl after re-chunking.")


def score_retrieval(idx: index.Index, golden: list[dict]) -> dict:
    results = {}
    for mode in ("bm25", "dense", "hybrid"):
        hits_at = {k: 0 for k in KS}
        page_hits = 0
        reciprocal = 0.0
        for item in golden:
            gold = gold_id(idx, item)
            hits = idx.search(item["question"], k=max(KS), mode=mode)
            ids = [hit.id for hit in hits]
            for k in KS:
                hits_at[k] += gold in ids[:k]
            page_hits += any(hit.chunk.doc_path == item["doc_path"] for hit in hits)
            reciprocal += 1 / (ids.index(gold) + 1) if gold in ids else 0
        n = len(golden)
        results[mode] = {
            **{f"hit@{k}": round(hits_at[k] / n, 3) for k in KS},
            "page_hit@8": round(page_hits / n, 3),
            "mrr": round(reciprocal / n, 3),
        }
    return results


def judge(question: str, reference: str, text: str) -> Verdict:
    response = client().messages.parse(
        model=config.JUDGE_MODEL,
        max_tokens=1000,
        messages=[{"role": "user", "content": JUDGE.format(question=question, reference=reference, answer=text)}],
        output_format=Verdict,
    )
    return response.parsed_output


def run_one(idx: index.Index, item: dict) -> dict:
    hits = idx.search(item["question"])
    result = answer(item["question"], hits)
    answerable = "reference_answer" in item
    reference = item.get("reference_answer", "(none: the documents do not cover this, so the only right response is to abstain)")
    verdict = judge(item["question"], reference, result["text"])
    cited_chars = sum(len(b["text"]) for b in result["blocks"] if b["citations"])
    cited_docs = {hits[c["doc"]].id for b in result["blocks"] for c in b["citations"]}
    return {
        "question": item["question"],
        "answerable": answerable,
        "answer": result["text"],
        "verdict": verdict.verdict,
        "reason": verdict.reason,
        "cited_share": round(cited_chars / max(len(result["text"]), 1), 3),
        "gold_retrieved": answerable and gold_id(idx, item) in {hit.id for hit in hits},
        "cited_gold": answerable and gold_id(idx, item) in cited_docs,
        "citations": sum(len(b["citations"]) for b in result["blocks"]),
        "cost_usd": result["cost_usd"],
        "total_ms": result["total_ms"],
    }


def score_answers(idx: index.Index, golden: list[dict], unanswerable: list[dict]) -> tuple[dict, list[dict]]:
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(lambda item: run_one(idx, item), golden + unanswerable))
    yes = [r for r in rows if r["answerable"]]
    no = [r for r in rows if not r["answerable"]]
    found = [r for r in yes if r["gold_retrieved"]]
    missed = [r for r in yes if not r["gold_retrieved"]]
    latencies = sorted(r["total_ms"] for r in rows)
    summary = {
        "answerable": {
            "n": len(yes),
            "correct": round(sum(r["verdict"] == "correct" for r in yes) / len(yes), 3),
            "partial": round(sum(r["verdict"] == "partial" for r in yes) / len(yes), 3),
            "wrong": round(sum(r["verdict"] == "wrong" for r in yes) / len(yes), 3),
            "abstained": round(sum(r["verdict"] == "abstained" for r in yes) / len(yes), 3),
            "correct_when_the_passage_was_retrieved": round(sum(r["verdict"] == "correct" for r in found) / max(len(found), 1), 3),
            "wrong_when_the_passage_was_missed": round(sum(r["verdict"] == "wrong" for r in missed) / max(len(missed), 1), 3),
            "abstained_when_the_passage_was_missed": round(sum(r["verdict"] == "abstained" for r in missed) / max(len(missed), 1), 3),
            "cited_the_gold_passage": round(sum(r["cited_gold"] for r in yes) / len(yes), 3),
            "share_of_answer_text_with_a_citation": round(sum(r["cited_share"] for r in yes) / len(yes), 3),
        },
        "unanswerable": {
            "n": len(no),
            "abstained": round(sum(r["verdict"] == "abstained" for r in no) / len(no), 3),
        },
        "median_latency_s": round(latencies[len(latencies) // 2] / 1000, 1),
        "mean_cost_usd": round(sum(r["cost_usd"] or 0 for r in rows) / len(rows), 4),
        "model": config.MODEL,
        "effort": config.EFFORT,
        "judge": config.JUDGE_MODEL,
    }
    return summary, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", action="store_true")
    args = parser.parse_args()

    idx = index.load()
    golden = read_jsonl("golden.jsonl")
    out_path = HERE / "results.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    results["corpus"] = {"chunks": len(idx.chunks), "documents": len({c.doc_path for c in idx.chunks}), "questions": len(golden)}
    results["retrieval"] = score_retrieval(idx, golden)
    print(json.dumps(results["retrieval"], indent=2))

    if args.answers:
        summary, rows = score_answers(idx, golden, read_jsonl("unanswerable.jsonl"))
        results["answers"] = summary
        (HERE / "answer_rows.json").write_text(json.dumps(rows, indent=2))
        print(json.dumps(summary, indent=2))

    out_path.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
