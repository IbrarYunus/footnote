<h1 align="center">footnote<sup>1</sup></h1>

<p align="center">
  <strong>Agentic RAG: an AI research agent for your documents. It decides what to search for, searches again when the results are thin, and every claim in its answer links to the exact sentence it came from.</strong>
</p>

<p align="center">
  <a href="#what-it-does">What it does</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#does-it-work">Evals</a> ·
  <a href="#run-it">Run it (no API key needed)</a> ·
  <a href="docs/deck.pdf">Slides (PDF)</a>
</p>

<p align="center">
  <img alt="CI" src="https://github.com/IbrarYunus/footnote/actions/workflows/ci.yml/badge.svg">
  <img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-b7791f">
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-1c1b19">
  <img alt="Agentic AI" src="https://img.shields.io/badge/agentic%20AI-tool%20use%20%2B%20RAG-e2a83a">
</p>

<p align="center">
  <img src="docs/screenshots/answer.png" alt="footnote answering a two-part question about annual leave. Right: the agent's two searches and the passages they returned. Left: the answer with numbered citation chips." width="920">
</p>

---

## Why

"Chat with your documents" is the most common thing companies ask an LLM to do. The usual build is a fixed pipeline: search once with the user's words, paste the top results into a prompt, hope. It fails in two ways. A two-part question gets one search and half an answer. And the answer reads well even when one sentence in it came from nowhere.

footnote replaces the fixed pipeline with an **agent**. Claude is given search tools and works like a researcher: it picks the wording, runs a search per part of the question, reads what came back, rephrases and tries again if the results are thin, opens the full page when a passage is cut off, and stops when it has the evidence or is satisfied the documents do not cover it.

It is built around one rule: **an answer you cannot check is not an answer.** The agent may use only what its tools returned, every claim carries a citation validated by the API, one click shows the source sentence highlighted in place, and "the handbook does not cover this" is a valid, expected answer.

The demo corpus is the real [18F / TTS staff handbook](https://github.com/18F/handbook) (232 pages, public domain): leave, travel, expenses, hiring, tools.

## What it does

### Plans its own research

A question with two parts gets two searches, in the handbook's wording rather than the asker's. The right-hand panel shows each step as it happens: the agent's working note, the query it chose, how many new passages came back.

<img src="docs/screenshots/answer.png" alt="Two searches chosen by the agent for a two-part question, and the cited answer" width="920">

### Answers with sentence-level citations, and shows the sentence

Each numbered chip is a citation returned by the Claude API, not text the model typed. Click one and the passage scrolls into view with the exact sentence highlighted. Passages the agent saw but did not use stay visible, dimmed, so you can see what it ignored.

<img src="docs/screenshots/citation.png" alt="A passage card focused, with the cited sentence highlighted in amber" width="920">

### Tries again, then says "not in the documents"

Asked about a Christmas party dress code, the agent searches for a dress code, finds nothing about events, tries the party angle, and then says plainly that the handbook does not cover it. The nearby material it did find is still cited.

<img src="docs/screenshots/abstain.png" alt="footnote searching twice and then declining to answer a question the handbook does not cover" width="920">

### Lets you change the search tool under the agent

The agent's search tool can run as hybrid, keyword-only or vector-only. Every passage shows which search returned it and where keyword and vector search each ranked it.

<img src="docs/screenshots/keyword-only.png" alt="The same agent running on keyword-only search, with rank labels on each passage" width="920">

### Works from the terminal

```text
$ footnote ask "How much annual leave do I earn, and can I carry it over?"
  1. search_handbook 'annual leave accrual rate hours per pay period' -> 5 new passages
  2. search_handbook 'annual leave carry over maximum use or lose' -> 4 new passages

Annual leave builds up with your length of federal service[1]: fewer than 3 years earns 4 hours
per pay period, 3 to 15 years earns 6, and 15 or more earns 8.[1] ...
You can carry over at most 240 hours of annual leave from one leave year to the next.[2]

[1] Leave types › Types of leave › Annual leave  (travel-and-leave/leave.md)
[2] Leave types › Types of leave › Annual leave › Use or Lose  (travel-and-leave/leave.md)
```

> **About the screenshots.** They were taken in **demo mode**, which needs no API key: the model's turns (which searches to run, the wording of the answer) are scripted for three example questions. Everything else is real: the searches run against the real index, and each citation is resolved against the passages those searches actually returned. A quote that was not retrieved gets no citation. The banner in the UI says when demo mode is on. With an `ANTHROPIC_API_KEY`, the same loop runs on Claude.

## How it works

```
 question
    │
    ▼
 answer.py ── agent loop, max 6 model turns (Claude Opus 5, streaming)
    │              │ tool_use
    │              ▼
    │     search_handbook(query) ──▶ index.py: BM25 + vectors ─▶ weighted rank fusion ─▶ top 5 not yet seen
    │     read_page(page)        ──▶ every passage of one page
    │              │
    │              ▼ tool_result = search_result blocks, one text block per sentence, citations on
    │◀─────────────┘
    ▼
 final turn: text blocks + citation spans (search result #, sentence range)
    │
    ▼
 server.py ── Server-Sent Events ─▶ web/index.html (one file, no build step)

 ingest.py ── markdown ─▶ clean ─▶ split on headings, ≤220 words ─▶ "Page › Section" prefix ─▶ SQLite + 384-d vectors
```

Design decisions, and why:

- **An agent, not a pipeline.** One search with the user's words is the ceiling of a fixed pipeline. Letting the model choose queries, split multi-part questions and retry is what gets the second half of the answer. The loop is capped at six model turns and never re-sends a passage the model has already seen.
- **Citations come from the API, not from the prompt.** Tool results are returned as [`search_result` blocks](https://platform.claude.com/docs/en/build-with-claude/search-results) with citations enabled, split into one block per sentence. Claude returns which result and which sentence range each claim rests on, and the API validates it. Asking a model to type `[1]` gives markers that look right and are sometimes wrong.
- **Hybrid search, because each half fails differently.** Vector search misses exact terms (a form number, a tool name). Keyword search misses paraphrase ("time off" vs "annual leave"). Reciprocal rank fusion needs no score calibration between the two.
- **Headings travel with the passage.** "You can carry over 240 hours" is useless to a search engine without "Annual leave" attached. Each passage is indexed as `Page title › Section path` + text.
- **No vector database.** 1,432 passages × 384 floats is about 2 MB. A numpy matrix multiply searches it in well under a millisecond. SQLite holds everything in one file. A vector DB becomes worth its cost somewhere past a few hundred thousand passages.
- **Embeddings run locally.** `bge-small-en-v1.5` through ONNX: no second API key, no per-query cost, documents never leave the machine for indexing.
- **Abstention is a requirement.** The system prompt explains *why* guessing is harmful (readers act on the answer) and tells the agent when to stop looking. The eval set includes questions the corpus cannot answer.

## Does it work?

Numbers below are produced by `evals/run.py` and stored in [`evals/results.json`](evals/results.json). Nothing is hand-entered.

### The search tool

59 questions, one correct passage each, 1,432 passages to choose from. This measures a single search call with the question as the query, which is the floor the agent starts from, not what it achieves with several searches.

| Search mode | Correct passage is #1 | In top 3 | In top 8 | Right page in top 8 | MRR |
|---|---|---|---|---|---|
| Keyword only (BM25) | 28.8% | 55.9% | 64.4% | 76.3% | 0.415 |
| Vector only (bge-small) | 44.1% | 67.8% | 84.7% | 88.1% | 0.581 |
| **Hybrid (default)** | **45.8%** | **74.6%** | **84.7%** | **88.1%** | **0.608** |

Hybrid matches vector search on coverage and ranks the right passage higher (top 3: +6.8 points). The first version, textbook fusion with equal weights, scored *below* vector-only (74.6% in top 8, MRR 0.580): on paraphrased questions keyword search is the weaker signal and was pushing good passages out of the list. Weighting vectors 70/30 fixed it. That regression only showed up because the eval existed.

How the questions were made: `evals/make_golden.py` samples passages from different pages and has Claude write the question a staff member would type, with instructions to avoid the passage's own wording. **Caveats:** synthetic questions are cleaner than real ones; 59 is a small set, so one question is worth 1.7 points; and the fusion weights were tuned on this same set, so the hybrid row is optimistic.

### The agent's answers

`uv run python evals/run.py --answers` runs the full agent on all 59 questions plus 10 that the handbook cannot answer, and has a second model grade each answer against the reference: correct, partial, wrong or abstained. It also records whether the right passage was ever retrieved, whether the answer cited it, how much of the answer text carries a citation, searches per question, latency and cost.

**This suite has not been run on the agentic version yet**, so no answer-quality numbers are claimed here. It needs an API key and costs a few dollars on Claude Opus 5.

## Run it

Needs Python 3.11+ and [uv](https://docs.astral.sh/uv/). An API key is optional.

```bash
git clone https://github.com/IbrarYunus/footnote && cd footnote
uv sync
uv run footnote ingest corpus/  # about 2 minutes on a laptop CPU; downloads a 64 MB embedding model once
uv run footnote serve           # http://127.0.0.1:8000
```

With no key it starts in **demo mode** (scripted model turns for the three example questions, real search and citations). To run the real agent on any question:

```bash
cp .env.example .env            # add ANTHROPIC_API_KEY
uv run footnote serve
uv run footnote ask "Can I work from another country for a few weeks?"
```

```bash
uv run pytest                          # unit tests, no API calls
uv run python evals/run.py             # search evals, free and local
uv run python evals/run.py --answers   # agent evals, needs a key
```

### Use your own documents

Point `ingest` at any folder of Markdown. Set `FOOTNOTE_SOURCE_URL_BASE` so "source" links go to your wiki or repo, then rebuild the question set with `uv run python evals/make_golden.py`.

| Setting | Default | Meaning |
|---|---|---|
| `FOOTNOTE_MODEL` | `claude-opus-5` | Agent model. `claude-sonnet-5` and `claude-haiku-4-5` also work. |
| `FOOTNOTE_EFFORT` | `medium` | How much the model thinks per turn. |
| `FOOTNOTE_DEMO` | unset | `1` forces demo mode even when a key is present. |
| `FOOTNOTE_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Any [fastembed](https://github.com/qdrant/fastembed) text model. |
| `FOOTNOTE_SOURCE_URL_BASE` | 18F handbook on GitHub | Prefix for source links. |

## Project layout

```
footnote/answer.py    the agent: tools, loop, citations       tests/          chunking, BM25, fusion
footnote/index.py     SQLite store, BM25, vectors, fusion     evals/          question set, scorer, results
footnote/ingest.py    clean and chunk markdown                corpus/         18F handbook (CC0)
footnote/demo.py      scripted model for keyless demo mode    web/index.html  the whole front end
footnote/server.py    FastAPI + Server-Sent Events            docs/           slides, screenshots
footnote/cli.py       ingest · ask · serve
```

About 850 lines of Python on the Anthropic SDK. No LangChain, no LlamaIndex: every step is short enough to read.

## What I would build next

- Run and publish the agent eval suite, then compare it against the single-search baseline on the same questions.
- A cross-encoder reranker inside the search tool, kept only if the eval says so.
- Multi-turn conversations: follow-up questions that build on what was already retrieved.
- A harder, human-written eval set, including questions whose answer spans two pages.
- Passage-level access control, so the search tool never returns what the asker may not read.

## Credits

Corpus: the [18F Handbook](https://github.com/18F/handbook), a work of the United States Government, public domain and CC0. It is included under `corpus/` with its licence. 18F is not affiliated with this project.

## Author

**Ibrar Yunus**, AI engineer.

[ibraryunus.com/ai-engineer](https://ibraryunus.com/ai-engineer) · [LinkedIn](https://www.linkedin.com/in/ibrar-yunus/) · [GitHub](https://github.com/IbrarYunus)

More agentic AI work: [frontdesk](https://github.com/IbrarYunus/frontdesk) (a support agent that acts on orders, with the money rules enforced in code and human approval for large refunds) and [groundwork](https://github.com/IbrarYunus/groundwork) (an agent that audits a codebase).

MIT licensed.
