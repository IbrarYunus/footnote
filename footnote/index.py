"""SQLite-backed index with BM25, dense and hybrid (reciprocal rank fusion) search."""

import math
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config
from .ingest import Chunk

_TOKEN = re.compile(r"[a-z0-9]+")

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    doc_path TEXT NOT NULL,
    title TEXT NOT NULL,
    heading TEXT NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25:
    def __init__(self, documents: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.lengths = np.array([len(doc) for doc in documents], dtype=np.float32)
        self.avg_length = float(self.lengths.mean()) if documents else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for doc_id, doc in enumerate(documents):
            for term, freq in Counter(doc).items():
                self.postings.setdefault(term, []).append((doc_id, freq))
        n = len(documents)
        self.idf = {
            term: math.log(1 + (n - len(posting) + 0.5) / (len(posting) + 0.5))
            for term, posting in self.postings.items()
        }

    def scores(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.lengths), dtype=np.float32)
        for term in set(tokenize(query)):
            for doc_id, freq in self.postings.get(term, ()):
                norm = self.k1 * (1 - self.b + self.b * self.lengths[doc_id] / self.avg_length)
                scores[doc_id] += self.idf[term] * freq * (self.k1 + 1) / (freq + norm)
        return scores


def reciprocal_rank_fusion(
    rankings: list[list[int]], weights: list[float] | None = None, k: int = config.RRF_K
) -> list[tuple[int, float]]:
    weights = weights or [1.0] * len(rankings)
    fused: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking):
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (k + rank + 1)
    return sorted(fused.items(), key=lambda item: item[1], reverse=True)


@dataclass
class Hit:
    id: int
    chunk: Chunk
    score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.chunk.title,
            "heading": self.chunk.heading,
            "text": self.chunk.text,
            "doc_path": self.chunk.doc_path,
            "url": config.SOURCE_URL_BASE + self.chunk.doc_path,
            "score": round(self.score, 5),
            "bm25_rank": self.bm25_rank,
            "dense_rank": self.dense_rank,
        }


@dataclass
class Index:
    chunks: list[Chunk]
    embeddings: np.ndarray
    embed_model: str
    bm25: BM25 = field(init=False)
    _embedder: object = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.bm25 = BM25([tokenize(chunk.indexed_text) for chunk in self.chunks])

    @property
    def embedder(self):
        if self._embedder is None:
            from fastembed import TextEmbedding

            self._embedder = TextEmbedding(self.embed_model)
        return self._embedder

    def _bm25_ranking(self, query: str, n: int) -> list[int]:
        scores = self.bm25.scores(query)
        order = np.argsort(-scores)[:n]
        return [int(i) for i in order if scores[i] > 0]

    def _dense_ranking(self, query: str, n: int) -> list[int]:
        vector = next(iter(self.embedder.query_embed(query)))
        vector = vector / np.linalg.norm(vector)
        return [int(i) for i in np.argsort(-(self.embeddings @ vector))[:n]]

    def search(self, query: str, k: int = config.TOP_K, mode: str = "hybrid") -> list[Hit]:
        n = config.CANDIDATES
        bm25 = self._bm25_ranking(query, n) if mode in ("hybrid", "bm25") else []
        dense = self._dense_ranking(query, n) if mode in ("hybrid", "dense") else []
        if mode == "hybrid":
            fused = reciprocal_rank_fusion([bm25, dense], [1 - config.DENSE_WEIGHT, config.DENSE_WEIGHT])
        else:
            ranking = bm25 or dense
            fused = [(doc_id, 1.0 / (rank + 1)) for rank, doc_id in enumerate(ranking)]
        bm25_rank = {doc_id: rank + 1 for rank, doc_id in enumerate(bm25)}
        dense_rank = {doc_id: rank + 1 for rank, doc_id in enumerate(dense)}
        return [
            Hit(doc_id, self.chunks[doc_id], score, bm25_rank.get(doc_id), dense_rank.get(doc_id))
            for doc_id, score in fused[:k]
        ]


def build(chunks: list[Chunk], path: Path = config.INDEX_PATH, embed_model: str = config.EMBED_MODEL) -> None:
    from fastembed import TextEmbedding

    embedder = TextEmbedding(embed_model)
    vectors = np.array(list(embedder.embed([c.indexed_text for c in chunks])), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    with sqlite3.connect(path) as db:
        db.executescript(SCHEMA)
        db.executemany(
            "INSERT INTO chunks (id, doc_path, title, heading, text, embedding) VALUES (?,?,?,?,?,?)",
            [
                (i, c.doc_path, c.title, c.heading, c.text, vectors[i].tobytes())
                for i, c in enumerate(chunks)
            ],
        )
        db.execute("INSERT INTO meta VALUES ('embed_model', ?)", (embed_model,))


def load(path: Path = config.INDEX_PATH) -> Index:
    if not path.exists():
        raise FileNotFoundError(f"No index at {path}. Run: footnote ingest corpus/")
    with sqlite3.connect(path) as db:
        rows = db.execute(
            "SELECT doc_path, title, heading, text, embedding FROM chunks ORDER BY id"
        ).fetchall()
        embed_model = db.execute("SELECT value FROM meta WHERE key='embed_model'").fetchone()[0]
    chunks = [Chunk(*row[:4]) for row in rows]
    embeddings = np.vstack([np.frombuffer(row[4], dtype=np.float32) for row in rows])
    return Index(chunks, embeddings, embed_model)
