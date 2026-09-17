"""Markdown in, heading-aligned chunks out."""

import re
from dataclasses import dataclass
from pathlib import Path

MAX_WORDS = 220
MIN_WORDS_BEFORE_SPLIT = 80
MIN_DOC_WORDS = 40

_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.S)
_TITLE = re.compile(r"^title:\s*[\"']?(.+?)[\"']?\s*$", re.M)
_HEADING = re.compile(r"^(#{1,4})\s+(.+?)\s*#*$")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    doc_path: str
    title: str
    heading: str
    text: str

    @property
    def header(self) -> str:
        return f"{self.title} › {self.heading}" if self.heading else self.title

    @property
    def indexed_text(self) -> str:
        return f"{self.header}\n{self.text}"


def clean_markdown(raw: str) -> tuple[str | None, str]:
    """Return (title, body) with front matter, Liquid tags, HTML and link syntax removed."""
    title = None
    match = _FRONT_MATTER.match(raw)
    if match:
        found = _TITLE.search(match.group(1))
        title = found.group(1).strip() if found else None
        raw = raw[match.end():]
    raw = re.sub(r"\{%\s*page\s+[\"']([^\"']+)[\"']\s*%\}", r"\1", raw)
    raw = re.sub(r"\{%.*?%\}", "", raw, flags=re.S)
    raw = re.sub(r"\{\{.*?\}\}", "", raw, flags=re.S)
    raw = re.sub(r"<!--.*?-->", "", raw, flags=re.S)
    raw = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", raw)
    raw = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", raw)
    raw = re.sub(r"<[^>]+>", "", raw)
    raw = re.sub(r"\\([^\w\s])", r"\1", raw)
    raw = re.sub(r"(?<![\w*_])(\*{1,3}|_{1,3})(\S(?:.*?\S)?)\1(?![\w*_])", r"\2", raw)
    raw = re.sub(r"[ \t]+\n", "\n", raw)
    return title, raw.strip()


def _split_long(paragraph: str) -> list[str]:
    if len(paragraph.split()) <= MAX_WORDS:
        return [paragraph]
    pieces, current, count = [], [], 0
    for sentence in _SENTENCE_END.split(paragraph):
        words = len(sentence.split())
        if current and count + words > MAX_WORDS:
            pieces.append(" ".join(current))
            current, count = [], 0
        current.append(sentence)
        count += words
    if current:
        pieces.append(" ".join(current))
    return pieces


def _units(body: str) -> list[tuple[str, str]]:
    """(heading path, paragraph) pairs in document order."""
    stack: list[tuple[int, str]] = []
    units: list[tuple[str, str]] = []
    paragraph: list[str] = []

    def flush():
        text = "\n".join(paragraph).strip()
        paragraph.clear()
        if text:
            path = " › ".join(name for _, name in stack)
            units.extend((path, piece) for piece in _split_long(text))

    in_code = False
    for line in body.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
        heading = None if in_code else _HEADING.match(line)
        if heading:
            flush()
            level = len(heading.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading.group(2).strip()))
        elif not line.strip() and not in_code:
            flush()
        else:
            paragraph.append(line)
    flush()
    return units


def chunk_document(doc_path: str, raw: str) -> list[Chunk]:
    title, body = clean_markdown(raw)
    if len(body.split()) < MIN_DOC_WORDS:
        return []
    title = title or Path(doc_path).stem.replace("-", " ").title()

    chunks: list[Chunk] = []
    parts: list[tuple[str, str]] = []
    count = 0

    def close():
        nonlocal parts, count
        if parts:
            # A chunk can absorb short neighbouring sections; label it with the heading that wrote most of it.
            weight: dict[str, int] = {}
            for path, paragraph in parts:
                weight[path] = weight.get(path, 0) + len(paragraph.split())
            heading = max(weight, key=weight.get)
            chunks.append(Chunk(doc_path, title, heading, "\n\n".join(p for _, p in parts)))
        parts, count = [], 0

    for path, paragraph in _units(body):
        words = len(paragraph.split())
        heading_changed = parts and path != parts[-1][0] and count >= MIN_WORDS_BEFORE_SPLIT
        if parts and (count + words > MAX_WORDS or heading_changed):
            close()
        parts.append((path, paragraph))
        count += words
    close()
    return chunks


def chunk_directory(root: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(root.rglob("*.md")):
        if path.name in {"LICENSE.md", "CONTRIBUTING.md", "README.md"}:
            continue
        rel = path.relative_to(root).as_posix()
        chunks.extend(chunk_document(rel, path.read_text(encoding="utf-8")))
    return chunks
