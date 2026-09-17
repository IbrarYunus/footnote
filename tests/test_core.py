from footnote.index import BM25, reciprocal_rank_fusion, tokenize
from footnote.ingest import MAX_WORDS, chunk_document, clean_markdown

DOC = """---
title: Leave types
keywords: vacation
---

{% alert %}See the <a href="https://example.com">chart</a>.{% endalert %}

Before you take leave, check [out of office expectations]({% page "travel/ooo/" %}).

## Annual leave

""" + ("You earn annual leave every pay period. " * 30) + """

## Sick leave

""" + ("Sick leave covers medical appointments. " * 30)


def test_clean_strips_front_matter_liquid_html_and_links():
    title, body = clean_markdown(DOC)
    assert title == "Leave types"
    assert "{%" not in body and "<a" not in body and "](" not in body
    assert "out of office expectations" in body


def test_chunks_follow_headings_and_respect_the_word_cap():
    chunks = chunk_document("leave.md", DOC)
    headings = [c.heading for c in chunks]
    assert "Annual leave" in headings and "Sick leave" in headings
    assert all(len(c.text.split()) <= MAX_WORDS for c in chunks)
    sick = next(c for c in chunks if c.heading == "Sick leave")
    assert "annual leave" not in sick.text.lower()
    assert sick.indexed_text.startswith("Leave types › Sick leave")


def test_short_documents_are_skipped():
    assert chunk_document("stub.md", "---\ntitle: Stub\n---\nMoved.") == []


def test_bm25_prefers_the_document_with_the_rare_term():
    docs = [tokenize(t) for t in ["travel card rules", "leave and travel", "parental leave policy", "leave balance"]]
    scores = BM25(docs).scores("parental leave")
    assert scores.argmax() == 2
    assert scores[0] == 0


def test_rrf_rewards_agreement_between_rankers():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 2, 9]])
    ids = [doc_id for doc_id, _ in fused]
    assert ids[0] in (2, 3) and ids[-1] in (1, 9)
    assert set(ids) == {1, 2, 3, 9}
