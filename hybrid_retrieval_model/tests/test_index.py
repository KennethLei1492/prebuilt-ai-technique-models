"""Documents, chunking strategies, metadata filters and store persistence."""
from __future__ import annotations

import pytest

from pipeline_core import Evidence, estimate_tokens

from hybrid_retrieval.index import Chunk, Chunker, Document, DocumentStore, match_filter, split_sentences

MARKDOWN = (
    "# Pump specification PS-4471\n\n"
    "## 4.1 Casing\nThe casing shall be carbon steel. Corrosion allowance 3 mm.\n\n"
    "## 4.2 Impeller\nThe impeller shall be duplex steel.\n\n"
    "```python\nclient = Client(max_attempts=5)\nclient.getUserName()\n```\n\n"
    "1. First clause about testing.\n\n"
    "2. Second clause about inspection.\n"
)


def test_document_coerce_forms() -> None:
    assert Document.coerce("plain text", index=2).id == "doc-3"
    d = Document.coerce({"id": "x", "content": "body", "type": "spec", "metadata": {"rev": "A"}})
    assert d.text == "body" and d.metadata == {"rev": "A", "type": "spec"}
    ev = Evidence(id="e1", content="evidence body", source="src.pdf", metadata={"k": 1})
    d2 = Document.coerce(ev)
    assert d2.id == "e1" and d2.metadata["source"] == "src.pdf" and d2.title == "src.pdf"
    assert Document.coerce(d2) is d2
    with pytest.raises(ValueError):
        Document.coerce({"id": "no-text"})
    with pytest.raises(TypeError):
        Document.coerce(42)


def test_split_sentences() -> None:
    assert split_sentences("First sentence. Second one! Third?\nFourth line") == ["First sentence.", "Second one!", "Third?", "Fourth line"]


def test_fixed_chunker_windows_and_overlap() -> None:
    text = "abcdefghij" * 40  # 400 chars
    ch = Chunker("fixed", size_tokens=25, overlap_tokens=5)  # 100 chars, 20 overlap
    pieces = [c.text for c in ch.chunk(Document(id="d", text=text))]
    assert len(pieces) == 5 and all(len(p) <= 100 for p in pieces)  # starts at 0, 80, 160, 240, 320
    assert pieces[0][-20:] == pieces[1][:20]
    with pytest.raises(ValueError):
        Chunker("fixed", size_tokens=10, overlap_tokens=10)
    with pytest.raises(ValueError):
        Chunker("semantic").chunk(Document(id="d", text="x"))


def test_sentence_and_token_budget_respect_the_budget() -> None:
    text = " ".join(f"Sentence number {i} talks about bearings and seals." for i in range(40))
    for strategy in ("sentence", "token_budget"):
        chunks = Chunker(strategy, size_tokens=40).chunk(Document(id="d", text=text))
        assert len(chunks) > 3
        assert all(estimate_tokens(c.text) <= 40 for c in chunks)
        assert all(c.text.strip() for c in chunks)
        assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_structure_chunker_records_section_paths_and_keeps_code_fences() -> None:
    chunks = Chunker("structure", size_tokens=60).chunk(Document(id="spec", text=MARKDOWN, metadata={"title": "PS-4471"}))
    paths = [c.section_path for c in chunks]
    assert ["Pump specification PS-4471", "4.1 Casing"] in paths
    assert ["Pump specification PS-4471", "4.2 Impeller"] in paths
    code = [c for c in chunks if "```python" in c.text]
    assert len(code) == 1 and "getUserName" in code[0].text and code[0].text.strip().endswith("```")
    clauses = [c for c in chunks if c.text.startswith("1 First") or c.text.startswith("2 Second")]
    assert len(clauses) == 2
    casing = next(c for c in chunks if "4.1 Casing" in c.section_path)
    assert casing.header() == "PS-4471 > Pump specification PS-4471 > 4.1 Casing"
    assert casing.index_text().startswith("PS-4471 > ") and casing.index_text(prepend_header=False) == casing.text
    assert casing.span[0] >= 0 and MARKDOWN[casing.span[0] : casing.span[1]].startswith("The casing")


def test_chunk_to_evidence_carries_provenance() -> None:
    c = Chunk(id="d#0", doc_id="d", text="body", metadata={"title": "T", "type": "spec"}, span=(0, 4), ordinal=0, section_path=["S"])
    ev = c.to_evidence(score=0.5, rank=1, retriever="fused")
    assert ev.id == "d#0" and ev.source == "T" and ev.score == 0.5 and ev.rank == 1 and ev.retriever == "fused"
    assert ev.metadata["doc_id"] == "d" and ev.metadata["section_path"] == ["S"] and ev.span == (0, 4)


@pytest.mark.parametrize(
    "filters, expected",
    [
        (None, True),
        ({"type": "spec"}, True),
        ({"type": "incident"}, False),
        ({"type": ["spec", "incident"]}, True),
        ({"type": {"$in": ["spec", "memo"]}}, True),
        ({"type": {"$nin": ["spec"]}}, False),
        ({"type": {"$ne": "spec"}}, False),
        ({"severity": {"$gte": 2}}, True),
        ({"severity": {"$lt": 2}}, False),
        ({"title": {"$contains": "pump"}}, True),
        ({"tags": "steel"}, True),
        ({"tags": {"$contains": "STEEL"}}, True),
        ({"missing": "x"}, False),
        ({"missing": {"$exists": False}}, True),
        ({"severity": {"$exists": True}}, True),
        ({"severity": {"$gte": "not-a-number"}}, False),
    ],
)
def test_match_filter(filters, expected) -> None:
    meta = {"type": "spec", "severity": 2, "title": "Pump spec", "tags": ["steel", "casing"]}
    assert match_filter(meta, filters) is expected


def test_document_store_crud_filter_fields_and_persistence(tmp_path) -> None:
    store = DocumentStore(Chunker("structure", 60))
    chunks = store.add([{"id": "spec", "text": MARKDOWN, "metadata": {"type": "spec", "severity": 2}}, "a plain string document"])
    assert len(chunks) == len(store) and store.version == 1
    assert "doc-2" in store.documents and store.get(chunks[0].id) is chunks[0] and chunks[0].id in store
    spec_ids = store.filter({"type": "spec"})
    assert spec_ids and all(cid.startswith("spec#") for cid in spec_ids)
    assert store.filter({"type": "nope"}) == set() and store.filter(None) is None
    fields = store.metadata_fields()
    assert fields["type"] == ["spec"] and fields["severity"] == ["2"] and "title" not in fields
    assert store.total_tokens() > 0
    # replace by id drops old chunks
    n_before = len(store)
    store.add([{"id": "spec", "text": "short replacement"}])
    assert len(store) < n_before and store.version == 3
    removed = store.remove(["doc-2"])
    assert removed and "doc-2" not in store.documents
    # persistence round trip
    path = store.save(tmp_path / "idx" / "store.json")
    loaded = DocumentStore.load(path)
    assert loaded.to_dict() == store.to_dict()
    assert loaded.chunker.size_tokens == 60 and loaded.chunker.strategy == "structure"
