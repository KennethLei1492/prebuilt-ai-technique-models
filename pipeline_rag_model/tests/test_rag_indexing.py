"""Built-in chunker, contextual prefixes, BM25 index and hierarchy nodes."""
from __future__ import annotations

from pipeline_core import Evidence, estimate_tokens

from pipeline_rag import BM25Index, IndexBuilder, PipelineRAGConfig, SourceDocument, chunk_document, normalize_documents
from pipeline_rag.indexing import heuristic_prefix, lead_summary


def test_normalize_documents_accepts_many_shapes() -> None:
    docs = normalize_documents(["plain text one", {"id": "x", "content": "dict content", "title": "T"}, Evidence(id="e", content="evidence text", source="src"), {"id": "x", "text": "dup id"}, ""])
    ids = [d.id for d in docs]
    assert ids == ["doc-1", "x", "e", "x-1"]
    assert docs[1].source == "T" and docs[2].source == "src"
    mapping = normalize_documents({"k1": "v1", "k2": {"text": "v2"}})
    assert [d.id for d in mapping] == ["k1", "k2"]
    assert normalize_documents(None) == []


def test_chunk_document_windows_overlap_and_spans() -> None:
    text = " ".join(f"Sentence {i} talks about the casing and its corrosion allowance in detail." for i in range(60))
    doc = SourceDocument(id="d", text=text)
    chunks = chunk_document(doc, chunk_size_tokens=100, overlap_tokens=20)
    assert len(chunks) > 3
    for ch in chunks:
        assert estimate_tokens(ch.text) <= 100 + 20  # a single sentence may slightly exceed
        s, e = ch.span
        assert text[s:e].strip() == ch.text
    # consecutive chunks overlap
    assert chunks[1].span[0] < chunks[0].span[1]
    assert chunks[0].id == "d#1" and chunks[-1].doc_id == "d"
    small = chunk_document(SourceDocument(id="s", text="Short text."), 512, 64)
    assert len(small) == 1 and small[0].id == "s"


def test_bm25_ranks_exact_identifier_matches_first() -> None:
    builder = IndexBuilder(PipelineRAGConfig(chunk_size_tokens=64, chunk_overlap_tokens=8))
    retr = builder.build(
        [
            {"id": "req", "source": "Requirements", "text": "REQ-CW-020 Seawater-wetted pressure parts shall have a minimum 3 mm corrosion allowance."},
            {"id": "motor", "source": "Motor", "text": "The drive motor is rated 75 kW at 1480 rpm with service factor 1.15."},
            {"id": "test", "source": "Test", "text": "Hydrostatic test held 24 bar for 30 minutes without leakage."},
        ]
    )
    hits = retr.retrieve_sync("REQ-CW-020 corrosion allowance", k=3)
    assert hits[0].id == "req" and hits[0].retriever == "builtin_bm25" and hits[0].rank == 1
    assert hits[0].metadata["prefix"].startswith("From Requirements")
    assert retr.retrieve_sync("zzzz qqqq", k=3) == []
    assert len(retr) == 3 and retr.total_tokens() > 0
    v1 = retr.version
    builder.add(retr, ["Another document about pumps."])
    assert retr.version != v1 and len(retr) == 4


def test_bm25_filters_and_idf() -> None:
    idx = BM25Index()
    from pipeline_rag.indexing import Chunk

    idx.add([Chunk(id="a", doc_id="a", text="pump casing corrosion", metadata={"type": "spec"}), Chunk(id="b", doc_id="b", text="pump motor rating", metadata={"type": "datasheet"})])
    assert idx.idf("pump") < idx.idf("casing")  # common term has lower idf
    assert [c for c, _ in idx.search("pump", k=5, filters={"type": "spec"})] == ["a"]
    assert [c for c, _ in idx.search("pump", k=5, filters={"type": ["spec", "datasheet"]})] == ["a", "b"] or len(idx.search("pump", k=5, filters={"type": ["spec", "datasheet"]})) == 2


def test_heuristic_prefix_and_lead_summary() -> None:
    doc = SourceDocument(id="d", text="Casing design pressure is 16 bar. More text follows here. Third sentence.", source="Spec PS-100")
    p = heuristic_prefix(doc, 2, 3)
    assert p.startswith("From Spec PS-100 (part 2 of 3)") and "16 bar" in p
    s = lead_summary("Spec", "One. Two. Three. Four. Five has 42 units. Six.", sentences=2)
    assert s.startswith("Spec: One. Two.") and "42 units" in s


async def test_build_hierarchy_adds_summary_nodes_and_uses_summarizer() -> None:
    builder = IndexBuilder(PipelineRAGConfig(chunk_size_tokens=64, chunk_overlap_tokens=8))
    retr = builder.build(
        [
            {"id": "inc1", "source": "Incident 1", "text": "Pump tripped on vibration. Root cause misalignment 0.35 mm. Corrective action realignment."},
            {"id": "inc2", "source": "Incident 2", "text": "Strainer fouling caused high differential pressure. Root cause marine growth. Cleaning interval reduced."},
        ]
    )
    calls: list[str] = []

    async def summarizer(title: str, text: str) -> str:
        calls.append(title)
        return f"SUMMARY of {title}: root cause and corrective action."

    nodes = await builder.build_hierarchy(retr, summarizer)
    ids = sorted(n.id for n in nodes)
    assert ids == ["corpus#summary", "inc1#summary", "inc2#summary"]
    assert all(n.level >= 1 for n in nodes) and len(calls) == 3
    hits = retr.retrieve_sync("SUMMARY root cause corrective action", k=5)
    assert any(h.metadata["level"] >= 1 for h in hits)
    again = await builder.build_hierarchy(retr, summarizer)
    assert again == []  # idempotent

    async def failing(title: str, text: str) -> str:
        raise RuntimeError("boom")

    retr2 = builder.build([{"id": "x", "text": "Only one document with a sentence. And a second sentence with 7 units."}])
    nodes2 = await builder.build_hierarchy(retr2, failing)
    assert len(nodes2) == 1 and nodes2[0].text.startswith("x:")  # lead-sentence fallback, no corpus node for one doc
