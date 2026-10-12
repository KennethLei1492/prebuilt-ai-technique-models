"""Lexical (tokenizer, BM25) and dense (hashing embedder, brute-force index) channels."""
from __future__ import annotations

import math

import pytest

from hybrid_retrieval.dense import CallableEmbedder, DenseIndex, HashingEmbedder, cosine, l2_normalize, mean_vector
from hybrid_retrieval.sparse import DEFAULT_STOPWORDS, BM25Index, Tokenizer, looks_like_identifier, make_bm25

DOCS = [
    ("d1", "ERR-1203 is raised when the heartbeat from the field bus is lost"),
    ("d2", "The casing shall be ASTM A216 WCB carbon steel with a corrosion allowance of 3 mm"),
    ("d3", "Bearing preload is set after the housing reaches ambient temperature; record the rolling torque"),
    ("d4", "Call getUserName() to fetch the user name from the directory service"),
]


# ------------------------------------------------------------------ sparse
def test_looks_like_identifier() -> None:
    assert looks_like_identifier("ERR-1203")
    assert looks_like_identifier("A890")
    assert looks_like_identifier("getUserName")
    assert looks_like_identifier("retry.max_attempts")
    assert not looks_like_identifier("casing")
    assert not looks_like_identifier("Pump")


def test_tokenizer_preserves_identifiers_and_subtokens() -> None:
    tok = Tokenizer()
    assert tok("ERR-1203") == ["err-1203", "err", "1203"]
    assert tok("getUserName") == ["getusername", "get", "user", "name"]
    assert tok("A890 Grade 4A") == ["a890", "a", "890", "grade", "4a", "4", "a"]
    plain = Tokenizer(preserve_identifiers=False)
    assert plain("ERR-1203") == ["err-1203"]


def test_tokenizer_stopwords_and_stemmer() -> None:
    tok = Tokenizer(stopwords=DEFAULT_STOPWORDS, stemmer=lambda w: w.rstrip("s"))
    assert tok("the bearings of the pumps") == ["bearing", "pump"]
    # identifiers are never stemmed or dropped
    assert tok("the ERR-1203 alarms")[:3] == ["err-1203", "err", "1203"]
    assert "the" not in tok.content_tokens("the ERR-1203 alarms")


def test_bm25_identifier_match_ranks_first_and_is_deterministic() -> None:
    idx = BM25Index(tokenizer=Tokenizer(stopwords=DEFAULT_STOPWORDS))
    idx.add_many(DOCS)
    assert len(idx) == 4
    top = idx.search("ERR-1203 cause", k=3)
    assert top[0][0] == "d1" and top[0][1] > 0
    assert idx.search("getUserName", k=1)[0][0] == "d4"
    assert idx.search("nothing matches here xyz", k=3) == []
    # ties break on id
    tie = BM25Index()
    tie.add_many([("b", "alpha"), ("a", "alpha")])
    assert [d for d, _ in tie.search("alpha", k=2)] == ["a", "b"]


def test_bm25_idf_decreases_with_df_and_allowed_ids_filter() -> None:
    idx = BM25Index()
    idx.add_many(DOCS)
    assert idx.idf("err-1203") > idx.idf("the")
    filtered = idx.search("the", k=10, allowed_ids={"d2"})
    assert [d for d, _ in filtered] == ["d2"]


def test_bm25_remove_and_replace() -> None:
    idx = BM25Index()
    idx.add_many(DOCS)
    idx.remove(["d1"])
    assert len(idx) == 3 and idx.search("ERR-1203", k=2) == []
    idx.add("d2", "completely different text about valves")
    assert idx.search("casing", k=2) == [] and idx.search("valves", k=1)[0][0] == "d2"
    assert math.isclose(idx.avgdl, sum(idx._doc_len.values()) / 3)


def test_bm25_plus_lower_bounds_term_scores() -> None:
    okapi = BM25Index(variant="okapi")
    plus = BM25Index(variant="plus", delta=1.0)
    for idx in (okapi, plus):
        idx.add_many(DOCS)
    s_ok = dict(okapi.search("corrosion allowance", k=5))["d2"]
    s_plus = dict(plus.search("corrosion allowance", k=5))["d2"]
    assert s_plus > s_ok
    with pytest.raises(ValueError):
        BM25Index(variant="bm25f")


def test_rank_bm25_adapter_agrees_on_top_result() -> None:
    pytest.importorskip("rank_bm25")
    idx = make_bm25("rank_bm25")
    assert idx.describe()["backend"] == "rank_bm25"
    idx.add_many(DOCS)
    assert idx.search("ERR-1203", k=1)[0][0] == "d1"
    idx.remove(["d1"])
    assert len(idx) == 3
    assert make_bm25("python").describe()["backend"] == "python"


# ------------------------------------------------------------------- dense
def test_hashing_embedder_is_deterministic_and_normalised() -> None:
    emb = HashingEmbedder(dimension=256)
    a, b = emb.embed(["pump impeller material", "pump impeller material"])
    assert a == b and len(a) == 256
    assert math.isclose(math.sqrt(sum(x * x for x in a)), 1.0, rel_tol=1e-9)
    near = cosine(emb.embed(["pump impeller material"])[0], emb.embed_queries(["impeller material of the pump"])[0])
    far = cosine(emb.embed(["pump impeller material"])[0], emb.embed_queries(["quarterly schedule memo"])[0])
    assert near > far
    assert emb.embed([]) == []


def test_dense_index_search_filter_remove_replace() -> None:
    idx = DenseIndex(HashingEmbedder(128))
    assert idx.search("anything", k=3) == []
    idx.add(DOCS)
    assert len(idx) == 4
    top = idx.search("heartbeat lost on the field bus", k=2)
    assert top[0][0] == "d1"
    assert all(-1.0 <= s <= 1.0 for _, s in top)
    only = idx.search("heartbeat lost on the field bus", k=3, allowed_ids={"d2", "d3"})
    assert {d for d, _ in only} <= {"d2", "d3"}
    idx.remove(["d1"])
    assert len(idx) == 3 and idx.vector("d1") is None
    idx.add([("d2", "new text about pump bearings")])
    assert len(idx) == 3 and idx.search("pump bearings", k=1)[0][0] == "d2"


def test_dense_search_vector_with_mean_vector() -> None:
    idx = DenseIndex(HashingEmbedder(128))
    idx.add(DOCS)
    q = idx.embed_query("corrosion allowance")
    p = l2_normalize(idx.embedder.embed(["carbon steel casing with corrosion allowance"])[0])
    v = mean_vector([q, p])
    assert len(v) == 128
    assert idx.search_vector(v, k=1)[0][0] == "d2"
    assert mean_vector([]) == []


def test_callable_embedder_wraps_functions() -> None:
    emb = CallableEmbedder(lambda texts: [[float(len(t)), 1.0] for t in texts], dimension=2, query_fn=lambda texts: [[0.0, 1.0] for _ in texts])
    assert emb.embed(["abc"]) == [[3.0, 1.0]]
    assert emb.embed_queries(["abc"]) == [[0.0, 1.0]]
    assert emb.describe()["dimension"] == 2
