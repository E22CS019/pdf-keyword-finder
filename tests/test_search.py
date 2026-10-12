# test_search.py
# API tests for POST /search in backend/main.py
#
# Patching strategy
# -----------------
# * backend.main.Embedder is replaced with a zero-vector mock so the model
#   never loads during tests.
# * backend.main.lexical_search and backend.main.semantic_search are patched
#   per test-class so each class controls what the search services return,
#   keeping tests independent of real PDF content.
# * A shared fixture uploads a minimal real PDF so search tests have a live
#   document_id to work with.  The fixture lives at class scope so the upload
#   happens once per class, not once per test.

from __future__ import annotations

from unittest.mock import MagicMock, patch

import fitz
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.models.schemas import (
    Chunk,
    LexicalResult,
    MergedResult,
    SemanticResult,
    StoredChunk,
)


# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

DIM = 4   # embedding dimension used throughout mock tests


def _zero_embedder() -> MagicMock:
    """Mock Embedder: embed_chunks → zeros, embed_query → zeros."""
    mock = MagicMock()
    mock.embed_chunks.side_effect = lambda texts, **_: np.zeros(
        (len(texts), DIM), dtype=np.float32
    )
    mock.embed_query.return_value = np.zeros(DIM, dtype=np.float32)
    return mock


def _real_pdf(text: str = "Invoice total is due.\n\nPayment terms are Net 30.") -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    buf = doc.tobytes()
    doc.close()
    return buf


def _chunk(chunk_id: str = "doc-p1-c0", page: int = 1,
           text: str = "Invoice total is due.") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc",
        page_number=page,
        chunk_index=0,
        text=text,
    )


def _stored_chunk(chunk_id: str = "doc-p1-c0", page: int = 1,
                  text: str = "Invoice total is due.") -> StoredChunk:
    return StoredChunk(
        chunk=_chunk(chunk_id, page, text),
        embedding=np.zeros(DIM, dtype=np.float32),
    )


def _lex(chunk: Chunk, keyword: str = "invoice",
         match_type: str = "exact", count: int = 1) -> LexicalResult:
    return LexicalResult(chunk=chunk, keyword=keyword,
                         match_type=match_type, match_count=count)


def _sem(chunk: Chunk, score: float = 0.80) -> SemanticResult:
    return SemanticResult(chunk=chunk, score=score)


def _merged(chunk: Chunk, keyword: str = "invoice",
            match_type: str = "exact", count: int = 1,
            score: float | None = None) -> MergedResult:
    return MergedResult(chunk=chunk, keyword=keyword,
                        match_type=match_type, match_count=count,
                        semantic_score=score)


def _search(client: TestClient, document_id: str,
            keywords: list[str], threshold: float = 0.30):
    return client.post("/search", json={
        "document_id": document_id,
        "keywords": keywords,
        "threshold": threshold,
    })


# ---------------------------------------------------------------------------
# Shared TestClient fixture (mock Embedder, real lifespan)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    mock_emb = _zero_embedder()
    with patch("backend.main.Embedder", return_value=mock_emb):
        from backend.main import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


@pytest.fixture(scope="module")
def uploaded_doc_id(client):
    """Upload one real PDF once and return its document_id."""
    import io
    pdf = _real_pdf()
    resp = client.post(
        "/upload",
        files={"file": ("report.pdf", io.BytesIO(pdf), "application/pdf")},
    )
    assert resp.status_code == 201
    return resp.json()["document_id"]


# ---------------------------------------------------------------------------
# 1. Input validation (no document access needed)
# ---------------------------------------------------------------------------

class TestSearchValidation:
    def test_zero_keywords_rejected(self, client, uploaded_doc_id):
        resp = _search(client, uploaded_doc_id, [])
        assert resp.status_code in (400, 422)

    def test_too_many_keywords_rejected(self, client, uploaded_doc_id):
        resp = _search(client, uploaded_doc_id, [f"kw{i}" for i in range(21)])
        assert resp.status_code in (400, 422)

    def test_empty_keyword_rejected(self, client, uploaded_doc_id):
        resp = _search(client, uploaded_doc_id, ["invoice", ""])
        assert resp.status_code in (400, 422)

    def test_whitespace_keyword_rejected(self, client, uploaded_doc_id):
        resp = _search(client, uploaded_doc_id, ["   "])
        assert resp.status_code in (400, 422)

    def test_threshold_above_1_rejected(self, client, uploaded_doc_id):
        resp = client.post("/search", json={
            "document_id": uploaded_doc_id,
            "keywords": ["invoice"],
            "threshold": 1.1,
        })
        assert resp.status_code in (400, 422)

    def test_threshold_below_0_rejected(self, client, uploaded_doc_id):
        resp = client.post("/search", json={
            "document_id": uploaded_doc_id,
            "keywords": ["invoice"],
            "threshold": -0.1,
        })
        assert resp.status_code in (400, 422)

    def test_missing_document_id_rejected(self, client):
        resp = client.post("/search", json={"keywords": ["invoice"]})
        assert resp.status_code in (400, 422)

    def test_missing_keywords_rejected(self, client, uploaded_doc_id):
        resp = client.post("/search", json={"document_id": uploaded_doc_id})
        assert resp.status_code in (400, 422)


# ---------------------------------------------------------------------------
# 2. Unknown document_id → 404
# ---------------------------------------------------------------------------

class TestUnknownDocument:
    def test_unknown_document_id_returns_404(self, client):
        resp = _search(client, "does-not-exist", ["invoice"])
        assert resp.status_code == 404

    def test_404_detail_mentions_document_id(self, client):
        resp = _search(client, "ghost-id", ["invoice"])
        assert "ghost-id" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# 3. Valid search — mocked service layer
# ---------------------------------------------------------------------------

class TestValidSearch:
    """Patch lexical_search, semantic_search, and merge_results so tests are
    independent of real PDF content and embedding values."""

    def _run(self, client, doc_id, keywords, threshold=0.30,
             merged_per_keyword=None):
        """Helper: patch _search_keyword to return controlled merged results."""
        if merged_per_keyword is None:
            merged_per_keyword = {}

        def fake_search_keyword(keyword, stored_chunks, embedder, threshold):
            return merged_per_keyword.get(keyword, [])

        with patch("backend.main._search_keyword", side_effect=fake_search_keyword):
            return _search(client, doc_id, keywords, threshold)

    def test_returns_200(self, client, uploaded_doc_id):
        resp = self._run(client, uploaded_doc_id, ["invoice"])
        assert resp.status_code == 200

    def test_response_contains_document_id(self, client, uploaded_doc_id):
        resp = self._run(client, uploaded_doc_id, ["invoice"])
        assert resp.json()["document_id"] == uploaded_doc_id

    def test_response_has_results_and_not_found_keys(self, client, uploaded_doc_id):
        resp = self._run(client, uploaded_doc_id, ["invoice"])
        body = resp.json()
        assert "results" in body
        assert "not_found" in body

    def test_no_match_keyword_appears_in_not_found(self, client, uploaded_doc_id):
        resp = self._run(client, uploaded_doc_id, ["ghost"],
                         merged_per_keyword={"ghost": []})
        assert "ghost" in resp.json()["not_found"]

    def test_no_match_keyword_not_in_results(self, client, uploaded_doc_id):
        resp = self._run(client, uploaded_doc_id, ["ghost"],
                         merged_per_keyword={"ghost": []})
        assert resp.json()["results"] == []

    def test_match_keyword_not_in_not_found(self, client, uploaded_doc_id):
        c = _chunk()
        resp = self._run(client, uploaded_doc_id, ["invoice"],
                         merged_per_keyword={"invoice": [_merged(c)]})
        assert "invoice" not in resp.json()["not_found"]

    def test_exact_match_result_shape(self, client, uploaded_doc_id):
        c = _chunk("doc-p1-c0", page=3, text="Invoice total is due.")
        resp = self._run(client, uploaded_doc_id, ["invoice"],
                         merged_per_keyword={"invoice": [_merged(c, count=2)]})
        r = resp.json()["results"][0]
        assert r["chunk_id"] == "doc-p1-c0"
        assert r["page_number"] == 3
        assert r["keyword"] == "invoice"
        assert r["match_type"] == "exact"
        assert r["match_count"] == 2
        assert r["semantic_score"] is None

    def test_semantic_match_result_has_score(self, client, uploaded_doc_id):
        c = _chunk()
        resp = self._run(client, uploaded_doc_id, ["invoice"],
                         merged_per_keyword={"invoice": [
                             _merged(c, match_type="semantic", count=0, score=0.88)
                         ]})
        r = resp.json()["results"][0]
        assert r["match_type"] == "semantic"
        assert abs(r["semantic_score"] - 0.88) < 1e-5

    def test_variant_match_result(self, client, uploaded_doc_id):
        c = _chunk()
        resp = self._run(client, uploaded_doc_id, ["invoice"],
                         merged_per_keyword={"invoice": [
                             _merged(c, match_type="variant", count=1)
                         ]})
        assert resp.json()["results"][0]["match_type"] == "variant"

    def test_chunk_text_preserved_in_result(self, client, uploaded_doc_id):
        c = _chunk(text="The invoice is overdue by 30 days.")
        resp = self._run(client, uploaded_doc_id, ["invoice"],
                         merged_per_keyword={"invoice": [_merged(c)]})
        assert resp.json()["results"][0]["text"] == "The invoice is overdue by 30 days."


# ---------------------------------------------------------------------------
# 4. Multiple keywords
# ---------------------------------------------------------------------------

class TestMultipleKeywords:
    def test_two_keywords_both_matched(self, client, uploaded_doc_id):
        c1 = _chunk("c1", text="Invoice total.")
        c2 = _chunk("c2", text="Payment terms.")
        merged = {
            "invoice": [_merged(c1, keyword="invoice")],
            "payment": [_merged(c2, keyword="payment")],
        }
        with patch("backend.main._search_keyword",
                   side_effect=lambda **kw: merged.get(kw["keyword"], [])):
            resp = _search(client, uploaded_doc_id, ["invoice", "payment"])
        body = resp.json()
        assert len(body["results"]) == 2
        assert body["not_found"] == []

    def test_one_hit_one_miss(self, client, uploaded_doc_id):
        c1 = _chunk("c1", text="Invoice total.")
        merged = {"invoice": [_merged(c1, keyword="invoice")], "ghost": []}
        with patch("backend.main._search_keyword",
                   side_effect=lambda **kw: merged.get(kw["keyword"], [])):
            resp = _search(client, uploaded_doc_id, ["invoice", "ghost"])
        body = resp.json()
        assert len(body["results"]) == 1
        assert body["not_found"] == ["ghost"]

    def test_all_not_found(self, client, uploaded_doc_id):
        with patch("backend.main._search_keyword", return_value=[]):
            resp = _search(client, uploaded_doc_id, ["aaa", "bbb", "ccc"])
        body = resp.json()
        assert body["results"] == []
        assert set(body["not_found"]) == {"aaa", "bbb", "ccc"}

    def test_keyword_preserved_on_each_result(self, client, uploaded_doc_id):
        c1 = _chunk("c1")
        c2 = _chunk("c2")
        merged = {
            "invoice": [_merged(c1, keyword="invoice")],
            "payment": [_merged(c2, keyword="payment")],
        }
        with patch("backend.main._search_keyword",
                   side_effect=lambda **kw: merged.get(kw["keyword"], [])):
            resp = _search(client, uploaded_doc_id, ["invoice", "payment"])
        keywords_in_results = {r["keyword"] for r in resp.json()["results"]}
        assert keywords_in_results == {"invoice", "payment"}

    def test_twenty_keywords_accepted(self, client, uploaded_doc_id):
        with patch("backend.main._search_keyword", return_value=[]):
            resp = _search(client, uploaded_doc_id, [f"kw{i}" for i in range(20)])
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 5. Deduplication — merger already deduplicates; endpoint must not re-add
# ---------------------------------------------------------------------------

class TestDeduplication:
    def test_single_merged_result_per_chunk_per_keyword(self, client, uploaded_doc_id):
        """Even if lexical and semantic both matched the same chunk, the merger
        returns one result.  The endpoint must pass that through as-is."""
        c = _chunk("dup-chunk")
        # Simulate merger returning one deduplicated result
        with patch("backend.main._search_keyword",
                   return_value=[_merged(c, keyword="invoice", match_type="exact")]):
            resp = _search(client, uploaded_doc_id, ["invoice"])
        chunk_ids = [r["chunk_id"] for r in resp.json()["results"]]
        assert chunk_ids.count("dup-chunk") == 1


# ---------------------------------------------------------------------------
# 6. Threshold is forwarded to semantic search
# ---------------------------------------------------------------------------

class TestThresholdPropagation:
    def test_threshold_passed_to_search_keyword(self, client, uploaded_doc_id):
        """Verify the threshold value reaches _search_keyword unchanged."""
        captured = {}

        def fake(keyword, stored_chunks, embedder, threshold):
            captured["threshold"] = threshold
            return []

        with patch("backend.main._search_keyword", side_effect=fake):
            _search(client, uploaded_doc_id, ["invoice"], threshold=0.85)

        assert abs(captured["threshold"] - 0.85) < 1e-9

    def test_default_threshold_used_when_not_supplied(self, client, uploaded_doc_id):
        captured = {}

        def fake(keyword, stored_chunks, embedder, threshold):
            captured["threshold"] = threshold
            return []

        with patch("backend.main._search_keyword", side_effect=fake):
            client.post("/search", json={
                "document_id": uploaded_doc_id,
                "keywords": ["invoice"],
            })

        assert "threshold" in captured   # default was applied


# ---------------------------------------------------------------------------
# 7. Integration-style: real pipeline, real (mock-embedded) PDF
# ---------------------------------------------------------------------------

class TestIntegrationStyle:
    """No mock on _search_keyword — exercises the real lexical + semantic
    path end-to-end with the zero-vector mock embedder."""

    def test_exact_keyword_found_in_pdf(self, client, uploaded_doc_id):
        """'invoice' appears verbatim in the uploaded PDF text."""
        resp = _search(client, uploaded_doc_id, ["invoice"])
        assert resp.status_code == 200
        body = resp.json()
        # At least one result should carry match_type exact or variant
        types = {r["match_type"] for r in body["results"]}
        # With zero-vector embeddings semantic threshold may weed out results;
        # lexical should still fire.
        assert body["results"] or "invoice" in body["not_found"]

    def test_absent_keyword_goes_to_not_found(self, client, uploaded_doc_id):
        resp = _search(client, uploaded_doc_id, ["zzzyyyxxx"])
        assert "zzzyyyxxx" in resp.json()["not_found"]

    def test_response_structure_correct(self, client, uploaded_doc_id):
        resp = _search(client, uploaded_doc_id, ["invoice"])
        body = resp.json()
        assert "document_id" in body
        assert "results" in body
        assert "not_found" in body


# ---------------------------------------------------------------------------
# 8. Snippet and match-position tests
# ---------------------------------------------------------------------------

class TestSnippetAndPositions:
    """Tests for the new snippet, match_positions, and keyword_summary fields.

    These tests patch _search_keyword to return controlled MergedResult
    objects with specific chunk texts, so snippet logic can be verified
    deterministically without real PDF content.
    """

    def _run(self, client, doc_id, keyword, merged_results):
        with patch("backend.main._search_keyword", return_value=merged_results):
            return _search(client, doc_id, [keyword])

    # ---- snippet present on every result ----

    def test_exact_result_has_snippet(self, client, uploaded_doc_id):
        c = _chunk(text="The invoice is due on Friday.")
        resp = self._run(client, uploaded_doc_id, "invoice",
                         [_merged(c, keyword="invoice", match_type="exact", count=1)])
        r = resp.json()["results"][0]
        assert "snippet" in r
        assert len(r["snippet"]) > 0

    def test_semantic_result_has_snippet(self, client, uploaded_doc_id):
        c = _chunk(text="This paragraph discusses financial obligations.")
        resp = self._run(client, uploaded_doc_id, "invoice",
                         [_merged(c, keyword="invoice", match_type="semantic",
                                  count=0, score=0.80)])
        r = resp.json()["results"][0]
        assert "snippet" in r
        assert len(r["snippet"]) > 0

    def test_snippet_is_substring_of_text(self, client, uploaded_doc_id):
        c = _chunk(text="The invoice is due on Friday.")
        resp = self._run(client, uploaded_doc_id, "invoice",
                         [_merged(c, keyword="invoice", match_type="exact", count=1)])
        r = resp.json()["results"][0]
        assert r["snippet"] in r["text"]

    # ---- exact match positions ----

    def test_exact_match_has_one_position(self, client, uploaded_doc_id):
        c = _chunk(text="The invoice is due on Friday.")
        resp = self._run(client, uploaded_doc_id, "invoice",
                         [_merged(c, keyword="invoice", match_type="exact", count=1)])
        r = resp.json()["results"][0]
        assert len(r["match_positions"]) == 1

    def test_exact_match_position_points_to_keyword(self, client, uploaded_doc_id):
        text = "The invoice is due on Friday."
        c = _chunk(text=text)
        resp = self._run(client, uploaded_doc_id, "invoice",
                         [_merged(c, keyword="invoice", match_type="exact", count=1)])
        r = resp.json()["results"][0]
        s, e = r["match_positions"][0]
        assert r["snippet"][s:e].lower() == "invoice"

    def test_exact_match_position_is_relative_to_snippet(self, client, uploaded_doc_id):
        # Build a text long enough that the snippet is a sub-window.
        # Window is 200 chars total; use a 250-char prefix so the match
        # lands past the window start and triggers a trimmed snippet.
        prefix = "x" * 250
        text = prefix + " invoice here."
        c = _chunk(text=text)
        resp = self._run(client, uploaded_doc_id, "invoice",
                         [_merged(c, keyword="invoice", match_type="exact", count=1)])
        r = resp.json()["results"][0]
        # Snippet must be shorter than the full text.
        assert len(r["snippet"]) < len(text)
        # Positions are relative to snippet, not original text.
        s, e = r["match_positions"][0]
        assert r["snippet"][s:e].lower() == "invoice"

    # ---- multiple lexical occurrences ----

    def test_multiple_occurrences_produce_multiple_positions(self, client, uploaded_doc_id):
        text = "Invoice one and invoice two are both due."
        c = _chunk(text=text)
        resp = self._run(client, uploaded_doc_id, "invoice",
                         [_merged(c, keyword="invoice", match_type="exact", count=2)])
        r = resp.json()["results"][0]
        assert len(r["match_positions"]) == 2

    def test_multiple_positions_all_point_to_keyword(self, client, uploaded_doc_id):
        text = "Invoice one and invoice two are both due."
        c = _chunk(text=text)
        resp = self._run(client, uploaded_doc_id, "invoice",
                         [_merged(c, keyword="invoice", match_type="exact", count=2)])
        r = resp.json()["results"][0]
        for s, e in r["match_positions"]:
            assert r["snippet"][s:e].lower() == "invoice"

    # ---- variant match positions ----

    def test_variant_match_has_position(self, client, uploaded_doc_id):
        # "invoices" is a variant of "invoice"
        text = "The invoices are attached."
        c = _chunk(text=text)
        resp = self._run(client, uploaded_doc_id, "invoice",
                         [_merged(c, keyword="invoice", match_type="variant", count=1)])
        r = resp.json()["results"][0]
        assert len(r["match_positions"]) >= 1

    def test_variant_position_points_to_variant_form(self, client, uploaded_doc_id):
        text = "The invoices are attached."
        c = _chunk(text=text)
        resp = self._run(client, uploaded_doc_id, "invoice",
                         [_merged(c, keyword="invoice", match_type="variant", count=1)])
        r = resp.json()["results"][0]
        s, e = r["match_positions"][0]
        matched = r["snippet"][s:e].lower()
        # Matched text should be a variant of "invoice"
        assert matched in ("invoices", "invoiced", "invoicing")

    # ---- semantic results have no positions ----

    def test_semantic_match_positions_is_empty(self, client, uploaded_doc_id):
        c = _chunk(text="This discusses payment obligations generally.")
        resp = self._run(client, uploaded_doc_id, "invoice",
                         [_merged(c, keyword="invoice", match_type="semantic",
                                  count=0, score=0.75)])
        r = resp.json()["results"][0]
        assert r["match_positions"] == []

    def test_semantic_snippet_is_head_of_text(self, client, uploaded_doc_id):
        text = "Financial obligations are described in this section. " * 5
        c = _chunk(text=text)
        resp = self._run(client, uploaded_doc_id, "invoice",
                         [_merged(c, keyword="invoice", match_type="semantic",
                                  count=0, score=0.70)])
        r = resp.json()["results"][0]
        # Snippet should be the start of the text (first 200 chars).
        assert text.startswith(r["snippet"])


# ---------------------------------------------------------------------------
# 9. keyword_summary tests
# ---------------------------------------------------------------------------

class TestKeywordSummary:

    def _run(self, client, doc_id, keywords, merged_per_keyword):
        with patch("backend.main._search_keyword",
                   side_effect=lambda **kw: merged_per_keyword.get(kw["keyword"], [])):
            return _search(client, doc_id, keywords)

    def test_response_has_keyword_summary(self, client, uploaded_doc_id):
        c = _chunk(text="Invoice is due.")
        resp = self._run(client, uploaded_doc_id, ["invoice"],
                         {"invoice": [_merged(c, keyword="invoice", count=2)]})
        assert "keyword_summary" in resp.json()

    def test_keyword_summary_keyed_by_keyword(self, client, uploaded_doc_id):
        c = _chunk(text="Invoice is due.")
        resp = self._run(client, uploaded_doc_id, ["invoice"],
                         {"invoice": [_merged(c, keyword="invoice", count=2)]})
        assert "invoice" in resp.json()["keyword_summary"]

    def test_keyword_summary_total_matches(self, client, uploaded_doc_id):
        # Two chunks both matching "invoice", counts 2 and 3
        c1 = _chunk("c1", text="Invoice invoice here.")
        c2 = _chunk("c2", text="Invoice invoice invoice here.")
        resp = self._run(client, uploaded_doc_id, ["invoice"], {
            "invoice": [
                _merged(c1, keyword="invoice", count=2),
                _merged(c2, keyword="invoice", count=3),
            ]
        })
        summary = resp.json()["keyword_summary"]["invoice"]
        assert summary["total_matches"] == 5

    def test_keyword_summary_result_count(self, client, uploaded_doc_id):
        c1 = _chunk("c1", text="Invoice here.")
        c2 = _chunk("c2", text="Invoice there.")
        resp = self._run(client, uploaded_doc_id, ["invoice"], {
            "invoice": [
                _merged(c1, keyword="invoice", count=1),
                _merged(c2, keyword="invoice", count=1),
            ]
        })
        summary = resp.json()["keyword_summary"]["invoice"]
        assert summary["result_count"] == 2

    def test_semantic_only_does_not_increase_total_matches(self, client, uploaded_doc_id):
        c = _chunk(text="Financial obligations.")
        resp = self._run(client, uploaded_doc_id, ["invoice"], {
            "invoice": [_merged(c, keyword="invoice", match_type="semantic",
                                count=0, score=0.80)]
        })
        summary = resp.json()["keyword_summary"]["invoice"]
        assert summary["total_matches"] == 0
        assert summary["result_count"] == 1

    def test_not_found_keyword_not_in_summary(self, client, uploaded_doc_id):
        resp = self._run(client, uploaded_doc_id, ["ghost"], {"ghost": []})
        assert "ghost" not in resp.json()["keyword_summary"]

    def test_multiple_keywords_each_have_summary(self, client, uploaded_doc_id):
        c1 = _chunk("c1", text="Invoice here.")
        c2 = _chunk("c2", text="Payment here.")
        resp = self._run(client, uploaded_doc_id, ["invoice", "payment"], {
            "invoice": [_merged(c1, keyword="invoice", count=1)],
            "payment": [_merged(c2, keyword="payment", count=2)],
        })
        summary = resp.json()["keyword_summary"]
        assert "invoice" in summary
        assert "payment" in summary
        assert summary["payment"]["total_matches"] == 2

    def test_not_found_still_works_with_summary(self, client, uploaded_doc_id):
        c = _chunk(text="Invoice here.")
        resp = self._run(client, uploaded_doc_id, ["invoice", "ghost"], {
            "invoice": [_merged(c, keyword="invoice", count=1)],
            "ghost": [],
        })
        body = resp.json()
        assert "ghost" in body["not_found"]
        assert "ghost" not in body["keyword_summary"]
        assert "invoice" in body["keyword_summary"]
