# test_semantic_search.py
# Unit tests for backend/services/semantic_search.py
#
# All tests use a mock Embedder injected directly — no model download, no GPU.
# Embeddings are hand-crafted unit vectors so expected scores are exact and
# predictable.

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from backend.models.schemas import Chunk, StoredChunk
from backend.services.semantic_search import semantic_search, DEFAULT_THRESHOLD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit(v: list[float]) -> np.ndarray:
    """Return a unit-normalised float32 vector."""
    a = np.array(v, dtype=np.float32)
    return (a / np.linalg.norm(a)).astype(np.float32)


def _make_chunk(chunk_id: str, page: int = 1, text: str = "text") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc-001",
        page_number=page,
        chunk_index=0,
        text=text,
    )


def _stored(chunk: Chunk, embedding: np.ndarray) -> StoredChunk:
    return StoredChunk(chunk=chunk, embedding=embedding)


def _mock_embedder(query_vec: np.ndarray) -> MagicMock:
    """Return a mock Embedder whose embed_query always returns *query_vec*."""
    embedder = MagicMock()
    embedder.embed_query.return_value = query_vec
    return embedder


# ---------------------------------------------------------------------------
# Pre-built vectors used across tests
#
# All four basis vectors are orthogonal.  Dot product between any two
# different basis vectors is 0.0; between identical ones is 1.0.
# ---------------------------------------------------------------------------

E0 = _unit([1.0, 0.0, 0.0, 0.0])   # "direction 0"
E1 = _unit([0.0, 1.0, 0.0, 0.0])   # "direction 1"
E2 = _unit([0.0, 0.0, 1.0, 0.0])   # "direction 2"
E3 = _unit([0.0, 0.0, 0.0, 1.0])   # "direction 3"
DIAG = _unit([1.0, 1.0, 0.0, 0.0]) # 45° between E0 and E1


# ---------------------------------------------------------------------------
# 1. Single matching chunk
# ---------------------------------------------------------------------------

class TestSingleMatchingChunk:
    def test_returns_one_result_for_one_match(self):
        chunk = _make_chunk("c0")
        stored = [_stored(chunk, E0)]
        embedder = _mock_embedder(E0)

        results = semantic_search("query", stored, embedder, threshold=0.5)

        assert len(results) == 1

    def test_result_contains_correct_chunk(self):
        chunk = _make_chunk("c0")
        stored = [_stored(chunk, E0)]
        embedder = _mock_embedder(E0)

        results = semantic_search("query", stored, embedder, threshold=0.5)

        assert results[0].chunk is chunk

    def test_result_score_is_correct(self):
        chunk = _make_chunk("c0")
        stored = [_stored(chunk, E0)]
        embedder = _mock_embedder(E0)

        results = semantic_search("query", stored, embedder, threshold=0.5)

        # E0 · E0 = 1.0
        assert abs(results[0].score - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# 2. Multiple matching chunks
# ---------------------------------------------------------------------------

class TestMultipleMatchingChunks:
    def test_all_matching_chunks_returned(self):
        chunks = [_make_chunk(f"c{i}") for i in range(3)]
        # All chunks point in the same direction as the query → all match
        stored = [_stored(c, E0) for c in chunks]
        embedder = _mock_embedder(E0)

        results = semantic_search("query", stored, embedder, threshold=0.5)

        assert len(results) == 3

    def test_partial_match_only_returns_above_threshold(self):
        chunk_match = _make_chunk("match")
        chunk_no = _make_chunk("no-match")
        stored = [
            _stored(chunk_match, E0),   # E0 · E0 = 1.0  ≥ 0.5  → included
            _stored(chunk_no,    E1),   # E0 · E1 = 0.0  < 0.5  → excluded
        ]
        embedder = _mock_embedder(E0)

        results = semantic_search("query", stored, embedder, threshold=0.5)

        assert len(results) == 1
        assert results[0].chunk.chunk_id == "match"


# ---------------------------------------------------------------------------
# 3. Descending score ordering
# ---------------------------------------------------------------------------

class TestDescendingScoreOrder:
    def test_results_sorted_highest_score_first(self):
        # DIAG · E0 ≈ 0.707,  DIAG · E1 ≈ 0.707,  DIAG · DIAG = 1.0
        c_low  = _make_chunk("low")
        c_mid  = _make_chunk("mid")
        c_high = _make_chunk("high")
        stored = [
            _stored(c_low,  E2),    # DIAG · E2 = 0.0
            _stored(c_mid,  E0),    # DIAG · E0 ≈ 0.707
            _stored(c_high, DIAG),  # DIAG · DIAG = 1.0
        ]
        embedder = _mock_embedder(DIAG)

        results = semantic_search("query", stored, embedder, threshold=0.0)

        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_highest_scoring_chunk_is_first(self):
        c_best  = _make_chunk("best")
        c_other = _make_chunk("other")
        stored = [
            _stored(c_other, E1),   # DIAG · E1 ≈ 0.707
            _stored(c_best,  DIAG), # DIAG · DIAG = 1.0
        ]
        embedder = _mock_embedder(DIAG)

        results = semantic_search("query", stored, embedder, threshold=0.0)

        assert results[0].chunk.chunk_id == "best"

    def test_three_chunks_in_correct_descending_order(self):
        c1 = _make_chunk("c1")
        c2 = _make_chunk("c2")
        c3 = _make_chunk("c3")
        # Scores when query = DIAG:
        # c3 → DIAG   = 1.0
        # c1 → E0     ≈ 0.707
        # c2 → E2     = 0.0
        stored = [
            _stored(c1, E0),
            _stored(c2, E2),
            _stored(c3, DIAG),
        ]
        embedder = _mock_embedder(DIAG)

        results = semantic_search("query", stored, embedder, threshold=0.0)

        assert [r.chunk.chunk_id for r in results] == ["c3", "c1", "c2"]


# ---------------------------------------------------------------------------
# 4. Threshold filtering
# ---------------------------------------------------------------------------

class TestThresholdFiltering:
    def test_default_threshold_filters_low_scores(self):
        c_pass = _make_chunk("pass")
        c_fail = _make_chunk("fail")
        stored = [
            _stored(c_pass, E0),    # score = 1.0  ≥ DEFAULT_THRESHOLD
            _stored(c_fail, E1),    # score = 0.0  < DEFAULT_THRESHOLD
        ]
        embedder = _mock_embedder(E0)

        results = semantic_search("query", stored, embedder)

        ids = [r.chunk.chunk_id for r in results]
        assert "pass" in ids
        assert "fail" not in ids

    def test_high_threshold_returns_only_very_similar_chunks(self):
        c_high = _make_chunk("high")
        c_mid  = _make_chunk("mid")
        stored = [
            _stored(c_high, DIAG),  # DIAG · DIAG = 1.0 ≥ 0.9
            _stored(c_mid,  E0),    # DIAG · E0  ≈ 0.707 < 0.9
        ]
        embedder = _mock_embedder(DIAG)

        results = semantic_search("query", stored, embedder, threshold=0.9)

        assert len(results) == 1
        assert results[0].chunk.chunk_id == "high"

    def test_zero_threshold_includes_all_non_negative_scores(self):
        chunks = [_make_chunk(f"c{i}") for i in range(4)]
        stored = [
            _stored(chunks[0], E0),    # score = 1.0
            _stored(chunks[1], E1),    # score = 0.0
            _stored(chunks[2], DIAG),  # score ≈ 0.707
            _stored(chunks[3], E2),    # score = 0.0
        ]
        embedder = _mock_embedder(E0)

        results = semantic_search("query", stored, embedder, threshold=0.0)

        # E0 · E0 = 1.0 ≥ 0.0; E0 · E1 = 0.0 ≥ 0.0; DIAG · E0 ≈ 0.707; E2 · E0 = 0.0
        assert len(results) == 4

    def test_threshold_boundary_exact_equality_included(self):
        """A score exactly equal to threshold must be included (>=)."""
        chunk = _make_chunk("boundary")
        # Set query = E0 and chunk = E0 → score = 1.0
        # Use threshold = 1.0 to test the boundary
        stored = [_stored(chunk, E0)]
        embedder = _mock_embedder(E0)

        results = semantic_search("query", stored, embedder, threshold=1.0)

        assert len(results) == 1
        assert abs(results[0].score - 1.0) < 1e-5

    def test_score_just_below_threshold_excluded(self):
        chunk = _make_chunk("c")
        # DIAG · E0 ≈ 0.7071 — set threshold slightly above it
        stored = [_stored(chunk, E0)]
        embedder = _mock_embedder(DIAG)

        results = semantic_search("query", stored, embedder, threshold=0.8)

        assert results == []


# ---------------------------------------------------------------------------
# 5. No matches
# ---------------------------------------------------------------------------

class TestNoMatches:
    def test_returns_empty_list_when_no_chunk_meets_threshold(self):
        chunk = _make_chunk("c")
        # E0 · E1 = 0.0 < 0.5
        stored = [_stored(chunk, E1)]
        embedder = _mock_embedder(E0)

        results = semantic_search("query", stored, embedder, threshold=0.5)

        assert results == []

    def test_returns_empty_list_for_orthogonal_embeddings(self):
        chunks = [_make_chunk(f"c{i}") for i in range(3)]
        stored = [
            _stored(chunks[0], E1),
            _stored(chunks[1], E2),
            _stored(chunks[2], E3),
        ]
        embedder = _mock_embedder(E0)

        results = semantic_search("query", stored, embedder, threshold=DEFAULT_THRESHOLD)

        assert results == []


# ---------------------------------------------------------------------------
# 6. Empty inputs
# ---------------------------------------------------------------------------

class TestEmptyInputs:
    def test_raises_on_empty_query_string(self):
        embedder = _mock_embedder(E0)
        with pytest.raises(ValueError, match="non-empty"):
            semantic_search("", [], embedder)

    def test_raises_on_whitespace_query(self):
        embedder = _mock_embedder(E0)
        with pytest.raises(ValueError, match="non-empty"):
            semantic_search("   ", [], embedder)

    def test_returns_empty_list_for_empty_chunk_list(self):
        embedder = _mock_embedder(E0)
        results = semantic_search("invoice", [], embedder)
        assert results == []

    def test_embed_query_not_called_when_no_chunks(self):
        """embed_query is cheap but there is no reason to call it with zero chunks."""
        embedder = _mock_embedder(E0)
        semantic_search("invoice", [], embedder)
        embedder.embed_query.assert_not_called()


# ---------------------------------------------------------------------------
# 7. Threshold validation
# ---------------------------------------------------------------------------

class TestThresholdValidation:
    def test_raises_on_threshold_above_1(self):
        embedder = _mock_embedder(E0)
        with pytest.raises(ValueError, match="threshold"):
            semantic_search("query", [], embedder, threshold=1.1)

    def test_raises_on_threshold_below_minus_1(self):
        embedder = _mock_embedder(E0)
        with pytest.raises(ValueError, match="threshold"):
            semantic_search("query", [], embedder, threshold=-1.1)

    def test_threshold_minus_1_is_valid(self):
        """threshold=-1.0 is the minimum valid value; must not raise."""
        chunk = _make_chunk("c")
        stored = [_stored(chunk, E0)]
        embedder = _mock_embedder(E0)
        # Should not raise, even with the minimum threshold
        results = semantic_search("query", stored, embedder, threshold=-1.0)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# 8. Inconsistent embedding dimensions
# ---------------------------------------------------------------------------

class TestInconsistentDimensions:
    def test_raises_when_chunk_has_wrong_dimension(self):
        chunk = _make_chunk("bad-dim")
        wrong_dim_embedding = _unit([1.0, 0.0])     # dim=2, query will be dim=4
        stored = [_stored(chunk, wrong_dim_embedding)]
        embedder = _mock_embedder(E0)                # E0 is dim=4

        with pytest.raises(ValueError, match="dimension"):
            semantic_search("query", stored, embedder)

    def test_raises_when_one_of_many_chunks_has_wrong_dimension(self):
        good   = _make_chunk("good")
        bad    = _make_chunk("bad")
        stored = [
            _stored(good, E0),
            _stored(bad,  _unit([1.0, 0.0])),        # wrong dim
        ]
        embedder = _mock_embedder(E0)

        with pytest.raises(ValueError, match="dimension"):
            semantic_search("query", stored, embedder)


# ---------------------------------------------------------------------------
# 9. Query embedding usage
# ---------------------------------------------------------------------------

class TestQueryEmbeddingUsage:
    def test_embed_query_called_exactly_once(self):
        chunk = _make_chunk("c")
        stored = [_stored(chunk, E0)]
        embedder = _mock_embedder(E0)

        semantic_search("invoice", stored, embedder)

        embedder.embed_query.assert_called_once_with("invoice")

    def test_query_string_passed_verbatim_to_embedder(self):
        query = "machine learning pipeline"
        chunk = _make_chunk("c")
        stored = [_stored(chunk, E0)]
        embedder = _mock_embedder(E0)

        semantic_search(query, stored, embedder)

        embedder.embed_query.assert_called_once_with(query)
