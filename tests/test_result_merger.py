# test_result_merger.py
# Unit tests for backend/services/result_merger.py

import pytest
from backend.models.schemas import Chunk, LexicalResult, SemanticResult, MergedResult
from backend.services.result_merger import merge_results

KEYWORD = "invoice"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(chunk_id: str, page: int = 1, text: str = "sample text") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc-001",
        page_number=page,
        chunk_index=0,
        text=text,
    )


def _lex(chunk: Chunk, match_type: str = "exact", count: int = 1) -> LexicalResult:
    return LexicalResult(
        chunk=chunk,
        keyword=KEYWORD,
        match_type=match_type,
        match_count=count,
    )


def _sem(chunk: Chunk, score: float = 0.80) -> SemanticResult:
    return SemanticResult(chunk=chunk, score=score)


def _ids(results: list[MergedResult]) -> list[str]:
    return [r.chunk.chunk_id for r in results]


def _types(results: list[MergedResult]) -> list[str]:
    return [r.match_type for r in results]


# ---------------------------------------------------------------------------
# 1. Single-source results (no overlap)
# ---------------------------------------------------------------------------

class TestSingleSourceResults:
    def test_lexical_exact_only(self):
        c = _chunk("c1")
        results = merge_results([_lex(c, "exact")], [], KEYWORD)
        assert len(results) == 1
        assert results[0].match_type == "exact"

    def test_lexical_variant_only(self):
        c = _chunk("c1")
        results = merge_results([_lex(c, "variant")], [], KEYWORD)
        assert len(results) == 1
        assert results[0].match_type == "variant"

    def test_semantic_only(self):
        c = _chunk("c1")
        results = merge_results([], [_sem(c, 0.75)], KEYWORD)
        assert len(results) == 1
        assert results[0].match_type == "semantic"

    def test_lexical_only_semantic_score_is_none(self):
        c = _chunk("c1")
        results = merge_results([_lex(c)], [], KEYWORD)
        assert results[0].semantic_score is None

    def test_semantic_only_match_count_is_zero(self):
        c = _chunk("c1")
        results = merge_results([], [_sem(c, 0.70)], KEYWORD)
        assert results[0].match_count == 0

    def test_semantic_only_score_preserved(self):
        c = _chunk("c1")
        results = merge_results([], [_sem(c, 0.82)], KEYWORD)
        assert abs(results[0].semantic_score - 0.82) < 1e-6

    def test_both_empty_returns_empty(self):
        results = merge_results([], [], KEYWORD)
        assert results == []

    def test_empty_lexical_with_semantic_returns_semantic_results(self):
        c1, c2 = _chunk("c1"), _chunk("c2")
        results = merge_results([], [_sem(c1), _sem(c2)], KEYWORD)
        assert len(results) == 2

    def test_empty_semantic_with_lexical_returns_lexical_results(self):
        c1, c2 = _chunk("c1"), _chunk("c2")
        results = merge_results([_lex(c1), _lex(c2)], [], KEYWORD)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# 2. Deduplication — same chunk_id in both sources
# ---------------------------------------------------------------------------

class TestDeduplication:
    def test_overlap_produces_one_result(self):
        c = _chunk("c1")
        results = merge_results([_lex(c, "exact")], [_sem(c)], KEYWORD)
        assert len(results) == 1

    def test_no_duplicate_chunk_ids_in_output(self):
        c1, c2, c3 = _chunk("c1"), _chunk("c2"), _chunk("c3")
        lex = [_lex(c1, "exact"), _lex(c2, "variant")]
        sem = [_sem(c1, 0.90), _sem(c3, 0.70)]
        results = merge_results(lex, sem, KEYWORD)
        ids = _ids(results)
        assert len(ids) == len(set(ids))

    def test_exact_plus_semantic_gives_exact(self):
        c = _chunk("c1")
        results = merge_results([_lex(c, "exact")], [_sem(c)], KEYWORD)
        assert results[0].match_type == "exact"

    def test_variant_plus_semantic_gives_variant(self):
        c = _chunk("c1")
        results = merge_results([_lex(c, "variant")], [_sem(c)], KEYWORD)
        assert results[0].match_type == "variant"

    def test_exact_plus_variant_plus_semantic_gives_exact(self):
        c = _chunk("c1")
        # Both an exact and a variant LexicalResult for the same chunk
        lex = [_lex(c, "exact", count=1), _lex(c, "variant", count=2)]
        results = merge_results(lex, [_sem(c)], KEYWORD)
        assert len(results) == 1
        assert results[0].match_type == "exact"

    def test_variant_before_exact_in_input_still_gives_exact(self):
        # Defensive: even if variant arrives first in the list, exact wins.
        c = _chunk("c1")
        lex = [_lex(c, "variant"), _lex(c, "exact")]
        results = merge_results(lex, [], KEYWORD)
        assert results[0].match_type == "exact"

    def test_semantic_score_preserved_on_lexical_overlap(self):
        c = _chunk("c1")
        results = merge_results([_lex(c, "exact")], [_sem(c, 0.88)], KEYWORD)
        assert abs(results[0].semantic_score - 0.88) < 1e-6

    def test_match_count_preserved_on_overlap(self):
        c = _chunk("c1")
        results = merge_results([_lex(c, "exact", count=5)], [_sem(c)], KEYWORD)
        assert results[0].match_count == 5

    def test_different_chunk_ids_are_not_deduplicated(self):
        # Same text, different pages → different chunk_ids → both kept.
        c_p3  = _chunk("doc-p3-c0",  page=3,  text="Payment must be completed.")
        c_p15 = _chunk("doc-p15-c0", page=15, text="Payment must be completed.")
        results = merge_results(
            [_lex(c_p3, "exact"), _lex(c_p15, "exact")], [], KEYWORD
        )
        assert len(results) == 2
        assert {"doc-p3-c0", "doc-p15-c0"} == set(_ids(results))


# ---------------------------------------------------------------------------
# 3. Ranking order: exact → variant → semantic (desc score)
# ---------------------------------------------------------------------------

class TestRankingOrder:
    def test_exact_ranked_before_variant(self):
        c1, c2 = _chunk("c1"), _chunk("c2")
        # Deliberately pass variant first to confirm ranking, not insertion order.
        results = merge_results(
            [_lex(c2, "variant"), _lex(c1, "exact")], [], KEYWORD
        )
        assert _types(results) == ["exact", "variant"]

    def test_exact_ranked_before_semantic(self):
        c1, c2 = _chunk("c1"), _chunk("c2")
        results = merge_results([_lex(c1, "exact")], [_sem(c2, 0.99)], KEYWORD)
        assert results[0].match_type == "exact"
        assert results[1].match_type == "semantic"

    def test_variant_ranked_before_semantic(self):
        c1, c2 = _chunk("c1"), _chunk("c2")
        results = merge_results([_lex(c1, "variant")], [_sem(c2, 0.99)], KEYWORD)
        assert results[0].match_type == "variant"
        assert results[1].match_type == "semantic"

    def test_full_tier_order_exact_variant_semantic(self):
        ce, cv, cs = _chunk("ce"), _chunk("cv"), _chunk("cs")
        lex = [_lex(cv, "variant"), _lex(ce, "exact")]
        sem = [_sem(cs, 0.75)]
        results = merge_results(lex, sem, KEYWORD)
        assert _types(results) == ["exact", "variant", "semantic"]

    def test_semantic_results_sorted_by_descending_score(self):
        c1, c2, c3 = _chunk("c1"), _chunk("c2"), _chunk("c3")
        sem = [_sem(c1, 0.60), _sem(c3, 0.95), _sem(c2, 0.80)]
        results = merge_results([], sem, KEYWORD)
        scores = [r.semantic_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_semantic_highest_score_is_first_within_tier(self):
        c1, c2 = _chunk("c1"), _chunk("c2")
        results = merge_results([], [_sem(c1, 0.55), _sem(c2, 0.90)], KEYWORD)
        assert results[0].chunk.chunk_id == "c2"

    def test_three_semantic_results_ordered_correctly(self):
        c1, c2, c3 = _chunk("c1"), _chunk("c2"), _chunk("c3")
        sem = [_sem(c1, 0.50), _sem(c2, 0.90), _sem(c3, 0.70)]
        results = merge_results([], sem, KEYWORD)
        assert _ids(results) == ["c2", "c3", "c1"]

    def test_mixed_overlap_and_new_chunks_ranked_correctly(self):
        # c1: exact + semantic  → exact tier
        # c2: semantic only     → semantic tier
        # c3: variant only      → variant tier
        c1, c2, c3 = _chunk("c1"), _chunk("c2"), _chunk("c3")
        lex = [_lex(c1, "exact"), _lex(c3, "variant")]
        sem = [_sem(c1, 0.88), _sem(c2, 0.75)]
        results = merge_results(lex, sem, KEYWORD)
        assert results[0].chunk.chunk_id == "c1"   # exact first
        assert results[1].chunk.chunk_id == "c3"   # variant second
        assert results[2].chunk.chunk_id == "c2"   # semantic last


# ---------------------------------------------------------------------------
# 4. Returned object integrity
# ---------------------------------------------------------------------------

class TestReturnedObjectIntegrity:
    def test_result_is_merged_result_instance(self):
        c = _chunk("c1")
        results = merge_results([_lex(c)], [], KEYWORD)
        assert isinstance(results[0], MergedResult)

    def test_keyword_preserved_on_lexical_result(self):
        c = _chunk("c1")
        results = merge_results([_lex(c)], [], KEYWORD)
        assert results[0].keyword == KEYWORD

    def test_keyword_preserved_on_semantic_result(self):
        c = _chunk("c1")
        results = merge_results([], [_sem(c)], KEYWORD)
        assert results[0].keyword == KEYWORD

    def test_page_number_preserved(self):
        c = _chunk("c1", page=9)
        results = merge_results([_lex(c)], [], KEYWORD)
        assert results[0].chunk.page_number == 9

    def test_chunk_text_preserved(self):
        c = _chunk("c1", text="The invoice is due on Friday.")
        results = merge_results([_lex(c)], [], KEYWORD)
        assert results[0].chunk.text == "The invoice is due on Friday."

    def test_original_objects_not_mutated(self):
        c = _chunk("c1")
        lr = _lex(c, "exact", count=2)
        sr = _sem(c, 0.85)
        _ = merge_results([lr], [sr], KEYWORD)
        # Source objects must be unchanged
        assert lr.match_type == "exact"
        assert lr.match_count == 2
        assert abs(sr.score - 0.85) < 1e-6
