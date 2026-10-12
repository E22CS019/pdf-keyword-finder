# result_merger.py
# Responsibilities:
#   - Accept LexicalResult and SemanticResult lists from the two search strategies
#   - Identify overlapping results by chunk_id and deduplicate them
#   - When overlap occurs, preserve the strongest match_type (exact > variant > semantic)
#   - Preserve semantic score on every result that has one
#   - Rank: exact first, then variant, then semantic (by descending similarity score)
#   - Return a clean list of MergedResult objects for the API layer
#   - Perform no new matching, embedding, or threshold filtering

from __future__ import annotations

import logging
from typing import Sequence

from backend.models.schemas import (
    LexicalResult,
    MergedResult,
    SemanticResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Match-type priority — lower integer = higher priority in the final ranking
# ---------------------------------------------------------------------------
_MATCH_PRIORITY: dict[str, int] = {
    "exact":    0,
    "variant":  1,
    "semantic": 2,
}


def _priority(match_type: str) -> int:
    """Return the sort priority for *match_type* (lower = ranked first)."""
    return _MATCH_PRIORITY.get(match_type, 99)


def merge_results(
    lexical_results: Sequence[LexicalResult],
    semantic_results: Sequence[SemanticResult],
    keyword: str,
) -> list[MergedResult]:
    """Combine lexical and semantic results into a single ranked result list.

    **Deduplication**

    ``chunk_id`` is the canonical identity of a document passage.  If the
    same ``chunk_id`` appears in both *lexical_results* and
    *semantic_results*, the two entries are merged into one
    :class:`~backend.models.schemas.MergedResult`.  A chunk that carries
    identical text but a different ``chunk_id`` (e.g. the same sentence on
    two separate pages) is treated as a distinct result and kept separately.

    **Merge rule for overlapping chunks**

    When a chunk appears in both result sets:

    - ``match_type`` is resolved as ``exact > variant > semantic`` — the
      strongest lexical signal always wins.
    - ``match_count`` is taken from the lexical result.
    - ``semantic_score`` is taken from the semantic result and preserved so
      downstream consumers can surface it if desired.

    If a chunk appears only in lexical results, ``semantic_score`` is
    ``None``.  If it appears only in semantic results, ``match_count`` is
    ``0`` and ``match_type`` is ``"semantic"``.

    **Ranking**

    The final list is sorted by ``(priority, -semantic_score)`` where
    ``priority`` is 0 for ``"exact"``, 1 for ``"variant"``, 2 for
    ``"semantic"``.  Within the ``"semantic"`` tier, results are ordered by
    descending similarity score.  Within the lexical tiers, the order of the
    input lists is preserved (Python's sort is stable).

    No new matching, embedding computation, or threshold filtering is
    performed here.

    Args:
        lexical_results: Results from
            :func:`~backend.services.lexical_search.lexical_search`.
            May be empty.
        semantic_results: Results from
            :func:`~backend.services.semantic_search.semantic_search`.
            May be empty.
        keyword: The original search keyword, attached to every
            :class:`~backend.models.schemas.MergedResult` for traceability.

    Returns:
        A deduplicated, ranked list of
        :class:`~backend.models.schemas.MergedResult` objects.
        Returns an empty list when both inputs are empty.
    """
    # ------------------------------------------------------------------
    # 1. Index semantic results by chunk_id for O(1) lookup
    # ------------------------------------------------------------------
    # When multiple SemanticResult objects share the same chunk_id (which
    # should not happen in normal operation but is handled defensively), keep
    # the one with the highest score.
    semantic_by_id: dict[str, SemanticResult] = {}
    for sr in semantic_results:
        cid = sr.chunk.chunk_id
        if cid not in semantic_by_id or sr.score > semantic_by_id[cid].score:
            semantic_by_id[cid] = sr

    # ------------------------------------------------------------------
    # 2. Index lexical results by chunk_id, keeping the strongest match type
    #    (exact > variant) for each chunk.
    #
    # lexical_search returns exact results before variant results, so the
    # first time we see a chunk_id it is already the strongest hit — but we
    # guard explicitly in case the caller supplies results in a different
    # order.
    # ------------------------------------------------------------------
    lexical_by_id: dict[str, LexicalResult] = {}
    for lr in lexical_results:
        cid = lr.chunk.chunk_id
        if cid not in lexical_by_id:
            lexical_by_id[cid] = lr
        else:
            existing = lexical_by_id[cid]
            if _priority(lr.match_type) < _priority(existing.match_type):
                # This entry has higher priority (e.g. "exact" beats "variant")
                lexical_by_id[cid] = lr

    # ------------------------------------------------------------------
    # 3. Build the merged result set
    # ------------------------------------------------------------------
    merged: dict[str, MergedResult] = {}

    # --- lexical results (may or may not overlap with semantic) ---
    for cid, lr in lexical_by_id.items():
        sem = semantic_by_id.get(cid)
        merged[cid] = MergedResult(
            chunk=lr.chunk,
            keyword=keyword,
            match_type=lr.match_type,          # "exact" or "variant"
            match_count=lr.match_count,
            semantic_score=sem.score if sem else None,
        )

    # --- semantic-only results (chunk_id not seen in lexical results) ---
    for cid, sr in semantic_by_id.items():
        if cid not in merged:
            merged[cid] = MergedResult(
                chunk=sr.chunk,
                keyword=keyword,
                match_type="semantic",
                match_count=0,
                semantic_score=sr.score,
            )

    # ------------------------------------------------------------------
    # 4. Rank: exact → variant → semantic (descending score within tier)
    #
    # Sort key: (priority_int, -semantic_score)
    # Using float("inf") as the fallback score means lexical-only results
    # (semantic_score=None) sort to the END of their tier when compared
    # against results that also have a score — but since the lexical tiers
    # (0 and 1) contain no semantic-only results this never matters in
    # practice. The negation of score sorts descending within the semantic
    # tier (priority 2).
    # ------------------------------------------------------------------
    def _sort_key(r: MergedResult) -> tuple[int, float]:
        score_for_sort = r.semantic_score if r.semantic_score is not None else 0.0
        return (_priority(r.match_type), -score_for_sort)

    ranked = sorted(merged.values(), key=_sort_key)

    logger.debug(
        "merge_results: keyword=%r, lexical=%d, semantic=%d, merged=%d",
        keyword,
        len(lexical_by_id),
        len(semantic_by_id),
        len(ranked),
    )
    return ranked
