# semantic_search.py
# Responsibilities:
#   - Accept a query string and a list of StoredChunks with pre-computed embeddings
#   - Embed the query via the provided Embedder instance
#   - Compute similarity between the query embedding and every chunk embedding
#   - Filter out chunks below the configured similarity threshold
#   - Return results sorted by similarity score, descending
#   - Perform no exact/variant matching, no deduplication, no result merging

from __future__ import annotations

import logging

import numpy as np

from backend.models.schemas import SemanticResult, StoredChunk
from backend.services.embedder import Embedder

logger = logging.getLogger(__name__)

# Default similarity threshold.  Vectors with a score below this are not
# returned.  Tune upward for higher precision, downward for higher recall.
DEFAULT_THRESHOLD: float = 0.30


def semantic_search(
    query: str,
    stored_chunks: list[StoredChunk],
    embedder: Embedder,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[SemanticResult]:
    """Find chunks whose meaning is semantically similar to *query*.

    Similarity is computed as the **dot product** between the (unit-normalised)
    query embedding and each (unit-normalised) chunk embedding.  For
    unit-normalised vectors, dot product is numerically identical to cosine
    similarity:

    .. math::

        \\cos(\\theta) = \\frac{\\mathbf{q} \\cdot \\mathbf{c}}{\\|\\mathbf{q}\\| \\|\\mathbf{c}\\|}
                       = \\mathbf{q} \\cdot \\mathbf{c} \\quad \\text{when } \\|\\mathbf{q}\\| = \\|\\mathbf{c}\\| = 1

    All embeddings are normalised by :class:`~backend.services.embedder.Embedder`
    before being stored or queried, so no extra division is needed here.

    The full similarity computation is vectorised: chunk embeddings are stacked
    into a matrix and a single ``numpy`` matrix–vector product is used, making
    the operation efficient even for thousands of chunks.

    Args:
        query: The user's search query.  Must be a non-empty string.
        stored_chunks: All chunks in the document store, each carrying a
            pre-computed unit-normalised embedding.  May be an empty list.
        embedder: The :class:`~backend.services.embedder.Embedder` instance
            used to encode the query.  The same instance (and therefore the
            same model and normalisation) used during ingestion must be
            provided so scores are comparable.
        threshold: Minimum similarity score for a chunk to be included in the
            results.  Must be in ``[-1.0, 1.0]``.  Defaults to ``0.30``.
            Chunks with ``score >= threshold`` are returned; those below are
            silently discarded.

    Returns:
        A list of :class:`~backend.models.schemas.SemanticResult` objects,
        each containing the matching :class:`~backend.models.schemas.Chunk`
        and its similarity score.  The list is sorted by score descending.
        Returns an empty list when:

        - *stored_chunks* is empty, or
        - no chunk meets the *threshold*.

    Raises:
        ValueError: If *query* is empty or whitespace-only.
        ValueError: If *threshold* is outside ``[-1.0, 1.0]``.
        ValueError: If any chunk embedding has a different dimensionality from
            the query embedding.
    """
    # ------------------------------------------------------------------
    # 1. Validate inputs
    # ------------------------------------------------------------------
    if not query or not query.strip():
        raise ValueError("semantic_search requires a non-empty query string.")

    if not (-1.0 <= threshold <= 1.0):
        raise ValueError(
            f"threshold must be in [-1.0, 1.0]; got {threshold!r}."
        )

    if not stored_chunks:
        logger.debug("semantic_search called with an empty chunk list; returning [].")
        return []

    # ------------------------------------------------------------------
    # 2. Embed the query
    # ------------------------------------------------------------------
    # embed_query raises ValueError for empty/whitespace queries (already
    # guarded above) and returns a 1-D unit-normalised float32 array.
    query_vec: np.ndarray = embedder.embed_query(query)  # shape: (D,)

    # ------------------------------------------------------------------
    # 3. Stack chunk embeddings into a matrix and validate dimensions
    # ------------------------------------------------------------------
    embeddings: list[np.ndarray] = [sc.embedding for sc in stored_chunks]

    dim = query_vec.shape[0]
    bad = [
        (i, sc.chunk.chunk_id, sc.embedding.shape)
        for i, sc in enumerate(stored_chunks)
        if sc.embedding.ndim != 1 or sc.embedding.shape[0] != dim
    ]
    if bad:
        details = ", ".join(
            f"chunk '{cid}' at index {i} has shape {shape}"
            for i, cid, shape in bad
        )
        raise ValueError(
            f"Embedding dimension mismatch (query dim={dim}): {details}."
        )

    # shape: (N, D)
    embedding_matrix: np.ndarray = np.vstack(embeddings)

    # ------------------------------------------------------------------
    # 4. Compute similarity scores in one vectorised operation
    #
    # Because every embedding is unit-normalised:
    #   dot(query_vec, chunk_vec) == cosine_similarity(query_vec, chunk_vec)
    #
    # embedding_matrix @ query_vec gives one score per row (chunk).
    # ------------------------------------------------------------------
    scores: np.ndarray = embedding_matrix @ query_vec  # shape: (N,)

    # ------------------------------------------------------------------
    # 5. Filter by threshold and build results
    # ------------------------------------------------------------------
    results: list[SemanticResult] = [
        SemanticResult(chunk=stored_chunks[i].chunk, score=float(scores[i]))
        for i in range(len(stored_chunks))
        if scores[i] >= threshold
    ]

    # ------------------------------------------------------------------
    # 6. Sort by similarity score, highest first
    # ------------------------------------------------------------------
    results.sort(key=lambda r: r.score, reverse=True)

    logger.debug(
        "semantic_search: query=%r, chunks=%d, above_threshold=%d, threshold=%.3f",
        query,
        len(stored_chunks),
        len(results),
        threshold,
    )
    return results
