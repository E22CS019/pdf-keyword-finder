# embedder.py
# Responsibilities:
#   - Load sentence-transformers model once and hold it for the lifetime of
#     the object
#   - Embed a list of document chunk texts during PDF ingestion
#   - Embed a single query string at search time
#   - Return consistent numpy arrays so callers can compute cosine similarity
#   - Abstract the model so swapping to a different checkpoint requires only
#     a config change, not a code change

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Default model: lightweight, well-benchmarked, fits in <100 MB.
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    """Thin wrapper around a SentenceTransformer model.

    The model is downloaded (or loaded from the local cache) exactly once,
    in :meth:`__init__`, and reused for every subsequent call.  This keeps
    per-request latency low and avoids redundant disk I/O or network access.

    Usage::

        embedder = Embedder()                      # load model once
        doc_vecs  = embedder.embed_chunks(texts)   # shape (N, D)
        query_vec = embedder.embed_query("invoice")# shape (D,)

    Args:
        model_name: HuggingFace model identifier or local path.
            Defaults to ``"sentence-transformers/all-MiniLM-L6-v2"``.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model_name = model_name
        logger.info("Loading embedding model: %s", model_name)
        self._model: SentenceTransformer = SentenceTransformer(model_name)
        logger.info("Embedding model loaded.")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def embed_chunks(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for a list of document chunk texts.

        Intended for use during PDF ingestion.  Texts are encoded in a single
        batched call so the model processes them efficiently.

        Args:
            texts: Non-empty list of cleaned paragraph strings produced by
                the chunker.  Each string corresponds to one
                :class:`~backend.models.schemas.Chunk`.

        Returns:
            A 2-D ``numpy.ndarray`` of shape ``(len(texts), embedding_dim)``
            and dtype ``float32``.  The *i*-th row is the embedding for
            ``texts[i]``; order is preserved.

        Raises:
            ValueError: If *texts* is empty.
        """
        if not texts:
            raise ValueError(
                "embed_chunks requires at least one text. "
                "Pass a non-empty list of chunk strings."
            )

        embeddings: np.ndarray = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,   # unit-norm vectors → dot product == cosine similarity
        )
        return embeddings  # shape: (N, D)

    def embed_query(self, query: str) -> np.ndarray:
        """Generate an embedding for a single search query string.

        Intended for use at search time.  The query is encoded with the same
        model and normalisation as document chunks so scores are directly
        comparable.

        Args:
            query: The raw search query entered by the user.

        Returns:
            A 1-D ``numpy.ndarray`` of shape ``(embedding_dim,)`` and dtype
            ``float32``.

        Raises:
            ValueError: If *query* is empty or whitespace-only.
        """
        if not query or not query.strip():
            raise ValueError(
                "embed_query requires a non-empty query string."
            )

        embedding: np.ndarray = self._model.encode(
            query,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embedding  # shape: (D,)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the model's output vectors."""
        return self._model.get_sentence_embedding_dimension()
