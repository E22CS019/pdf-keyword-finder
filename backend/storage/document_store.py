# document_store.py
# Responsibilities:
#   - Hold processed documents in memory, keyed by document_id
#   - Support add (with deterministic replace), get, exists, delete, clear, list
#   - Provide all StoredChunks for a document so the search layer can operate on them
#   - Store no query-specific information (keywords, scores, match types)
#   - Remain independent from FastAPI, embedders, and search modules

from __future__ import annotations

import logging
from typing import Optional

from backend.models.schemas import DocumentMetadata, StoredDocument, StoredChunk

logger = logging.getLogger(__name__)


class DocumentNotFoundError(KeyError):
    """Raised when a requested document_id does not exist in the store."""


class DocumentStore:
    """In-memory store for processed PDF documents.

    Each document is recorded as a :class:`~backend.models.schemas.StoredDocument`
    containing its :class:`~backend.models.schemas.DocumentMetadata` and the
    full list of :class:`~backend.models.schemas.StoredChunk` objects produced
    by the ingestion pipeline.

    **What is stored**

    - Document-level facts: ``document_id``, ``filename``, page count, chunk count.
    - Per-chunk data: the :class:`~backend.models.schemas.Chunk` (id, page,
      text) and its pre-computed embedding vector.

    **What is NOT stored**

    Query-specific information — keywords, match types, semantic scores — is
    never persisted here.  Those values are computed fresh for each search
    request and live only in the request/response cycle.

    **Singleton usage**

    This class has no hidden module-level state.  The application creates one
    instance at startup (in ``main.py``) and passes it to service functions via
    dependency injection.  Tests create their own isolated instances.

    **Replacement path**

    The public interface (``add``, ``get``, ``exists``, ``delete``, ``clear``,
    ``list_documents``, ``get_all_chunks``) is the only surface the rest of
    the codebase touches.  Replacing this class with a PostgreSQL-backed or
    vector-database-backed implementation requires changing only this file and
    the dependency injection in ``main.py``.
    """

    def __init__(self) -> None:
        # Primary storage: document_id → StoredDocument
        self._store: dict[str, StoredDocument] = {}

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add(self, document: StoredDocument) -> None:
        """Store a processed document.

        If a document with the same ``document_id`` already exists it is
        replaced entirely.  This is the deterministic, side-effect-free
        behaviour required by re-upload scenarios — no silent duplicates
        are ever created.

        Args:
            document: The :class:`~backend.models.schemas.StoredDocument` to
                store.  Embeddings and chunk data are stored as-is; no
                normalisation or recalculation is performed.
        """
        doc_id = document.metadata.document_id
        replaced = doc_id in self._store
        self._store[doc_id] = document
        if replaced:
            logger.info("DocumentStore: replaced existing document '%s'.", doc_id)
        else:
            logger.info(
                "DocumentStore: added document '%s' (%d chunks).",
                doc_id,
                len(document.stored_chunks),
            )

    def delete(self, document_id: str) -> None:
        """Remove a document from the store.

        Args:
            document_id: The ID of the document to remove.

        Raises:
            DocumentNotFoundError: If *document_id* is not in the store.
        """
        if document_id not in self._store:
            raise DocumentNotFoundError(
                f"Cannot delete: document '{document_id}' not found in the store."
            )
        del self._store[document_id]
        logger.info("DocumentStore: deleted document '%s'.", document_id)

    def clear(self) -> None:
        """Remove all documents from the store.

        After this call ``len(self) == 0``.  Primarily useful in tests and
        admin reset scenarios.
        """
        count = len(self._store)
        self._store.clear()
        logger.info("DocumentStore: cleared %d document(s).", count)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get(self, document_id: str) -> StoredDocument:
        """Retrieve a stored document by its ID.

        Args:
            document_id: The ID of the document to retrieve.

        Returns:
            The :class:`~backend.models.schemas.StoredDocument` for
            *document_id*.

        Raises:
            DocumentNotFoundError: If *document_id* is not in the store.
        """
        try:
            return self._store[document_id]
        except KeyError:
            raise DocumentNotFoundError(
                f"Document '{document_id}' not found in the store."
            )

    def exists(self, document_id: str) -> bool:
        """Return ``True`` if *document_id* is present in the store.

        Args:
            document_id: The ID to check.

        Returns:
            ``True`` if the document exists, ``False`` otherwise.
        """
        return document_id in self._store

    def list_documents(self) -> list[DocumentMetadata]:
        """Return metadata for all stored documents.

        Returns:
            A list of :class:`~backend.models.schemas.DocumentMetadata`
            objects, one per stored document, in insertion order.  Returns
            an empty list when the store is empty.
        """
        return [doc.metadata for doc in self._store.values()]

    def get_all_chunks(self, document_id: str) -> list[StoredChunk]:
        """Return all stored chunks for a document.

        This is the primary access point for the search layer: both
        :mod:`~backend.services.lexical_search` and
        :mod:`~backend.services.semantic_search` receive this list and
        operate on it independently.

        Args:
            document_id: The ID of the document whose chunks are needed.

        Returns:
            A list of :class:`~backend.models.schemas.StoredChunk` objects
            in the order they were ingested.

        Raises:
            DocumentNotFoundError: If *document_id* is not in the store.
        """
        return self.get(document_id).stored_chunks

    # ------------------------------------------------------------------
    # Convenience / dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of documents currently in the store."""
        return len(self._store)

    def __contains__(self, document_id: object) -> bool:
        """Support ``document_id in store`` syntax."""
        return document_id in self._store

    def __repr__(self) -> str:  # pragma: no cover
        return f"DocumentStore(documents={len(self._store)})"
