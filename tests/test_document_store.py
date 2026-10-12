# test_document_store.py
# Unit tests for backend/storage/document_store.py
#
# Each test gets a fresh DocumentStore instance — no shared state between tests.

import numpy as np
import pytest

from backend.models.schemas import (
    Chunk,
    DocumentMetadata,
    StoredChunk,
    StoredDocument,
)
from backend.storage.document_store import DocumentNotFoundError, DocumentStore


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------

def _meta(doc_id: str = "doc-001", filename: str = "report.pdf",
          pages: int = 3, chunks: int = 5) -> DocumentMetadata:
    return DocumentMetadata(
        document_id=doc_id,
        filename=filename,
        page_count=pages,
        chunk_count=chunks,
    )


def _chunk(chunk_id: str, page: int = 1, text: str = "Some text.") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=chunk_id.split("-")[0],
        page_number=page,
        chunk_index=0,
        text=text,
    )


def _stored_chunk(chunk_id: str, page: int = 1,
                  text: str = "Some text.",
                  embedding: np.ndarray | None = None) -> StoredChunk:
    if embedding is None:
        embedding = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    return StoredChunk(chunk=_chunk(chunk_id, page, text), embedding=embedding)


def _doc(doc_id: str = "doc-001",
         stored_chunks: list[StoredChunk] | None = None) -> StoredDocument:
    if stored_chunks is None:
        stored_chunks = [_stored_chunk(f"{doc_id}-p1-c0")]
    return StoredDocument(
        metadata=_meta(doc_id, chunks=len(stored_chunks)),
        stored_chunks=stored_chunks,
    )


@pytest.fixture()
def store() -> DocumentStore:
    """Fresh empty DocumentStore for each test."""
    return DocumentStore()


# ---------------------------------------------------------------------------
# 1. Storing a document
# ---------------------------------------------------------------------------

class TestAddDocument:
    def test_add_increases_length(self, store):
        store.add(_doc("d1"))
        assert len(store) == 1

    def test_add_two_documents(self, store):
        store.add(_doc("d1"))
        store.add(_doc("d2"))
        assert len(store) == 2

    def test_empty_store_has_length_zero(self, store):
        assert len(store) == 0

    def test_add_returns_none(self, store):
        result = store.add(_doc("d1"))
        assert result is None


# ---------------------------------------------------------------------------
# 2. Retrieving a stored document
# ---------------------------------------------------------------------------

class TestGetDocument:
    def test_get_returns_stored_document(self, store):
        doc = _doc("d1")
        store.add(doc)
        assert store.get("d1") is doc

    def test_get_preserves_metadata_filename(self, store):
        store.add(StoredDocument(
            metadata=_meta("d1", filename="invoice.pdf"),
            stored_chunks=[_stored_chunk("d1-p1-c0")],
        ))
        assert store.get("d1").metadata.filename == "invoice.pdf"

    def test_get_preserves_metadata_page_count(self, store):
        store.add(StoredDocument(
            metadata=_meta("d1", pages=10),
            stored_chunks=[_stored_chunk("d1-p1-c0")],
        ))
        assert store.get("d1").metadata.page_count == 10

    def test_get_preserves_chunk_count(self, store):
        chunks = [_stored_chunk(f"d1-p1-c{i}") for i in range(4)]
        store.add(StoredDocument(metadata=_meta("d1", chunks=4), stored_chunks=chunks))
        assert len(store.get("d1").stored_chunks) == 4


# ---------------------------------------------------------------------------
# 3. Existence check
# ---------------------------------------------------------------------------

class TestExists:
    def test_exists_true_after_add(self, store):
        store.add(_doc("d1"))
        assert store.exists("d1") is True

    def test_exists_false_for_unknown_id(self, store):
        assert store.exists("ghost") is False

    def test_exists_false_after_delete(self, store):
        store.add(_doc("d1"))
        store.delete("d1")
        assert store.exists("d1") is False

    def test_contains_syntax(self, store):
        store.add(_doc("d1"))
        assert "d1" in store
        assert "ghost" not in store


# ---------------------------------------------------------------------------
# 4. Retrieving an unknown document
# ---------------------------------------------------------------------------

class TestGetUnknownDocument:
    def test_get_unknown_raises_document_not_found(self, store):
        with pytest.raises(DocumentNotFoundError):
            store.get("nonexistent")

    def test_error_is_subclass_of_key_error(self, store):
        with pytest.raises(KeyError):
            store.get("nonexistent")

    def test_error_message_contains_document_id(self, store):
        with pytest.raises(DocumentNotFoundError, match="missing-doc"):
            store.get("missing-doc")


# ---------------------------------------------------------------------------
# 5. Deleting a document
# ---------------------------------------------------------------------------

class TestDeleteDocument:
    def test_delete_reduces_length(self, store):
        store.add(_doc("d1"))
        store.delete("d1")
        assert len(store) == 0

    def test_delete_makes_document_inaccessible(self, store):
        store.add(_doc("d1"))
        store.delete("d1")
        with pytest.raises(DocumentNotFoundError):
            store.get("d1")

    def test_delete_removes_only_target_document(self, store):
        store.add(_doc("d1"))
        store.add(_doc("d2"))
        store.delete("d1")
        assert store.exists("d2")
        assert not store.exists("d1")

    def test_delete_returns_none(self, store):
        store.add(_doc("d1"))
        result = store.delete("d1")
        assert result is None


# ---------------------------------------------------------------------------
# 6. Deleting an unknown document
# ---------------------------------------------------------------------------

class TestDeleteUnknownDocument:
    def test_delete_unknown_raises_document_not_found(self, store):
        with pytest.raises(DocumentNotFoundError):
            store.delete("ghost")

    def test_delete_unknown_does_not_affect_other_documents(self, store):
        store.add(_doc("d1"))
        with pytest.raises(DocumentNotFoundError):
            store.delete("ghost")
        assert store.exists("d1")


# ---------------------------------------------------------------------------
# 7. Replacing an existing document
# ---------------------------------------------------------------------------

class TestReplaceDocument:
    def test_adding_same_id_replaces_document(self, store):
        store.add(_doc("d1"))
        new_doc = StoredDocument(
            metadata=_meta("d1", filename="updated.pdf"),
            stored_chunks=[_stored_chunk("d1-p1-c0")],
        )
        store.add(new_doc)
        assert store.get("d1").metadata.filename == "updated.pdf"

    def test_store_length_unchanged_after_replace(self, store):
        store.add(_doc("d1"))
        store.add(_doc("d1"))  # same id
        assert len(store) == 1

    def test_replaced_document_is_new_object(self, store):
        original = _doc("d1")
        replacement = _doc("d1")
        store.add(original)
        store.add(replacement)
        assert store.get("d1") is replacement
        assert store.get("d1") is not original

    def test_chunks_from_replaced_document_are_new_chunks(self, store):
        old_chunk = _stored_chunk("d1-p1-c0", text="old text")
        new_chunk = _stored_chunk("d1-p1-c0", text="new text")
        store.add(StoredDocument(metadata=_meta("d1"), stored_chunks=[old_chunk]))
        store.add(StoredDocument(metadata=_meta("d1"), stored_chunks=[new_chunk]))
        assert store.get_all_chunks("d1")[0].chunk.text == "new text"


# ---------------------------------------------------------------------------
# 8. Clearing all documents
# ---------------------------------------------------------------------------

class TestClear:
    def test_clear_empties_store(self, store):
        store.add(_doc("d1"))
        store.add(_doc("d2"))
        store.clear()
        assert len(store) == 0

    def test_clear_empty_store_does_not_raise(self, store):
        store.clear()   # should not raise
        assert len(store) == 0

    def test_documents_inaccessible_after_clear(self, store):
        store.add(_doc("d1"))
        store.clear()
        with pytest.raises(DocumentNotFoundError):
            store.get("d1")

    def test_can_add_after_clear(self, store):
        store.add(_doc("d1"))
        store.clear()
        store.add(_doc("d2"))
        assert len(store) == 1
        assert store.exists("d2")


# ---------------------------------------------------------------------------
# 9. Chunk data preservation
# ---------------------------------------------------------------------------

class TestChunkPreservation:
    def test_chunk_ids_preserved(self, store):
        sc = _stored_chunk("doc-p3-c1")
        store.add(StoredDocument(metadata=_meta("doc"), stored_chunks=[sc]))
        retrieved = store.get_all_chunks("doc")
        assert retrieved[0].chunk.chunk_id == "doc-p3-c1"

    def test_page_numbers_preserved(self, store):
        sc = _stored_chunk("doc-p7-c0", page=7)
        store.add(StoredDocument(metadata=_meta("doc"), stored_chunks=[sc]))
        retrieved = store.get_all_chunks("doc")
        assert retrieved[0].chunk.page_number == 7

    def test_chunk_text_preserved(self, store):
        text = "The invoice is due on Friday."
        sc = _stored_chunk("doc-p1-c0", text=text)
        store.add(StoredDocument(metadata=_meta("doc"), stored_chunks=[sc]))
        assert store.get_all_chunks("doc")[0].chunk.text == text

    def test_chunk_order_preserved(self, store):
        chunks = [_stored_chunk(f"doc-p1-c{i}", text=f"chunk {i}") for i in range(5)]
        store.add(StoredDocument(metadata=_meta("doc", chunks=5), stored_chunks=chunks))
        retrieved = store.get_all_chunks("doc")
        for i, sc in enumerate(retrieved):
            assert sc.chunk.text == f"chunk {i}"


# ---------------------------------------------------------------------------
# 10. Embedding preservation — exact array identity, no recalculation
# ---------------------------------------------------------------------------

class TestEmbeddingPreservation:
    def test_embedding_array_is_identical_object(self, store):
        """The stored embedding must be the exact same ndarray object."""
        emb = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        sc = StoredChunk(chunk=_chunk("doc-p1-c0"), embedding=emb)
        store.add(StoredDocument(metadata=_meta("doc"), stored_chunks=[sc]))
        retrieved_emb = store.get_all_chunks("doc")[0].embedding
        assert retrieved_emb is emb

    def test_embedding_values_are_unchanged(self, store):
        emb = np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32)
        sc = StoredChunk(chunk=_chunk("doc-p1-c0"), embedding=emb)
        store.add(StoredDocument(metadata=_meta("doc"), stored_chunks=[sc]))
        retrieved_emb = store.get_all_chunks("doc")[0].embedding
        np.testing.assert_array_equal(retrieved_emb, emb)

    def test_embedding_dtype_preserved(self, store):
        emb = np.zeros(8, dtype=np.float32)
        sc = StoredChunk(chunk=_chunk("doc-p1-c0"), embedding=emb)
        store.add(StoredDocument(metadata=_meta("doc"), stored_chunks=[sc]))
        assert store.get_all_chunks("doc")[0].embedding.dtype == np.float32

    def test_multiple_embeddings_each_preserved(self, store):
        embs = [np.full(4, float(i), dtype=np.float32) for i in range(3)]
        chunks = [
            StoredChunk(chunk=_chunk(f"doc-p1-c{i}"), embedding=embs[i])
            for i in range(3)
        ]
        store.add(StoredDocument(metadata=_meta("doc", chunks=3), stored_chunks=chunks))
        retrieved = store.get_all_chunks("doc")
        for i, sc in enumerate(retrieved):
            assert sc.embedding is embs[i]


# ---------------------------------------------------------------------------
# 11. list_documents
# ---------------------------------------------------------------------------

class TestListDocuments:
    def test_list_empty_store(self, store):
        assert store.list_documents() == []

    def test_list_returns_one_metadata_per_document(self, store):
        store.add(_doc("d1"))
        store.add(_doc("d2"))
        assert len(store.list_documents()) == 2

    def test_list_metadata_contains_correct_ids(self, store):
        store.add(_doc("alpha"))
        store.add(_doc("beta"))
        ids = {m.document_id for m in store.list_documents()}
        assert ids == {"alpha", "beta"}

    def test_list_returns_document_metadata_instances(self, store):
        store.add(_doc("d1"))
        for item in store.list_documents():
            assert isinstance(item, DocumentMetadata)


# ---------------------------------------------------------------------------
# 12. get_all_chunks
# ---------------------------------------------------------------------------

class TestGetAllChunks:
    def test_returns_all_chunks(self, store):
        chunks = [_stored_chunk(f"doc-p1-c{i}") for i in range(6)]
        store.add(StoredDocument(metadata=_meta("doc", chunks=6), stored_chunks=chunks))
        assert len(store.get_all_chunks("doc")) == 6

    def test_raises_for_unknown_document(self, store):
        with pytest.raises(DocumentNotFoundError):
            store.get_all_chunks("nonexistent")

    def test_returns_empty_list_for_document_with_no_chunks(self, store):
        store.add(StoredDocument(metadata=_meta("doc", chunks=0), stored_chunks=[]))
        assert store.get_all_chunks("doc") == []
