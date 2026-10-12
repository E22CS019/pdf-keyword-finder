# test_upload.py
# API tests for POST /upload in backend/main.py
#
# Patching strategy
# -----------------
# * backend.main.Embedder is replaced with a lightweight mock so the
#   sentence-transformer model is never downloaded.  The mock's embed_chunks
#   returns a (N, 4) float32 array of zeros — enough for the pipeline to run.
# * backend.main.extract_pages is replaced in most tests so we control the
#   PDF content without needing real PDF files.
# * One integration-style fixture (_real_pdf_bytes) builds a genuine minimal
#   PDF in memory using fitz to prove the full pipeline wires together.
#
# Each test class that hits the app receives a fresh TestClient (and therefore
# a fresh DocumentStore) via a dedicated fixture — no state bleeds between tests.

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import fitz
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.models.schemas import PageText
from backend.main import MAX_FILE_BYTES, MAX_PAGES, PDF_MAGIC


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_mock_embedder(dim: int = 4) -> MagicMock:
    """Return a mock Embedder whose embed_chunks returns zero vectors."""
    mock = MagicMock()
    def _encode(texts, **_kw):
        return np.zeros((len(texts), dim), dtype=np.float32)
    mock.embed_chunks.side_effect = _encode
    return mock


def _real_pdf_bytes(text: str = "Invoice total is due on Friday.\n\nPlease pay promptly.") -> bytes:
    """Build a minimal single-page PDF in memory using fitz."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    buf = doc.tobytes()
    doc.close()
    return buf


def _fake_pages(texts: list[str]) -> list[PageText]:
    """Build a PageText list from plain strings."""
    return [PageText(page_number=i + 1, text=t) for i, t in enumerate(texts)]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    """TestClient backed by a mock Embedder.

    extract_pages is NOT patched here — individual tests patch it themselves
    when they need to control PDF content.
    """
    mock_embedder = _make_mock_embedder()
    with patch("backend.main.Embedder", return_value=mock_embedder):
        from backend.main import app
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


def _upload(client: TestClient,
            content: bytes,
            filename: str = "test.pdf",
            content_type: str = "application/pdf"):
    """POST to /upload with the given bytes."""
    return client.post(
        "/upload",
        files={"file": (filename, io.BytesIO(content), content_type)},
    )


# ---------------------------------------------------------------------------
# 1. Valid PDF upload — happy path (real fitz PDF + mocked embedder)
# ---------------------------------------------------------------------------

class TestValidUpload:
    def test_returns_201(self, client):
        pdf = _real_pdf_bytes()
        response = _upload(client, pdf)
        assert response.status_code == 201

    def test_response_contains_document_id(self, client):
        pdf = _real_pdf_bytes()
        response = _upload(client, pdf)
        body = response.json()
        assert "document_id" in body
        assert len(body["document_id"]) > 0

    def test_response_contains_filename(self, client):
        pdf = _real_pdf_bytes()
        response = _upload(client, pdf, filename="my_report.pdf")
        assert response.json()["filename"] == "my_report.pdf"

    def test_response_page_count_is_positive(self, client):
        pdf = _real_pdf_bytes()
        response = _upload(client, pdf)
        assert response.json()["page_count"] >= 1

    def test_response_chunk_count_is_positive(self, client):
        pdf = _real_pdf_bytes()
        response = _upload(client, pdf)
        assert response.json()["chunk_count"] >= 1

    def test_response_schema_has_all_required_fields(self, client):
        pdf = _real_pdf_bytes()
        body = _upload(client, pdf).json()
        for field in ("document_id", "filename", "page_count", "chunk_count"):
            assert field in body

    def test_document_id_is_unique_per_upload(self, client):
        pdf = _real_pdf_bytes()
        id1 = _upload(client, pdf).json()["document_id"]
        id2 = _upload(client, pdf).json()["document_id"]
        assert id1 != id2


# ---------------------------------------------------------------------------
# 2. Document stored in DocumentStore after upload
# ---------------------------------------------------------------------------

class TestDocumentStorage:
    def test_document_is_stored_after_upload(self, client):
        """After a successful upload the document must exist in the store."""
        pdf = _real_pdf_bytes()
        doc_id = _upload(client, pdf).json()["document_id"]
        # Access the store via app.state through the test client
        from backend.main import app
        assert app.state.document_store.exists(doc_id)

    def test_stored_chunk_count_matches_response(self, client):
        pdf = _real_pdf_bytes()
        body = _upload(client, pdf).json()
        doc_id = body["document_id"]
        from backend.main import app
        stored = app.state.document_store.get(doc_id)
        assert len(stored.stored_chunks) == body["chunk_count"]

    def test_stored_filename_matches_response(self, client):
        pdf = _real_pdf_bytes()
        body = _upload(client, pdf, filename="invoice.pdf").json()
        from backend.main import app
        stored = app.state.document_store.get(body["document_id"])
        assert stored.metadata.filename == "invoice.pdf"


# ---------------------------------------------------------------------------
# 3. Non-PDF rejection
# ---------------------------------------------------------------------------

class TestNonPdfRejection:
    def test_text_file_rejected(self, client):
        response = _upload(client, b"Hello world", content_type="text/plain")
        assert response.status_code == 400

    def test_image_file_rejected(self, client):
        response = _upload(client, b"\x89PNG\r\n", content_type="image/png")
        assert response.status_code == 400

    def test_rejection_has_error_detail(self, client):
        response = _upload(client, b"not a pdf", content_type="text/plain")
        assert "detail" in response.json()

    def test_wrong_magic_bytes_rejected_even_with_pdf_content_type(self, client):
        # Content-type says PDF but bytes are not — magic-byte check must fire.
        response = _upload(client, b"NOTPDF content here", content_type="application/pdf")
        assert response.status_code == 400

    def test_rejection_message_mentions_pdf(self, client):
        response = _upload(client, b"plain text", content_type="text/plain")
        detail = response.json()["detail"].lower()
        assert "pdf" in detail


# ---------------------------------------------------------------------------
# 4. Empty file rejection
# ---------------------------------------------------------------------------

class TestEmptyFileRejection:
    def test_empty_bytes_rejected(self, client):
        response = _upload(client, b"", content_type="application/pdf")
        assert response.status_code == 400

    def test_empty_file_error_detail_present(self, client):
        response = _upload(client, b"", content_type="application/pdf")
        assert "detail" in response.json()

    def test_empty_file_message_mentions_empty(self, client):
        response = _upload(client, b"", content_type="application/pdf")
        assert "empty" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 5. Oversized file rejection
# ---------------------------------------------------------------------------

class TestOversizedFileRejection:
    def test_file_just_over_limit_rejected(self, client):
        # Construct bytes that start with %PDF so magic check passes,
        # but are larger than MAX_FILE_BYTES.
        oversized = PDF_MAGIC + b"x" * (MAX_FILE_BYTES + 1)
        response = _upload(client, oversized, content_type="application/pdf")
        assert response.status_code == 413

    def test_oversized_error_detail_present(self, client):
        oversized = PDF_MAGIC + b"x" * (MAX_FILE_BYTES + 1)
        response = _upload(client, oversized, content_type="application/pdf")
        assert "detail" in response.json()

    def test_file_exactly_at_limit_passes_size_check(self, client):
        # A file exactly at the limit should not be rejected for size.
        # It will fail later (not a real PDF), but NOT with 413.
        at_limit = PDF_MAGIC + b"x" * (MAX_FILE_BYTES - len(PDF_MAGIC))
        response = _upload(client, at_limit, content_type="application/pdf")
        assert response.status_code != 413


# ---------------------------------------------------------------------------
# 6. Page limit rejection
# ---------------------------------------------------------------------------

class TestPageLimitRejection:
    def test_pdf_over_page_limit_rejected(self, client):
        too_many_pages = _fake_pages(["text"] * (MAX_PAGES + 1))
        with patch("backend.main.extract_pages", return_value=too_many_pages):
            pdf = PDF_MAGIC + b"valid-looking-content"
            response = _upload(client, pdf, content_type="application/pdf")
        assert response.status_code == 413

    def test_page_limit_error_mentions_page_count(self, client):
        too_many_pages = _fake_pages(["text"] * (MAX_PAGES + 1))
        with patch("backend.main.extract_pages", return_value=too_many_pages):
            pdf = PDF_MAGIC + b"content"
            response = _upload(client, pdf, content_type="application/pdf")
        detail = response.json()["detail"]
        assert str(MAX_PAGES + 1) in detail or str(MAX_PAGES) in detail

    def test_pdf_exactly_at_page_limit_accepted(self, client):
        exactly = _fake_pages(["Invoice paragraph text here."] * MAX_PAGES)
        with patch("backend.main.extract_pages", return_value=exactly):
            pdf = PDF_MAGIC + b"content"
            response = _upload(client, pdf, content_type="application/pdf")
        assert response.status_code != 413


# ---------------------------------------------------------------------------
# 7. Corrupt / invalid PDF rejection
# ---------------------------------------------------------------------------

class TestCorruptPdfRejection:
    def test_random_bytes_rejected(self, client):
        # Starts with %PDF so magic bytes pass, but fitz will reject it.
        corrupt = PDF_MAGIC + b"\x00\x01\x02\x03garbage"
        response = _upload(client, corrupt, content_type="application/pdf")
        assert response.status_code == 400

    def test_corrupt_pdf_error_detail_present(self, client):
        corrupt = PDF_MAGIC + b"garbage data that fitz cannot parse"
        response = _upload(client, corrupt, content_type="application/pdf")
        assert "detail" in response.json()

    def test_extract_pages_exception_becomes_400(self, client):
        """Any unexpected exception from extract_pages → 400."""
        with patch("backend.main.extract_pages", side_effect=RuntimeError("boom")):
            pdf = PDF_MAGIC + b"content"
            response = _upload(client, pdf, content_type="application/pdf")
        assert response.status_code == 400


# ---------------------------------------------------------------------------
# 8. No extractable text
# ---------------------------------------------------------------------------

class TestNoExtractableText:
    def test_image_only_pdf_rejected(self, client):
        """All pages return empty text → 400, not a silent empty document."""
        blank_pages = _fake_pages(["", "   ", "\n\n"])
        with patch("backend.main.extract_pages", return_value=blank_pages):
            pdf = PDF_MAGIC + b"content"
            response = _upload(client, pdf, content_type="application/pdf")
        assert response.status_code == 400

    def test_no_text_error_mentions_extractable(self, client):
        blank_pages = _fake_pages(["", ""])
        with patch("backend.main.extract_pages", return_value=blank_pages):
            pdf = PDF_MAGIC + b"content"
            response = _upload(client, pdf, content_type="application/pdf")
        detail = response.json()["detail"].lower()
        assert "text" in detail or "extract" in detail

    def test_no_chunks_means_document_not_stored(self, client):
        """A failed ingestion must not leave a partial document in the store."""
        blank_pages = _fake_pages([""])
        with patch("backend.main.extract_pages", return_value=blank_pages):
            pdf = PDF_MAGIC + b"content"
            _upload(client, pdf, content_type="application/pdf")
        from backend.main import app
        assert len(app.state.document_store) == 0


# ---------------------------------------------------------------------------
# 9. Response metadata accuracy (mocked pipeline)
# ---------------------------------------------------------------------------

class TestResponseMetadata:
    def test_page_count_matches_extracted_pages(self, client):
        pages = _fake_pages(["Paragraph one text here.", "Paragraph two text here."])
        with patch("backend.main.extract_pages", return_value=pages):
            pdf = PDF_MAGIC + b"content"
            body = _upload(client, pdf, content_type="application/pdf").json()
        assert body["page_count"] == 2

    def test_chunk_count_reflects_paragraphs(self, client):
        # Two pages, one paragraph each → expect 2 chunks
        pages = _fake_pages([
            "This is the first paragraph on page one.",
            "This is the second paragraph on page two.",
        ])
        with patch("backend.main.extract_pages", return_value=pages):
            pdf = PDF_MAGIC + b"content"
            body = _upload(client, pdf, content_type="application/pdf").json()
        assert body["chunk_count"] == 2

    def test_document_id_is_uuid_format(self, client):
        import uuid
        pdf = _real_pdf_bytes()
        doc_id = _upload(client, pdf).json()["document_id"]
        # Must parse as a valid UUID without raising
        uuid.UUID(doc_id)

    def test_embed_chunks_called_with_chunk_texts(self, client):
        """embed_chunks must be called exactly once with all chunk texts."""
        pages = _fake_pages(["First chunk text here.", "Second chunk text here."])
        with patch("backend.main.extract_pages", return_value=pages):
            pdf = PDF_MAGIC + b"content"
            _upload(client, pdf, content_type="application/pdf")
        from backend.main import app
        embedder = app.state.embedder
        embedder.embed_chunks.assert_called_once()
        texts_arg = embedder.embed_chunks.call_args[0][0]
        assert len(texts_arg) == 2
