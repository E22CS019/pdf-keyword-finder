# pdf_processor.py
# Responsibilities:
#   - Accept an uploaded PDF file (binary / file-like object)
#   - Extract text content on a page-by-page basis using PyMuPDF (fitz)
#   - Return a structured list of PageText objects preserving page numbers
#   - Handle encrypted, malformed, image-only PDFs, and empty pages gracefully

from __future__ import annotations

import logging
from typing import Union

import fitz  # PyMuPDF

from backend.models.schemas import PageText

logger = logging.getLogger(__name__)


def extract_pages(pdf_source: Union[bytes, str]) -> list[PageText]:
    """Extract raw text from every page of a PDF, preserving page numbers.

    Accepts either the raw binary content of a PDF (e.g. read from an uploaded
    file) or a filesystem path to a PDF file.  Text is returned as-is with no
    cleaning, normalisation, or chunking applied — those steps belong to
    downstream modules.

    Args:
        pdf_source: PDF content as a ``bytes`` object, or a path string to a
            PDF file on disk.

    Returns:
        A list of :class:`~backend.models.schemas.PageText` objects, one per
        page, in page order.  Pages that yield no text (e.g. scanned images,
        blank pages) are included with an empty string so that the caller
        always receives an entry for every physical page.  Returns an empty
        list if the document contains no pages.

    Raises:
        ValueError: If the PDF is encrypted and cannot be opened without a
            password.
        fitz.FileDataError: If the binary data or file path does not represent
            a valid PDF.
    """
    # ------------------------------------------------------------------
    # 1. Open the document
    # ------------------------------------------------------------------
    try:
        if isinstance(pdf_source, bytes):
            doc: fitz.Document = fitz.open(stream=pdf_source, filetype="pdf")
        else:
            doc = fitz.open(pdf_source)
    except fitz.FileDataError as exc:
        logger.error("Failed to open PDF: %s", exc)
        raise

    # ------------------------------------------------------------------
    # 2. Guard: encrypted documents
    # ------------------------------------------------------------------
    if doc.is_encrypted:
        doc.close()
        raise ValueError(
            "The PDF is encrypted. Provide a decrypted file or supply a password."
        )

    # ------------------------------------------------------------------
    # 3. Guard: empty document
    # ------------------------------------------------------------------
    if doc.page_count == 0:
        logger.warning("PDF has no pages; returning empty list.")
        doc.close()
        return []

    # ------------------------------------------------------------------
    # 4. Page-by-page extraction
    # ------------------------------------------------------------------
    pages: list[PageText] = []

    for index in range(doc.page_count):
        page: fitz.Page = doc.load_page(index)

        # get_text("text") returns a plain-text representation of the page.
        # It returns an empty string for image-only pages — no exception is raised.
        raw_text: str = page.get_text("text")

        if not raw_text.strip():
            logger.debug("Page %d has no extractable text (blank or image-only).", index + 1)

        pages.append(
            PageText(
                page_number=index + 1,  # convert 0-based index to 1-based page number
                text=raw_text,
            )
        )

    doc.close()
    return pages
