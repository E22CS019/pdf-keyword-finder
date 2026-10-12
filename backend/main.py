# main.py
# Entry point for the FastAPI application.
# Responsibilities:
#   - Initialise shared resources (Embedder, DocumentStore) once via lifespan
#   - Expose those resources through FastAPI dependency injection
#   - Register API endpoints (POST /upload; /search added later)
#   - Orchestrate the service pipeline for each upload request
#   - Convert service-layer exceptions into user-friendly HTTP errors

from __future__ import annotations

import logging
import re
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from backend.models.schemas import (
    Chunk,
    ChunkResult,
    DocumentMetadata,
    KeywordSummary,
    MergedResult,
    SearchRequest,
    SearchResponse,
    StoredChunk,
    StoredDocument,
    UploadResponse,
)
from backend.services.chunker import chunk_page
from backend.services.embedder import Embedder
from backend.services.lexical_search import lexical_search
from backend.services.pdf_processor import extract_pages
from backend.services.result_merger import merge_results
from backend.services.semantic_search import semantic_search
from backend.services.text_cleaner import clean_page_text
from backend.storage.document_store import DocumentNotFoundError, DocumentStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Upload constraints
# ---------------------------------------------------------------------------
MAX_FILE_BYTES: int = 20 * 1024 * 1024   # 20 MB
MAX_PAGES: int = 50
PDF_MAGIC: bytes = b"%PDF"               # first 4 bytes of every valid PDF


# ---------------------------------------------------------------------------
# Lifespan — create shared resources once, tear them down on shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the Embedder and DocumentStore at startup.

    Both objects are attached to ``app.state`` so they survive the entire
    lifetime of the process and are reused across every request.
    """
    logger.info("Starting up: loading embedding model…")
    app.state.embedder = Embedder()
    app.state.document_store = DocumentStore()
    logger.info("Startup complete.")
    yield
    logger.info("Shutting down.")


# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="PDF Keyword Finder",
    description="Upload a PDF and search for keywords using hybrid lexical + semantic retrieval.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tightened in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Dependency providers
# ---------------------------------------------------------------------------

def get_embedder(request: Request) -> Embedder:
    """Provide the application-wide Embedder instance."""
    return request.app.state.embedder


def get_document_store(request: Request) -> DocumentStore:
    """Provide the application-wide DocumentStore instance."""
    return request.app.state.document_store


EmbedderDep = Annotated[Embedder, Depends(get_embedder)]
StoreDep = Annotated[DocumentStore, Depends(get_document_store)]


# ---------------------------------------------------------------------------
# Helper — ingest pipeline (pure function, easily unit-tested)
# ---------------------------------------------------------------------------

def _ingest(
    pdf_bytes: bytes,
    filename: str,
    document_id: str,
    embedder: Embedder,
) -> StoredDocument:
    """Run the full ingestion pipeline for one PDF.

    Pipeline:
        bytes → extract_pages → clean_page_text → chunk_page
              → embed_chunks → StoredChunk → StoredDocument

    Args:
        pdf_bytes: Raw PDF content.
        filename:  Original filename for metadata.
        document_id: Pre-generated unique ID for this document.
        embedder:  Shared :class:`~backend.services.embedder.Embedder` instance.

    Returns:
        A fully populated :class:`~backend.models.schemas.StoredDocument`.

    Raises:
        HTTPException 400: Encrypted, corrupt, or image-only PDF.
        HTTPException 413: Page count exceeds ``MAX_PAGES``.
    """
    # --- 1. PDF extraction ---
    try:
        pages = extract_pages(pdf_bytes)
    except ValueError as exc:
        # Encrypted PDF
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except Exception as exc:
        # Corrupt / not a real PDF (fitz.FileDataError and others)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not read PDF: {exc}",
        )

    page_count = len(pages)

    # --- 2. Page-count guard ---
    if page_count > MAX_PAGES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"PDF has {page_count} pages; the maximum allowed is {MAX_PAGES}."
            ),
        )

    # --- 3. Clean → chunk, accumulating all chunks across pages ---
    all_chunks: list[Chunk] = []
    for page in pages:
        cleaned = clean_page_text(page.text)
        if not cleaned:
            continue   # blank / image-only page — skip silently
        page_chunks = chunk_page(
            cleaned_text=cleaned,
            page_number=page.page_number,
            doc_id=document_id,
        )
        all_chunks.extend(page_chunks)

    # --- 4. No-text guard ---
    if not all_chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "No text could be extracted from this PDF. "
                "It may be a scanned image without OCR or contain only non-text content."
            ),
        )

    # --- 5. Batch embedding ---
    texts = [chunk.text for chunk in all_chunks]
    embedding_matrix = embedder.embed_chunks(texts)   # shape (N, D)

    # --- 6. Build StoredChunk list ---
    stored_chunks: list[StoredChunk] = [
        StoredChunk(chunk=chunk, embedding=embedding_matrix[i])
        for i, chunk in enumerate(all_chunks)
    ]

    # --- 7. Assemble StoredDocument ---
    metadata = DocumentMetadata(
        document_id=document_id,
        filename=filename,
        page_count=page_count,
        chunk_count=len(stored_chunks),
    )
    return StoredDocument(metadata=metadata, stored_chunks=stored_chunks)


# ---------------------------------------------------------------------------
# POST /upload
# ---------------------------------------------------------------------------

@app.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index a PDF document",
    description=(
        "Accepts a PDF file, extracts and cleans its text, splits it into "
        "paragraph-level chunks, generates embeddings, and stores the result "
        "for subsequent keyword search."
    ),
)
async def upload_pdf(
    file: UploadFile,
    embedder: EmbedderDep,
    store: StoreDep,
) -> UploadResponse:
    """Validate, ingest, and store an uploaded PDF.

    Validation (in order):
    1. Content-type header must indicate ``application/pdf``.
    2. File must not be empty.
    3. File bytes must not exceed ``MAX_FILE_BYTES`` (20 MB).
    4. First four bytes must be the PDF magic bytes ``%PDF``.
    5. Page count must not exceed ``MAX_PAGES`` (50).
    6. At least one chunk of extractable text must exist.

    Args:
        file:     The uploaded file from the multipart form.
        embedder: Injected shared :class:`~backend.services.embedder.Embedder`.
        store:    Injected shared :class:`~backend.storage.document_store.DocumentStore`.

    Returns:
        :class:`~backend.models.schemas.UploadResponse` with document metadata.

    Raises:
        HTTPException 400: Non-PDF, empty, corrupt, encrypted, or image-only file.
        HTTPException 413: File exceeds 20 MB or PDF exceeds 50 pages.
    """
    # ------------------------------------------------------------------
    # Validation 1 — content-type header
    # ------------------------------------------------------------------
    content_type = (file.content_type or "").lower()
    if "pdf" not in content_type:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Invalid file type '{file.content_type}'. "
                "Only PDF files are accepted."
            ),
        )

    # ------------------------------------------------------------------
    # Validation 2 & 3 — read bytes, check emptiness and size
    # ------------------------------------------------------------------
    pdf_bytes: bytes = await file.read()

    if len(pdf_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty.",
        )

    if len(pdf_bytes) > MAX_FILE_BYTES:
        mb = len(pdf_bytes) / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File size {mb:.1f} MB exceeds the 20 MB limit."
            ),
        )

    # ------------------------------------------------------------------
    # Validation 4 — magic-byte check (content-based, not filename-only)
    # ------------------------------------------------------------------
    if not pdf_bytes.startswith(PDF_MAGIC):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File does not appear to be a valid PDF (missing PDF header).",
        )

    # ------------------------------------------------------------------
    # Ingestion — orchestrate service modules
    # ------------------------------------------------------------------
    document_id = str(uuid.uuid4())
    filename = file.filename or "upload.pdf"

    stored_doc = _ingest(
        pdf_bytes=pdf_bytes,
        filename=filename,
        document_id=document_id,
        embedder=embedder,
    )

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    store.add(stored_doc)
    logger.info(
        "Stored document '%s' ('%s'): %d pages, %d chunks.",
        document_id,
        filename,
        stored_doc.metadata.page_count,
        stored_doc.metadata.chunk_count,
    )

    return UploadResponse(
        document_id=stored_doc.metadata.document_id,
        filename=stored_doc.metadata.filename,
        page_count=stored_doc.metadata.page_count,
        chunk_count=stored_doc.metadata.chunk_count,
    )


# ---------------------------------------------------------------------------
# Helper — snippet and match-position extraction
# ---------------------------------------------------------------------------

# Half-width of the snippet window in characters.
_SNIPPET_HALF: int = 100   # window = up to 200 chars total


def _build_snippet(
    text: str,
    keyword: str,
    match_type: str,
) -> tuple[str, list[tuple[int, int]]]:
    """Extract a short snippet from *text* and return match positions relative to it.

    **For exact and variant matches** a window of up to ``2 * _SNIPPET_HALF``
    characters is centred on the start of the first keyword occurrence in
    *text*.  All match spans (exact form OR any variant form) that fall
    entirely within that window are converted to snippet-relative offsets and
    returned.

    **For semantic-only matches** the snippet is the first
    ``2 * _SNIPPET_HALF`` characters of *text* and the positions list is
    empty, because there is no literal keyword position to report.

    The function intentionally uses only the standard ``re`` module and the
    same word-boundary + case-insensitive approach used by ``lexical_search``
    — no service module is imported.

    Args:
        text: The full cleaned paragraph text of the matched chunk.
        keyword: The original user keyword (un-normalised).
        match_type: ``"exact"``, ``"variant"``, or ``"semantic"``.

    Returns:
        ``(snippet, positions)`` where ``positions`` is a list of
        ``(start, end)`` character offsets **relative to the snippet**.
    """
    max_len = _SNIPPET_HALF * 2

    if match_type == "semantic":
        snippet = text[:max_len]
        return snippet, []

    # Build the exact-form pattern.
    norm_kw = " ".join(keyword.lower().split())
    tokens = [re.escape(t) for t in norm_kw.split()]
    exact_pat = re.compile(
        r"\b" + r"\s+".join(tokens) + r"\b", re.IGNORECASE
    )

    # Build variant patterns — same logic as lexical_search._generate_variants
    # but inlined here so we never import from the service layer.
    _VARIANT_RULES_LOCAL = [
        (r"(?:s|x|z|ch|sh)$", 0, "es"),
        ("", 0, "s"),
        (r"e$", 0, "d"),
        (r"[^aeiou]y$", 1, "ied"),
        ("", 0, "ed"),
        (r"e$", 1, "ing"),
        ("", 0, "ing"),
    ]

    variant_pats: list[re.Pattern[str]] = []
    if norm_kw.isalpha() and " " not in norm_kw:
        seen: set[str] = {norm_kw}
        for condition, drop, suffix in _VARIANT_RULES_LOCAL:
            if condition and not re.search(condition, norm_kw):
                continue
            root = norm_kw[: len(norm_kw) - drop] if drop else norm_kw
            candidate = root + suffix
            if candidate not in seen:
                seen.add(candidate)
                variant_pats.append(
                    re.compile(r"\b" + re.escape(candidate) + r"\b", re.IGNORECASE)
                )

    # Collect all match spans from exact pattern (and variant patterns for
    # variant match_type) across the full text.
    all_spans: list[tuple[int, int]] = [
        (m.start(), m.end()) for m in exact_pat.finditer(text)
    ]
    if match_type == "variant":
        for vpat in variant_pats:
            for m in vpat.finditer(text):
                all_spans.append((m.start(), m.end()))
        all_spans.sort(key=lambda s: s[0])

    if not all_spans:
        # Defensive: no spans found despite a lexical match_type — fall back
        # to the head of the text.
        snippet = text[:max_len]
        return snippet, []

    # Centre the window on the first match.
    first_start = all_spans[0][0]
    win_start = max(0, first_start - _SNIPPET_HALF)
    win_end = win_start + max_len
    # If window extends past the text, slide it back so it stays within bounds.
    if win_end > len(text):
        win_end = len(text)
        win_start = max(0, win_end - max_len)

    snippet = text[win_start:win_end]

    # Convert absolute spans to snippet-relative, keeping only spans that
    # fall entirely within the window.
    positions: list[tuple[int, int]] = []
    for abs_start, abs_end in all_spans:
        if abs_start >= win_start and abs_end <= win_end:
            positions.append((abs_start - win_start, abs_end - win_start))

    return snippet, positions


# ---------------------------------------------------------------------------
# Helper — per-keyword search pipeline (pure function, easily unit-tested)
# ---------------------------------------------------------------------------

def _search_keyword(
    keyword: str,
    stored_chunks: list[StoredChunk],
    embedder: Embedder,
    threshold: float,
) -> list[MergedResult]:
    """Run the full retrieval pipeline for one keyword.

    Calls lexical search, semantic search, and the result merger in order.
    No new embeddings are generated for document chunks — only the query
    string is embedded here (inside ``semantic_search``).

    Args:
        keyword: The search term or phrase.
        stored_chunks: All pre-embedded chunks for the target document.
        embedder: Shared :class:`~backend.services.embedder.Embedder`.
        threshold: Minimum cosine-similarity score for semantic results.

    Returns:
        Deduplicated, ranked :class:`~backend.models.schemas.MergedResult`
        list.  Empty list when neither strategy finds anything.
    """
    # Lexical search operates on the Chunk objects inside each StoredChunk.
    chunks = [sc.chunk for sc in stored_chunks]
    lex_results = lexical_search(keyword, chunks)

    # Semantic search operates on the StoredChunks directly (needs embeddings).
    sem_results = semantic_search(
        query=keyword,
        stored_chunks=stored_chunks,
        embedder=embedder,
        threshold=threshold,
    )

    return merge_results(lex_results, sem_results, keyword)


# ---------------------------------------------------------------------------
# POST /search
# ---------------------------------------------------------------------------

@app.post(
    "/search",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Search an uploaded PDF for keywords",
    description=(
        "Searches a previously uploaded document for one or more keywords "
        "using hybrid lexical (exact + variant) and semantic retrieval. "
        "Returns ranked matched chunks and a list of keywords with no matches."
    ),
)
def search_pdf(
    body: SearchRequest,
    embedder: EmbedderDep,
    store: StoreDep,
) -> SearchResponse:
    """Search a stored document for the supplied keywords.

    For every keyword the endpoint:

    1. Runs :func:`~backend.services.lexical_search.lexical_search` against
       the document's plain-text chunks.
    2. Runs :func:`~backend.services.semantic_search.semantic_search` against
       the stored embeddings (only the *query* is embedded here).
    3. Passes both result sets to
       :func:`~backend.services.result_merger.merge_results`, which
       deduplicates and ranks the combined results.

    Args:
        body:     Validated :class:`~backend.models.schemas.SearchRequest`.
        embedder: Injected shared :class:`~backend.services.embedder.Embedder`.
        store:    Injected shared
            :class:`~backend.storage.document_store.DocumentStore`.

    Returns:
        :class:`~backend.models.schemas.SearchResponse` containing all
        matched chunks and a list of keywords that produced no results.

    Raises:
        HTTPException 404: ``document_id`` not found in the document store.
    """
    # ------------------------------------------------------------------
    # 1. Retrieve document (404 if unknown)
    # ------------------------------------------------------------------
    try:
        stored_doc = store.get(body.document_id)
    except DocumentNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{body.document_id}' not found.",
        )

    stored_chunks = stored_doc.stored_chunks

    # ------------------------------------------------------------------
    # 2. Per-keyword search loop
    # ------------------------------------------------------------------
    all_results: list[ChunkResult] = []
    not_found: list[str] = []
    keyword_summary: dict[str, KeywordSummary] = {}

    for keyword in body.keywords:
        merged = _search_keyword(
            keyword=keyword,
            stored_chunks=stored_chunks,
            embedder=embedder,
            threshold=body.threshold,
        )

        if not merged:
            not_found.append(keyword)
            continue

        total_matches = 0
        for mr in merged:
            snippet, positions = _build_snippet(
                text=mr.chunk.text,
                keyword=mr.keyword,
                match_type=mr.match_type,
            )
            total_matches += mr.match_count
            all_results.append(
                ChunkResult(
                    chunk_id=mr.chunk.chunk_id,
                    page_number=mr.chunk.page_number,
                    text=mr.chunk.text,
                    snippet=snippet,
                    keyword=mr.keyword,
                    match_type=mr.match_type,
                    match_count=mr.match_count,
                    semantic_score=mr.semantic_score,
                    match_positions=positions,
                )
            )

        keyword_summary[keyword] = KeywordSummary(
            total_matches=total_matches,
            result_count=len(merged),
        )

    logger.info(
        "Search '%s': %d keyword(s), %d result(s), %d not found.",
        body.document_id,
        len(body.keywords),
        len(all_results),
        len(not_found),
    )

    return SearchResponse(
        document_id=body.document_id,
        results=all_results,
        not_found=not_found,
        keyword_summary=keyword_summary,
    )
