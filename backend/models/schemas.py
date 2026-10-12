# schemas.py
# Responsibilities:
#   - Define Pydantic models for all API request and response payloads
#   - Define internal data models: Chunk, SearchResult, DocumentMetadata
#   - Provide a single source of truth for data shapes shared across the service layer and API layer
#   - Enforce validation and serialisation rules for the FastAPI endpoints

from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Internal data models
# ---------------------------------------------------------------------------

@dataclass
class PageText:
    """Raw text extracted from a single PDF page.

    Attributes:
        page_number: 1-based page index matching the physical page in the PDF.
        text: Raw extracted text with no cleaning applied.
    """

    page_number: int
    text: str


# ---------------------------------------------------------------------------
# Chunk
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """A single paragraph-level chunk produced by the chunker.

    Attributes:
        chunk_id: Globally unique identifier for this chunk, constructed from
            the document ID, page number, and chunk index within the page.
            Format: ``"<doc_id>-p<page_number>-c<chunk_index>"``.
        doc_id: Identifier of the source document this chunk belongs to.
        page_number: 1-based page number from which the chunk was extracted.
        chunk_index: 0-based position of this chunk within its page.
        text: Cleaned paragraph text — no further normalisation applied.
    """

    chunk_id: str
    doc_id: str
    page_number: int
    chunk_index: int
    text: str


# ---------------------------------------------------------------------------
# StoredChunk — a Chunk paired with its embedding vector
# ---------------------------------------------------------------------------

@dataclass
class StoredChunk:
    """A :class:`Chunk` together with its pre-computed embedding vector.

    Produced by the ingestion pipeline after the embedder runs and stored in
    :mod:`backend.storage.document_store`.  Semantic search operates on a
    list of ``StoredChunk`` objects.

    Attributes:
        chunk: The original :class:`Chunk` from the chunker.
        embedding: Unit-normalised 1-D ``numpy.ndarray`` of shape
            ``(embedding_dim,)`` produced by
            :meth:`~backend.services.embedder.Embedder.embed_chunks`.
    """

    chunk: Chunk
    embedding: np.ndarray = field(repr=False)


# ---------------------------------------------------------------------------
# SemanticResult — one hit returned by semantic_search
# ---------------------------------------------------------------------------

@dataclass
class SemanticResult:
    """A single result from semantic search.

    Attributes:
        chunk: The matching :class:`Chunk`.
        score: Cosine similarity score in the range ``[-1.0, 1.0]``.
            Higher is more similar.
    """

    chunk: Chunk
    score: float


# ---------------------------------------------------------------------------
# LexicalResult — one hit returned by lexical_search
# ---------------------------------------------------------------------------

@dataclass
class LexicalResult:
    """A single result from lexical (exact or variant) search.

    Attributes:
        chunk: The matching :class:`Chunk`.
        keyword: The original keyword/phrase as supplied by the caller
            (un-normalised).
        match_type: ``"exact"`` when the keyword itself was found verbatim
            (case-insensitive); ``"variant"`` when a morphological variant
            (e.g. plural, past-tense) was found instead.
        match_count: Number of times the match (exact or variant) appears in
            the chunk text.  Used by the result merger for scoring.
    """

    chunk: Chunk
    keyword: str
    match_type: str          # "exact" | "variant"
    match_count: int


# ---------------------------------------------------------------------------
# MergedResult — the unified result type produced by result_merger
# ---------------------------------------------------------------------------

@dataclass
class MergedResult:
    """A single deduplicated, ranked result returned to the API layer.

    Produced by :func:`~backend.services.result_merger.merge_results` after
    combining :class:`LexicalResult` and :class:`SemanticResult` objects.

    Attributes:
        chunk: The matching document chunk.
        keyword: The search keyword as supplied by the caller.
        match_type: The strongest match type found for this chunk:
            ``"exact"`` > ``"variant"`` > ``"semantic"``.
        match_count: Number of lexical occurrences in the chunk text.
            ``0`` for semantic-only results.
        semantic_score: Cosine similarity score when this chunk was also
            (or only) returned by semantic search; ``None`` otherwise.
    """

    chunk: Chunk
    keyword: str
    match_type: str            # "exact" | "variant" | "semantic"
    match_count: int           # 0 for semantic-only results
    semantic_score: float | None  # None when not in semantic results


# ---------------------------------------------------------------------------
# DocumentMetadata — lightweight summary stored alongside each document
# ---------------------------------------------------------------------------

@dataclass
class DocumentMetadata:
    """Lightweight metadata about an ingested document.

    Stored alongside :class:`StoredDocument` inside the document store and
    returned by listing / info operations.

    Attributes:
        document_id: Unique identifier assigned at upload time.
        filename: Original filename of the uploaded PDF.
        page_count: Number of pages extracted from the PDF.
        chunk_count: Total number of chunks produced by the chunker.
    """

    document_id: str
    filename: str
    page_count: int
    chunk_count: int


# ---------------------------------------------------------------------------
# StoredDocument — the complete record kept in the document store
# ---------------------------------------------------------------------------

@dataclass
class StoredDocument:
    """The complete in-memory record for one ingested PDF.

    Held by :class:`~backend.storage.document_store.DocumentStore` keyed by
    ``document_id``.  Contains everything needed by the search layer —
    no query-specific information (keywords, scores, match types) is stored
    here.

    Attributes:
        metadata: :class:`DocumentMetadata` with document-level facts.
        stored_chunks: All :class:`StoredChunk` objects for this document
            in the order they were produced by the chunker.  Each carries
            the original :class:`Chunk` and its pre-computed embedding.
    """

    metadata: DocumentMetadata
    stored_chunks: list[StoredChunk]


# ---------------------------------------------------------------------------
# API response models (Pydantic) — used by FastAPI for serialisation
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    """Response body returned by ``POST /upload``.

    Attributes:
        document_id: Unique identifier assigned to the uploaded document.
        filename: Original filename of the uploaded PDF.
        page_count: Number of pages successfully extracted.
        chunk_count: Total number of paragraph chunks created.
    """

    document_id: str
    filename: str
    page_count: int
    chunk_count: int


# ---------------------------------------------------------------------------
# Search request / response models
# ---------------------------------------------------------------------------

from typing import Optional                       # noqa: E402  (kept near usage)
from pydantic import field_validator, Field       # noqa: E402


class SearchRequest(BaseModel):
    """Request body for ``POST /search``.

    Attributes:
        document_id: ID of a previously uploaded document.
        keywords: 1–20 non-empty search terms or phrases.
        threshold: Minimum cosine-similarity score for semantic results.
            Must be in ``[0.0, 1.0]``.  Defaults to ``0.30``.
    """

    document_id: str = Field(..., min_length=1)
    keywords: list[str] = Field(..., min_length=1, max_length=20)
    threshold: float = Field(default=0.30, ge=0.0, le=1.0)

    @field_validator("keywords")
    @classmethod
    def keywords_not_empty(cls, v: list[str]) -> list[str]:
        for kw in v:
            if not kw or not kw.strip():
                raise ValueError("Keywords must not be empty or whitespace-only.")
        return v


class ChunkResult(BaseModel):
    """A single matched chunk within a ``SearchResponse``.

    Attributes:
        chunk_id: Unique identifier of the matched chunk.
        page_number: 1-based page number where the chunk was found.
        text: The full cleaned paragraph text of the chunk.
        snippet: A short excerpt (~200 chars) centred on the first lexical
            match, or the first 200 characters of ``text`` for semantic-only
            results.
        keyword: The keyword that produced this result.
        match_type: ``"exact"``, ``"variant"``, or ``"semantic"``.
        match_count: Lexical occurrence count; ``0`` for semantic-only results.
        semantic_score: Cosine similarity score, or ``None`` for lexical-only.
        match_positions: Character offset pairs ``(start, end)`` of every
            match occurrence that falls within the snippet, relative to the
            start of the snippet (not the original chunk text).  Empty for
            semantic-only results.
    """

    chunk_id: str
    page_number: int
    text: str
    snippet: str
    keyword: str
    match_type: str
    match_count: int
    semantic_score: Optional[float]
    match_positions: list[tuple[int, int]]


class KeywordSummary(BaseModel):
    """Aggregate statistics for one keyword across all returned chunks.

    Attributes:
        total_matches: Sum of ``match_count`` across all chunks for this
            keyword.  Semantic-only results contribute 0 (their
            ``match_count`` is always 0).
        result_count: Number of merged results (chunks) returned for this
            keyword.
    """

    total_matches: int
    result_count: int


class SearchResponse(BaseModel):
    """Response body returned by ``POST /search``.

    Attributes:
        document_id: The document that was searched.
        results: All matched chunks, ranked per-keyword then by match type.
        not_found: Keywords that produced zero results.
        keyword_summary: Per-keyword aggregate statistics keyed by keyword
            string.
    """

    document_id: str
    results: list[ChunkResult]
    not_found: list[str]
    keyword_summary: dict[str, KeywordSummary]
