# PDF Keyword Finder

A take-home assignment implementing a hybrid lexical + semantic keyword search engine over uploaded PDF documents.

Upload a PDF, type keywords, and receive ranked results that combine exact word matching, morphological variant matching, and embedding-based semantic similarity — all in a single response.

---

## Features

- **PDF upload** — drag-and-drop or file picker, validated by content (not just extension), max 20 MB / 50 pages
- **Hybrid retrieval** — exact match, controlled morphological variants (plural, past-tense), and semantic similarity run in parallel and are merged
- **Snippet extraction** — every result includes a ~200-character excerpt centred on the first match
- **Keyword highlighting** — character-level match positions returned so the frontend can highlight exactly where the keyword appears
- **Keyword summary** — per-keyword aggregate of matched chunks and total occurrence count
- **Not-found reporting** — keywords that produced zero results are clearly identified
- **Configurable threshold** — semantic similarity cutoff is adjustable per query (0.0–1.0)
- **React frontend** — responsive single-page app with Tailwind CSS styling

---

## Architecture

```
POST /upload
    ↓
PDF validation (content-type, size, magic bytes, page count)
    ↓
Page-by-page text extraction          pdf_processor.py   (PyMuPDF)
    ↓
Text cleaning                         text_cleaner.py    (regex)
    ↓
Paragraph-based chunking              chunker.py
    ↓
Batch embedding generation            embedder.py        (sentence-transformers)
    ↓
In-memory document storage            document_store.py

POST /search
    ↓
For each keyword:
    ├── Lexical search                lexical_search.py  (regex + morphology)
    └── Semantic search               semantic_search.py (dot product on normalised vectors)
          ↓
      Result merging & deduplication  result_merger.py
          ↓
      Snippet + match positions       main.py
    ↓
SearchResponse  →  React frontend
```

---

## Tech Stack

### Backend
| Component | Technology |
|---|---|
| API framework | FastAPI 0.111 |
| PDF parsing | PyMuPDF (fitz) 1.24 |
| Embeddings | sentence-transformers / all-MiniLM-L6-v2 |
| Similarity computation | NumPy (dot product on unit-normalised vectors) |
| Data validation | Pydantic v2 |
| Server | Uvicorn 0.29 |
| Python | 3.12 |

### Frontend
| Component | Technology |
|---|---|
| Framework | React 18 |
| Build tool | Vite 5 |
| Styling | Tailwind CSS 3 |
| HTTP | fetch (native browser API) |

---

## Installation and Setup

### Prerequisites
- Python 3.10+
- Node.js 18+

---

### Backend

#### 1. Create and activate a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

#### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

The first install will also download the `all-MiniLM-L6-v2` model (~90 MB) on first run.

#### 3. Start the backend server

```bash
python -m uvicorn backend.main:app --reload
```

The server starts at `http://localhost:8000`.  
Interactive API docs: `http://localhost:8000/docs`

> **Note:** First startup takes 10–30 seconds while the embedding model loads into memory.

---

### Frontend

#### 1. Install Node dependencies

```bash
cd frontend
npm install
```

#### 2. Start the dev server

```bash
npm run dev
```

Opens at `http://localhost:5173`.

#### 3. Production build

```bash
npm run build
```

Output goes to `frontend/dist/`.

---

### Running tests (backend only)

```bash
pytest tests/ -m "not integration"
```

To also run the real-model integration tests (requires model download):

```bash
pytest tests/ -m integration
```

---

## API Endpoints

### `POST /upload`

Upload and index a PDF file.

**Request:** `multipart/form-data` with field `file` (PDF).

**Response `201`:**
```json
{
  "document_id": "3f7a1b2c-...",
  "filename": "contract.pdf",
  "page_count": 12,
  "chunk_count": 47
}
```

**Error codes:** `400` (invalid file, no text), `413` (too large or too many pages)

---

### `POST /search`

Search an uploaded document for keywords.

**Request:**
```json
{
  "document_id": "3f7a1b2c-...",
  "keywords": ["invoice", "payment terms"],
  "threshold": 0.75
}
```

Constraints: 1–20 keywords, threshold in `[0.0, 1.0]`.

**Response `200`:**
```json
{
  "document_id": "3f7a1b2c-...",
  "results": [
    {
      "chunk_id": "3f7a1b2c-p3-c1",
      "page_number": 3,
      "text": "Full paragraph text...",
      "snippet": "...centred excerpt ~200 chars...",
      "keyword": "invoice",
      "match_type": "exact",
      "match_count": 2,
      "semantic_score": null,
      "match_positions": [[4, 11], [28, 35]]
    }
  ],
  "not_found": ["payment terms"],
  "keyword_summary": {
    "invoice": { "total_matches": 5, "result_count": 3 }
  }
}
```

`match_type` is one of `"exact"`, `"variant"`, `"semantic"`.  
`match_positions` are `[start, end]` character offsets **relative to `snippet`**.  
`semantic_score` is `null` for lexical-only results.

---

## How Hybrid Search Works

Each keyword goes through two independent retrieval strategies, then the results are merged.

### 1. Lexical search (`lexical_search.py`)

- **Exact match:** case-insensitive word-boundary regex (`\b<keyword>\b`).
- **Variant match:** a small explicit suffix-rule table generates controlled morphological forms — plural (`+s`, `+es`), past-tense (`+d`, `+ed`), present participle (`+ing`). No stemmer is used; every variant is a real English word.
- Substring false positives are prevented by `\b` anchoring — `"invoice"` does not match `"invoiceless"`.

### 2. Semantic search (`semantic_search.py`)

- The query string is embedded by the same `all-MiniLM-L6-v2` model used at ingestion.
- All chunk embeddings are stacked into a matrix; a single `numpy` matrix–vector multiply computes dot-product similarity against the query vector.
- Because all vectors are unit-normalised, dot product equals cosine similarity.
- Chunks below `threshold` are discarded before returning.

### 3. Result merging (`result_merger.py`)

- Results are deduplicated by `chunk_id`.
- Priority: `exact > variant > semantic` — the strongest signal wins.
- When a chunk appears in both result sets, the lexical `match_type` is kept and the `semantic_score` is preserved alongside it.
- Final ranking: exact first, then variant, then semantic (by descending score within the semantic tier).

---

## Design Decisions and Trade-offs

| Decision | Rationale |
|---|---|
| **`all-MiniLM-L6-v2`** | 384-dim, <100 MB, top benchmark performance at its size. Fast enough for a demo without GPU. |
| **In-memory document store** | Sufficient for a single-process assignment demo. Trivially replaceable with a vector DB by changing only `document_store.py`. |
| **Paragraph-level chunking** | Balances context (enough text for meaningful embeddings) with granularity (small enough to locate a specific passage). |
| **Conservative morphology table** | Avoids incorrect stems (Porter stemmer reduces "invoice" to "invoic"). Every generated variant is a real English word. |
| **Dot product instead of cosine similarity** | Identical when vectors are unit-normalised. Avoids per-pair division, enabling a single vectorised matrix multiply. |
| **Snippet centred on first match** | Gives the reader immediate context around the keyword without exposing the entire paragraph. |
| **Positions relative to snippet** | The frontend only needs to index into the snippet string — no arithmetic against original offsets required. |
| **State in `App.jsx` only** | No Redux/Zustand. The state is shallow enough that prop drilling is manageable and the code stays readable. |

---

## Limitations

- **Image-only PDFs** are not supported. Pages without embedded text return no chunks. OCR is not implemented.
- **Single document** — the store holds all uploaded documents in memory; there is no multi-document search across documents simultaneously.
- **Process restart** clears all stored documents (in-memory only).
- **No authentication** — the API is open; suitable only for local/demo use.
- **Variant generation is English-only** and covers only the most common inflection patterns.
- **Long PDFs** are capped at 50 pages. Larger documents require chunking at a higher level before upload.
- **Threshold sensitivity** — the default of `0.75` is conservative; documents with domain-specific vocabulary may need a lower threshold to surface relevant semantic results.

---

## Evaluation

See [`evaluation/evaluation.md`](evaluation/evaluation.md) for the evaluation methodology and results template.
