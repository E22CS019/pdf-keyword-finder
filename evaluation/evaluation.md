# Evaluation Report — PDF Keyword Finder

> **Status:** Template with methodology defined. Quantitative results marked `[ TO BE MEASURED ]` must be filled in after running the evaluation against the test dataset.

---

## 1. Test Dataset

| Field | Value |
|---|---|
| **Dataset name** | [ TO BE MEASURED — specify document name and source ] |
| **Official source / URL** | [ TO BE MEASURED ] |
| **Number of pages** | [ TO BE MEASURED ] |
| **Document type** | [ e.g. research paper / legal contract / technical manual ] |
| **Language** | English |

**Rationale for dataset choice:** [ TO BE MEASURED — explain why this document was chosen, e.g. domain coverage, keyword diversity, known ground truth. ]

---

## 2. Evaluation Setup

| Parameter | Value |
|---|---|
| Backend model | `sentence-transformers/all-MiniLM-L6-v2` |
| Semantic threshold tested | 0.30 / 0.50 / 0.75 (see per-run table below) |
| Chunking strategy | Paragraph-based, no max-chars limit |
| Lexical variants | plural, past-tense, present-participle (suffix table) |
| Hardware | [ TO BE MEASURED — CPU/GPU, RAM ] |

---

## 3. Keywords Tested

| # | Keyword | Expected match type | Expected pages |
|---|---|---|---|
| 1 | [ TO BE MEASURED ] | exact / variant / semantic | [ pages ] |
| 2 | [ TO BE MEASURED ] | | |
| 3 | [ TO BE MEASURED ] | | |
| 4 | [ TO BE MEASURED ] | | |
| 5 | [ TO BE MEASURED ] | | |

*Add rows as needed. At least 5 keywords covering exact, variant, and semantic cases are recommended.*

---

## 4. Ground Truth Labelling

Ground truth is defined manually by reading the document and recording which page(s) contain a relevant passage for each keyword.

| Keyword | Relevant pages (ground truth) | Notes |
|---|---|---|
| [ keyword 1 ] | [ TO BE MEASURED ] | |
| [ keyword 2 ] | [ TO BE MEASURED ] | |
| [ keyword 3 ] | [ TO BE MEASURED ] | |

---

## 5. Results — Expected vs Actual

Run the search for each keyword at each threshold and record what the system returned.

### Threshold = 0.30

| Keyword | Expected pages | Returned pages | Match types returned | TP | FP | FN |
|---|---|---|---|---|---|---|
| [ keyword 1 ] | [ ] | [ TO BE MEASURED ] | | | | |
| [ keyword 2 ] | [ ] | [ TO BE MEASURED ] | | | | |
| [ keyword 3 ] | [ ] | [ TO BE MEASURED ] | | | | |

### Threshold = 0.50

| Keyword | Expected pages | Returned pages | Match types returned | TP | FP | FN |
|---|---|---|---|---|---|---|
| [ keyword 1 ] | [ ] | [ TO BE MEASURED ] | | | | |

### Threshold = 0.75

| Keyword | Expected pages | Returned pages | Match types returned | TP | FP | FN |
|---|---|---|---|---|---|---|
| [ keyword 1 ] | [ ] | [ TO BE MEASURED ] | | | | |

---

## 6. Metrics

### Definitions

A **result** is a `(keyword, chunk)` pair returned by the system.  
A result is a **True Positive (TP)** if the chunk's page is in the ground-truth set for that keyword.  
A result is a **False Positive (FP)** if the chunk's page is not relevant.  
A **False Negative (FN)** is a relevant page that the system did not return.

### Precision

```
Precision = TP / (TP + FP)
```

Higher precision means fewer irrelevant results returned.

### Recall

```
Recall = TP / (TP + FN)
```

Higher recall means fewer relevant passages missed.

### F1 Score

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### Summary Table

| Threshold | Strategy | Precision | Recall | F1 |
|---|---|---|---|---|
| 0.30 | Hybrid (lexical + semantic) | [ TO BE MEASURED ] | [ TO BE MEASURED ] | [ TO BE MEASURED ] |
| 0.30 | Lexical only | [ TO BE MEASURED ] | [ TO BE MEASURED ] | [ TO BE MEASURED ] |
| 0.30 | Semantic only | [ TO BE MEASURED ] | [ TO BE MEASURED ] | [ TO BE MEASURED ] |
| 0.50 | Hybrid | [ TO BE MEASURED ] | [ TO BE MEASURED ] | [ TO BE MEASURED ] |
| 0.75 | Hybrid | [ TO BE MEASURED ] | [ TO BE MEASURED ] | [ TO BE MEASURED ] |

---

## 7. Baseline Comparisons

To isolate the value of each retrieval strategy, compare:

| Baseline | How to approximate |
|---|---|
| **Lexical only** | Set threshold = 1.0 so no semantic results pass; inspect only `match_type` ∈ {exact, variant} |
| **Semantic only** | Use keywords that have no exact match in the document (pure paraphrase queries) |
| **Hybrid** | Default setup with threshold = 0.50 |

Results for baselines: [ TO BE MEASURED ]

---

## 8. Failure Cases and Limitations

| Failure mode | Observed? | Example |
|---|---|---|
| Image-only pages produce no chunks | [ TO BE MEASURED ] | |
| Keyword split across hyphenated line break | [ TO BE MEASURED ] | |
| Variant generation produces wrong form | [ TO BE MEASURED ] | e.g. "datum" → "datums" instead of "data" |
| Semantic threshold too high — misses paraphrases | [ TO BE MEASURED ] | |
| Semantic threshold too low — returns unrelated chunks | [ TO BE MEASURED ] | |
| Multi-word phrase not matched due to different word order in document | [ TO BE MEASURED ] | |
| Domain-specific abbreviations not matched | [ TO BE MEASURED ] | |

---

## 9. Latency

| Operation | Observed latency | Notes |
|---|---|---|
| First startup (model load) | [ TO BE MEASURED ] | One-time cost |
| Upload 10-page PDF | [ TO BE MEASURED ] | Includes extraction, cleaning, chunking, embedding |
| Search — 1 keyword | [ TO BE MEASURED ] | |
| Search — 5 keywords | [ TO BE MEASURED ] | |
| Search — 20 keywords | [ TO BE MEASURED ] | |

---

## 10. Future Improvements

| Improvement | Expected impact |
|---|---|
| OCR support (e.g. Tesseract) | Enables image-only PDFs |
| Larger embedding model (all-mpnet-base-v2) | Higher semantic recall at cost of speed |
| BM25 lexical scoring | Better term-frequency weighting than raw match count |
| Persistent vector store (e.g. ChromaDB, Qdrant) | Survives server restarts; supports multi-document search |
| Cross-encoder re-ranking | Higher precision on top-k results |
| Language detection + multilingual model | Non-English document support |
