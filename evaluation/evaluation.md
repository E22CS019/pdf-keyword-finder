# Evaluation — PDF Keyword Finder

## 1. Dataset

**Document:** `sample-50-page-pdf-a4-size.pdf`

**Document size:** 50 pages

**Chunks generated:** 50

The evaluation was performed using a 50-page text-based PDF.

---

## 2. Test Configuration

- Embedding model: `all-MiniLM-L6-v2`
- Similarity metric: Cosine similarity
- Semantic threshold: `0.75`
- Search mode: Hybrid lexical + semantic retrieval
- Keywords tested: 6
- Maximum supported keywords: 20

### Keywords

1. artificial intelligence
2. machine learning
3. human
4. chatbot
5. cases
6. learning

---

## 3. Observed Results

| Keyword | Chunks Matched | Total Occurrences | Result |
|---|---:|---:|---|
| artificial intelligence | 1 | 2 | Found |
| machine learning | 3 | 3 | Found |
| human | 6 | 6 | Found |
| chatbot | 0 | 0 | Not found |
| cases | 6 | 8 | Found |
| learning | 3 | 4 | Found |

The application returned **19 result chunks** across the six keywords.

The difference between result chunks and total occurrences occurs because a single chunk can contain multiple occurrences of a keyword.

---

## 4. Match Types

The observed lexical matches were classified as:

- **Exact:** The keyword appears literally in the extracted text.
- **Variant:** Controlled lexical variations can be detected by the lexical matching layer.
- **Semantic:** A chunk can be retrieved based on embedding similarity even when the exact wording is not present.

For this particular test run, the visible matches were primarily exact lexical matches. The `chatbot` keyword produced no result.

No semantic-only result is claimed from this run.

---

## 5. Not-Found Handling

The keyword:

`chatbot`

returned no matching result at the configured threshold.

The application explicitly reports unmatched keywords instead of silently dropping them.

---

## 6. Threshold Behavior

The semantic threshold is configurable between `0.0` and `1.0`.

- Higher threshold → stricter semantic matching and higher precision potential.
- Lower threshold → broader semantic retrieval and higher recall potential.

A threshold of `0.75` was used for this evaluation.

The threshold is passed directly to the semantic search component, allowing the user to control retrieval strictness.

---

## 7. Evaluation Methodology

The primary purpose of this evaluation was to verify:

1. Page-aware PDF extraction.
2. Preservation of page information during chunking.
3. Exact keyword retrieval.
4. Multiple occurrences within a chunk.
5. Multiple keyword searches.
6. Explicit not-found handling.
7. Configurable semantic similarity threshold.
8. Result merging between lexical and semantic retrieval.

This is a functional evaluation rather than a statistically significant benchmark.

A larger hand-labelled dataset would be required to produce reliable precision, recall, and F1 estimates for semantic retrieval.

---

## 8. Failure Cases and Limitations

### Semantic retrieval

The selected test run did not produce a semantic-only result for the six tested keywords. Therefore, semantic precision/recall is not reported from this run rather than presenting unsupported numbers.

### OCR

The current implementation primarily targets text-based PDFs. Scanned PDFs requiring OCR are outside the core implementation.

### Evaluation size

Only one 50-page document was used for this quick evaluation. A broader evaluation should include multiple documents and manually labelled semantic-only matches.

---

## 9. Automated Testing

The project includes automated tests covering:

- PDF upload validation
- Text cleaning
- Chunking
- Lexical search
- Semantic search
- Result merging
- Document storage
- Search API behavior

Latest test run:

**320 tests passed**

---

## 10. Conclusion

The evaluation demonstrates that the system can process a 50-page PDF, preserve page-level information, search multiple keywords, report multiple occurrences, provide contextual results, and explicitly identify keywords with no matches.

The hybrid architecture combines deterministic lexical retrieval with local embedding-based semantic retrieval. The semantic threshold provides a configurable precision/recall trade-off for semantic matching.