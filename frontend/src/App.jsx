// App.jsx
// Root component. Owns all application state.
// Renders the two-column layout: left panel (upload + search controls),
// right panel (results).
//
// State shape:
//   uploadedDoc   – null | { document_id, filename, page_count, chunk_count }
//   keywords      – string[]
//   threshold     – number  (0.0 – 1.0)
//   uploadState   – "idle" | "loading" | "error"
//   uploadError   – string | null
//   searchState   – "idle" | "loading" | "error" | "done"
//   searchError   – string | null
//   searchResults – null | { results, not_found, keyword_summary }

import { useState } from "react";
import { uploadPDF, searchDocument } from "./services/api.js";
import FileUpload      from "./components/FileUpload.jsx";
import KeywordInput    from "./components/KeywordInput.jsx";
import ThresholdSlider from "./components/ThresholdSlider.jsx";
import SearchButton    from "./components/SearchButton.jsx";
import KeywordSummary  from "./components/KeywordSummary.jsx";
import ResultCard      from "./components/ResultCard.jsx";
import NotFound        from "./components/NotFound.jsx";

export default function App() {
  // ── document ──────────────────────────────────────────────────────────────
  const [uploadedDoc,  setUploadedDoc]  = useState(null);
  const [uploadState,  setUploadState]  = useState("idle");   // "idle"|"loading"|"error"
  const [uploadError,  setUploadError]  = useState(null);

  // ── search controls ────────────────────────────────────────────────────────
  const [keywords,   setKeywords]  = useState([]);
  const [threshold,  setThreshold] = useState(0.75);

  // ── search results ─────────────────────────────────────────────────────────
  const [searchState,   setSearchState]   = useState("idle"); // "idle"|"loading"|"error"|"done"
  const [searchError,   setSearchError]   = useState(null);
  const [searchResults, setSearchResults] = useState(null);

  // ── handlers ───────────────────────────────────────────────────────────────

  async function handleUpload(file) {
    setUploadState("loading");
    setUploadError(null);
    setUploadedDoc(null);
    // Reset search when a new document is uploaded
    setSearchState("idle");
    setSearchResults(null);
    setSearchError(null);

    try {
      const doc = await uploadPDF(file);
      setUploadedDoc(doc);
      setUploadState("idle");
    } catch (err) {
      setUploadError(err.message ?? "Unknown upload error.");
      setUploadState("error");
    }
  }

  async function handleSearch() {
    if (!uploadedDoc || keywords.length === 0) return;

    setSearchState("loading");
    setSearchError(null);
    setSearchResults(null);

    try {
      const data = await searchDocument(uploadedDoc.document_id, keywords, threshold);
      setSearchResults(data);
      setSearchState("done");
    } catch (err) {
      setSearchError(err.message ?? "Unknown search error.");
      setSearchState("error");
    }
  }

  // ── derived flags ──────────────────────────────────────────────────────────
  const canSearch    = !!uploadedDoc && keywords.length > 0;
  const totalResults = searchResults?.results?.length ?? 0;

  // ── render ─────────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Top bar */}
      <header className="border-b border-slate-200 bg-white px-6 py-4 shadow-sm">
        <div className="mx-auto flex max-w-7xl items-center gap-3">
          <svg className="h-7 w-7 text-blue-600 shrink-0" fill="none" viewBox="0 0 24 24"
               stroke="currentColor" strokeWidth={1.8}>
            <path strokeLinecap="round" strokeLinejoin="round"
              d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5
                 a3.375 3.375 0 00-3.375-3.375H8.25m5.231 13.481L15 17.25m-4.5-15H5.625c-.621 0-1.125.504-1.125
                 1.125v16.5c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
          </svg>
          <div>
            <h1 className="text-xl font-bold text-slate-800 leading-none">PDF Keyword Finder</h1>
            <p className="text-xs text-slate-400 mt-0.5">Upload a PDF · search with hybrid lexical + semantic retrieval</p>
          </div>
        </div>
      </header>

      {/* Main layout */}
      <main className="mx-auto max-w-7xl px-4 py-8 lg:px-6">
        <div className="flex flex-col gap-8 lg:flex-row lg:items-start">

          {/* ── Left panel: controls ───────────────────────────────────────── */}
          <div className="w-full lg:w-96 lg:shrink-0 space-y-6">
            <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm space-y-6">
              <FileUpload
                onUpload={handleUpload}
                uploadedDoc={uploadedDoc}
                isLoading={uploadState === "loading"}
                error={uploadError}
              />

              <hr className="border-slate-100" />

              <KeywordInput keywords={keywords} onChange={setKeywords} />

              <hr className="border-slate-100" />

              <ThresholdSlider value={threshold} onChange={setThreshold} />
            </div>

            <SearchButton
              onClick={handleSearch}
              disabled={!canSearch}
              isLoading={searchState === "loading"}
            />

            {/* Search error */}
            {searchState === "error" && searchError && (
              <div role="alert"
                   className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                <span className="font-semibold">Search failed: </span>{searchError}
              </div>
            )}
          </div>

          {/* ── Right panel: results ───────────────────────────────────────── */}
          <div className="flex-1 min-w-0 space-y-6">
            {/* Empty state — no document yet */}
            {!uploadedDoc && searchState === "idle" && (
              <EmptyState
                icon="📄"
                title="No document uploaded"
                body="Upload a PDF using the panel on the left to get started."
              />
            )}

            {/* Empty state — document uploaded but no search yet */}
            {uploadedDoc && searchState === "idle" && (
              <EmptyState
                icon="🔍"
                title="Ready to search"
                body={`"${uploadedDoc.filename}" is indexed. Add keywords and click Search.`}
              />
            )}

            {/* Loading spinner */}
            {searchState === "loading" && (
              <div className="flex flex-col items-center justify-center py-24 text-slate-400 gap-4">
                <svg className="h-10 w-10 animate-spin text-blue-400" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10"
                          stroke="currentColor" strokeWidth="4" />
                  <path className="opacity-75" fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                <p className="text-sm">Searching…</p>
              </div>
            )}

            {/* Results */}
            {searchState === "done" && searchResults && (
              <>
                {/* Result count header */}
                <div className="flex flex-wrap items-baseline gap-2">
                  <h2 className="text-lg font-bold text-slate-800">
                    {totalResults === 0
                      ? "No results"
                      : `${totalResults} result${totalResults !== 1 ? "s" : ""}`}
                  </h2>
                  <p className="text-sm text-slate-500">
                    for {keywords.length} keyword{keywords.length !== 1 ? "s" : ""}
                  </p>
                </div>

                {/* Keyword summary table */}
                {Object.keys(searchResults.keyword_summary).length > 0 && (
                  <KeywordSummary summary={searchResults.keyword_summary} />
                )}

                {/* Not-found keywords */}
                <NotFound keywords={searchResults.not_found} />

                {/* No results empty state */}
                {totalResults === 0 && searchResults.not_found.length > 0 && (
                  <EmptyState
                    icon="🤷"
                    title="No matches found"
                    body="Try broader keywords or lower the similarity threshold."
                  />
                )}

                {/* Result cards — grouped by keyword for readability */}
                {totalResults > 0 && (
                  <div className="results-scroll space-y-3">
                    {searchResults.results.map((result) => (
                      <ResultCard key={result.chunk_id + result.keyword} {...result} />
                    ))}
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="mt-16 border-t border-slate-200 py-6 text-center text-xs text-slate-400">
        PDF Keyword Finder · Hybrid lexical + semantic retrieval
      </footer>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Small reusable empty-state card
// ---------------------------------------------------------------------------
function EmptyState({ icon, title, body }) {
  return (
    <div className="flex flex-col items-center justify-center rounded-2xl border border-dashed
                    border-slate-200 bg-white py-16 px-8 text-center space-y-3">
      <span className="text-5xl">{icon}</span>
      <h3 className="text-base font-semibold text-slate-700">{title}</h3>
      <p className="max-w-sm text-sm text-slate-400">{body}</p>
    </div>
  );
}
