// FileUpload.jsx
// Drag-and-drop + file-picker upload panel.
// Calls onUpload(file) when the user confirms a selection.
// Displays upload state, success summary, and errors from the parent.

import { useRef, useState } from "react";

/**
 * @param {{ onUpload: (file: File) => void, uploadedDoc: object|null, isLoading: boolean, error: string|null }} props
 */
export default function FileUpload({ onUpload, uploadedDoc, isLoading, error }) {
  const inputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);

  // ---- drag handlers ----
  function handleDragOver(e) {
    e.preventDefault();
    setDragOver(true);
  }
  function handleDragLeave() {
    setDragOver(false);
  }
  function handleDrop(e) {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) pickFile(file);
  }

  // ---- picker ----
  function handlePickerChange(e) {
    const file = e.target.files?.[0];
    if (file) pickFile(file);
    // reset so the same file can be re-selected after an error
    e.target.value = "";
  }

  function pickFile(file) {
    if (!file.type.includes("pdf")) {
      // surface this early so the user doesn't wait for the server round-trip
      setSelectedFile(null);
      return;
    }
    setSelectedFile(file);
  }

  function handleUpload() {
    if (selectedFile) onUpload(selectedFile);
  }

  // ---- derived UI state ----
  const dropzoneBase =
    "relative flex flex-col items-center justify-center rounded-xl border-2 border-dashed p-8 transition-colors cursor-pointer";
  const dropzoneColor = dragOver
    ? "border-blue-400 bg-blue-50"
    : "border-slate-300 bg-slate-50 hover:border-blue-300 hover:bg-blue-50/40";

  return (
    <section className="space-y-4">
      <h2 className="text-lg font-semibold text-slate-700">1. Upload PDF</h2>

      {/* Drop zone */}
      <div
        className={`${dropzoneBase} ${dropzoneColor}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        role="button"
        aria-label="Drop PDF here or click to browse"
      >
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf"
          className="hidden"
          onChange={handlePickerChange}
        />

        {/* Icon */}
        <svg className="mb-3 h-10 w-10 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
          <path strokeLinecap="round" strokeLinejoin="round"
            d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m6.75 12l-3-3m0 0l-3 3m3-3v6m-1.5-15H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
        </svg>

        {selectedFile ? (
          <p className="text-sm font-medium text-blue-600">{selectedFile.name}</p>
        ) : (
          <>
            <p className="text-sm font-medium text-slate-600">
              Drag &amp; drop a PDF, or <span className="text-blue-500 underline">browse</span>
            </p>
            <p className="mt-1 text-xs text-slate-400">PDF files only · max 20 MB · max 50 pages</p>
          </>
        )}
      </div>

      {/* Upload button */}
      <button
        onClick={handleUpload}
        disabled={!selectedFile || isLoading}
        className="w-full rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white
                   transition hover:bg-blue-700 active:bg-blue-800
                   disabled:cursor-not-allowed disabled:bg-slate-300 disabled:text-slate-500"
      >
        {isLoading ? (
          <span className="flex items-center justify-center gap-2">
            <Spinner /> Uploading…
          </span>
        ) : "Upload"}
      </button>

      {/* Error */}
      {error && (
        <div role="alert" className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
          <span className="font-semibold">Upload failed: </span>{error}
        </div>
      )}

      {/* Success summary */}
      {uploadedDoc && !error && (
        <div className="rounded-lg border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-800 space-y-1">
          <p className="font-semibold text-emerald-700">✓ Document ready</p>
          <p><span className="font-medium">File:</span> {uploadedDoc.filename}</p>
          <p>
            <span className="font-medium">Pages:</span> {uploadedDoc.page_count}
            &ensp;·&ensp;
            <span className="font-medium">Chunks:</span> {uploadedDoc.chunk_count}
          </p>
        </div>
      )}
    </section>
  );
}

function Spinner() {
  return (
    <svg className="h-4 w-4 animate-spin" fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
  );
}
