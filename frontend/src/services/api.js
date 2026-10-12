// api.js
// All communication with the PDF Keyword Finder backend.
// Keep API_BASE here so swapping it for an env variable later is a one-line change.

const API_BASE = "http://localhost:8000";

/**
 * Upload a PDF file to the backend.
 *
 * @param {File} file - A File object selected by the user (must be a PDF).
 * @returns {Promise<{document_id: string, filename: string, page_count: number, chunk_count: number}>}
 * @throws {ApiError} on non-2xx responses or network failures.
 */
export async function uploadPDF(file) {
  const formData = new FormData();
  formData.append("file", file);

  let response;
  try {
    response = await fetch(`${API_BASE}/upload`, {
      method: "POST",
      body: formData,
    });
  } catch {
    throw new ApiError("Cannot reach the backend. Is the server running on localhost:8000?", null);
  }

  await _assertOk(response);
  return response.json();
}

/**
 * Search a previously uploaded document for keywords.
 *
 * @param {string} documentId - The document_id returned by uploadPDF.
 * @param {string[]} keywords  - 1–20 non-empty keyword strings.
 * @param {number}  threshold  - Semantic similarity threshold [0.0, 1.0].
 * @returns {Promise<SearchResponse>}
 * @throws {ApiError}
 */
export async function searchDocument(documentId, keywords, threshold) {
  let response;
  try {
    response = await fetch(`${API_BASE}/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        document_id: documentId,
        keywords,
        threshold,
      }),
    });
  } catch {
    throw new ApiError("Cannot reach the backend. Is the server running on localhost:8000?", null);
  }

  await _assertOk(response);
  return response.json();
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/**
 * Throw an ApiError if the response status is not 2xx.
 * Tries to read a `detail` field from the JSON body (FastAPI convention).
 */
async function _assertOk(response) {
  if (response.ok) return;

  let detail = `HTTP ${response.status}`;
  try {
    const body = await response.json();
    if (body?.detail) {
      detail = typeof body.detail === "string"
        ? body.detail
        : JSON.stringify(body.detail);
    }
  } catch {
    // body wasn't JSON — keep the status code message
  }
  throw new ApiError(detail, response.status);
}

/**
 * Structured error class so UI components can distinguish API errors from
 * unexpected JS errors without parsing message strings.
 */
export class ApiError extends Error {
  /** @param {string} message @param {number|null} status */
  constructor(message, status) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}
