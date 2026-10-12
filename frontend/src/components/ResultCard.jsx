// ResultCard.jsx
// Renders one matched chunk result.
//
// Highlighting:
//   match_positions is a list of [start, end] character offsets relative to `snippet`.
//   We split the snippet into segments: plain text, then highlighted span, alternating.
//   For semantic results match_positions is [] so the snippet is shown as plain text.

/**
 * @param {{
 *   chunk_id: string,
 *   page_number: number,
 *   snippet: string,
 *   text: string,
 *   keyword: string,
 *   match_type: string,
 *   match_count: number,
 *   semantic_score: number|null,
 *   match_positions: [number, number][]
 * }} props
 */
export default function ResultCard({
  chunk_id,
  page_number,
  snippet,
  keyword,
  match_type,
  match_count,
  semantic_score,
  match_positions,
}) {
  const segments = buildSegments(snippet, match_positions);

  return (
    <article className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm space-y-3 hover:shadow-md transition-shadow">
      {/* Header row */}
      <div className="flex flex-wrap items-center gap-2">
        {/* Page badge */}
        <span className="inline-flex items-center gap-1 rounded-full bg-slate-100 px-2.5 py-0.5 text-xs font-medium text-slate-600">
          <svg className="h-3 w-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round"
              d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
          </svg>
          Page {page_number}
        </span>

        {/* Match type badge */}
        <MatchTypeBadge type={match_type} />

        {/* Keyword chip */}
        <span className="rounded-full bg-blue-50 px-2.5 py-0.5 text-xs font-medium text-blue-700">
          {keyword}
        </span>

        {/* Occurrence count for lexical */}
        {match_count > 0 && (
          <span className="ml-auto text-xs text-slate-400 tabular-nums">
            {match_count} occurrence{match_count !== 1 ? "s" : ""}
          </span>
        )}

        {/* Semantic score */}
        {semantic_score !== null && semantic_score !== undefined && (
          <span className="ml-auto text-xs font-medium tabular-nums text-violet-600">
            score {semantic_score.toFixed(3)}
          </span>
        )}
      </div>

      {/* Snippet with optional highlights */}
      <p className="rounded-md bg-slate-50 px-3 py-2 font-mono text-sm leading-relaxed text-slate-700">
        {segments.map((seg, i) =>
          seg.highlight ? (
            <mark key={i} className="rounded bg-yellow-200 px-0.5 text-yellow-900 not-italic">
              {seg.text}
            </mark>
          ) : (
            <span key={i}>{seg.text}</span>
          )
        )}
      </p>
    </article>
  );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Split `snippet` into alternating plain/highlighted segments using
 * the character offsets from `match_positions`.
 *
 * @param {string} snippet
 * @param {[number, number][]} positions  — sorted, non-overlapping, relative to snippet
 * @returns {{ text: string, highlight: boolean }[]}
 */
function buildSegments(snippet, positions) {
  if (!positions || positions.length === 0) {
    return [{ text: snippet, highlight: false }];
  }

  const segments = [];
  let cursor = 0;

  for (const [start, end] of positions) {
    if (start > cursor) {
      segments.push({ text: snippet.slice(cursor, start), highlight: false });
    }
    segments.push({ text: snippet.slice(start, end), highlight: true });
    cursor = end;
  }

  if (cursor < snippet.length) {
    segments.push({ text: snippet.slice(cursor), highlight: false });
  }

  return segments;
}

function MatchTypeBadge({ type }) {
  const styles = {
    exact:    "bg-emerald-100 text-emerald-700",
    variant:  "bg-sky-100 text-sky-700",
    semantic: "bg-violet-100 text-violet-700",
  };
  const label = {
    exact:    "EXACT",
    variant:  "VARIANT",
    semantic: "SEMANTIC",
  };

  return (
    <span className={`rounded-full px-2.5 py-0.5 text-xs font-semibold tracking-wide ${styles[type] ?? "bg-slate-100 text-slate-600"}`}>
      {label[type] ?? type.toUpperCase()}
    </span>
  );
}
