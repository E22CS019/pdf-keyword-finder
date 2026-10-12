// KeywordSummary.jsx
// Renders a summary table of per-keyword match statistics returned by the API.
// keyword_summary: { [keyword]: { total_matches: number, result_count: number } }

/**
 * @param {{ summary: Record<string, { total_matches: number, result_count: number }> }} props
 */
export default function KeywordSummary({ summary }) {
  const entries = Object.entries(summary);
  if (entries.length === 0) return null;

  return (
    <div className="overflow-hidden rounded-lg border border-slate-200 bg-white shadow-sm">
      <div className="border-b border-slate-100 bg-slate-50 px-4 py-2.5">
        <h3 className="text-sm font-semibold text-slate-600">Keyword Summary</h3>
      </div>
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-100 text-left text-xs font-medium uppercase tracking-wide text-slate-400">
            <th className="px-4 py-2">Keyword</th>
            <th className="px-4 py-2 text-center">Chunks matched</th>
            <th className="px-4 py-2 text-center">Total occurrences</th>
          </tr>
        </thead>
        <tbody>
          {entries.map(([kw, stats]) => (
            <tr key={kw} className="border-b border-slate-50 last:border-0 hover:bg-slate-50">
              <td className="px-4 py-2 font-medium text-slate-700">{kw}</td>
              <td className="px-4 py-2 text-center tabular-nums text-slate-600">{stats.result_count}</td>
              <td className="px-4 py-2 text-center tabular-nums text-slate-600">
                {stats.total_matches > 0 ? stats.total_matches : <span className="text-slate-300">—</span>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
