// NotFound.jsx
// Displays the list of keywords that returned zero results.

/**
 * @param {{ keywords: string[] }} props
 */
export default function NotFound({ keywords }) {
  if (!keywords || keywords.length === 0) return null;

  return (
    <div className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-3">
      <p className="mb-2 text-sm font-semibold text-amber-700">
        No matches found for:
      </p>
      <ul className="flex flex-wrap gap-2">
        {keywords.map(kw => (
          <li
            key={kw}
            className="rounded-full border border-amber-300 bg-white px-3 py-0.5 text-sm text-amber-700"
          >
            {kw}
          </li>
        ))}
      </ul>
    </div>
  );
}
