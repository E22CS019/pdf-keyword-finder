// ThresholdSlider.jsx
// Slider controlling the semantic similarity threshold passed to POST /search.

/**
 * @param {{ value: number, onChange: (v: number) => void }} props
 */
export default function ThresholdSlider({ value, onChange }) {
  function handleChange(e) {
    onChange(parseFloat(e.target.value));
  }

  // Map the value to a human-readable description so non-ML users understand it.
  function description(v) {
    if (v >= 0.85) return "Very strict — only near-exact semantic matches";
    if (v >= 0.70) return "Strict — closely related meaning required";
    if (v >= 0.50) return "Balanced — reasonably similar passages";
    if (v >= 0.30) return "Relaxed — loosely related content included";
    return "Very relaxed — almost anything will match";
  }

  return (
    <section className="space-y-2">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-slate-700">3. Semantic threshold</h2>
        <span className="rounded-full bg-blue-100 px-2.5 py-0.5 text-sm font-semibold tabular-nums text-blue-700">
          {value.toFixed(2)}
        </span>
      </div>

      <input
        type="range"
        min="0"
        max="1"
        step="0.05"
        value={value}
        onChange={handleChange}
        className="w-full h-2 cursor-pointer accent-blue-600"
        aria-label="Semantic similarity threshold"
        aria-valuemin={0}
        aria-valuemax={1}
        aria-valuenow={value}
      />

      <div className="flex justify-between text-xs text-slate-400">
        <span>0.0 — broad</span>
        <span>1.0 — strict</span>
      </div>

      <p className="text-xs text-slate-500 italic">{description(value)}</p>
    </section>
  );
}
