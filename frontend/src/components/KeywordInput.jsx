// KeywordInput.jsx
// Tag/chip-style keyword input.
// Pressing Enter or comma commits the current text as a keyword chip.
// Each chip can be removed individually.

import { useState, useRef } from "react";

const MAX_KEYWORDS = 20;

/**
 * @param {{ keywords: string[], onChange: (keywords: string[]) => void }} props
 */
export default function KeywordInput({ keywords, onChange }) {
  const [inputValue, setInputValue] = useState("");
  const inputRef = useRef(null);

  function addKeyword(raw) {
    const kw = raw.trim();
    if (!kw) return;
    if (keywords.length >= MAX_KEYWORDS) return;
    if (keywords.includes(kw)) return; // no duplicates
    onChange([...keywords, kw]);
    setInputValue("");
  }

  function removeKeyword(index) {
    onChange(keywords.filter((_, i) => i !== index));
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" || e.key === ",") {
      e.preventDefault();
      addKeyword(inputValue);
    } else if (e.key === "Backspace" && inputValue === "" && keywords.length > 0) {
      // delete last chip on empty backspace
      removeKeyword(keywords.length - 1);
    }
  }

  function handleBlur() {
    // commit whatever is typed when the user leaves the field
    if (inputValue.trim()) addKeyword(inputValue);
  }

  const atMax = keywords.length >= MAX_KEYWORDS;

  return (
    <section className="space-y-2">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-slate-700">2. Keywords</h2>
        <span className={`text-xs font-medium tabular-nums ${atMax ? "text-amber-600" : "text-slate-400"}`}>
          {keywords.length} / {MAX_KEYWORDS}
        </span>
      </div>

      {/* Chip container + input */}
      <div
        className="flex min-h-[3rem] flex-wrap gap-1.5 rounded-lg border border-slate-300 bg-white px-2 py-1.5
                   focus-within:border-blue-400 focus-within:ring-2 focus-within:ring-blue-100 cursor-text"
        onClick={() => inputRef.current?.focus()}
      >
        {keywords.map((kw, i) => (
          <Chip key={kw + i} label={kw} onRemove={() => removeKeyword(i)} />
        ))}

        {!atMax && (
          <input
            ref={inputRef}
            value={inputValue}
            onChange={e => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            onBlur={handleBlur}
            placeholder={keywords.length === 0 ? "Type a keyword and press Enter…" : ""}
            className="flex-1 min-w-[8rem] bg-transparent py-0.5 text-sm text-slate-700
                       placeholder:text-slate-400 outline-none"
            aria-label="Add keyword"
          />
        )}
      </div>

      <p className="text-xs text-slate-400">
        Press <kbd className="rounded border border-slate-200 bg-slate-100 px-1 py-0.5 font-mono text-[11px]">Enter</kbd>{" "}
        or <kbd className="rounded border border-slate-200 bg-slate-100 px-1 py-0.5 font-mono text-[11px]">,</kbd>{" "}
        to add · multi-word phrases are supported · max {MAX_KEYWORDS} keywords
      </p>
    </section>
  );
}

function Chip({ label, onRemove }) {
  return (
    <span className="inline-flex items-center gap-1 rounded-full bg-blue-100 px-2.5 py-0.5 text-sm font-medium text-blue-800">
      {label}
      <button
        onClick={e => { e.stopPropagation(); onRemove(); }}
        className="rounded-full p-0.5 hover:bg-blue-200 transition-colors"
        aria-label={`Remove keyword "${label}"`}
      >
        <svg className="h-3 w-3" viewBox="0 0 12 12" fill="currentColor">
          <path d="M2.293 2.293a1 1 0 011.414 0L6 4.586l2.293-2.293a1 1 0 111.414 1.414L7.414 6l2.293 2.293a1 1 0 01-1.414 1.414L6 7.414 3.707 9.707a1 1 0 01-1.414-1.414L4.586 6 2.293 3.707a1 1 0 010-1.414z" />
        </svg>
      </button>
    </span>
  );
}
