// SearchButton.jsx
// Disabled until a document is uploaded and at least one keyword exists.
// Shows a spinner while the search request is in flight.

/**
 * @param {{ onClick: () => void, disabled: boolean, isLoading: boolean }} props
 */
export default function SearchButton({ onClick, disabled, isLoading }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled || isLoading}
      className="w-full rounded-lg bg-indigo-600 px-4 py-3 text-base font-semibold text-white
                 transition hover:bg-indigo-700 active:bg-indigo-800
                 disabled:cursor-not-allowed disabled:bg-slate-300 disabled:text-slate-500
                 shadow-sm"
    >
      {isLoading ? (
        <span className="flex items-center justify-center gap-2">
          <Spinner /> Searching…
        </span>
      ) : (
        <span className="flex items-center justify-center gap-2">
          <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round"
              d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z" />
          </svg>
          Search
        </span>
      )}
    </button>
  );
}

function Spinner() {
  return (
    <svg className="h-5 w-5 animate-spin" fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
  );
}
