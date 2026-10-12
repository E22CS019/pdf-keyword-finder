# lexical_search.py
# Responsibilities:
#   - Perform case-insensitive exact keyword/phrase matching across chunks
#   - Generate a small, explicit set of controlled morphological variants
#     (plural, past-tense, progressive) and search for those too
#   - Avoid substring false positives using word-boundary anchoring
#   - Return structured LexicalResult objects carrying match type and count
#   - Serve as one of two independent retrieval strategies fed to result_merger

from __future__ import annotations

import re
import logging
from typing import Sequence

from backend.models.schemas import Chunk, LexicalResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Variant generation — explicit suffix rules, no external stemmer
# ---------------------------------------------------------------------------
#
# Each rule is a tuple:
#   (condition_suffix, remove_n_chars, add_suffix)
#
# Rules are tried in order; the first whose condition matches is applied.
# Only purely alphabetic single-token keywords get variants; multi-word
# phrases and non-alphabetic tokens are searched exactly.
#
# Examples produced for "invoice":
#   invoices  (add -s)
#   invoiced  (add -d, since keyword already ends in 'e')
#
# Examples for "process":
#   processes (ends in s/x/z/ch/sh → add -es)
#   processed (add -d after final 'e'? No: 'process' ends in 's', add -ed)
#   processing (drop trailing 'e'? No trailing 'e'; add -ing)
#
# The table is intentionally small.  Uncommon patterns are left unhandled
# rather than risked making incorrect guesses.

_VARIANT_RULES: list[tuple[str, int, str]] = [
    # ---- plural ----
    # words ending in s, x, z, ch, sh  →  +es
    (r"(?:s|x|z|ch|sh)$",   0,  "es"),
    # all other words                   →  +s
    ("",                     0,  "s"),
    # ---- past tense / past participle ----
    # already ends in 'e'               →  +d    (invoke→invoked, file→filed)
    (r"e$",                  0,  "d"),
    # ends in consonant+y               →  -y +ied
    (r"[^aeiou]y$",          1,  "ied"),
    # short CVC pattern (stop→stopped)  →  double final consonant +ed
    # (kept out deliberately — too many false positives)
    # default                           →  +ed
    ("",                     0,  "ed"),
    # ---- present participle ----
    # ends in 'e'                       →  drop e, +ing  (invoke→invoking)
    (r"e$",                  1,  "ing"),
    # default                           →  +ing
    ("",                     0,  "ing"),
]


def _generate_variants(keyword: str) -> list[str]:
    """Return controlled morphological variants for *keyword*.

    Only single, purely alphabetic tokens receive variants.  Multi-word
    phrases and tokens containing digits or punctuation are returned unchanged
    (an empty list — callers fall back to exact-only matching).

    Args:
        keyword: Lowercased, stripped keyword token.

    Returns:
        A deduplicated list of variant strings, each different from *keyword*.
    """
    # Only generate variants for single, purely alphabetic words.
    if not keyword.isalpha() or " " in keyword:
        return []

    seen: set[str] = {keyword}  # exclude the keyword itself
    variants: list[str] = []

    for condition, drop, suffix in _VARIANT_RULES:
        if condition and not re.search(condition, keyword):
            continue
        root = keyword[: len(keyword) - drop] if drop else keyword
        candidate = root + suffix
        if candidate not in seen:
            seen.add(candidate)
            variants.append(candidate)

    return variants


# ---------------------------------------------------------------------------
# Word-boundary pattern helpers
# ---------------------------------------------------------------------------

def _boundary_pattern(phrase: str) -> re.Pattern[str]:
    r"""Return a compiled, case-insensitive word-boundary regex for *phrase*.

    For single-word keywords ``\b`` anchors are placed around the whole
    phrase.  For multi-word phrases the anchors are placed around the first
    and last tokens; internal spaces are matched as ``\s+`` to tolerate
    minor whitespace variation.

    Args:
        phrase: Lowercased keyword or phrase.

    Returns:
        Compiled ``re.Pattern`` with ``re.IGNORECASE``.
    """
    tokens = phrase.strip().split()
    # Escape each token for regex safety (handles punctuation in phrases)
    escaped = [re.escape(t) for t in tokens]
    inner = r"\s+".join(escaped)
    pattern = r"\b" + inner + r"\b"
    return re.compile(pattern, re.IGNORECASE)


# ---------------------------------------------------------------------------
# Core normalisation
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lowercase and collapse internal whitespace."""
    return " ".join(text.lower().split())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def lexical_search(
    keyword: str,
    chunks: Sequence[Chunk],
) -> list[LexicalResult]:
    """Search *chunks* for exact and controlled-variant matches of *keyword*.

    Matching rules (applied in order per chunk):

    1. **Exact match** — the keyword (case-insensitive, word-boundary anchored)
       is found in the chunk text.  Produces a ``LexicalResult`` with
       ``match_type="exact"``.

    2. **Variant match** — one or more controlled morphological variants of
       the keyword (plural, past-tense, present participle) are found in the
       chunk text.  Produces a ``LexicalResult`` with
       ``match_type="variant"``.

    If a chunk contains *both* an exact and a variant hit, **both results are
    returned** so that :mod:`~backend.services.result_merger` can keep the
    ``"exact"`` result and discard the ``"variant"`` when deduplicating.

    Word-boundary anchoring (``\b``) prevents substring false positives —
    searching for ``"invoice"`` will not match ``"invoiceless"`` or
    ``"pre-invoice"`` as separate tokens, though it will match ``"invoice,"``
    and ``"invoice."`` because ``\b`` sits between a word character and a
    non-word character.

    Variant generation is intentionally conservative:  only single,
    purely-alphabetic keywords receive variants.  Multi-word phrases are
    matched exactly.  No stemmer or NLP library is used; the variant table
    is small, explicit, and easy to audit.

    Args:
        keyword: The user's search keyword or phrase.  Must be a non-empty
            string.
        chunks: The chunks to search.  Typically all chunks for a document
            as provided by :mod:`~backend.storage.document_store`.

    Returns:
        A list of :class:`~backend.models.schemas.LexicalResult` objects.
        A single chunk may appear more than once (once for ``"exact"``, once
        for ``"variant"``) if both match types are found.  Results are
        ordered: exact matches first, then variant matches; within each
        group, in the same order as the input *chunks* list.
        Returns an empty list for an empty keyword or empty chunk list.

    Raises:
        Nothing — all errors are handled gracefully.
    """
    # ------------------------------------------------------------------
    # 1. Validate / normalise keyword
    # ------------------------------------------------------------------
    if not keyword or not keyword.strip():
        logger.debug("lexical_search: empty keyword, returning [].")
        return []

    normalised_keyword = _normalise(keyword)

    # ------------------------------------------------------------------
    # 2. Build search patterns
    # ------------------------------------------------------------------
    exact_pattern = _boundary_pattern(normalised_keyword)

    # Variants are only generated for single alphabetic tokens
    raw_variants = _generate_variants(normalised_keyword)
    variant_patterns: list[tuple[str, re.Pattern[str]]] = [
        (v, _boundary_pattern(v)) for v in raw_variants
    ]

    # ------------------------------------------------------------------
    # 3. Search each chunk
    # ------------------------------------------------------------------
    exact_results: list[LexicalResult] = []
    variant_results: list[LexicalResult] = []

    for chunk in chunks:
        if not chunk.text or not chunk.text.strip():
            continue  # skip blank chunks

        text = chunk.text  # patterns are IGNORECASE, no need to lowercase

        # ---- exact match ----
        exact_hits = exact_pattern.findall(text)
        if exact_hits:
            exact_results.append(
                LexicalResult(
                    chunk=chunk,
                    keyword=keyword,
                    match_type="exact",
                    match_count=len(exact_hits),
                )
            )

        # ---- variant matches ----
        # Collect total hit count across all variants for this chunk.
        # Produce one LexicalResult per chunk (not per variant) to keep
        # the merger's input simple.
        total_variant_hits = 0
        for _variant_str, vpat in variant_patterns:
            total_variant_hits += len(vpat.findall(text))

        if total_variant_hits > 0:
            variant_results.append(
                LexicalResult(
                    chunk=chunk,
                    keyword=keyword,
                    match_type="variant",
                    match_count=total_variant_hits,
                )
            )

    # Exact results first, then variant results.
    results = exact_results + variant_results

    logger.debug(
        "lexical_search: keyword=%r, chunks=%d, exact=%d, variant=%d",
        keyword,
        len(chunks),
        len(exact_results),
        len(variant_results),
    )
    return results
