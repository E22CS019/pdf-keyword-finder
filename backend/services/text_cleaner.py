# text_cleaner.py
# Responsibilities:
#   - Receive raw text extracted from a PDF page
#   - Repair hyphenated line breaks introduced by PDF line wrapping
#   - Preserve legitimate compound hyphenated words
#   - Normalise excessive whitespace without destroying paragraph boundaries
#   - Collapse unnecessary soft line breaks while keeping paragraph separators
#   - Return clean, plain-text strings ready for chunking

from __future__ import annotations

import re


# ---------------------------------------------------------------------------
# Compiled patterns (compiled once at import time for performance)
# ---------------------------------------------------------------------------

# A hyphenated line break looks like:   "invo-\nice"
# The word fragment before "-\n" must end with a letter.
# The fragment after "\n" must start with a lowercase letter
# (uppercase would suggest a new sentence / proper noun, not a continuation).
_HYPHEN_LINE_BREAK = re.compile(r"([a-zA-Z])-\n([a-z])")

# A "soft" line break is a single newline that is NOT preceded or followed by
# another newline.  These arise when a PDF renderer wraps long text lines
# visually — they don't represent real paragraph boundaries.
_SOFT_LINE_BREAK = re.compile(r"(?<!\n)\n(?!\n)")

# Two or more consecutive newlines signal a paragraph boundary.
# We normalise any run of 2+ newlines down to exactly two newlines so
# the chunker has a consistent separator to split on.
_MULTI_NEWLINE = re.compile(r"\n{2,}")

# Horizontal whitespace: any run of spaces or tabs longer than one character.
_MULTI_SPACE = re.compile(r"[ \t]{2,}")


def clean_page_text(raw_text: str) -> str:
    """Clean raw page text extracted from a PDF page.

    Applies a deterministic sequence of text-normalisation rules to remove
    PDF extraction artefacts while preserving the paragraph structure needed
    by the downstream chunker.  No chunking, embedding, or keyword logic is
    performed here.

    Cleaning pipeline (applied in order):

    1. **Hyphenated line-break repair** — joins words split across lines by
       PDF line-wrapping, e.g. ``"invo-\\nice"`` → ``"invoice"``.
       Only repairs splits where the continuation starts with a lowercase
       letter; legitimate compound words like ``"well-known"`` or
       ``"state-of-the-art"`` are left untouched because they never have a
       newline after the hyphen.

    2. **Soft line-break collapse** — replaces single newlines (visual
       wrapping artefacts) with a space so sentences read continuously.

    3. **Paragraph boundary normalisation** — collapses any run of two or
       more consecutive newlines down to exactly ``"\\n\\n"``, giving the
       chunker a uniform separator.

    4. **Horizontal whitespace normalisation** — collapses runs of spaces
       or tabs inside a line down to a single space.

    5. **Strip** — removes leading and trailing whitespace from the result.

    Args:
        raw_text: Raw text for a single PDF page as returned by
            :func:`~backend.services.pdf_processor.extract_pages`.

    Returns:
        Cleaned text string.  Returns an empty string if the input is
        ``None``, empty, or contains only whitespace.
    """
    if not raw_text or not raw_text.strip():
        return ""

    text = raw_text

    # Step 1 — repair hyphenated line breaks
    # "invo-\nice" → "invoice"
    # The hyphen is removed and the two fragments are joined directly.
    text = _HYPHEN_LINE_BREAK.sub(r"\1\2", text)

    # Step 2 — collapse soft (single) line breaks into a space
    # "This is a long\nsentence." → "This is a long sentence."
    text = _SOFT_LINE_BREAK.sub(" ", text)

    # Step 3 — normalise paragraph boundaries to exactly two newlines
    # "\n\n\n\n" → "\n\n"
    text = _MULTI_NEWLINE.sub("\n\n", text)

    # Step 4 — collapse multiple spaces / tabs on a single line
    # "word    word" → "word word"
    text = _MULTI_SPACE.sub(" ", text)

    # Step 5 — strip leading/trailing whitespace
    return text.strip()
