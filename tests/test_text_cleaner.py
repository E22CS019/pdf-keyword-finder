# test_text_cleaner.py
# Unit tests for backend/services/text_cleaner.py

import pytest
from backend.services.text_cleaner import clean_page_text


# ---------------------------------------------------------------------------
# 1. Hyphenated line-break repair
# ---------------------------------------------------------------------------

class TestHyphenatedLineBreaks:
    """Words split across lines by PDF line-wrapping must be rejoined."""

    def test_basic_hyphen_line_break(self):
        assert clean_page_text("invo-\nice") == "invoice"

    def test_hyphen_line_break_mid_sentence(self):
        raw = "The system gen-\nerates a report daily."
        assert clean_page_text(raw) == "The system generates a report daily."

    def test_multiple_hyphen_line_breaks(self):
        raw = "This is a compli-\ncated and time-con-\nsuming process."
        assert clean_page_text(raw) == "This is a complicated and time-consuming process."

    def test_hyphen_break_at_end_of_paragraph(self):
        # Repair must work even when the hyphenated break occurs just before
        # a paragraph boundary.
        raw = "See the follow-\ning section.\n\nNext paragraph."
        result = clean_page_text(raw)
        assert "following" in result
        assert "Next paragraph." in result


# ---------------------------------------------------------------------------
# 2. Legitimate hyphenated words are preserved
# ---------------------------------------------------------------------------

class TestLegitimateHyphenatedWords:
    """Compound words must NOT be joined — the hyphen carries meaning."""

    def test_well_known(self):
        assert clean_page_text("well-known algorithm") == "well-known algorithm"

    def test_state_of_the_art(self):
        assert clean_page_text("state-of-the-art model") == "state-of-the-art model"

    def test_hyphen_before_uppercase_not_repaired(self):
        # If the continuation starts with an uppercase letter it is a new
        # sentence or proper noun, not a split word — the hyphen stays.
        raw = "See section-\nResults for details."
        result = clean_page_text(raw)
        # The "\n" after the hyphen becomes a space; hyphen is kept.
        assert "section-" in result
        assert "Results" in result

    def test_inline_hyphen_unchanged(self):
        text = "A high-level, well-known, state-of-the-art approach."
        assert clean_page_text(text) == text


# ---------------------------------------------------------------------------
# 3. Excessive / multiple spaces
# ---------------------------------------------------------------------------

class TestExcessiveSpaces:
    """Multiple consecutive spaces and tabs must be collapsed to one space."""

    def test_multiple_spaces(self):
        assert clean_page_text("word    word") == "word word"

    def test_tabs_between_words(self):
        assert clean_page_text("col1\t\t\tcol2") == "col1 col2"

    def test_mixed_spaces_and_tabs(self):
        assert clean_page_text("a \t  b") == "a b"

    def test_single_space_unchanged(self):
        assert clean_page_text("one two three") == "one two three"


# ---------------------------------------------------------------------------
# 4. Unnecessary (soft) line breaks
# ---------------------------------------------------------------------------

class TestUnnecessaryLineBreaks:
    """Single newlines from visual line-wrapping must collapse into a space."""

    def test_single_newline_becomes_space(self):
        assert clean_page_text("This is a long\nsentence.") == "This is a long sentence."

    def test_multiple_soft_breaks_in_paragraph(self):
        raw = "Line one\nline two\nline three."
        assert clean_page_text(raw) == "Line one line two line three."

    def test_soft_break_does_not_create_double_space(self):
        # Space before newline should not produce two spaces after collapse.
        raw = "end of line \nnext line"
        result = clean_page_text(raw)
        assert "  " not in result


# ---------------------------------------------------------------------------
# 5. Paragraph boundary preservation
# ---------------------------------------------------------------------------

class TestParagraphPreservation:
    """Double (or more) newlines mark paragraph boundaries and must be kept."""

    def test_double_newline_preserved(self):
        raw = "Paragraph one.\n\nParagraph two."
        result = clean_page_text(raw)
        assert "\n\n" in result
        assert result == "Paragraph one.\n\nParagraph two."

    def test_triple_newline_normalised_to_double(self):
        raw = "First.\n\n\nSecond."
        result = clean_page_text(raw)
        assert "\n\n\n" not in result
        assert "\n\n" in result
        assert "First." in result
        assert "Second." in result

    def test_many_newlines_normalised(self):
        raw = "A.\n\n\n\n\n\nB."
        result = clean_page_text(raw)
        assert result == "A.\n\nB."

    def test_multiple_paragraphs_all_preserved(self):
        raw = "Para one.\n\nPara two.\n\nPara three."
        result = clean_page_text(raw)
        parts = result.split("\n\n")
        assert len(parts) == 3
        assert parts[0] == "Para one."
        assert parts[1] == "Para two."
        assert parts[2] == "Para three."

    def test_paragraph_with_soft_breaks_inside(self):
        # Soft breaks inside a paragraph collapse; the boundary between
        # paragraphs is still a double newline.
        raw = "Line one\nline two.\n\nNew paragraph."
        result = clean_page_text(raw)
        assert result == "Line one line two.\n\nNew paragraph."


# ---------------------------------------------------------------------------
# 6. Empty / whitespace-only input
# ---------------------------------------------------------------------------

class TestEmptyInput:
    """Edge-case inputs must return an empty string without raising."""

    def test_empty_string(self):
        assert clean_page_text("") == ""

    def test_none_equivalent_whitespace_only(self):
        assert clean_page_text("   ") == ""

    def test_only_newlines(self):
        assert clean_page_text("\n\n\n") == ""

    def test_only_tabs_and_spaces(self):
        assert clean_page_text("\t  \t  ") == ""

    def test_single_word(self):
        assert clean_page_text("hello") == "hello"

    def test_leading_trailing_whitespace_stripped(self):
        assert clean_page_text("  hello world  ") == "hello world"
