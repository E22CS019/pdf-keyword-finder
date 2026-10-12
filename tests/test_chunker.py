# test_chunker.py
# Unit tests for backend/services/chunker.py

import pytest
from backend.services.chunker import chunk_page
from backend.models.schemas import Chunk

# Helpers
DOC_ID = "doc-001"
PAGE = 1


def make_chunks(text: str, page: int = PAGE, doc_id: str = DOC_ID, max_chars=None):
    return chunk_page(text, page_number=page, doc_id=doc_id, max_chars=max_chars)


# ---------------------------------------------------------------------------
# 1. Single paragraph
# ---------------------------------------------------------------------------

class TestSingleParagraph:
    def test_returns_one_chunk(self):
        chunks = make_chunks("This is a single paragraph.")
        assert len(chunks) == 1

    def test_chunk_text_matches_input(self):
        chunks = make_chunks("Hello world.")
        assert chunks[0].text == "Hello world."

    def test_chunk_is_chunk_instance(self):
        chunks = make_chunks("Some text.")
        assert isinstance(chunks[0], Chunk)


# ---------------------------------------------------------------------------
# 2. Multiple paragraphs
# ---------------------------------------------------------------------------

class TestMultipleParagraphs:
    def test_two_paragraphs_give_two_chunks(self):
        text = "First paragraph.\n\nSecond paragraph."
        chunks = make_chunks(text)
        assert len(chunks) == 2

    def test_three_paragraphs_give_three_chunks(self):
        text = "One.\n\nTwo.\n\nThree."
        chunks = make_chunks(text)
        assert len(chunks) == 3

    def test_paragraph_text_is_preserved(self):
        text = "Alpha.\n\nBeta.\n\nGamma."
        chunks = make_chunks(text)
        assert chunks[0].text == "Alpha."
        assert chunks[1].text == "Beta."
        assert chunks[2].text == "Gamma."

    def test_chunk_order_matches_paragraph_order(self):
        text = "First.\n\nSecond.\n\nThird."
        chunks = make_chunks(text)
        assert chunks[0].text == "First."
        assert chunks[1].text == "Second."
        assert chunks[2].text == "Third."


# ---------------------------------------------------------------------------
# 3. Empty and whitespace-only input
# ---------------------------------------------------------------------------

class TestEmptyInput:
    def test_empty_string_returns_no_chunks(self):
        assert make_chunks("") == []

    def test_whitespace_only_returns_no_chunks(self):
        assert make_chunks("   \n\n   ") == []

    def test_only_newlines_returns_no_chunks(self):
        assert make_chunks("\n\n\n\n") == []

    def test_none_like_empty_returns_no_chunks(self):
        # Passing an empty string simulates a blank page from pdf_processor.
        assert make_chunks("") == []


# ---------------------------------------------------------------------------
# 4. Empty paragraphs between valid ones
# ---------------------------------------------------------------------------

class TestEmptyParagraphsSkipped:
    def test_leading_empty_paragraph_ignored(self):
        # A leading "\n\n" produces an empty first split — must be skipped.
        text = "\n\nActual content."
        chunks = make_chunks(text)
        assert len(chunks) == 1
        assert chunks[0].text == "Actual content."

    def test_trailing_empty_paragraph_ignored(self):
        text = "Content.\n\n"
        chunks = make_chunks(text)
        assert len(chunks) == 1

    def test_consecutive_separators_collapsed(self):
        # Two "\n\n" in a row produce an empty paragraph between them.
        text = "Para one.\n\n\n\nPara two."
        # text_cleaner normalises these to "\n\n", but chunker must also be
        # robust if it receives raw multi-blank-line input.
        chunks = make_chunks(text)
        assert len(chunks) == 2
        assert chunks[0].text == "Para one."
        assert chunks[1].text == "Para two."

    def test_only_empty_paragraphs_returns_no_chunks(self):
        text = "\n\n\n\n\n\n"
        assert make_chunks(text) == []


# ---------------------------------------------------------------------------
# 5. Page number preservation
# ---------------------------------------------------------------------------

class TestPageNumberPreservation:
    def test_single_chunk_carries_correct_page(self):
        chunks = make_chunks("Text.", page=7)
        assert chunks[0].page_number == 7

    def test_all_chunks_carry_same_page_number(self):
        text = "Para one.\n\nPara two.\n\nPara three."
        chunks = make_chunks(text, page=42)
        assert all(c.page_number == 42 for c in chunks)

    def test_page_one(self):
        chunks = make_chunks("Content.", page=1)
        assert chunks[0].page_number == 1

    def test_large_page_number(self):
        chunks = make_chunks("Content.", page=999)
        assert chunks[0].page_number == 999


# ---------------------------------------------------------------------------
# 6. Unique chunk IDs
# ---------------------------------------------------------------------------

class TestUniqueChunkIDs:
    def test_single_chunk_id_format(self):
        chunks = make_chunks("Text.", page=3, doc_id="mydoc")
        assert chunks[0].chunk_id == "mydoc-p3-c0"

    def test_multiple_chunks_have_unique_ids(self):
        text = "Para one.\n\nPara two.\n\nPara three."
        chunks = make_chunks(text, page=1, doc_id="doc-x")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_chunk_ids_are_sequential(self):
        text = "A.\n\nB.\n\nC."
        chunks = make_chunks(text, page=2, doc_id="seq-doc")
        assert chunks[0].chunk_id == "seq-doc-p2-c0"
        assert chunks[1].chunk_id == "seq-doc-p2-c1"
        assert chunks[2].chunk_id == "seq-doc-p2-c2"

    def test_chunk_index_matches_position(self):
        text = "First.\n\nSecond."
        chunks = make_chunks(text)
        assert chunks[0].chunk_index == 0
        assert chunks[1].chunk_index == 1

    def test_doc_id_is_embedded_in_chunk_id(self):
        chunks = make_chunks("Text.", doc_id="report-2024")
        assert "report-2024" in chunks[0].chunk_id


# ---------------------------------------------------------------------------
# 7. Large paragraph splitting
# ---------------------------------------------------------------------------

class TestLargeParagraphSplitting:
    # A 200-char paragraph limit is used across these tests for predictability.
    MAX = 200

    def _long_paragraph(self, n_sentences: int = 10) -> str:
        """Build a paragraph guaranteed to exceed MAX characters."""
        sentence = "This is a moderately long sentence used for testing purposes."
        return "  ".join([sentence] * n_sentences)

    def test_paragraph_under_limit_not_split(self):
        text = "Short paragraph."
        chunks = make_chunks(text, max_chars=self.MAX)
        assert len(chunks) == 1

    def test_paragraph_exactly_at_limit_not_split(self):
        # Edge case: a paragraph that is exactly max_chars long.
        text = "A" * self.MAX
        chunks = make_chunks(text, max_chars=self.MAX)
        assert len(chunks) == 1

    def test_large_paragraph_is_split(self):
        text = self._long_paragraph(n_sentences=10)
        assert len(text) > self.MAX
        chunks = make_chunks(text, max_chars=self.MAX)
        assert len(chunks) > 1

    def test_split_chunks_each_within_limit(self):
        text = self._long_paragraph(n_sentences=10)
        chunks = make_chunks(text, max_chars=self.MAX)
        oversized = [c for c in chunks if len(c.text) > self.MAX]
        # Only a single sentence that itself exceeds the limit is allowed to
        # exceed it; with our test sentences (~60 chars each) none should.
        assert oversized == []

    def test_split_chunks_preserve_page_number(self):
        text = self._long_paragraph(n_sentences=10)
        chunks = make_chunks(text, page=5, max_chars=self.MAX)
        assert all(c.page_number == 5 for c in chunks)

    def test_split_chunks_have_unique_ids(self):
        text = self._long_paragraph(n_sentences=10)
        chunks = make_chunks(text, max_chars=self.MAX)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_no_content_lost_in_split(self):
        """All words from the original paragraph must appear somewhere in the
        resulting chunks."""
        text = self._long_paragraph(n_sentences=5)
        chunks = make_chunks(text, max_chars=self.MAX)
        combined = " ".join(c.text for c in chunks)
        for word in text.split():
            assert word in combined

    def test_normal_paragraph_not_split_when_max_chars_set(self):
        text = "Short paragraph.\n\nAnother short one."
        chunks = make_chunks(text, max_chars=self.MAX)
        # Both paragraphs are well under 200 chars — neither should be split.
        assert len(chunks) == 2


# ---------------------------------------------------------------------------
# 8. No-limit default (max_chars=None)
# ---------------------------------------------------------------------------

class TestNoLimitDefault:
    def test_very_large_paragraph_not_split_by_default(self):
        text = "Word. " * 500  # ~3000 chars
        chunks = make_chunks(text)
        assert len(chunks) == 1

    def test_default_max_chars_is_none(self):
        import inspect
        from backend.services.chunker import chunk_page
        sig = inspect.signature(chunk_page)
        assert sig.parameters["max_chars"].default is None
