# test_lexical_search.py
# Unit tests for backend/services/lexical_search.py

import pytest
from backend.models.schemas import Chunk, LexicalResult
from backend.services.lexical_search import lexical_search, _generate_variants


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(text: str, chunk_id: str = "c0", page: int = 1) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc-001",
        page_number=page,
        chunk_index=0,
        text=text,
    )


def _search(keyword: str, *texts: str) -> list[LexicalResult]:
    """Convenience: build chunks from plain strings and call lexical_search."""
    chunks = [_chunk(t, chunk_id=f"c{i}") for i, t in enumerate(texts)]
    return lexical_search(keyword, chunks)


def _exact(results: list[LexicalResult]) -> list[LexicalResult]:
    return [r for r in results if r.match_type == "exact"]


def _variant(results: list[LexicalResult]) -> list[LexicalResult]:
    return [r for r in results if r.match_type == "variant"]


# ---------------------------------------------------------------------------
# 1. Exact matching
# ---------------------------------------------------------------------------

class TestExactMatch:
    def test_exact_lowercase_match(self):
        results = _search("invoice", "Please send the invoice by Friday.")
        assert len(_exact(results)) == 1

    def test_exact_match_returns_lexical_result_instance(self):
        results = _search("invoice", "invoice")
        assert isinstance(results[0], LexicalResult)

    def test_exact_match_type_is_exact(self):
        results = _search("invoice", "The invoice is overdue.")
        assert _exact(results)[0].match_type == "exact"

    def test_exact_match_preserves_original_keyword(self):
        results = _search("Invoice", "The Invoice is overdue.")
        assert results[0].keyword == "Invoice"

    def test_exact_match_count_reflects_occurrences(self):
        results = _search("invoice", "invoice invoice invoice")
        assert _exact(results)[0].match_count == 3

    def test_case_insensitive_uppercase_query(self):
        results = _search("INVOICE", "The invoice total is $100.")
        assert len(_exact(results)) == 1

    def test_case_insensitive_mixed_case_in_text(self):
        results = _search("invoice", "Invoice INVOICE invoice")
        assert _exact(results)[0].match_count == 3

    def test_exact_match_with_trailing_punctuation(self):
        # "invoice," — the comma is a non-word char, so \b still fires.
        results = _search("invoice", "Please review the invoice, then sign.")
        assert len(_exact(results)) == 1

    def test_exact_match_with_full_stop(self):
        results = _search("invoice", "Attached is the invoice.")
        assert len(_exact(results)) == 1

    def test_no_match_on_unrelated_text(self):
        results = _search("invoice", "The weather is sunny today.")
        assert results == []


# ---------------------------------------------------------------------------
# 2. Substring false-positive prevention
# ---------------------------------------------------------------------------

class TestSubstringFalsePositives:
    def test_invoice_does_not_match_invoiceless(self):
        results = _search("invoice", "This system is invoiceless.")
        assert _exact(results) == []

    def test_invoice_does_not_match_preinvoice(self):
        results = _search("invoice", "A preinvoice stage exists.")
        assert _exact(results) == []

    def test_word_does_not_match_as_prefix(self):
        # "key" must not match "keyword"
        results = _search("key", "The keyword is important.")
        assert _exact(results) == []

    def test_word_does_not_match_as_suffix(self):
        # "port" must not match "report"
        results = _search("port", "See the report for details.")
        assert _exact(results) == []

    def test_word_matches_when_standalone(self):
        # "key" DOES match when it stands alone
        results = _search("key", "The key is in the lock.")
        assert len(_exact(results)) == 1


# ---------------------------------------------------------------------------
# 3. Morphological variant matching
# ---------------------------------------------------------------------------

class TestVariantMatching:
    def test_plural_variant_invoices(self):
        results = _search("invoice", "We received three invoices today.")
        assert len(_variant(results)) == 1
        assert _variant(results)[0].match_type == "variant"

    def test_past_tense_variant_invoiced(self):
        results = _search("invoice", "The client was invoiced last month.")
        assert len(_variant(results)) == 1

    def test_variant_count_reflects_occurrences(self):
        results = _search("invoice", "invoices invoices invoices")
        assert _variant(results)[0].match_count == 3

    def test_variant_keyword_is_original_keyword(self):
        results = _search("invoice", "The invoices are overdue.")
        assert _variant(results)[0].keyword == "invoice"

    def test_process_plural_processes(self):
        # "process" ends in 's' → variant is "processes"
        results = _search("process", "The processes run in parallel.")
        assert len(_variant(results)) == 1

    def test_file_past_tense_filed(self):
        # "file" ends in 'e' → variant is "filed"
        results = _search("file", "The document was filed correctly.")
        assert len(_variant(results)) == 1

    def test_no_variant_match_when_only_exact_present(self):
        # Only the exact word is in the text; no variant hit expected.
        results = _search("invoice", "The invoice is here.")
        assert _variant(results) == []


# ---------------------------------------------------------------------------
# 4. Exact takes priority over variant for same chunk
# ---------------------------------------------------------------------------

class TestExactPriorityOverVariant:
    def test_both_results_returned_when_chunk_has_exact_and_variant(self):
        # "invoice" (exact) and "invoices" (variant) both in same chunk.
        results = _search("invoice", "The invoice and invoices are ready.")
        chunk_ids_exact   = {r.chunk.chunk_id for r in _exact(results)}
        chunk_ids_variant = {r.chunk.chunk_id for r in _variant(results)}
        # Same chunk appears in both lists.
        assert chunk_ids_exact == chunk_ids_variant == {"c0"}

    def test_exact_result_appears_before_variant_result(self):
        results = _search("invoice", "invoice and invoices and invoiced")
        # First result must be exact.
        assert results[0].match_type == "exact"

    def test_exact_result_listed_first_regardless_of_text_order(self):
        # Even when the variant token appears first in the text.
        results = _search("invoice", "invoices then invoice")
        assert results[0].match_type == "exact"

    def test_exact_and_variant_are_separate_result_objects(self):
        results = _search("invoice", "invoice and invoices")
        exact_r   = _exact(results)
        variant_r = _variant(results)
        assert len(exact_r) == 1
        assert len(variant_r) == 1
        assert exact_r[0] is not variant_r[0]


# ---------------------------------------------------------------------------
# 5. Multi-word phrase matching
# ---------------------------------------------------------------------------

class TestMultiWordPhrase:
    def test_exact_multi_word_match(self):
        results = _search("payment terms", "The payment terms are Net 30.")
        assert len(_exact(results)) == 1

    def test_case_insensitive_multi_word(self):
        results = _search("Payment Terms", "The PAYMENT TERMS are Net 30.")
        assert len(_exact(results)) == 1

    def test_multi_word_does_not_match_partial(self):
        # Only "payment" is present, not the full phrase.
        results = _search("payment terms", "The payment is overdue.")
        assert _exact(results) == []

    def test_multi_word_does_not_match_reversed_order(self):
        results = _search("payment terms", "The terms payment are unclear.")
        assert _exact(results) == []

    def test_multi_word_match_count(self):
        results = _search(
            "payment terms",
            "The payment terms and payment terms are both listed.",
        )
        assert _exact(results)[0].match_count == 2

    def test_multi_word_produces_no_variants(self):
        # Multi-word phrases do not get morphological variants.
        results = _search("payment terms", "The payments terms are Net 30.")
        assert _variant(results) == []


# ---------------------------------------------------------------------------
# 6. Empty / edge-case inputs
# ---------------------------------------------------------------------------

class TestEmptyInputs:
    def test_empty_keyword_returns_empty(self):
        results = _search("", "Some text with content.")
        assert results == []

    def test_whitespace_keyword_returns_empty(self):
        results = _search("   ", "Some text.")
        assert results == []

    def test_empty_chunk_list_returns_empty(self):
        results = lexical_search("invoice", [])
        assert results == []

    def test_blank_chunk_text_returns_no_result(self):
        chunks = [_chunk(""), _chunk("   "), _chunk("\n\n")]
        results = lexical_search("invoice", chunks)
        assert results == []

    def test_match_in_one_of_several_chunks(self):
        results = _search(
            "invoice",
            "Nothing relevant here.",
            "The invoice is attached.",
            "Also nothing here.",
        )
        assert len(_exact(results)) == 1
        assert results[0].chunk.chunk_id == "c1"


# ---------------------------------------------------------------------------
# 7. Multiple chunks — correct chunk identity in results
# ---------------------------------------------------------------------------

class TestMultipleChunks:
    def test_matches_returned_for_each_matching_chunk(self):
        results = _search(
            "invoice",
            "Invoice one is here.",
            "Invoice two is here.",
            "Nothing to see here.",
        )
        assert len(_exact(results)) == 2

    def test_chunk_ids_are_preserved_correctly(self):
        chunks = [
            _chunk("The invoice total.", chunk_id="chunk-A"),
            _chunk("No match here.",    chunk_id="chunk-B"),
            _chunk("Another invoice.",  chunk_id="chunk-C"),
        ]
        results = lexical_search("invoice", chunks)
        matched_ids = {r.chunk.chunk_id for r in _exact(results)}
        assert matched_ids == {"chunk-A", "chunk-C"}

    def test_page_number_preserved_in_result(self):
        chunks = [_chunk("invoice here", chunk_id="c0", page=7)]
        results = lexical_search("invoice", chunks)
        assert results[0].chunk.page_number == 7


# ---------------------------------------------------------------------------
# 8. Variant generation unit tests (_generate_variants)
# ---------------------------------------------------------------------------

class TestGenerateVariants:
    def test_invoice_produces_invoices(self):
        variants = _generate_variants("invoice")
        assert "invoices" in variants

    def test_invoice_produces_invoiced(self):
        variants = _generate_variants("invoice")
        assert "invoiced" in variants

    def test_process_produces_processes(self):
        variants = _generate_variants("process")
        assert "processes" in variants

    def test_keyword_itself_not_in_variants(self):
        for word in ("invoice", "process", "file", "report"):
            assert word not in _generate_variants(word)

    def test_variants_are_unique(self):
        for word in ("invoice", "file", "attach", "process"):
            v = _generate_variants(word)
            assert len(v) == len(set(v)), f"duplicates in variants for '{word}'"

    def test_multi_word_returns_no_variants(self):
        assert _generate_variants("payment terms") == []

    def test_numeric_token_returns_no_variants(self):
        assert _generate_variants("2024") == []

    def test_alphanumeric_returns_no_variants(self):
        assert _generate_variants("item1") == []
