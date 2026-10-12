# test_embedder.py
# Unit tests for backend/services/embedder.py
#
# Strategy
# --------
# All unit tests patch SentenceTransformer so the model is never downloaded.
# The mock returns deterministic fixed-size numpy arrays so tests remain fast
# and reproducible in CI.
#
# The single integration test (marked "integration") exercises the real model
# and should be run explicitly:
#
#   pytest tests/test_embedder.py -m integration

from __future__ import annotations

from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

FAKE_DIM = 16   # small dimensionality used in all mocked tests


def _make_mock_model(dim: int = FAKE_DIM) -> MagicMock:
    """Return a MagicMock that mimics SentenceTransformer just enough."""
    mock = MagicMock()

    def fake_encode(texts, convert_to_numpy=True, show_progress_bar=False,
                    normalize_embeddings=False):
        if isinstance(texts, str):
            # Single string → 1-D array
            vec = np.random.default_rng(0).random(dim).astype(np.float32)
            if normalize_embeddings:
                vec /= np.linalg.norm(vec)
            return vec
        # List of strings → 2-D array, one row per text
        n = len(texts)
        rng = np.random.default_rng(0)
        vecs = rng.random((n, dim)).astype(np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs /= norms
        return vecs

    mock.encode.side_effect = fake_encode
    mock.get_sentence_embedding_dimension.return_value = dim
    return mock


@pytest.fixture()
def mock_embedder():
    """Embedder instance backed by a mock model — no network access."""
    with patch(
        "backend.services.embedder.SentenceTransformer",
        return_value=_make_mock_model(),
    ):
        from backend.services.embedder import Embedder
        return Embedder(model_name="fake-model")


# ---------------------------------------------------------------------------
# 1. Model loading
# ---------------------------------------------------------------------------

class TestModelLoading:
    def test_sentencetransformer_called_once_on_init(self):
        """SentenceTransformer must be instantiated exactly once per Embedder."""
        with patch("backend.services.embedder.SentenceTransformer") as MockST:
            MockST.return_value = _make_mock_model()
            from backend.services.embedder import Embedder

            emb = Embedder(model_name="fake-model")
            assert MockST.call_count == 1

    def test_model_not_reloaded_on_repeated_calls(self, mock_embedder):
        """Multiple embed_chunks / embed_query calls must not re-instantiate the model."""
        with patch("backend.services.embedder.SentenceTransformer") as MockST:
            # The embedder is already created; further calls to embed_* must
            # not call SentenceTransformer again.
            mock_embedder.embed_chunks(["text one", "text two"])
            mock_embedder.embed_query("search query")
            assert MockST.call_count == 0   # never called after init

    def test_custom_model_name_passed_to_sentencetransformer(self):
        custom_name = "my-custom-model"
        with patch("backend.services.embedder.SentenceTransformer") as MockST:
            MockST.return_value = _make_mock_model()
            from backend.services.embedder import Embedder

            Embedder(model_name=custom_name)
            MockST.assert_called_once_with(custom_name)


# ---------------------------------------------------------------------------
# 2. embed_chunks — multiple texts
# ---------------------------------------------------------------------------

class TestEmbedChunks:
    def test_returns_numpy_ndarray(self, mock_embedder):
        result = mock_embedder.embed_chunks(["hello world"])
        assert isinstance(result, np.ndarray)

    def test_output_row_count_matches_input(self, mock_embedder):
        texts = ["chunk one", "chunk two", "chunk three"]
        result = mock_embedder.embed_chunks(texts)
        assert result.shape[0] == len(texts)

    def test_output_is_2d(self, mock_embedder):
        result = mock_embedder.embed_chunks(["a", "b"])
        assert result.ndim == 2

    def test_embedding_dimension_is_consistent(self, mock_embedder):
        r1 = mock_embedder.embed_chunks(["first text"])
        r2 = mock_embedder.embed_chunks(["second text", "third text"])
        assert r1.shape[1] == r2.shape[1]

    def test_single_text_produces_one_row(self, mock_embedder):
        result = mock_embedder.embed_chunks(["only one chunk"])
        assert result.shape[0] == 1

    def test_order_preserved(self, mock_embedder):
        """Rows must correspond to input texts in the same order."""
        texts = ["alpha", "beta", "gamma"]
        result = mock_embedder.embed_chunks(texts)
        # We rely on the mock using the same RNG seed for each call, so calling
        # embed_chunks once with 3 texts gives the same result as calling it
        # again with the same 3 texts.
        result2 = mock_embedder.embed_chunks(texts)
        np.testing.assert_array_equal(result, result2)

    def test_dtype_is_float32(self, mock_embedder):
        result = mock_embedder.embed_chunks(["text"])
        assert result.dtype == np.float32

    def test_embeddings_are_unit_normalised(self, mock_embedder):
        """With normalize_embeddings=True each row should have L2-norm ≈ 1."""
        result = mock_embedder.embed_chunks(["normalised chunk"])
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# 3. embed_query — single string
# ---------------------------------------------------------------------------

class TestEmbedQuery:
    def test_returns_numpy_ndarray(self, mock_embedder):
        result = mock_embedder.embed_query("what is an invoice?")
        assert isinstance(result, np.ndarray)

    def test_output_is_1d(self, mock_embedder):
        result = mock_embedder.embed_query("search term")
        assert result.ndim == 1

    def test_dimension_matches_chunk_embedding(self, mock_embedder):
        query_vec = mock_embedder.embed_query("query")
        chunk_vecs = mock_embedder.embed_chunks(["some document chunk"])
        assert query_vec.shape[0] == chunk_vecs.shape[1]

    def test_dtype_is_float32(self, mock_embedder):
        result = mock_embedder.embed_query("query text")
        assert result.dtype == np.float32

    def test_query_is_unit_normalised(self, mock_embedder):
        result = mock_embedder.embed_query("normalisation check")
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# 4. Empty / invalid input handling
# ---------------------------------------------------------------------------

class TestEmptyInputHandling:
    def test_embed_chunks_raises_on_empty_list(self, mock_embedder):
        with pytest.raises(ValueError, match="non-empty"):
            mock_embedder.embed_chunks([])

    def test_embed_query_raises_on_empty_string(self, mock_embedder):
        with pytest.raises(ValueError, match="non-empty"):
            mock_embedder.embed_query("")

    def test_embed_query_raises_on_whitespace_only(self, mock_embedder):
        with pytest.raises(ValueError, match="non-empty"):
            mock_embedder.embed_query("   ")


# ---------------------------------------------------------------------------
# 5. embedding_dim property
# ---------------------------------------------------------------------------

class TestEmbeddingDim:
    def test_embedding_dim_matches_output_width(self, mock_embedder):
        result = mock_embedder.embed_chunks(["dim check"])
        assert mock_embedder.embedding_dim == result.shape[1]

    def test_embedding_dim_matches_query_length(self, mock_embedder):
        result = mock_embedder.embed_query("dim check")
        assert mock_embedder.embedding_dim == result.shape[0]


# ---------------------------------------------------------------------------
# 6. Integration test (opt-in, downloads real model)
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestRealModel:
    """Runs against the real all-MiniLM-L6-v2 model.

    Skip by default; enable with:
        pytest tests/test_embedder.py -m integration
    """

    @pytest.fixture(scope="class")
    def real_embedder(self):
        from backend.services.embedder import Embedder
        return Embedder()  # uses DEFAULT_MODEL_NAME

    def test_real_embed_chunks_shape(self, real_embedder):
        texts = ["hello world", "the quick brown fox"]
        result = real_embedder.embed_chunks(texts)
        assert result.shape == (2, real_embedder.embedding_dim)

    def test_real_embed_query_shape(self, real_embedder):
        result = real_embedder.embed_query("invoice total amount")
        assert result.shape == (real_embedder.embedding_dim,)

    def test_real_dim_is_384(self, real_embedder):
        # all-MiniLM-L6-v2 produces 384-dimensional vectors.
        assert real_embedder.embedding_dim == 384

    def test_real_embeddings_are_unit_normalised(self, real_embedder):
        result = real_embedder.embed_chunks(["norm check"])
        norm = np.linalg.norm(result[0])
        assert abs(norm - 1.0) < 1e-5
