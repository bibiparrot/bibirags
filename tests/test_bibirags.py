"""
Unit tests for bibirags.

These tests mock all external dependencies (LiteLLM, vector stores) so they
run offline without any API keys or locally running models.
"""

from __future__ import annotations

import pathlib
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

class TestLLMHelpers:
    def test_litellm_embedding(self):
        fake_vec = [0.1, 0.2, 0.3]
        mock_response = MagicMock()
        mock_response.data = [{"embedding": fake_vec}]

        with patch("litellm.embedding", return_value=mock_response):
            from bibirags.llm import litellm_embedding

            result = litellm_embedding(["hello"], "text-embedding-3-small")

        assert result == [fake_vec]

    def test_litellm_complete(self):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Paris"

        with patch("litellm.completion", return_value=mock_response):
            from bibirags.llm import litellm_complete

            result = litellm_complete("What is the capital of France?", "gpt-4o-mini")

        assert result == "Paris"

    @pytest.mark.asyncio
    async def test_alitellm_embedding(self):
        fake_vec = [0.4, 0.5, 0.6]
        mock_response = MagicMock()
        mock_response.data = [{"embedding": fake_vec}]

        import asyncio

        async def mock_aembedding(**kwargs):
            return mock_response

        with patch("litellm.aembedding", side_effect=mock_aembedding):
            from bibirags.llm import alitellm_embedding

            result = await alitellm_embedding(["hello"], "text-embedding-3-small")

        assert result == [fake_vec]

    @pytest.mark.asyncio
    async def test_alitellm_embedding_cleans_empty_strings(self):
        """Empty strings must be replaced with 'empty'."""
        captured = {}

        async def mock_aembedding(**kwargs):
            captured["input"] = kwargs.get("input", [])
            mock_resp = MagicMock()
            mock_resp.data = [{"embedding": [0.0]}] * len(captured["input"])
            return mock_resp

        with patch("litellm.aembedding", side_effect=mock_aembedding):
            from bibirags.llm import alitellm_embedding

            await alitellm_embedding(["", "  "], "text-embedding-3-small")

        assert captured["input"] == ["empty", "empty"]

    @pytest.mark.asyncio
    async def test_get_embedding_dim(self):
        async def mock_aembedding(**kwargs):
            mock_resp = MagicMock()
            mock_resp.data = [{"embedding": [0.1] * 768}]
            return mock_resp

        with patch("litellm.aembedding", side_effect=mock_aembedding):
            from bibirags.llm import get_embedding_dim

            dim = await get_embedding_dim("some-model")

        assert dim == 768


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

class TestChunkDocs:
    def test_returns_empty_list_for_empty_directory(self, tmp_path):
        from bibirags.chunking import chunk_docs

        result = chunk_docs(tmp_path)
        assert result == []

    def test_chunks_txt_file(self, tmp_path):
        txt = tmp_path / "sample.txt"
        txt.write_text("Hello world. " * 200)  # enough to produce multiple chunks

        from bibirags.chunking import chunk_docs

        chunks = chunk_docs(tmp_path, chunk_size=100, chunk_overlap=10)
        assert len(chunks) > 1
        assert all(isinstance(c, str) for c in chunks)


# ---------------------------------------------------------------------------
# Qdrant backend
# ---------------------------------------------------------------------------

class TestQdrantBackend:
    def _fake_embedding(self, texts, model):
        return [[0.1, 0.2, 0.3] for _ in texts]

    def test_save_qdrant(self, tmp_path):
        with (
            patch("bibirags.backends.qdrant.litellm_embedding", side_effect=self._fake_embedding),
            patch("qdrant_client.QdrantClient") as MockClient,
        ):
            instance = MockClient.return_value
            instance.collection_exists.return_value = False

            from bibirags.backends.qdrant import save_qdrant

            save_qdrant(["chunk one", "chunk two"], str(tmp_path), "fake-model")

            instance.create_collection.assert_called_once()
            assert instance.upsert.call_count == 2

    def test_search_qdrant(self, tmp_path):
        fake_point = MagicMock()
        fake_point.payload = {"text": "chunk one"}
        fake_point.score = 0.99

        fake_result = MagicMock()
        fake_result.points = [fake_point]

        with (
            patch("bibirags.backends.qdrant.litellm_embedding", side_effect=self._fake_embedding),
            patch("qdrant_client.QdrantClient") as MockClient,
        ):
            instance = MockClient.return_value
            instance.query_points.return_value = fake_result

            from bibirags.backends.qdrant import search_qdrant

            result = search_qdrant("what happened?", str(tmp_path), "fake-model", top_k=1)

        assert result == ["chunk one"]

    def test_query_qdrant(self, tmp_path):
        fake_point = MagicMock()
        fake_point.payload = {"text": "chunk one"}
        fake_point.score = 0.99

        fake_result = MagicMock()
        fake_result.points = [fake_point]

        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = "The answer is 42."

        with (
            patch("bibirags.backends.qdrant.litellm_embedding", side_effect=self._fake_embedding),
            patch("bibirags.backends.qdrant.litellm_complete", return_value="The answer is 42."),
            patch("qdrant_client.QdrantClient") as MockClient,
        ):
            instance = MockClient.return_value
            instance.query_points.return_value = fake_result

            from bibirags.backends.qdrant import query_qdrant

            answer, sources = query_qdrant(
                "What is the answer?", str(tmp_path), "gpt-4o-mini", "fake-model"
            )

        assert answer == "The answer is 42."
        assert "chunk one" in sources


# ---------------------------------------------------------------------------
# Package metadata
# ---------------------------------------------------------------------------

class TestPackageMetadata:
    def test_version_exposed(self):
        import bibirags

        assert hasattr(bibirags, "__version__")
        assert bibirags.__version__ == "0.1.0"

    def test_public_api_exported(self):
        import bibirags

        for name in [
            "chunk_docs",
            "litellm_embedding",
            "litellm_complete",
            "save_qdrant",
            "search_qdrant",
            "query_qdrant",
            "save_chroma",
            "search_chroma",
            "query_chroma",
            "save_txtai",
            "search_txtai",
            "query_txtai",
            "save_lightrag",
            "search_lightrag",
            "query_lightrag",
        ]:
            assert hasattr(bibirags, name), f"Missing export: {name}"
