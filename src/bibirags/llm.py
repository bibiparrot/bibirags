"""
LLM utility functions using LiteLLM.

All public functions accept a :class:`LitellmConfDict` instead of bare
``embed_model`` / ``llm_model`` strings so that API base URLs and keys can be
passed uniformly across every backend.
"""

from __future__ import annotations

from typing import TypedDict

from loguru import logger


# ---------------------------------------------------------------------------
# Configuration dict
# ---------------------------------------------------------------------------


class LitellmConfDict(TypedDict, total=False):
    """Unified LiteLLM configuration passed to every bibirags function.

    Fields
    ------
    embed_model:
        LiteLLM-compatible embedding model string, e.g.
        ``"ollama/bge-m3:latest"`` or ``"text-embedding-3-small"``.
    llm_model:
        LiteLLM-compatible generative model string, e.g.
        ``"ollama/gemma4:e2b"`` or ``"gpt-4o"``.
    api_base:
        Optional custom API base URL (e.g. ``"http://localhost:11434"`` for
        Ollama, or a proxy endpoint).
    api_key:
        Optional API key.  When omitted LiteLLM falls back to the relevant
        environment variable (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, etc.).
    """

    embed_model: str
    llm_model: str
    api_base: str
    api_key: str


# ---------------------------------------------------------------------------
# Sync helpers
# ---------------------------------------------------------------------------


def litellm_embedding(texts: list[str], conf: LitellmConfDict) -> list[list[float]]:
    """Return embeddings for *texts* using ``conf["embed_model"]``.

    Parameters
    ----------
    texts:
        List of strings to embed.
    conf:
        :class:`LitellmConfDict` with at least ``embed_model`` set.

    Returns
    -------
    list[list[float]]
        One embedding vector per input text.
    """
    import litellm

    response = litellm.embedding(
        model=conf["embed_model"],
        input=texts,
        api_base=conf.get("api_base"),
        api_key=conf.get("api_key"),
        num_retries=3,  # 重试次数
        retry_after=1,  # 重试间隔
        timeout=60,  # 超时时间
    )
    return [r["embedding"] for r in response.data]


def litellm_complete(content: str, conf: LitellmConfDict) -> str:
    """Return a completion for *content* using ``conf["llm_model"]``.

    Parameters
    ----------
    content:
        The user prompt.
    conf:
        :class:`LitellmConfDict` with at least ``llm_model`` set.

    Returns
    -------
    str
        The model's response text.
    """
    import litellm

    response = litellm.completion(
        model=conf["llm_model"],
        messages=[{"content": content, "role": "user"}],
        temperature=0,
        api_base=conf.get("api_base"),
        api_key=conf.get("api_key"),
        num_retries=3,  # 重试次数
        retry_after=1,  # 重试间隔
        timeout=60,  # 超时时间
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Async helpers
# ---------------------------------------------------------------------------


async def alitellm_embedding(texts: list[str], conf: LitellmConfDict) -> list[list[float]]:
    """Async version of :func:`litellm_embedding`.

    Empty strings are replaced with ``"empty"`` to avoid upstream API errors.
    """
    import litellm

    clean_texts = [(t.strip() if isinstance(t, str) else str(t)) for t in texts]
    clean_texts = [t if t else "empty" for t in clean_texts]

    response = await litellm.aembedding(
        model=conf["embed_model"],
        input=clean_texts,
        api_base=conf.get("api_base"),
        api_key=conf.get("api_key"),
        num_retries=3,  # 重试次数
        retry_after=1,  # 重试间隔
        timeout=60,  # 超时时间
    )
    return [r["embedding"] for r in response.data]


async def get_embedding_dim(conf: LitellmConfDict) -> int:
    """Return the embedding dimension reported by ``conf["embed_model"]``."""
    embeddings = await alitellm_embedding(["test"], conf)
    return len(embeddings[0])
