"""
LLM utility functions using LiteLLM.

These functions provide a thin, model-agnostic wrapper around LiteLLM for
embeddings and completions so every backend can share the same calling
convention.
"""

from __future__ import annotations

import asyncio
from loguru import logger


def litellm_embedding(texts: list[str], embed_model: str) -> list[list[float]]:
    """Return embeddings for *texts* using the given *embed_model*.

    Parameters
    ----------
    texts:
        List of strings to embed.
    embed_model:
        A LiteLLM-compatible model string, e.g. ``"ollama/bge-m3:latest"``,
        ``"text-embedding-3-small"``, etc.

    Returns
    -------
    list[list[float]]
        One embedding vector per input text.
    """
    import litellm

    response = litellm.embedding(model=embed_model, input=texts)
    return [r["embedding"] for r in response.data]


def litellm_complete(content: str, llm_model: str) -> str:
    """Return a completion for *content* using the given *llm_model*.

    Parameters
    ----------
    content:
        The user prompt.
    llm_model:
        A LiteLLM-compatible model string, e.g. ``"ollama/gemma4:e2b"``,
        ``"gpt-4o"``, etc.

    Returns
    -------
    str
        The model's response text.
    """
    import litellm

    response = litellm.completion(
        model=llm_model,
        messages=[{"content": content, "role": "user"}],
        temperature=0,
    )
    return response.choices[0].message.content


async def alitellm_embedding(texts: list[str], embed_model: str) -> list[list[float]]:
    """Async version of :func:`litellm_embedding`.

    Empty strings are replaced with ``"empty"`` to avoid upstream API errors.
    """
    import litellm

    clean_texts = [
        (t.strip() if isinstance(t, str) else str(t)) for t in texts
    ]
    clean_texts = [t if t else "empty" for t in clean_texts]

    response = await litellm.aembedding(model=embed_model, input=clean_texts)
    return [r["embedding"] for r in response.data]


async def get_embedding_dim(embed_model: str) -> int:
    """Return the dimension of the embedding space for *embed_model*."""
    embeddings = await alitellm_embedding(["test"], embed_model)
    return len(embeddings[0])
