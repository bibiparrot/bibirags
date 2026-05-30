"""
LightRAG backend for bibirags.

Requires the ``lightrag`` extra::

    pip install bibirags[lightrag]
"""

from __future__ import annotations

import asyncio
import json
import pathlib

import numpy as np
from loguru import logger

from bibirags.llm import LitellmConfDict, alitellm_embedding, get_embedding_dim


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sanitize_kwargs(kwargs: dict) -> dict:
    """Drop non-JSON-serialisable keyword arguments."""
    clean = {}
    for k, v in kwargs.items():
        try:
            json.dumps(v)
            clean[k] = v
        except Exception:
            logger.warning(f"Dropping non-serialisable kwarg: {k!r}")
    return clean


async def _create_rag(rag_root: str, conf: LitellmConfDict):
    """Instantiate and initialise a :class:`LightRAG` instance."""
    import litellm
    from lightrag import LightRAG, QueryParam  # noqa: F401
    from lightrag.utils import EmbeddingFunc

    embedding_dimension = await get_embedding_dim(conf)

    async def embedding_func(texts):
        embeddings = await alitellm_embedding(texts, conf)
        return np.array(embeddings)

    async def llm_model_func(
        prompt,
        system_prompt=None,
        history_messages=None,
        **kwargs,
    ):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        response = await litellm.acompletion(
            model=conf["llm_model"],
            messages=messages,
            temperature=0,
            api_base=conf.get("api_base"),
            api_key=conf.get("api_key"),
            **_sanitize_kwargs(kwargs),
        )
        return response.choices[0].message.content

    rag = LightRAG(
        working_dir=rag_root,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dimension,
            max_token_size=8192,
            func=embedding_func,
        ),
        # concurrency – keep low to avoid rate-limit issues
        embedding_func_max_async=1,
        llm_model_max_async=1,
        max_parallel_insert=1,
        default_embedding_timeout=3000,
        default_llm_timeout=3000,
        summary_max_tokens=8192,
    )

    await rag.initialize_storages()
    return rag


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_lightrag(
    chunks: list[str] | str,
    rag_root: str | pathlib.Path,
    conf: LitellmConfDict,
) -> None:
    """Index *chunks* using LightRAG's graph-based approach.

    Parameters
    ----------
    chunks:
        One or more plain-text chunks to insert.
    rag_root:
        Working directory for LightRAG's storage files.
    conf:
        :class:`~bibirags.llm.LitellmConfDict` with ``embed_model`` and
        ``llm_model`` set.
    """

    async def _run():
        rag = await _create_rag(rag_root=str(rag_root), conf=conf)
        texts = [chunks] if isinstance(chunks, str) else chunks
        for chunk in texts:
            if chunk and chunk.strip():
                await rag.ainsert(chunk)
        logger.info(f"LightRAG: indexed {len(texts)} chunk(s) in {rag_root!r}")

    asyncio.run(_run())


def search_lightrag(
    query: str,
    rag_root: str | pathlib.Path,
    conf: LitellmConfDict,
    top_k: int = 3,
) -> list[str]:
    """Retrieve the *top_k* most relevant chunks for *query* via LightRAG.

    Parameters
    ----------
    query:
        Natural-language search query.
    rag_root:
        Working directory containing a saved LightRAG index.
    conf:
        :class:`~bibirags.llm.LitellmConfDict` with ``embed_model`` and
        ``llm_model`` set (both must match what was used during indexing).
    top_k:
        Number of results to return.

    Returns
    -------
    list[str]
        Matched chunk texts.
    """

    async def _run():
        rag = await _create_rag(rag_root=str(rag_root), conf=conf)
        [query_embedding] = await rag.embedding_func([query])
        results = await rag.chunks_vdb.query(
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,
        )
        chunks = []
        for point in results:
            logger.debug(f"{point['content']!r} | score: {point['distance']:.4f}")
            chunks.append(point["content"])
        return chunks

    return asyncio.run(_run())


def query_lightrag(
    query: str,
    rag_root: str | pathlib.Path,
    conf: LitellmConfDict,
    top_k: int = 3,
) -> tuple[str, str]:
    """Retrieve relevant context and generate an answer using LightRAG.

    Uses hybrid (vector + graph) retrieval.

    Parameters
    ----------
    query:
        Question to answer.
    rag_root:
        Working directory containing a saved LightRAG index.
    conf:
        :class:`~bibirags.llm.LitellmConfDict` with ``embed_model`` and
        ``llm_model`` set.
    top_k:
        Number of context chunks considered.

    Returns
    -------
    tuple[str, str]
        ``(answer, context_string)``
    """
    from lightrag import QueryParam

    async def _run():
        rag = await _create_rag(rag_root=str(rag_root), conf=conf)
        context = await rag.aquery(
            query,
            param=QueryParam(mode="hybrid", only_need_context=True, top_k=top_k),
        )
        answer = await rag.aquery(
            query,
            param=QueryParam(mode="hybrid", top_k=top_k),
        )
        return answer, context

    return asyncio.run(_run())
