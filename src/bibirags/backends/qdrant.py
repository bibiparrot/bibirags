"""
Qdrant backend for bibirags.

Requires the ``qdrant`` extra::

    pip install bibirags[qdrant]
"""

from __future__ import annotations

import pathlib
import uuid
from loguru import logger

from bibirags.llm import LitellmConfDict, litellm_embedding, litellm_complete


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _add_text(
    client,
    collection_name: str,
    conf: LitellmConfDict,
    text: str,
) -> None:
    """Embed *text* and upsert it as a point into *collection_name*."""
    from qdrant_client.http import models

    [vector] = litellm_embedding([text], conf)
    client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"text": text},
            )
        ],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_qdrant(
    chunks: list[str],
    rag_root: str | pathlib.Path,
    conf: LitellmConfDict,
) -> None:
    """Index *chunks* into a local Qdrant collection at *rag_root*.

    Parameters
    ----------
    chunks:
        Plain-text chunks to index.
    rag_root:
        Directory where Qdrant will persist the collection.
    conf:
        :class:`~bibirags.llm.LitellmConfDict` with at least ``embed_model`` set.
    """
    from qdrant_client import QdrantClient
    from qdrant_client.http import models

    rag_name = pathlib.Path(rag_root).stem
    client = QdrantClient(path=str(rag_root))
    try:
        [probe] = litellm_embedding(["this is a test."], conf)
        vector_size = len(probe)

        if not client.collection_exists(rag_name):
            client.create_collection(
                collection_name=rag_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info(f"Created Qdrant collection {rag_name!r} (dim={vector_size})")

        for chunk in chunks:
            _add_text(client, rag_name, conf, chunk)

        logger.info(f"Indexed {len(chunks)} chunks into Qdrant at {rag_root!r}")
    finally:
        client.close()


def search_qdrant(
    query: str,
    rag_root: str | pathlib.Path,
    conf: LitellmConfDict,
    top_k: int = 3,
) -> list[str]:
    """Retrieve the *top_k* most relevant chunks for *query* from Qdrant.

    Parameters
    ----------
    query:
        Natural-language search query.
    rag_root:
        Directory containing the persisted Qdrant collection.
    conf:
        :class:`~bibirags.llm.LitellmConfDict` with at least ``embed_model`` set.
    top_k:
        Number of results to return.

    Returns
    -------
    list[str]
        Matched chunk texts, ordered by descending cosine similarity.
    """
    from qdrant_client import QdrantClient

    rag_name = pathlib.Path(rag_root).stem
    client = QdrantClient(path=str(rag_root))
    try:
        [query_vector] = litellm_embedding([query], conf)
        result = client.query_points(
            collection_name=rag_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )

        chunks = []
        for point in result.points:
            text = point.payload["text"]
            logger.debug(f"{text!r} | score: {point.score:.4f}")
            chunks.append(text)
        return chunks
    finally:
        client.close()


def query_qdrant(
    query: str,
    rag_root: str | pathlib.Path,
    conf: LitellmConfDict,
    top_k: int = 3,
) -> tuple[str, list[str]]:
    """Retrieve relevant chunks and generate an answer using LiteLLM.

    Parameters
    ----------
    query:
        Question to answer.
    rag_root:
        Directory containing the persisted Qdrant collection.
    conf:
        :class:`~bibirags.llm.LitellmConfDict` with ``embed_model`` and
        ``llm_model`` set.
    top_k:
        Number of context chunks to pass to the LLM.

    Returns
    -------
    tuple[str, list[str]]
        ``(answer, source_chunks)``
    """
    chunks = search_qdrant(query, rag_root, conf, top_k=top_k)

    prompt = f"""You are a document analysis assistant.
Answer the following question using the provided context.

Question:
{query}

Context:
{chr(10).join(f'- {c}' for c in chunks)}
"""
    answer = litellm_complete(prompt, conf)
    return answer, chunks
