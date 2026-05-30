"""
txtai backend for bibirags.

Requires the ``txtai`` extra::

    pip install bibirags[txtai]
"""

from __future__ import annotations

import pathlib
from loguru import logger


def save_txtai(
    chunks: list[str],
    rag_root: str | pathlib.Path,
    embed_model: str,
) -> None:
    """Index *chunks* into a txtai embeddings store at *rag_root*.

    Parameters
    ----------
    chunks:
        Plain-text chunks to index.
    rag_root:
        Directory where the txtai index will be persisted.
    embed_model:
        HuggingFace model name or path understood by txtai (e.g.
        ``"sentence-transformers/all-MiniLM-L6-v2"``).
    """
    from txtai import Embeddings

    embeddings = Embeddings(content=True, path=embed_model)
    embeddings.index(chunks)
    embeddings.save(str(rag_root))
    logger.info(f"txtai index saved to {rag_root!r}")


def search_txtai(
    query: str,
    rag_root: str | pathlib.Path,
    embed_model: str,
    top_k: int = 3,
) -> list[str]:
    """Retrieve the *top_k* most relevant chunks for *query*.

    Parameters
    ----------
    query:
        Natural-language search query.
    rag_root:
        Directory containing the persisted txtai index.
    embed_model:
        Must match the model used when the index was built.
    top_k:
        Number of results to return.

    Returns
    -------
    list[str]
        Matched chunk texts, ordered by descending relevance.
    """
    from txtai import Embeddings

    embeddings = Embeddings(content=True, path=embed_model)
    embeddings.load(str(rag_root))
    results = embeddings.search(query, limit=top_k)

    chunks = []
    for r in results:
        logger.debug(f"{r['text']!r} | score: {r['score']:.4f}")
        chunks.append(r["text"])
    return chunks


def query_txtai(
    query: str,
    rag_root: str | pathlib.Path,
    llm_model: str,
    embed_model: str,
    top_k: int = 3,
) -> tuple[str, list[str]]:
    """Retrieve relevant chunks and generate an answer with txtai RAG.

    Parameters
    ----------
    query:
        Question to answer.
    rag_root:
        Directory containing the persisted txtai index.
    llm_model:
        HuggingFace or pipeline path for the generative model.
    embed_model:
        Must match the model used when the index was built.
    top_k:
        Number of context chunks to pass to the LLM.

    Returns
    -------
    tuple[str, list[str]]
        ``(answer, source_chunks)``
    """
    from txtai import Embeddings, RAG

    embeddings = Embeddings(content=True, path=embed_model)
    embeddings.load(str(rag_root))

    rag = RAG(
        similarity=embeddings,
        path=llm_model,
        output="reference",
        context=top_k,
        system="You are a document analysis assistant",
        template="""
Answer the following question using the provided context.

Question:
{question}

Context:
{context}
""",
    )

    results = rag(query, defaultrole="user", stripThink=True)
    answer: str = results["answer"]
    reference_id = results["reference"]

    sources: list[str] = []
    if reference_id:
        references = embeddings.search(
            f"select id, text, source from txtai where id = '{reference_id}'",
            limit=1,
        )
        sources = [r["text"] for r in references]

    return answer, sources
