"""
Chroma backend for bibirags.

Requires the ``chroma`` extra::

    pip install bibirags[chroma]
"""

from __future__ import annotations

import pathlib
from loguru import logger

from bibirags.llm import LitellmConfDict
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

def save_chroma(
    chunks: list[str],
    rag_root: str | pathlib.Path,
    conf: LitellmConfDict,
) -> None:
    """Index *chunks* into a Chroma vector store at *rag_root*.

    Parameters
    ----------
    chunks:
        Plain-text chunks to index.
    rag_root:
        Directory where Chroma will persist the collection.
    conf:
        :class:`~bibirags.llm.LitellmConfDict` with at least ``embed_model``
        set.
    """
    from langchain_litellm import LiteLLMEmbeddings
    from langchain_chroma import Chroma
    from langchain_core.documents import Document

    documents = [Document(page_content=text) for text in chunks]
    embeddings = LiteLLMEmbeddings(
        model=conf["embed_model"],
        api_base=conf.get("api_base"),
        api_key=conf.get("api_key"),
    )
    rag_name = pathlib.Path(rag_root).stem

    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(rag_root),
        collection_name=rag_name,
    )
    logger.info(f"Chroma collection {rag_name!r} saved to {rag_root!r}")


def search_chroma(
    query: str,
    rag_root: str | pathlib.Path,
    conf: LitellmConfDict,
    top_k: int = 3,
) -> list[str]:
    """Retrieve the *top_k* most relevant chunks for *query* from Chroma.

    Parameters
    ----------
    query:
        Natural-language search query.
    rag_root:
        Directory containing the persisted Chroma collection.
    conf:
        :class:`~bibirags.llm.LitellmConfDict` — ``embed_model`` must match the
        model used during indexing.
    top_k:
        Number of results to return.

    Returns
    -------
    list[str]
        Matched chunk texts, ordered by ascending distance (best first).
    """
    from langchain_litellm import LiteLLMEmbeddings
    from langchain_chroma import Chroma

    embeddings = LiteLLMEmbeddings(
        model=conf["embed_model"],
        api_base=conf.get("api_base"),
        api_key=conf.get("api_key"),
    )
    rag_name = pathlib.Path(rag_root).stem

    vector_store = Chroma(
        persist_directory=str(rag_root),
        embedding_function=embeddings,
        collection_name=rag_name,
    )

    results = vector_store.similarity_search_with_score(query, k=top_k)
    chunks = []
    for doc, score in results:
        logger.debug(f"{doc.page_content!r} | score: {score:.4f}")
        chunks.append(doc.page_content)
    return chunks


def query_chroma(
    query: str,
    rag_root: str | pathlib.Path,
    conf: LitellmConfDict,
    top_k: int = 3,
) -> tuple[str, list[str]]:
    """Retrieve relevant chunks and generate an answer via LangChain RetrievalQA.

    Parameters
    ----------
    query:
        Question to answer.
    rag_root:
        Directory containing the persisted Chroma collection.
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
    from langchain_litellm import LiteLLMEmbeddings, ChatLiteLLM
    from langchain_chroma import Chroma
    from langchain_classic.chains.retrieval_qa.base import RetrievalQA

    embeddings = LiteLLMEmbeddings(
        model=conf["embed_model"],
        api_base=conf.get("api_base"),
        api_key=conf.get("api_key"),
    )
    rag_name = pathlib.Path(rag_root).stem

    vector_store = Chroma(
        persist_directory=str(rag_root),
        embedding_function=embeddings,
        collection_name=rag_name,
    )

    llm = ChatLiteLLM(
        model=conf["llm_model"],
        temperature=0,
        api_base=conf.get("api_base"),
        api_key=conf.get("api_key"),
    )
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": top_k}),
        return_source_documents=True,
    )

    result = qa.invoke({"query": query})
    answer: str = result["result"]
    sources: list[str] = [d.page_content for d in result["source_documents"]]
    return answer, sources
