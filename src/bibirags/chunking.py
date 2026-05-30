"""
Document loading and chunking utilities.

Supports ``.pdf`` and ``.txt`` files out of the box.  Additional file types can
be handled by extending the ``loaders`` mapping before calling
:func:`chunk_docs`.
"""

from __future__ import annotations

import pathlib
from loguru import logger


def chunk_docs(
    docs_path: str | pathlib.Path,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[str]:
    """Load documents from *docs_path* and split them into text chunks.

    Recursively walks *docs_path* and loads every ``.pdf`` and ``.txt`` file it
    finds.  The resulting documents are split with LangChain's
    ``RecursiveCharacterTextSplitter``.

    Parameters
    ----------
    docs_path:
        Directory (or single file) containing source documents.
    chunk_size:
        Maximum number of characters per chunk.
    chunk_overlap:
        Number of characters to overlap between consecutive chunks.

    Returns
    -------
    list[str]
        Plain-text chunks ready to be indexed by any backend.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import PyPDFLoader, TextLoader

    loaders: dict[str, type] = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
    }

    docs = []
    for f in pathlib.Path(docs_path).rglob("*"):
        suffix = f.suffix.lower()
        if suffix in loaders:
            logger.debug(f"Loading {f}")
            loader = loaders[suffix](str(f))
            loaded = loader.load()
            for doc in loaded:
                doc.metadata["source"] = str(f)
            docs.extend(loaded)

    if not docs:
        logger.warning(f"No supported documents found in {docs_path!r}")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    split_docs = splitter.split_documents(docs)
    chunks = [d.page_content for d in split_docs]
    logger.info(f"Produced {len(chunks)} chunks from {len(docs)} document(s)")
    return chunks
