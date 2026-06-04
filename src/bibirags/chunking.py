"""
Document loading and chunking utilities.

Supports ``.pdf`` and ``.txt`` files out of the box.
"""

from __future__ import annotations

import pathlib
from loguru import logger

from bibirags.chunkingdirectory import chunk_directory, PDFMode, DocxMode


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
    all_chunks = chunk_directory(
        docs_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        pdf_mode=PDFMode.COMPLEX,
        docx_mode=DocxMode.STRUCTURED,
    )
    text_chunks = []
    for path, chunks in all_chunks.items():
        for chunk in chunks:
            text_chunks.append(chunk['text'])
    return text_chunks
