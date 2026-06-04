"""
bibirags — A simple, unified interface for RAG (Retrieval-Augmented Generation)
across multiple vector store backends.

Supported backends:
  - txtai
  - chroma
  - qdrant
  - lightrag

All functions accept a :class:`~bibirags.llm.LitellmConfDict` for model
configuration instead of bare model-name strings.
"""

from bibirags.llm import LitellmConfDict, litellm_embedding, litellm_complete
from bibirags.chunking import chunk_docs
from bibirags.chunkingdirectory import chunk_file, chunk_directory
from bibirags.backends.txtai import save_txtai, search_txtai, query_txtai
from bibirags.backends.chroma import save_chroma, search_chroma, query_chroma
from bibirags.backends.qdrant import save_qdrant, search_qdrant, query_qdrant
from bibirags.backends.lightrag import save_lightrag, search_lightrag, query_lightrag

__version__ = "0.1.0"
__all__ = [
    # config
    "LitellmConfDict",
    # llm helpers
    "litellm_embedding",
    "litellm_complete",
    # chunking
    "chunk_docs",
    "chunk_file",
    "chunk_directory",
    # txtai
    "save_txtai",
    "search_txtai",
    "query_txtai",
    # chroma
    "save_chroma",
    "search_chroma",
    "query_chroma",
    # qdrant
    "save_qdrant",
    "search_qdrant",
    "query_qdrant",
    # lightrag
    "save_lightrag",
    "search_lightrag",
    "query_lightrag",
]
