"""
bibirags — A simple, unified interface for RAG (Retrieval-Augmented Generation)
across multiple vector store backends.

Supported backends:
  - txtai
  - chroma
  - qdrant
  - lightrag
"""

from bibirags.chunking import chunk_docs
from bibirags.backends.txtai import save_txtai, search_txtai, query_txtai
from bibirags.backends.chroma import save_chroma, search_chroma, query_chroma
from bibirags.backends.qdrant import save_qdrant, search_qdrant, query_qdrant
from bibirags.backends.lightrag import save_lightrag, search_lightrag, query_lightrag
from bibirags.llm import litellm_embedding, litellm_complete

__version__ = "0.1.0"
__all__ = [
    # chunking
    "chunk_docs",
    # llm helpers
    "litellm_embedding",
    "litellm_complete",
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
