"""
Backend implementations for bibirags.

Each sub-module exposes three functions with a consistent signature:

* ``save_<backend>(chunks, rag_root, embed_model, ...)``   – index chunks
* ``search_<backend>(query, rag_root, embed_model, ...)``  – retrieve chunks
* ``query_<backend>(query, rag_root, llm_model, ...)``     – retrieve + answer
"""
