"""
Backend implementations for bibirags.

Each sub-module exposes three functions with a consistent signature:

* ``save_<backend>(chunks, rag_root, conf, ...)``   – index chunks
* ``search_<backend>(query, rag_root, conf, ...)``  – retrieve chunks
* ``query_<backend>(query, rag_root, conf, ...)``   – retrieve + answer

All three accept a :class:`~bibirags.llm.LitellmConfDict` as ``conf``.
"""
