"""
examples/quickstart_docs.py
────────────────────────────
Shows how to load real PDF / TXT files from disk, chunk them, and index
them into Qdrant.

Usage
─────
    python examples/quickstart_docs.py --docs ./my_docs --query "What is the return policy?"

Prerequisites
─────────────
    pip install bibirags[qdrant,docs]
"""

import argparse
import pathlib

from loguru import logger

from bibirags import LitellmConfDict, chunk_docs, save_qdrant, query_qdrant

# ── CLI ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Index a folder of documents and query them.")
parser.add_argument("--docs",  default="./docs",        help="Path to folder with .pdf / .txt files")
parser.add_argument("--index", default="./rag_root_qdrant", help="Where to store the Qdrant index")
parser.add_argument("--query", default="What is the main topic?", help="Question to answer")
parser.add_argument("--top-k", type=int, default=3,     help="Number of chunks to retrieve")
args = parser.parse_args()

# ── config ─────────────────────────────────────────────────────────────────
conf: LitellmConfDict = {
    "embed_model": "ollama/bge-m3:latest",
    "llm_model":   "ollama/gemma4:e2b",
    "api_base":    "http://localhost:11434",
}

# ── pipeline ───────────────────────────────────────────────────────────────
docs_path = pathlib.Path(args.docs)
rag_root  = pathlib.Path(args.index).absolute()

logger.info(f"Loading documents from {docs_path!r} …")
chunks = chunk_docs(docs_path, chunk_size=800, chunk_overlap=120)
logger.info(f"  {len(chunks)} chunks produced")

if not chunks:
    logger.warning("No documents found – create some .pdf or .txt files in the docs folder.")
else:
    logger.info("Indexing into Qdrant …")
    save_qdrant(chunks, rag_root, conf)

    logger.info(f"Querying: {args.query!r}")
    answer, sources = query_qdrant(args.query, rag_root, conf, top_k=args.top_k)

    logger.info(f"Answer: {answer}")
    logger.info("Sources:")
    for i, src in enumerate(sources, 1):
        logger.info(f"  [{i}] {src[:120]}…")
