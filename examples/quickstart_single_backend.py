"""
examples/quickstart_single_backend.py
───────────────────────────────────────
Minimal self-contained example for one backend (Qdrant).
Good starting point if you only need one backend.

Run
───
    python examples/quickstart_single_backend.py
"""

import pathlib
from loguru import logger

from bibirags import LitellmConfDict, save_qdrant, search_qdrant, query_qdrant

conf: LitellmConfDict = {
    "embed_model": "ollama/bge-m3:latest",
    "llm_model":   "ollama/gemma4:e2b",
    "api_base":    "http://localhost:11434",
}

chunks = [
    "US tops 5 million confirmed virus cases",
    "Canada's last fully intact ice shelf has suddenly collapsed, "
    "forming a Manhattan-sized iceberg",
    "Beijing mobilises invasion craft along coast as Taiwan tensions escalate",
    "The National Park Service warns against sacrificing slower friends "
    "in a bear attack",
    "Maine man wins $1M from $25 lottery ticket",
    "Make huge profits without work, earn up to $100,000 a day",
]

query = "What was won?"
rag_root = pathlib.Path("./rag_root_qdrant").absolute()

# Index
save_qdrant(chunks, rag_root, conf)
logger.info("Index saved.")

# Search (retrieval only)
results = search_qdrant(query, rag_root, conf, top_k=3)
logger.info("Top results:")
for i, r in enumerate(results, 1):
    logger.info(f"  [{i}] {r}")

# Query (retrieval + LLM answer)
answer, sources = query_qdrant(query, rag_root, conf, top_k=3)
logger.info(f"Answer: {answer}")
