"""
examples/quickstart_ollama.py
─────────────────────────────
Runs all four bibirags backends (txtai, Chroma, Qdrant, LightRAG) against a
small in-memory chunk list using a local Ollama server.

Prerequisites
─────────────
1. Ollama running at http://localhost:11434
2. Models pulled:
       ollama pull bge-m3
       ollama pull gemma4:e2b
3. bibirags installed with all backends:
       pip install bibirags[all]

Run
───
    python examples/quickstart_ollama.py
"""

import pathlib
import nest_asyncio
from loguru import logger

from bibirags import (
    LitellmConfDict,
    save_txtai,   search_txtai,   query_txtai,
    save_chroma,  search_chroma,  query_chroma,
    save_qdrant,  search_qdrant,  query_qdrant,
    save_lightrag, search_lightrag, query_lightrag,
)

# ── LightRAG uses asyncio internally; nest_asyncio lets it run in scripts ──
nest_asyncio.apply()

# ── shared config ──────────────────────────────────────────────────────────
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


# ── helpers ────────────────────────────────────────────────────────────────

def _section(title: str) -> None:
    logger.info("=" * 60)
    logger.info(f"  {title}")
    logger.info("=" * 60)


def _show_search(results: list[str]) -> None:
    for i, chunk in enumerate(results, 1):
        logger.info(f"  [{i}] {chunk}")


def _show_query(answer: str, sources: list[str] | str) -> None:
    logger.info(f"  Answer : {answer}")
    if isinstance(sources, str):
        logger.info(f"  Context: {sources[:200]}...")
    else:
        for i, src in enumerate(sources, 1):
            logger.info(f"  Source [{i}]: {src}")


# ── txtai ──────────────────────────────────────────────────────────────────
_section("txtai")
rag_root = pathlib.Path("./rag_root_txtai").absolute()

logger.info("Saving index…")
save_txtai(chunks, rag_root, conf)

logger.info(f"Searching: {query!r}")
_show_search(search_txtai(query, rag_root, conf))

logger.info(f"Querying: {query!r}")
_show_query(*query_txtai(query, rag_root, conf))


# ── Chroma ─────────────────────────────────────────────────────────────────
_section("Chroma")
rag_root = pathlib.Path("./rag_root_chroma").absolute()

logger.info("Saving index…")
save_chroma(chunks, rag_root, conf)

logger.info(f"Searching: {query!r}")
_show_search(search_chroma(query, rag_root, conf))

logger.info(f"Querying: {query!r}")
_show_query(*query_chroma(query, rag_root, conf))


# ── Qdrant ─────────────────────────────────────────────────────────────────
_section("Qdrant")
rag_root = pathlib.Path("./rag_root_qdrant").absolute()

logger.info("Saving index…")
save_qdrant(chunks, rag_root, conf)

logger.info(f"Searching: {query!r}")
_show_search(search_qdrant(query, rag_root, conf))

logger.info(f"Querying: {query!r}")
_show_query(*query_qdrant(query, rag_root, conf))


# ── LightRAG ───────────────────────────────────────────────────────────────
_section("LightRAG")
rag_root = pathlib.Path("./rag_root_lightrag").absolute()

logger.info("Saving index…")
save_lightrag(chunks, rag_root, conf)

logger.info(f"Searching: {query!r}")
_show_search(search_lightrag(query, rag_root, conf))

logger.info(f"Querying: {query!r}")
_show_query(*query_lightrag(query, rag_root, conf))

logger.info("Done!")
