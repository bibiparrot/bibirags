"""
examples/quickstart_dashscope.py
──────────────────────────────────
Runs all four bibirags backends (txtai, Chroma, Qdrant, LightRAG) using
Alibaba Cloud DashScope (Qwen models) via LiteLLM.

LiteLLM DashScope docs: https://docs.litellm.ai/docs/providers/dashscope

Models used
───────────
  Embedding : dashscope/text-embedding-v3   (1 024-dim, 8 192-token limit)
  LLM       : dashscope/qwen-plus           (strong general-purpose Qwen model)

API base options
────────────────
  International : https://dashscope-intl.aliyuncs.com/compatible-mode/v1
  China/Beijing : https://dashscope.aliyuncs.com/compatible-mode/v1

Prerequisites
─────────────
    export DASHSCOPE_API_KEY="sk-..."
    pip install bibirags[all]

Run
───
    python examples/quickstart_dashscope.py

    # To use the China endpoint instead:
    DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1 \\
    python examples/quickstart_dashscope.py
"""

import os
import pathlib
import nest_asyncio
from loguru import logger

from bibirags import (
    LitellmConfDict,
    save_txtai,    search_txtai,    query_txtai,
    save_chroma,   search_chroma,   query_chroma,
    save_qdrant,   search_qdrant,   query_qdrant,
    save_lightrag, search_lightrag, query_lightrag,
)

# ── LightRAG uses asyncio internally; nest_asyncio lets it run in scripts ──
nest_asyncio.apply()

# ── DashScope config ───────────────────────────────────────────────────────
# api_base defaults to the international endpoint; set DASHSCOPE_API_BASE
# to override (e.g. the Beijing endpoint for users inside China).
_DEFAULT_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"

conf: LitellmConfDict = {
    "embed_model": "dashscope/text-embedding-v3",
    "llm_model":   "dashscope/qwen-plus",
    "api_key":     os.environ["DASHSCOPE_API_KEY"],
    "api_base":    os.environ.get("DASHSCOPE_API_BASE", _DEFAULT_API_BASE),
}

# ── demo data ──────────────────────────────────────────────────────────────
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
