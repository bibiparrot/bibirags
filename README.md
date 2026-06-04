# bibirags

**A simple, unified interface for RAG (Retrieval-Augmented Generation) across multiple vector store backends.**

`bibirags` wraps [txtai](https://github.com/neuml/txtai), [Chroma](https://www.trychroma.com/), [Qdrant](https://qdrant.tech/), and [LightRAG](https://github.com/HKUDS/LightRAG) behind a consistent three-function API so you can swap backends without rewriting your pipeline.

```
save_<backend>(chunks, rag_root, conf, ...)  → index documents
search_<backend>(query, rag_root, conf, ...) → retrieve chunks
query_<backend>(query, rag_root, conf, ...)  → retrieve + answer
```

All LLM and embedding calls go through [LiteLLM](https://github.com/BerriAI/litellm) via a single `LitellmConfDict`, meaning any model provider (OpenAI, Anthropic, Cohere, Ollama, etc.) works out of the box.

---

## Installation

Install the core package plus the backends you need:

```bash
# Qdrant only
pip install bibirags[qdrant]

# Chroma only
pip install bibirags[chroma]

# txtai only
pip install bibirags[txtai]

# LightRAG only
pip install bibirags[lightrag]

# All backends + document loading helpers
pip install bibirags[all]
```

Add `[docs]` to get PDF and TXT loading via LangChain:

```bash
pip install bibirags[qdrant,docs]
```

---

## Quick start

### Build a `LitellmConfDict`

Every function takes a single `conf` dict instead of scattered `llm_model` / `embed_model` / `api_key` arguments:

```python
from bibirags import LitellmConfDict

# OpenAI
conf: LitellmConfDict = {
    "embed_model": "text-embedding-3-small",
    "llm_model": "gpt-4o-mini",
    "api_key": "sk-...",          # falls back to OPENAI_API_KEY env var
}

# Ollama (local)
conf: LitellmConfDict = {
    "embed_model": "ollama/bge-m3:latest",
    "llm_model": "ollama/gemma3:8b",
    "api_base": "http://localhost:11434",
}

# Any LiteLLM-compatible proxy
conf: LitellmConfDict = {
    "embed_model": "openai/text-embedding-3-small",
    "llm_model": "openai/gpt-4o",
    "api_base": "https://my-proxy.example.com/v1",
    "api_key": "proxy-key",
}
```

### Index raw text chunks

```python
from bibirags import save_qdrant, search_qdrant, query_qdrant

chunks = [
    "The Eiffel Tower was completed in 1889.",
    "The Louvre is the world's largest art museum.",
    "Paris is the capital of France.",
]

conf = {"embed_model": "text-embedding-3-small", "llm_model": "gpt-4o-mini"}
rag_root = "./my_rag_index"

# 1. Index
save_qdrant(chunks, rag_root, conf)

# 2. Semantic search
results = search_qdrant("When was the Eiffel Tower built?", rag_root, conf)

# 3. RAG query → answer + source chunks
answer, sources = query_qdrant("When was the Eiffel Tower built?", rag_root, conf)
print(answer)
```

### Load documents from disk

```python
from bibirags import chunk_docs, save_chroma, query_chroma

conf = {"embed_model": "text-embedding-3-small", "llm_model": "gpt-4o"}
chunks = chunk_docs("./my_docs/", chunk_size=800, chunk_overlap=120)

save_chroma(chunks, "./chroma_index", conf)

answer, sources = query_chroma(
    "What does the contract say about termination?",
    rag_root="./chroma_index",
    conf=conf,
)
```

### Using Ollama (local models)

```python
from bibirags import save_txtai, query_txtai

conf = {
    "embed_model": "ollama/bge-m3:latest",
    "llm_model": "ollama/gemma3:8b",
    "api_base": "http://localhost:11434",
}

save_txtai(chunks, "./txtai_index", conf)
answer, sources = query_txtai("What happened in the news?", "./txtai_index", conf)
```

---

## Backends at a glance

| Backend  | Best for | Index format | Notes |
|----------|----------|--------------|-------|
| **Qdrant** | Production workloads, filtering | Local files or server | Cosine similarity, rich payload filtering |
| **Chroma** | LangChain ecosystems | Local SQLite | Easy LangChain integration |
| **txtai**  | All-in-one HuggingFace pipelines | SQLite + FAISS | Built-in pipeline support |
| **LightRAG** | Knowledge-graph RAG | Local JSON + vector | Graph-enhanced hybrid retrieval |

---

## API reference

### `LitellmConfDict`

```python
class LitellmConfDict(TypedDict, total=False):
    embed_model: str   # required for save/search/query
    llm_model:   str   # required for query
    api_base:    str   # optional – custom API endpoint
    api_key:     str   # optional – falls back to env vars
```

### `chunk_docs`

```python
chunk_docs(docs_path, chunk_size=800, chunk_overlap=120) → list[str]
```

Recursively loads `.pdf` and `.txt` files from `docs_path` and returns text chunks.

### `save_<backend>`

```python
save_qdrant(chunks, rag_root, conf)
save_chroma(chunks, rag_root, conf)
save_txtai(chunks, rag_root, conf)
save_lightrag(chunks, rag_root, conf)
```

### `search_<backend>`

```python
results: list[str] = search_qdrant(query, rag_root, conf, top_k=3)
```

Returns the `top_k` most relevant chunk texts.

### `query_<backend>`

```python
answer, sources = query_qdrant(query, rag_root, conf, top_k=3)
```

Returns `(answer_string, list_of_source_chunks)`.

---

## Contributing

```bash
git clone https://github.com/yourname/bibirags
cd bibirags
pip install -e ".[dev]"
pre-commit install
pytest
```

---

## License

MIT – see [LICENSE](LICENSE).
