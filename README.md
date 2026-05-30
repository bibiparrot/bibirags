# bibirags

**A simple, unified interface for RAG (Retrieval-Augmented Generation) across multiple vector store backends.**

`bibirags` wraps [txtai](https://github.com/neuml/txtai), [Chroma](https://www.trychroma.com/), [Qdrant](https://qdrant.tech/), and [LightRAG](https://github.com/HKUDS/LightRAG) behind a consistent three-function API so you can swap backends without rewriting your pipeline.

```
save_<backend>(chunks, rag_root, embed_model, ...)  → index documents
search_<backend>(query, rag_root, embed_model, ...) → retrieve chunks
query_<backend>(query, rag_root, llm_model, ...)    → retrieve + answer
```

All LLM and embedding calls go through [LiteLLM](https://github.com/BerriAI/litellm), meaning any model provider (OpenAI, Anthropic, Cohere, Ollama, etc.) works out of the box.

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

### Index raw text chunks

```python
from bibirags import save_qdrant, search_qdrant, query_qdrant

chunks = [
    "The Eiffel Tower was completed in 1889.",
    "The Louvre is the world's largest art museum.",
    "Paris is the capital of France.",
]

embed_model = "text-embedding-3-small"   # any LiteLLM-compatible model
llm_model   = "gpt-4o-mini"
rag_root    = "./my_rag_index"

# 1. Index
save_qdrant(chunks, rag_root, embed_model)

# 2. Semantic search
results = search_qdrant("When was the Eiffel Tower built?", rag_root, embed_model)

# 3. RAG query → answer + source chunks
answer, sources = query_qdrant(
    "When was the Eiffel Tower built?", rag_root, llm_model, embed_model
)
print(answer)
```

### Load documents from disk

```python
from bibirags import chunk_docs, save_chroma, query_chroma

chunks = chunk_docs("./my_docs/", chunk_size=800, chunk_overlap=120)
save_chroma(chunks, "./chroma_index", embed_model="text-embedding-3-small")

answer, sources = query_chroma(
    "What does the contract say about termination?",
    rag_root="./chroma_index",
    llm_model="gpt-4o",
    embed_model="text-embedding-3-small",
)
```

### Using Ollama (local models)

```python
from bibirags import save_txtai, query_txtai

save_txtai(chunks, "./txtai_index", embed_model="ollama/bge-m3:latest")

answer, sources = query_txtai(
    "What happened in the news?",
    rag_root="./txtai_index",
    llm_model="ollama/gemma3:8b",
    embed_model="ollama/bge-m3:latest",
)
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

### `chunk_docs`

```python
chunk_docs(docs_path, chunk_size=800, chunk_overlap=120) → list[str]
```

Recursively loads `.pdf` and `.txt` files from `docs_path` and returns text chunks.

### `save_<backend>`

```python
save_qdrant(chunks, rag_root, embed_model)
save_chroma(chunks, rag_root, embed_model)
save_txtai(chunks, rag_root, embed_model)
save_lightrag(chunks, rag_root, embed_model, llm_model)
```

### `search_<backend>`

```python
results: list[str] = search_qdrant(query, rag_root, embed_model, top_k=3)
```

Returns the `top_k` most relevant chunk texts.

### `query_<backend>`

```python
answer, sources = query_qdrant(query, rag_root, llm_model, embed_model, top_k=3)
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
