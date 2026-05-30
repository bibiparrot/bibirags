# bibirags examples

| File | What it shows |
|------|---------------|
| `quickstart_ollama.py` | All four backends with a local Ollama server |
| `quickstart_openai.py` | All four backends with OpenAI |
| `quickstart_dashscope.py` | All four backends with Alibaba Cloud DashScope (Qwen models) |
| `quickstart_single_backend.py` | Minimal Qdrant-only example |
| `quickstart_docs.py` | Load real PDF/TXT files → chunk → index → query |

## Running the examples

### Ollama (local)

```bash
# Pull models first
ollama pull bge-m3
ollama pull gemma4:e2b

pip install bibirags[all]
python examples/quickstart_ollama.py
```

### OpenAI

```bash
export OPENAI_API_KEY="sk-..."
pip install bibirags[all]
python examples/quickstart_openai.py
```

### DashScope / Qwen (Alibaba Cloud)

```bash
export DASHSCOPE_API_KEY="sk-..."
pip install bibirags[all]
python examples/quickstart_dashscope.py

# Users inside mainland China – use the Beijing endpoint:
DASHSCOPE_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1 \
python examples/quickstart_dashscope.py
```

### Document search (PDF / TXT)

```bash
# Put some .pdf or .txt files in ./my_docs
python examples/quickstart_docs.py --docs ./my_docs --query "What is the main topic?"
```

## Switching backends

All backends share the same `LitellmConfDict` — swap the functions, keep the conf:

```python
conf = {"embed_model": "ollama/bge-m3:latest", "llm_model": "ollama/gemma4:e2b",
        "api_base": "http://localhost:11434"}

# swap any of these pairs freely
save_qdrant(chunks, "./idx", conf);  answer, _ = query_qdrant(q, "./idx", conf)
save_chroma(chunks, "./idx", conf);  answer, _ = query_chroma(q, "./idx", conf)
save_txtai(chunks,  "./idx", conf);  answer, _ = query_txtai(q,  "./idx", conf)
save_lightrag(chunks,"./idx", conf); answer, _ = query_lightrag(q,"./idx", conf)
```
