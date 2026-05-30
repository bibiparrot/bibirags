# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2024-01-01

### Added
- Initial release.
- Four vector-store backends: **txtai**, **Chroma**, **Qdrant**, **LightRAG**.
- Consistent `save / search / query` API across all backends.
- LiteLLM-powered embedding and completion helpers (sync + async).
- `chunk_docs` utility for PDF and TXT document loading and splitting.
- Optional extras for each backend (`pip install bibirags[qdrant]` etc.).
- Full type annotations and Loguru structured logging.
