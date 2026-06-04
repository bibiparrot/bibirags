from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    PDFPlumberLoader,
    PyMuPDFLoader,
    Docx2txtLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
    CSVLoader,
    UnstructuredXMLLoader,
    UnstructuredEmailLoader,
)
from enum import Enum
import os
import glob as glob_module

from langchain_text_splitters import RecursiveCharacterTextSplitter


class PDFMode(Enum):
    TEXT_ONLY = "text"  # PyPDFLoader       — fast, plain text
    TABLES = "tables"  # PDFPlumberLoader  — strong table extraction
    COMPLEX = "complex"  # PyMuPDFLoader     — complex layouts / speed


class DocxMode(Enum):
    TEXT_ONLY = "text"  # Docx2txtLoader                — lightweight, fast
    STRUCTURED = "structured"  # UnstructuredWordDocumentLoader — headings, paragraphs


PDF_LOADERS = {
    PDFMode.TEXT_ONLY: PyPDFLoader,
    PDFMode.TABLES: PDFPlumberLoader,
    PDFMode.COMPLEX: PyMuPDFLoader,
}

DOCX_LOADERS = {
    DocxMode.TEXT_ONLY: Docx2txtLoader,
    DocxMode.STRUCTURED: UnstructuredWordDocumentLoader,
}

OTHER_LOADERS = {
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader,
    ".html": UnstructuredHTMLLoader,
    ".htm": UnstructuredHTMLLoader,
    ".csv": CSVLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".xls": UnstructuredExcelLoader,
    ".pptx": UnstructuredPowerPointLoader,
    ".ppt": UnstructuredPowerPointLoader,
    ".xml": UnstructuredXMLLoader,
    ".eml": UnstructuredEmailLoader,
}


def get_loader(
    file_path: str,
    pdf_mode: PDFMode = PDFMode.TEXT_ONLY,
    docx_mode: DocxMode = DocxMode.TEXT_ONLY,
):
    """Return the appropriate loader based on extension + chosen mode."""
    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".pdf":
        loader_cls = PDF_LOADERS[pdf_mode]
    elif ext in (".docx", ".doc"):
        loader_cls = DOCX_LOADERS[docx_mode]
    else:
        loader_cls = OTHER_LOADERS.get(ext)
        if not loader_cls:
            raise ValueError(
                f"Unsupported extension: {ext}. "
                f"Supported: .pdf .docx .doc {' '.join(OTHER_LOADERS)}"
            )

    return loader_cls(file_path)


def chunk_file(
    file_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    pdf_mode: PDFMode = PDFMode.TEXT_ONLY,
    docx_mode: DocxMode = DocxMode.TEXT_ONLY,
) -> list[dict]:
    """Chunk a single file. Choose PDF / DOCX strategy via mode enums."""
    loader = get_loader(file_path, pdf_mode=pdf_mode, docx_mode=docx_mode)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    return [
        {
            "chunk_index": i,
            "text": chunk.page_content,
            "metadata": {**chunk.metadata, "source": file_path},
        }
        for i, chunk in enumerate(chunks)
    ]


def chunk_directory(
    dir_path: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    recursive: bool = True,
    pdf_mode: PDFMode = PDFMode.TEXT_ONLY,
    docx_mode: DocxMode = DocxMode.TEXT_ONLY,
) -> dict[str, list[dict]]:
    """
    Chunk all supported files in a directory.
    PDF and DOCX strategies are applied uniformly across the directory;
    override per file with chunk_file() directly if you need mixed modes.
    """
    GLOB_LOADER_MAP = {
        "**/*.pdf": PDF_LOADERS[pdf_mode],
        "**/*.docx": DOCX_LOADERS[docx_mode],
        "**/*.doc": DOCX_LOADERS[docx_mode],
        "**/*.txt": TextLoader,
        "**/*.md": UnstructuredMarkdownLoader,
        "**/*.html": UnstructuredHTMLLoader,
        "**/*.htm": UnstructuredHTMLLoader,
        "**/*.csv": CSVLoader,
        "**/*.xlsx": UnstructuredExcelLoader,
        "**/*.xls": UnstructuredExcelLoader,
        "**/*.pptx": UnstructuredPowerPointLoader,
        "**/*.ppt": UnstructuredPowerPointLoader,
        "**/*.xml": UnstructuredXMLLoader,
        "**/*.eml": UnstructuredEmailLoader,
    }

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    results: dict[str, list[dict]] = {}

    for glob_pattern, loader_cls in GLOB_LOADER_MAP.items():
        loader = DirectoryLoader(
            dir_path,
            glob=glob_pattern,
            loader_cls=loader_cls,
            recursive=recursive,
            silent_errors=True,
            show_progress=True,
        )
        try:
            docs = loader.load()
            if not docs:
                continue
            chunks = splitter.split_documents(docs)
            for chunk in chunks:
                src = chunk.metadata.get("source", "unknown")
                results.setdefault(src, []).append(
                    {
                        "chunk_index": len(results.get(src, [])),
                        "text": chunk.page_content,
                        "metadata": chunk.metadata,
                    }
                )
            print(f"{glob_pattern}: {len(docs)} docs → {len(chunks)} chunks")
        except Exception as e:
            print(f"  ✗ {glob_pattern} failed: {e}")

    return results


# ── Example usage ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Plain-text PDF (fastest)
    chunks = chunk_file("report.pdf", pdf_mode=PDFMode.TEXT_ONLY)

    # PDF with tables (e.g. financial statements)
    chunks = chunk_file("financials.pdf", pdf_mode=PDFMode.TABLES)

    # Complex layout PDF (e.g. scanned / multi-column)
    chunks = chunk_file("brochure.pdf", pdf_mode=PDFMode.COMPLEX)

    # DOCX — plain text only
    chunks = chunk_file("contract.docx", docx_mode=DocxMode.TEXT_ONLY)

    # DOCX — keep headings / paragraph structure
    chunks = chunk_file("contract.docx", docx_mode=DocxMode.STRUCTURED)

    # Whole directory — PDFs treated as table-heavy, DOCX structured
    all_chunks = chunk_directory(
        "./docs",
        chunk_size=500,
        chunk_overlap=50,
        pdf_mode=PDFMode.TABLES,
        docx_mode=DocxMode.STRUCTURED,
    )

    for path, chunks in all_chunks.items():
        print(f"{path}: {len(chunks)} chunks")
