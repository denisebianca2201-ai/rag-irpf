import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ask_the_lion.config import Settings

# The official document numbers its questions, e.g. "001 — Who must file...?".
QUESTION_START = re.compile(r"^\s*(\d{3,4})\s*[—–-]\s*(\S.*)$")
CHAPTER = re.compile(r"^\s*(CAP[ÍI]TULO\s+.+)$", re.IGNORECASE)


def load_pdf_pages(path: Path) -> list[tuple[int, str]]:
    """Return (1-based page number, text) pairs."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return [(i, page.extract_text() or "") for i, page in enumerate(reader.pages, start=1)]


def split_into_questions(pages: list[tuple[int, str]]) -> list[Document]:
    """One document per numbered question (question + answer)."""
    docs: list[Document] = []
    chapter = ""
    current: dict | None = None

    def flush() -> None:
        nonlocal current
        if current is not None:
            text = "\n".join(current["lines"]).strip()
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "question_number": current["number"],
                        "page": current["page"],
                        "chapter": current["chapter"],
                    },
                )
            )
        current = None

    for page_number, text in pages:
        for line in text.splitlines():
            chapter_match = CHAPTER.match(line)
            if chapter_match:
                flush()
                chapter = chapter_match.group(1).strip()
                continue
            start = QUESTION_START.match(line)
            if start:
                flush()
                current = {
                    "number": int(start.group(1)),
                    "page": page_number,
                    "chapter": chapter,
                    "lines": [f"{start.group(1)} — {start.group(2)}"],
                }
            elif current is not None and line.strip():
                current["lines"].append(line.strip())
    flush()
    return docs


def split_oversized(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    """Split long answers; every piece keeps the question metadata."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


def build_vector_store(embeddings: Embeddings, settings: Settings) -> Chroma:
    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )


def index_documents(docs: list[Document], embeddings: Embeddings, settings: Settings) -> int:
    """Rebuild the collection from scratch and return the number of chunks stored."""
    store = build_vector_store(embeddings, settings)
    store.reset_collection()
    chunks = split_oversized(docs, settings.chunk_size, settings.chunk_overlap)
    if chunks:
        store.add_documents(chunks)
    return len(chunks)


def ingest_pdf(path: Path, embeddings: Embeddings, settings: Settings) -> int:
    docs = split_into_questions(load_pdf_pages(path))
    if not docs:
        raise ValueError(f"No numbered questions found in {path}; has the format changed?")
    return index_documents(docs, embeddings, settings)
