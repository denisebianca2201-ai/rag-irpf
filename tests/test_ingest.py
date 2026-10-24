from ask_the_lion.ingest import (
    build_vector_store,
    index_documents,
    split_into_questions,
    split_oversized,
)
from conftest import HashEmbeddings, PAGES


def test_splits_one_document_per_question(docs):
    assert [d.metadata["question_number"] for d in docs] == [1, 2, 3]


def test_metadata_has_page_and_chapter(docs):
    first, _, third = docs
    assert first.metadata["page"] == 1
    assert first.metadata["chapter"].startswith("CAPÍTULO 1")
    assert third.metadata["page"] == 2
    assert third.metadata["chapter"].startswith("CAPÍTULO 2")


def test_answer_text_stays_with_its_question(docs):
    assert "limite anual" in docs[0].page_content
    assert "limite anual" not in docs[1].page_content
    # multi-line answers are kept together
    assert "dedução anual" in docs[2].page_content


def test_text_before_first_question_is_ignored():
    pages = [(1, "Cover page\nSome preface\n001 — First question?\nAnswer.")]
    docs = split_into_questions(pages)
    assert len(docs) == 1
    assert "preface" not in docs[0].page_content


def test_no_questions_found_returns_empty():
    assert split_into_questions([(1, "Just prose without numbered questions.")]) == []


def test_oversized_answers_are_split_and_keep_metadata(docs):
    long_doc = docs[0].model_copy(update={"page_content": "palavra " * 400})
    chunks = split_oversized([long_doc], chunk_size=200, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(c.metadata["question_number"] == 1 for c in chunks)


def test_index_replaces_previous_content(settings, docs):
    embeddings = HashEmbeddings()
    assert index_documents(docs, embeddings, settings) == 3
    assert index_documents(docs[:1], embeddings, settings) == 1
    store = build_vector_store(embeddings, settings)
    assert len(store.get()["ids"]) == 1


def test_pdf_roundtrip(tmp_path, settings):
    """Generate a real PDF and make sure extraction + splitting works on it."""
    from pypdf import PdfWriter

    from ask_the_lion.ingest import load_pdf_pages

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    path = tmp_path / "blank.pdf"
    with path.open("wb") as fh:
        writer.write(fh)
    pages = load_pdf_pages(path)
    assert pages == [(1, "")]
