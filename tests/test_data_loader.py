from pathlib import Path

from langchain_core.documents import Document

from src.data_loader import (
    _prepare_pdf_documents,
    _source_metadata,
    process_single_txt,
    split_document,
)


def test_source_metadata_distinguishes_nested_same_named_files(tmp_path: Path) -> None:
    root = tmp_path / "textfiles"
    first = root / "one" / "report.txt"
    second = root / "two" / "report.txt"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")

    first_metadata = _source_metadata(first, "txt", source_root=root)
    second_metadata = _source_metadata(second, "txt", source_root=root)

    assert first_metadata["source_id"] != second_metadata["source_id"]
    assert first_metadata["source_file"] == "one/report.txt"
    assert second_metadata["source_file"] == "two/report.txt"


def test_process_single_txt_preserves_stable_upload_identity(tmp_path: Path) -> None:
    path = tmp_path / "upload.txt"
    path.write_text("Hello, 世界", encoding="utf-8")

    documents = process_single_txt(str(path), display_name="upload.txt")

    assert len(documents) == 1
    assert documents[0].page_content == "Hello, 世界"
    assert documents[0].metadata["source_id"].startswith("upload:txt:")
    assert documents[0].metadata["source_file"] == "upload.txt"
    assert documents[0].metadata["page"] == 1


def test_split_document_preserves_source_and_adds_start_index() -> None:
    document = Document(
        page_content="alpha beta gamma delta " * 100,
        metadata={
            "source_id": "bulk:txt:notes.txt",
            "source_file": "notes.txt",
            "page_index": 0,
            "page": 1,
        },
    )

    chunks = split_document([document], chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(chunk.metadata["source_id"] == "bulk:txt:notes.txt" for chunk in chunks)
    assert all("start_index" in chunk.metadata for chunk in chunks)


def test_prepare_pdf_documents_drops_repeated_page_text() -> None:
    # Some PDFs emit the same text layer twice, so the loader sees page 3 twice
    # with identical content. Only the first occurrence should survive, and it
    # must be the one that carries the citation metadata.
    source_metadata = {
        "source_file": "paper.pdf",
        "source_id": "bulk:pdf:paper.pdf",
        "source": "paper.pdf",
    }
    pages = [
        Document(page_content="page one", metadata={"page": 0}),
        Document(page_content="shared page text", metadata={"page": 1}),
        Document(page_content="shared page text", metadata={"page": 2}),
    ]

    prepared, blank_pages = _prepare_pdf_documents(pages, source_metadata)

    assert [document.page_content for document in prepared] == ["page one", "shared page text"]
    assert blank_pages == []
    assert prepared[-1].metadata["page"] == 2


def test_prepare_pdf_documents_ignores_whitespace_and_case_differences() -> None:
    source_metadata = {
        "source_file": "paper.pdf",
        "source_id": "bulk:pdf:paper.pdf",
        "source": "paper.pdf",
    }
    pages = [
        Document(page_content="Repeated Content Here", metadata={"page": 0}),
        Document(page_content="repeated   content here!", metadata={"page": 1}),
    ]

    prepared, _ = _prepare_pdf_documents(pages, source_metadata)

    assert len(prepared) == 1
    assert prepared[0].page_content == "Repeated Content Here"


def test_prepare_pdf_documents_still_routes_blank_pages_to_ocr() -> None:
    source_metadata = {
        "source_file": "scan.pdf",
        "source_id": "bulk:pdf:scan.pdf",
        "source": "scan.pdf",
    }
    pages = [
        Document(page_content="text layer", metadata={"page": 0}),
        Document(page_content="   ", metadata={"page": 1}),
        Document(page_content="", metadata={"page": 2}),
    ]

    prepared, blank_pages = _prepare_pdf_documents(pages, source_metadata)

    assert len(prepared) == 1
    assert blank_pages == [1, 2]


def test_split_document_drops_duplicate_chunks() -> None:
    # Large text files repeat boilerplate blocks verbatim. Identical chunks add
    # no information and would compete for the same evidence slots at query
    # time, so only the first copy is indexed.
    repeated = "Repeated boilerplate block. " * 40
    document = Document(
        page_content=repeated + "unique middle content. " * 40 + repeated,
        metadata={"source_id": "bulk:txt:notes.txt", "source_file": "notes.txt"},
    )

    chunks = split_document([document], chunk_size=400, chunk_overlap=0)

    assert len(chunks) > 1
    assert len({chunk.page_content for chunk in chunks}) == len(chunks)


def test_split_document_keeps_distinct_chunks() -> None:
    document = Document(
        page_content="alpha beta gamma delta " * 100,
        metadata={"source_id": "bulk:txt:notes.txt", "source_file": "notes.txt"},
    )

    chunks = split_document([document], chunk_size=100, chunk_overlap=20)

    assert len(chunks) == len({chunk.page_content for chunk in chunks})
