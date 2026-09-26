"""Document loading, OCR fallback, and chunking utilities."""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf as fitz
from charset_normalizer import from_bytes
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from src.config import OCR_DPI

logger = logging.getLogger(__name__)


class OCRUnavailableError(RuntimeError):
    """Raised when the OCR dependencies cannot be initialized."""


@lru_cache(maxsize=2)
def _get_ocr_reader(*, gpu: bool):
    """Create one EasyOCR reader per process and device."""
    try:
        import easyocr
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise OCRUnavailableError(
            "OCR support requires the 'easyocr' and OpenCV packages. Run 'uv sync' and try again."
        ) from exc

    try:
        import cv2  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise OCRUnavailableError(
            "OCR support requires OpenCV. Install the project dependencies and try again."
        ) from exc

    try:
        return easyocr.Reader(["en"], gpu=gpu, verbose=False)
    except Exception as exc:  # pragma: no cover - model/runtime dependent
        raise OCRUnavailableError(f"Unable to initialize EasyOCR: {exc}") from exc


def _sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_id_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "document"


def _relative_source(file_path: Path, source_root: Path | None) -> str:
    path = file_path.resolve()
    if source_root is None:
        return path.name
    try:
        return path.relative_to(source_root.resolve()).as_posix()
    except ValueError:
        return path.name


def _source_metadata(
    file_path: str | Path,
    file_type: str,
    *,
    source_root: str | Path | None = None,
    display_name: str | None = None,
) -> dict[str, Any]:
    """Build stable source metadata for both bulk and uploaded documents."""
    path = Path(file_path).resolve()
    name = display_name or path.name
    root = Path(source_root).resolve() if source_root is not None else None
    relative_name = _relative_source(path, root)
    content_hash = _sha256_file(path)

    if root is None:
        # Uploaded documents are logical sources identified by their sanitized
        # display name. Re-uploading a revised file replaces the old version;
        # the full content hash remains in metadata for traceability.
        source_id = f"upload:{file_type}:{_safe_id_part(name)}"
        source = name
    else:
        source_id = f"bulk:{file_type}:{relative_name}"
        source = relative_name

    return {
        "source_file": name if root is None else relative_name,
        "source_id": source_id,
        "source": source,
        "file_type": file_type,
        "content_sha256": content_hash,
    }


def _normalise_page(raw_page: Any, fallback: int) -> tuple[int, int]:
    """Return (zero-based page index, one-based display page)."""
    try:
        page_index = int(raw_page)
    except (TypeError, ValueError):
        page_index = fallback
    return page_index, page_index + 1


def content_digest(text: str) -> str:
    """Return a normalized digest so formatting differences do not hide duplicates.

    Some PDFs emit the same text layer more than once, and OCR can return nearly
    identical text for repeated layout. Normalizing whitespace, case, and
    punctuation means a repeated passage is recognized regardless of spacing.
    Retrieval reuses this so the two dedupe layers cannot disagree.
    """
    normalized = re.sub(r"\W+", " ", str(text or "")).strip().lower()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def _prepare_pdf_documents(
    documents: Iterable[Document], source_metadata: dict[str, Any]
) -> tuple[list[Document], list[int]]:
    """Normalize text pages, drop repeated pages, and identify pages needing OCR."""
    prepared: list[Document] = []
    blank_pages: list[int] = []
    seen_digests: set[str] = set()
    duplicates = 0
    for fallback, document in enumerate(documents):
        raw_page = document.metadata.get("page", fallback)
        page_index, display_page = _normalise_page(raw_page, fallback)
        metadata = dict(source_metadata)
        metadata.update(document.metadata)
        metadata["source_file"] = source_metadata["source_file"]
        metadata["source_id"] = source_metadata["source_id"]
        metadata["source"] = source_metadata["source"]
        metadata["file_type"] = "pdf"
        metadata["page_index"] = page_index
        metadata["page"] = display_page
        document.metadata = metadata

        if not document.page_content.strip():
            blank_pages.append(page_index)
            continue

        # Drop a page whose text was already captured verbatim. The first
        # occurrence wins so page numbers stay stable and citations remain
        # meaningful.
        digest = content_digest(document.page_content)
        if digest in seen_digests:
            duplicates += 1
            continue
        seen_digests.add(digest)
        prepared.append(document)

    if duplicates:
        logger.warning(
            "Dropped %d repeated page(s) from %s; the PDF text layer is duplicated",
            duplicates,
            source_metadata.get("source_file", "document"),
        )
    return prepared, blank_pages


def extract_text_with_ocr(
    file_path: str,
    *,
    page_indices: Iterable[int] | None = None,
    source_metadata: dict[str, Any] | None = None,
    dpi: int = OCR_DPI,
) -> list[Document]:
    """Extract text from selected PDF pages using EasyOCR.

    The reader is cached across files. Only pages without a text layer should
    be passed in ``page_indices``.
    """
    source_metadata = source_metadata or {
        "source_file": Path(file_path).name,
        "source_id": f"ocr:{Path(file_path).name}",
        "source": Path(file_path).name,
        "file_type": "pdf",
    }
    reader = _get_ocr_reader(gpu=False)
    document = fitz.open(file_path)
    documents: list[Document] = []
    indices = sorted(set(page_indices)) if page_indices is not None else range(len(document))

    try:
        import cv2

        for page_index in indices:
            if page_index < 0 or page_index >= len(document):
                continue
            page = document[page_index]
            pixmap = page.get_pixmap(dpi=dpi)
            image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                pixmap.h, pixmap.w, pixmap.n
            )
            if pixmap.n == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif pixmap.n == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif pixmap.n == 1:
                image = np.repeat(image, 3, axis=2)
            elif pixmap.n != 3:
                raise ValueError(f"Unsupported PDF pixmap channel count: {pixmap.n}")

            result = reader.readtext(image, detail=0, paragraph=True)
            content = " ".join(result).strip()
            if not content:
                continue

            metadata = dict(source_metadata)
            metadata.update(
                {
                    "file_type": "pdf",
                    "page_index": page_index,
                    "page": page_index + 1,
                }
            )
            documents.append(Document(page_content=content, metadata=metadata))
    finally:
        document.close()

    logger.info("OCR extracted %d requested pages from %s", len(documents), file_path)
    return documents


def _load_pdf(
    file_path: str | Path,
    *,
    source_metadata: dict[str, Any],
) -> list[Document]:
    reader = PdfReader(str(file_path))
    try:
        documents: list[Document] = []
        for page_index, page in enumerate(reader.pages):
            try:
                content = page.extract_text() or ""
            except Exception:
                logger.exception(
                    "Unable to extract text from %s page %d", file_path, page_index + 1
                )
                content = ""
            documents.append(Document(page_content=content, metadata={"page": page_index}))
    finally:
        reader.close()

    prepared, blank_pages = _prepare_pdf_documents(documents, source_metadata)
    if blank_pages:
        try:
            prepared.extend(
                extract_text_with_ocr(
                    str(file_path),
                    page_indices=blank_pages,
                    source_metadata=source_metadata,
                )
            )
        except OCRUnavailableError:
            if not prepared:
                raise
            logger.warning(
                "OCR is unavailable; retaining %d text-layer pages from %s",
                len(prepared),
                file_path,
            )

    return sorted(prepared, key=lambda document: document.metadata.get("page_index", 0))


def process_all_pdf(pdf_directory: str | Path) -> list[Document]:
    """Load all PDF files below a directory, preserving source-relative paths."""
    pdf_dir = Path(pdf_directory)
    pdf_files = sorted(pdf_dir.glob("**/*.pdf"))
    logger.info("Found %d PDF files in %s", len(pdf_files), pdf_dir)
    all_documents: list[Document] = []

    for pdf_file in pdf_files:
        logger.info("Processing PDF: %s", pdf_file)
        try:
            source_metadata = _source_metadata(pdf_file, "pdf", source_root=pdf_dir)
            all_documents.extend(_load_pdf(pdf_file, source_metadata=source_metadata))
        except OCRUnavailableError:
            logger.exception("Skipping %s because OCR dependencies are unavailable", pdf_file.name)
        except Exception:
            logger.exception("Unable to process PDF %s", pdf_file)
        logger.info("Loaded %d document pages so far", len(all_documents))

    return all_documents


def _load_text_file(
    file_path: str | Path,
    *,
    source_metadata: dict[str, Any],
) -> list[Document]:
    path = Path(file_path)
    raw_text = path.read_bytes()
    try:
        text = raw_text.decode("utf-8")
    except UnicodeDecodeError:
        detected = from_bytes(raw_text).best()
        encoding = detected.encoding if detected and detected.encoding else "utf-8"
        text = raw_text.decode(encoding, errors="replace")

    # A TXT file is a single logical page, so page/page_index are fixed rather
    # than derived. _source_metadata already supplies the identity keys.
    metadata = {**source_metadata, "file_type": "txt", "page_index": 0, "page": 1}
    return [Document(page_content=text, metadata=metadata)]


def process_all_txt(txt_directory: str | Path) -> list[Document]:
    """Load all text files below a directory with encoding detection."""
    txt_dir = Path(txt_directory)
    txt_files = sorted(txt_dir.glob("**/*.txt"))
    logger.info("Found %d TXT files in %s", len(txt_files), txt_dir)
    all_documents: list[Document] = []

    for txt_file in txt_files:
        logger.info("Processing TXT: %s", txt_file)
        try:
            source_metadata = _source_metadata(txt_file, "txt", source_root=txt_dir)
            all_documents.extend(_load_text_file(txt_file, source_metadata=source_metadata))
        except Exception:
            logger.exception("Unable to process TXT file %s", txt_file)
        logger.info("Loaded %d text documents so far", len(all_documents))

    return all_documents


def process_single_pdf(
    file_path: str,
    *,
    display_name: str | None = None,
) -> list[Document]:
    """Load one PDF and use OCR only for pages without a text layer."""
    pdf_file = Path(file_path)
    logger.info("Processing single PDF: %s", pdf_file.name)
    source_metadata = _source_metadata(
        pdf_file,
        "pdf",
        display_name=display_name,
    )
    try:
        return _load_pdf(pdf_file, source_metadata=source_metadata)
    except OCRUnavailableError:
        raise
    except Exception:
        logger.exception("Unable to process PDF %s", pdf_file.name)
        return []


def process_single_txt(
    file_path: str,
    *,
    display_name: str | None = None,
) -> list[Document]:
    """Load one text file with encoding detection."""
    txt_file = Path(file_path)
    logger.info("Processing single TXT: %s", txt_file.name)
    try:
        source_metadata = _source_metadata(
            txt_file,
            "txt",
            display_name=display_name,
        )
        return _load_text_file(txt_file, source_metadata=source_metadata)
    except Exception:
        logger.exception("Unable to process TXT file %s", txt_file.name)
        return []


def split_document(
    documents: list[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Split documents into overlapping chunks, preserving metadata and uniqueness.

    chunk_size and chunk_overlap are required rather than defaulted so the
    caller must pass the configured values; a silent default here would let the
    index chunk at a different size than CHUNK_SIZE without any signal.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
        add_start_index=True,
    )
    split_docs = splitter.split_documents(documents)

    # Drop repeated chunk content across the whole batch, not just within one
    # page. Large text files often repeat boilerplate blocks, and identical
    # chunks compete for the same evidence slots at query time without adding
    # any information. The first occurrence wins so offsets stay stable.
    unique: list[Document] = []
    seen: set[str] = set()
    duplicates = 0
    for document in split_docs:
        digest = content_digest(document.page_content)
        if digest in seen:
            duplicates += 1
            continue
        seen.add(digest)
        unique.append(document)

    if duplicates:
        logger.info(
            "Dropped %d duplicate chunk(s) of %d after splitting",
            duplicates,
            len(split_docs),
        )
    logger.info("Split %d documents into %d chunks", len(documents), len(unique))
    return unique
