"""Bulk ingestion command for the RAG application.

The command is intentionally idempotent: chunks use deterministic IDs and are
upserted into the configured collection. ``--rebuild`` is available when a
completely clean collection is desired.
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_BATCH_SIZE,
    PDF_DIR,
    TEXT_DIR,
    VECTOR_COLLECTION_NAME,
    VECTOR_STORE_DIR,
)
from src.data_loader import process_all_pdf, process_all_txt, split_document
from src.embedding import EmbeddingManager
from src.vector_store import VectorStore, _stable_source_id

logger = logging.getLogger(__name__)

# Aliased from config so the CLI's --batch-size default and the app's upload
# path stay on the same number instead of two hand-synced literals.
DEFAULT_EMBEDDING_BATCH_SIZE = EMBEDDING_BATCH_SIZE


def _load_from_dir(
    directory: str | Path,
    loader: Callable[[str | Path], list[Any]],
    label: str,
) -> list[Any]:
    path = Path(directory)
    if not path.exists():
        logger.warning("Directory %s does not exist. Skipping %s", path, label)
        return []
    if not path.is_dir():
        logger.warning("Path %s is not a directory. Skipping %s", path, label)
        return []
    logger.info("Loading %s from %s", label, path)
    return loader(path)


def _collect_documents(
    pdf_directory: str | Path = PDF_DIR,
    text_directory: str | Path = TEXT_DIR,
) -> list[Any]:
    documents: list[Any] = []
    documents.extend(_load_from_dir(pdf_directory, process_all_pdf, "PDFs"))
    documents.extend(_load_from_dir(text_directory, process_all_txt, "TXT files"))
    return documents


def _embed_and_store(
    documents: list[Any],
    embedding_manager: EmbeddingManager,
    vectorstore: VectorStore,
    *,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    rebuild: bool = False,
) -> int:
    """Embed to temporary batches before replacing any existing source rows."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not documents:
        logger.warning("No documents found to process")
        return 0

    chunks = split_document(
        documents,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    if not chunks:
        logger.warning("No chunks generated")
        return 0

    staged_batches: list[tuple[list[Any], Path]] = []
    with tempfile.TemporaryDirectory(prefix="rag-embedding-staging-") as staging_directory:
        staging_root = Path(staging_directory)
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            texts = [chunk.page_content for chunk in batch]
            logger.info(
                "Generating embeddings for chunks %d-%d of %d",
                start + 1,
                start + len(batch),
                len(chunks),
            )
            embeddings = embedding_manager.generate_embeddings(
                texts,
                is_query=False,
                show_progress_bar=True,
            )
            embedding_path = staging_root / f"batch-{start:012d}.npy"
            np.save(embedding_path, np.asarray(embeddings, dtype=np.float32), allow_pickle=False)
            staged_batches.append((batch, embedding_path))

        # Do not remove the previous index until every replacement embedding has
        # been generated successfully. A model/network failure therefore leaves
        # the existing collection usable.
        if rebuild:
            logger.warning("Resetting collection %s before ingestion", vectorstore.collection_name)
            vectorstore.reset_collection()
        else:
            source_ids = sorted({_stable_source_id(chunk.metadata) for chunk in chunks})
            vectorstore.delete_sources(source_ids)

        total = 0
        for batch, embedding_path in staged_batches:
            embeddings = np.load(embedding_path, allow_pickle=False)
            vectorstore.add_documents(batch, embeddings)
            total += len(batch)

    logger.info("Ingested %d chunks into %s", total, vectorstore.collection_name)
    return total


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf-directory",
        type=Path,
        default=PDF_DIR,
        help=f"Directory containing PDF files (default: {PDF_DIR})",
    )
    parser.add_argument(
        "--text-directory",
        type=Path,
        default=TEXT_DIR,
        help=f"Directory containing TXT files (default: {TEXT_DIR})",
    )
    parser.add_argument(
        "--collection-name",
        default=VECTOR_COLLECTION_NAME,
        help="Chroma collection to write",
    )
    parser.add_argument(
        "--persist-directory",
        type=Path,
        default=VECTOR_STORE_DIR,
        help=f"Chroma persistence directory (default: {VECTOR_STORE_DIR})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_EMBEDDING_BATCH_SIZE,
        help="Number of chunks embedded at once",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete and recreate the configured collection before ingestion",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info("--- Starting RAG data ingestion ---")

    vectorstore = VectorStore(
        collection_name=args.collection_name,
        persist_directory=args.persist_directory,
    )
    try:
        documents = _collect_documents(args.pdf_directory, args.text_directory)
        if documents:
            embedding_manager = EmbeddingManager()
            _embed_and_store(
                documents,
                embedding_manager,
                vectorstore,
                batch_size=args.batch_size,
                rebuild=args.rebuild,
            )
        else:
            logger.warning("No source documents were found; the collection was not changed")

        logger.info("--- Ingestion complete ---")
        for source in vectorstore.get_source_catalog():
            logger.info(
                " - %s (%s)",
                source.get("source_file", source["source_id"]),
                source["source_id"],
            )
    finally:
        vectorstore.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
