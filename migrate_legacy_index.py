"""Migrate the historical L2 collection into the cosine v2 collection.

This utility copies existing embeddings instead of recomputing them. It is
non-destructive with respect to the legacy collection and is intended for a
one-time migration after verifying that the legacy embeddings were produced by
the configured embedding model.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.documents import Document

from src.config import (
    LEGACY_VECTOR_COLLECTION_NAME,
    PDF_DIR,
    TEXT_DIR,
    VECTOR_COLLECTION_NAME,
    VECTOR_STORE_DIR,
)
from src.data_loader import _source_metadata
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 500


def _build_source_map(pdf_directory: Path, text_directory: Path) -> dict[str, list[dict[str, Any]]]:
    mapping: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for directory, file_type, pattern in (
        (pdf_directory, "pdf", "**/*.pdf"),
        (text_directory, "txt", "**/*.txt"),
    ):
        if not directory.exists():
            continue
        for path in sorted(directory.glob(pattern)):
            metadata = _source_metadata(path, file_type, source_root=directory)
            mapping[path.name].append(metadata)
    return mapping


def _source_for_legacy_metadata(
    metadata: dict[str, Any], source_map: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    source_file = str(
        metadata.get("source_file")
        or Path(str(metadata.get("source", "unknown"))).name
        or "unknown"
    )
    candidates = source_map.get(source_file, [])
    if len(candidates) == 1:
        return dict(candidates[0])
    # Ambiguous or unknown names retain a legacy identity rather than silently
    # merging unrelated files.
    return {
        "source_file": source_file,
        "source_id": f"legacy:{metadata.get('file_type', 'unknown')}:{source_file}",
        "source": source_file,
        "file_type": metadata.get("file_type", "unknown"),
    }


def _page_values(metadata: dict[str, Any]) -> tuple[int, int]:
    try:
        page_index = int(metadata.get("page", 0))
    except (TypeError, ValueError):
        page_index = 0
    return page_index, page_index + 1


def _legacy_key(source_id: str, page_index: int, start_index: int | str, content: str) -> str:
    # start_index is widened because the pre-v2 collection stored it as either an
    # int or a string depending on the writer; the key only needs a stable
    # textual identity, so coercing at the call site would be misleading.
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return f"{source_id}|{page_index}|{start_index}|{digest}"


def migrate(
    *,
    persist_directory: Path = VECTOR_STORE_DIR,
    target_collection: str = VECTOR_COLLECTION_NAME,
    legacy_collection: str = LEGACY_VECTOR_COLLECTION_NAME,
    pdf_directory: Path = PDF_DIR,
    text_directory: Path = TEXT_DIR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    replace_target: bool = False,
) -> tuple[int, int]:
    """Copy unique legacy rows and return ``(seen_rows, copied_rows)``."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if target_collection == legacy_collection:
        raise ValueError("Target and legacy collection names must be different")

    target = VectorStore(target_collection, persist_directory)
    if replace_target:
        target.reset_collection()

    client = target.client
    try:
        source = client.get_collection(legacy_collection)
    except Exception as exc:
        target.close()
        raise RuntimeError(f"Legacy collection {legacy_collection!r} was not found") from exc

    source_map = _build_source_map(pdf_directory, text_directory)
    seen_keys: set[str] = set()
    pending_documents: list[Document] = []
    pending_embeddings: list[np.ndarray] = []
    seen_rows = 0
    copied_rows = 0
    offset = 0
    page_size = 1000

    def flush() -> None:
        nonlocal copied_rows
        if not pending_documents:
            return
        embeddings = np.stack(pending_embeddings).astype(np.float32)
        target.add_documents(pending_documents, embeddings, update_manifest=False)
        copied_rows += len(pending_documents)
        pending_documents.clear()
        pending_embeddings.clear()

    try:
        while True:
            result = source.get(
                include=["embeddings", "metadatas", "documents"],
                limit=page_size,
                offset=offset,
            )
            metadatas = result.get("metadatas") or []
            documents = result.get("documents") or []
            embeddings = result.get("embeddings")
            if not metadatas:
                break

            for index, raw_metadata in enumerate(metadatas):
                seen_rows += 1
                metadata = raw_metadata or {}
                content = documents[index] if index < len(documents) else ""
                if not content or embeddings is None or index >= len(embeddings):
                    continue
                source_metadata = _source_for_legacy_metadata(metadata, source_map)
                page_index, page = _page_values(metadata)
                start_index = metadata.get("start_index", metadata.get("doc_index", 0))
                key = _legacy_key(
                    source_metadata["source_id"], page_index, str(start_index), content
                )
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                new_metadata = dict(source_metadata)
                new_metadata.update(
                    {
                        "page_index": page_index,
                        "page": page,
                        "start_index": start_index,
                        "content_length": len(content),
                        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    }
                )
                pending_documents.append(Document(page_content=content, metadata=new_metadata))
                pending_embeddings.append(np.asarray(embeddings[index], dtype=np.float32))
                if len(pending_documents) >= batch_size:
                    flush()
                    if copied_rows % 10_000 == 0:
                        logger.info("Copied %d unique rows", copied_rows)

            offset += len(metadatas)
            if len(metadatas) < page_size:
                break
        flush()
        target.rebuild_manifest()
    finally:
        target.close()

    logger.info("Legacy migration complete: scanned=%d copied=%d", seen_rows, copied_rows)
    return seen_rows, copied_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persist-directory", type=Path, default=VECTOR_STORE_DIR)
    parser.add_argument("--target-collection", default=VECTOR_COLLECTION_NAME)
    parser.add_argument("--legacy-collection", default=LEGACY_VECTOR_COLLECTION_NAME)
    parser.add_argument("--pdf-directory", type=Path, default=PDF_DIR)
    parser.add_argument("--text-directory", type=Path, default=TEXT_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--replace-target",
        action="store_true",
        help="Delete the target v2 collection before copying",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    migrate(
        persist_directory=args.persist_directory,
        target_collection=args.target_collection,
        legacy_collection=args.legacy_collection,
        pdf_directory=args.pdf_directory,
        text_directory=args.text_directory,
        batch_size=args.batch_size,
        replace_target=args.replace_target,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
