"""Persistent ChromaDB vector-store adapter."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

import chromadb
import numpy as np

from src.config import (
    VECTOR_COLLECTION_NAME,
    VECTOR_STORE_DIR,
)

logger = logging.getLogger(__name__)

MANIFEST_VERSION = 1
MANIFEST_PAGE_SIZE = 1000
WRITE_BATCH_SIZE = 1000


class VectorStoreError(RuntimeError):
    """Raised when the vector store cannot complete an operation."""


def _metadata_value(value: Any) -> str | int | float | bool | None:
    """Convert loader metadata to values accepted by ChromaDB."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(item) for item in value)
    return str(value)


def _clean_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if not metadata:
        return {}
    return {
        str(key): _metadata_value(value)
        for key, value in metadata.items()
        if str(key) and value is not None
    }


def _validated_embeddings(documents: list[Any], embeddings: Any) -> np.ndarray:
    """Coerce embeddings to a 2-D matrix that lines up one-to-one with documents.

    Every write path must agree on this check, otherwise a misaligned batch
    would store text against the wrong vector and quietly poison retrieval.
    """
    matrix = np.asarray(embeddings)
    if matrix.ndim != 2 or len(documents) != len(matrix):
        raise ValueError("Number of documents must match a 2-D embedding matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("Embeddings must contain only finite values")
    return matrix


def _stable_source_id(metadata: dict[str, Any]) -> str:
    source_id = metadata.get("source_id")
    if source_id:
        return str(source_id)
    source_file = str(metadata.get("source_file", "unknown"))
    # Older callers did not provide a source ID. Group their rows by the
    # available source label so replacement/removal remains possible; new
    # ingestion always supplies a path- or content-scoped source_id.
    return f"legacy:{source_file}"


def _stable_chunk_id(metadata: dict[str, Any], content: str, fallback_index: int) -> str:
    source_id = _stable_source_id(metadata)
    page = metadata.get("page_index", metadata.get("page", 0))
    start = metadata.get("start_index", metadata.get("chunk_index", fallback_index))
    identity = f"{source_id}|{page}|{start}|{content}"
    return f"chunk_{hashlib.sha256(identity.encode('utf-8')).hexdigest()}"


class VectorStore:
    """ChromaDB vector store with idempotent chunk IDs and a source manifest."""

    def __init__(
        self,
        collection_name: str = VECTOR_COLLECTION_NAME,
        persist_directory: str | Path = VECTOR_STORE_DIR,
    ) -> None:
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory).resolve()
        self.client: Any = None
        self.collection: Any = None
        self._manifest: dict[str, Any] = {}
        self._lock = threading.RLock()
        self.initialize_store()

    @property
    def manifest_path(self) -> Path:
        return self.persist_directory / "source_manifest.json"

    def initialize_store(self) -> None:
        """Create or open the cosine collection and load its source manifest."""
        try:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(self.persist_directory))
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                configuration={"hnsw": {"space": "cosine"}},
                metadata={
                    "description": "Document embeddings for the RAG application",
                    "schema_version": MANIFEST_VERSION,
                },
            )
            self._validate_collection_metric()
            self._load_or_build_manifest()
            logger.info(
                "Vector store initialized: collection=%s documents=%d",
                self.collection_name,
                self.collection.count(),
            )
        except Exception as exc:
            logger.exception("Unable to initialize vector store")
            raise VectorStoreError(str(exc)) from exc

    def _validate_collection_metric(self) -> None:
        configuration = getattr(self.collection, "configuration_json", {}) or {}
        space = (configuration.get("hnsw") or {}).get("space")
        if space != "cosine":
            raise VectorStoreError(
                f"Collection {self.collection_name!r} uses {space!r}, expected cosine. "
                "Use a new collection name or run the documented re-ingestion command."
            )

    def _empty_manifest(self) -> dict[str, Any]:
        return {
            "version": MANIFEST_VERSION,
            "collection_name": self.collection_name,
            "collection_count": self.collection.count(),
            "sources": {},
        }

    def _load_or_build_manifest(self) -> None:
        with self._lock:
            manifest = self._read_manifest()
            current_count = self.collection.count()
            if (
                manifest is None
                or manifest.get("version") != MANIFEST_VERSION
                or manifest.get("collection_name") != self.collection_name
                or manifest.get("collection_count") != current_count
            ):
                self._manifest = self._build_manifest()
                self._write_manifest()
            else:
                self._manifest = manifest

    def _read_manifest(self) -> dict[str, Any] | None:
        try:
            with self.manifest_path.open("r", encoding="utf-8") as stream:
                value = json.load(stream)
            return value if isinstance(value, dict) else None
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError):
            logger.warning("Source manifest is unreadable; rebuilding it")
            return None

    def _write_manifest(self) -> None:
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix="source_manifest.", suffix=".tmp", dir=self.persist_directory
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(self._manifest, stream, indent=2, sort_keys=True)
                stream.write("\n")
            os.replace(temporary_name, self.manifest_path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    def _build_manifest(self) -> dict[str, Any]:
        sources: dict[str, dict[str, Any]] = {}
        offset = 0
        while True:
            result = self.collection.get(
                include=["metadatas"],
                limit=MANIFEST_PAGE_SIZE,
                offset=offset,
            )
            metadatas = result.get("metadatas") or []
            if not metadatas:
                break
            for raw_metadata in metadatas:
                metadata = raw_metadata or {}
                source_file = str(metadata.get("source_file", "unknown"))
                source_id = str(metadata.get("source_id") or f"legacy:{source_file}")
                entry = sources.setdefault(
                    source_id,
                    {
                        "source_id": source_id,
                        "source_file": source_file,
                        "file_type": metadata.get("file_type"),
                    },
                )
                if not entry.get("source_file"):
                    entry["source_file"] = source_file
            offset += len(metadatas)
            if len(metadatas) < MANIFEST_PAGE_SIZE:
                break

        return {
            "version": MANIFEST_VERSION,
            "collection_name": self.collection_name,
            "collection_count": self.collection.count(),
            "sources": sources,
        }

    def rebuild_manifest(self) -> list[dict[str, Any]]:
        """Rebuild the source manifest from Chroma metadata."""
        with self._lock:
            self._manifest = self._build_manifest()
            self._write_manifest()
            return self.get_source_catalog()

    def get_source_catalog(self) -> list[dict[str, Any]]:
        """Return the lightweight source catalog without scanning the index."""
        with self._lock:
            if not self._manifest:
                self._load_or_build_manifest()
            sources = list(self._manifest.get("sources", {}).values())
        return sorted(sources, key=lambda item: (item.get("source_file", ""), item["source_id"]))

    def _where_for_source(self, source_id: str, source_file: str | None = None) -> dict[str, Any]:
        # New records always have source_id. The fallback is for a catalog
        # built from a legacy collection.
        if source_id.startswith("legacy:") and source_file:
            return {"source_file": source_file}
        return {"source_id": source_id}

    def remove_source(self, source_name: str) -> bool:
        """Remove a source and report whether at least one row was deleted."""
        catalog_entry = next(
            (
                item
                for item in self.get_source_catalog()
                if item["source_id"] == source_name or item.get("source_file") == source_name
            ),
            None,
        )
        source_id = source_name
        source_file = None
        if catalog_entry:
            source_id = catalog_entry["source_id"]
            source_file = catalog_entry.get("source_file")

        with self._lock:
            before = self.collection.count()
            self.collection.delete(where=self._where_for_source(source_id, source_file))
            after = self.collection.count()
            removed = after < before
            if catalog_entry:
                self._manifest.get("sources", {}).pop(catalog_entry["source_id"], None)
            self._manifest["collection_count"] = after
            self._write_manifest()
            if removed:
                logger.info("Removed source %s (%d rows)", source_name, before - after)
            else:
                logger.warning("No rows matched source %s", source_name)
            return removed

    def _prepare_documents(
        self, documents: list[Any]
    ) -> tuple[list[str], list[dict[str, Any]], list[str]]:
        ids: list[str] = []
        metadatas: list[dict[str, Any]] = []
        texts: list[str] = []
        for fallback_index, document in enumerate(documents):
            content = document.page_content
            if not isinstance(content, str) or not content.strip():
                raise ValueError("Documents must contain non-empty text")
            metadata = _clean_metadata(document.metadata)
            source_id = _stable_source_id(metadata)
            metadata.setdefault("source_file", source_id)
            metadata["source_id"] = source_id
            metadata["chunk_index"] = metadata.get(
                "start_index", metadata.get("chunk_index", fallback_index)
            )
            metadata["content_length"] = len(content)
            ids.append(_stable_chunk_id(metadata, content, fallback_index))
            metadatas.append(metadata)
            texts.append(content)
        return ids, metadatas, texts

    def _upsert_batches(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        metadatas: list[dict[str, Any]],
        documents: list[str],
    ) -> None:
        for start in range(0, len(ids), WRITE_BATCH_SIZE):
            end = start + WRITE_BATCH_SIZE
            self.collection.upsert(
                ids=ids[start:end],
                embeddings=embeddings[start:end].tolist(),
                metadatas=metadatas[start:end],
                documents=documents[start:end],
            )

    def _update_manifest_after_write(self, metadatas: list[dict[str, Any]]) -> None:
        for metadata in metadatas:
            source_id = str(metadata["source_id"])
            source_file = str(metadata.get("source_file", source_id))
            self._manifest.setdefault("sources", {})[source_id] = {
                "source_id": source_id,
                "source_file": source_file,
                "file_type": metadata.get("file_type"),
            }
        self._manifest["collection_count"] = self.collection.count()
        self._write_manifest()

    def delete_sources(self, source_ids: list[str]) -> int:
        """Delete all rows for the supplied stable source IDs."""
        removed = 0
        with self._lock:
            before = self.collection.count()
            for source_id in sorted(set(source_ids)):
                entry = self._manifest.get("sources", {}).get(source_id, {})
                source_file = entry.get("source_file")
                self.collection.delete(where=self._where_for_source(source_id, source_file))
                self._manifest.get("sources", {}).pop(source_id, None)
            after = self.collection.count()
            removed = before - after
            self._manifest["collection_count"] = after
            self._write_manifest()
        return removed

    def add_documents(
        self,
        documents: list[Any],
        embeddings: np.ndarray,
        *,
        update_manifest: bool = True,
    ) -> None:
        """Upsert chunks using deterministic IDs, making retries idempotent."""
        if not documents:
            logger.warning("No documents to add to vector store")
            return
        embeddings = _validated_embeddings(documents, embeddings)

        with self._lock:
            ids, metadatas, texts = self._prepare_documents(documents)
            self._upsert_batches(ids, embeddings, metadatas, texts)
            if update_manifest:
                self._update_manifest_after_write(metadatas)
                logger.info(
                    "Upserted %d chunks; collection count is %d",
                    len(ids),
                    self.collection.count(),
                )
            else:
                logger.info("Upserted %d chunks (manifest update deferred)", len(ids))

    def replace_documents(
        self,
        documents: list[Any],
        embeddings: np.ndarray,
    ) -> None:
        """Replace all chunks belonging to the supplied sources."""
        if not documents:
            return
        embeddings = _validated_embeddings(documents, embeddings)

        with self._lock:
            ids, metadatas, texts = self._prepare_documents(documents)
            source_ids = sorted({metadata["source_id"] for metadata in metadatas})
            for source_id in source_ids:
                source_file = next(
                    metadata.get("source_file")
                    for metadata in metadatas
                    if metadata["source_id"] == source_id
                )
                self.collection.delete(where=self._where_for_source(source_id, source_file))
            self._upsert_batches(ids, embeddings, metadatas, texts)
            self._update_manifest_after_write(metadatas)
            logger.info(
                "Replaced %d chunks across %d sources; collection count is %d",
                len(ids),
                len(source_ids),
                self.collection.count(),
            )

    def reset_collection(self) -> None:
        """Drop and recreate the configured collection and manifest."""
        with self._lock:
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                logger.debug("Collection did not exist before reset", exc_info=True)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                configuration={"hnsw": {"space": "cosine"}},
                metadata={
                    "description": "Document embeddings for the RAG application",
                    "schema_version": MANIFEST_VERSION,
                },
            )
            self._manifest = self._empty_manifest()
            self._write_manifest()

    def close(self) -> None:
        """Release the Chroma client when a short-lived process exits."""
        if self.client is not None:
            close = getattr(self.client, "close", None)
            if callable(close):
                close()
            self.client = None
            self.collection = None


class SupportsEmbedding(Protocol):
    """The slice of the embedding manager that ingestion depends on.

    Declared here rather than importing ``EmbeddingManager`` so that ingestion
    can be driven by any object with the same method, and so tests can supply a
    fake without constructing a real sentence-transformers model.
    """

    def generate_embeddings(
        self,
        texts: Sequence[str],
        *,
        is_query: bool,
        show_progress_bar: bool,
    ) -> Any:
        """Return one embedding per entry in ``texts``."""
        ...


def stage_and_replace(
    chunks: list[Any],
    embedding_manager: SupportsEmbedding,
    vectorstore: VectorStore,
    *,
    batch_size: int,
    replace: Callable[[], None],
) -> int:
    """Embed every batch to disk, then swap the stored rows in.

    Both the bulk command and the app's upload path need the same property: a
    failure while *generating* embeddings must not damage what is already
    indexed. So every embedding is written to a temporary ``.npy`` file first,
    and the rows these chunks supersede are only removed once all of them
    exist. A model or network error part way through therefore leaves the
    previous version of the source fully intact rather than half-indexed or
    missing. That is the case worth engineering for, because the embedding
    model is the slow, remote-capable, failure-prone step.

    What this does *not* make atomic is the write itself. ``replace`` runs
    before the first ``add_documents`` call, so a store failure after that
    point (disk full, Chroma write error) leaves the source deleted or partly
    written. Closing that window needs a transactional replace in the store,
    which ChromaDB does not expose; the next run repairs the damage by
    replacing the same source again, and the deterministic chunk IDs mean the
    retry is idempotent. Do not read this function as a general rollback.

    Embeddings are staged on disk instead of held in memory so peak usage does
    not scale with the number of chunks, which matters for the ~127k chunk
    index.

    Args:
        chunks: Chunks to embed and store.
        embedding_manager: Source of the embeddings.
        vectorstore: Destination store.
        batch_size: Chunks per embedding request. Must be positive.
        replace: Removes or resets the rows that ``chunks`` supersede. Called
            once, after every embedding has been generated successfully.

    Returns:
        The number of chunks stored.

    Raises:
        ValueError: If ``batch_size`` is not positive.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    with tempfile.TemporaryDirectory(prefix="rag-embedding-staging-") as staging_directory:
        staging_root = Path(staging_directory)
        staged_batches: list[tuple[list[Any], Path]] = []
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            logger.info(
                "Generating embeddings for chunks %d-%d of %d",
                start + 1,
                start + len(batch),
                len(chunks),
            )
            embeddings = embedding_manager.generate_embeddings(
                [chunk.page_content for chunk in batch],
                is_query=False,
                show_progress_bar=True,
            )
            embedding_path = staging_root / f"batch-{start:012d}.npy"
            np.save(embedding_path, np.asarray(embeddings, dtype=np.float32), allow_pickle=False)
            staged_batches.append((batch, embedding_path))

        # The previous rows survive every failure above; only now is it safe to
        # drop them.
        replace()

        total = 0
        for batch, embedding_path in staged_batches:
            vectorstore.add_documents(batch, np.load(embedding_path, allow_pickle=False))
            total += len(batch)
    return total
