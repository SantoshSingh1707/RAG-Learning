import numpy as np
import pytest
from langchain_core.documents import Document

from ingest_data import _embed_and_store
from src.vector_store import stage_and_replace


class FakeEmbeddingManager:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def generate_embeddings(self, texts, is_query=False, show_progress_bar=None):
        if self.fail:
            raise RuntimeError("embedding failure")
        return np.ones((len(texts), 2), dtype=np.float32)


class FakeVectorStore:
    collection_name = "test_collection"

    def __init__(self) -> None:
        self.deleted = []
        self.added = []

    def delete_sources(self, source_ids):
        self.deleted.append(list(source_ids))
        return 0

    def add_documents(self, documents, embeddings):
        self.added.append((len(documents), embeddings.shape))

    def reset_collection(self):
        raise AssertionError("reset_collection must not run when rebuild is False")


def _document() -> Document:
    return Document(
        page_content="A short source document.",
        metadata={
            "source_id": "bulk:txt:notes.txt",
            "source_file": "notes.txt",
            "file_type": "txt",
        },
    )


def test_ingestion_stages_embeddings_before_replacing_sources() -> None:
    store = FakeVectorStore()
    manager = FakeEmbeddingManager()

    count = _embed_and_store([_document()], manager, store, batch_size=1)

    assert count == 1
    assert store.deleted == [["bulk:txt:notes.txt"]]
    assert store.added == [(1, (1, 2))]


def test_ingestion_does_not_delete_when_embedding_fails() -> None:
    store = FakeVectorStore()
    manager = FakeEmbeddingManager(fail=True)

    with pytest.raises(RuntimeError, match="embedding failure"):
        _embed_and_store([_document()], manager, store, batch_size=1)

    assert store.deleted == []
    assert store.added == []


class _RecordingManager:
    """Embedding manager that records how many texts each call received."""

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def generate_embeddings(self, texts, is_query=False, show_progress_bar=None):
        self.batch_sizes.append(len(texts))
        return np.ones((len(texts), 2), dtype=np.float32)


def _chunks(count: int) -> list[Document]:
    return [
        Document(page_content=f"chunk {index}", metadata={"source_id": "bulk:txt:notes.txt"})
        for index in range(count)
    ]


def test_stage_and_replace_batches_then_replaces_once() -> None:
    store = FakeVectorStore()
    manager = _RecordingManager()
    calls: list[str] = []

    total = stage_and_replace(
        _chunks(5),
        manager,
        store,
        batch_size=2,
        replace=lambda: calls.append("replace"),
    )

    assert total == 5
    # 5 chunks at batch_size 2 is three requests, and the store sees all three.
    assert manager.batch_sizes == [2, 2, 1]
    assert [size for size, _ in store.added] == [2, 2, 1]
    # Every embedding exists before the previous rows are dropped.
    assert len(manager.batch_sizes) == len(store.added)
    assert calls == ["replace"]


def test_stage_and_replace_rejects_non_positive_batch_size() -> None:
    store = FakeVectorStore()
    manager = _RecordingManager()

    with pytest.raises(ValueError, match="batch_size must be positive"):
        stage_and_replace(
            _chunks(1),
            manager,
            store,
            batch_size=0,
            replace=lambda: pytest.fail("must not replace"),
        )

    assert manager.batch_sizes == []
    assert store.added == []


def test_stage_and_replace_keeps_staged_arrays_two_dimensional() -> None:
    store = FakeVectorStore()
    manager = _RecordingManager()

    stage_and_replace(_chunks(3), manager, store, batch_size=3, replace=lambda: None)

    for _, shape in store.added:
        assert shape == (3, 2)
