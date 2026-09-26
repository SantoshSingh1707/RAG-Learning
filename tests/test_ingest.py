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


class _EventLog:
    """Single ordered log shared by the fake store and the fake embedder.

    Ordering is the property under test, so every side effect has to land in
    one place. Separate per-caller lists would record *that* each step ran but
    not *when*, and could not distinguish correct staging from a version that
    deletes the old rows before embedding.
    """

    def __init__(self) -> None:
        self.events: list[tuple[str, int]] = []

    def record(self, event: str, size: int) -> None:
        self.events.append((event, size))

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.events]


class _RecordingManager:
    """Embedding manager that appends to the shared event log."""

    def __init__(self, log: _EventLog) -> None:
        self.log = log

    def generate_embeddings(self, texts, is_query=False, show_progress_bar=None):
        self.log.record("embed", len(texts))
        return np.ones((len(texts), 2), dtype=np.float32)


class _OrderingStore(FakeVectorStore):
    """Store that records writes into the shared event log."""

    def __init__(self, log: _EventLog) -> None:
        super().__init__()
        self.log = log

    def add_documents(self, documents, embeddings):
        super().add_documents(documents, embeddings)
        self.log.record("add", len(documents))


def _chunks(count: int) -> list[Document]:
    return [
        Document(page_content=f"chunk {index}", metadata={"source_id": "bulk:txt:notes.txt"})
        for index in range(count)
    ]


def test_stage_and_replace_embeds_everything_before_replacing() -> None:
    log = _EventLog()
    store = _OrderingStore(log)
    manager = _RecordingManager(log)

    total = stage_and_replace(
        _chunks(5),
        manager,
        store,
        batch_size=2,
        replace=lambda: log.record("replace", 0),
    )

    assert total == 5
    # 5 chunks at batch_size 2 is three embedding requests and three writes.
    # The swap sits between them: nothing is deleted until every embedding
    # exists, and nothing is written until after the swap.
    assert log.names == ["embed", "embed", "embed", "replace", "add", "add", "add"]
    assert [size for name, size in log.events if name == "embed"] == [2, 2, 1]
    assert [size for name, size in log.events if name == "add"] == [2, 2, 1]
    assert log.names.count("replace") == 1


def test_stage_and_replace_rejects_non_positive_batch_size() -> None:
    log = _EventLog()
    store = _OrderingStore(log)
    manager = _RecordingManager(log)

    with pytest.raises(ValueError, match="batch_size must be positive"):
        stage_and_replace(
            _chunks(1),
            manager,
            store,
            batch_size=0,
            replace=lambda: pytest.fail("must not replace"),
        )

    # The guard fires before anything is embedded, written, or replaced.
    assert log.events == []
    assert store.added == []


def test_stage_and_replace_keeps_staged_arrays_two_dimensional() -> None:
    store = FakeVectorStore()
    manager = _RecordingManager(_EventLog())

    stage_and_replace(_chunks(3), manager, store, batch_size=3, replace=lambda: None)

    for _, shape in store.added:
        assert shape == (3, 2)
