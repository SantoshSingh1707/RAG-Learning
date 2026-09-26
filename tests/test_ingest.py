import numpy as np
import pytest
from langchain_core.documents import Document

from ingest_data import _embed_and_store


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
