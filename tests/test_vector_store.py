from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from langchain_core.documents import Document

from src.vector_store import VectorStore, VectorStoreError


def _document(text: str, source_id: str, source_file: str, start: int) -> Document:
    return Document(
        page_content=text,
        metadata={
            "source_id": source_id,
            "source_file": source_file,
            "file_type": "txt",
            "page_index": 0,
            "page": 1,
            "start_index": start,
        },
    )


def _source_ids(store: VectorStore) -> list[str]:
    return [item["source_id"] for item in store.get_source_catalog()]


def test_upsert_is_idempotent_and_replace_removes_old_chunks(tmp_path: Path) -> None:
    store = VectorStore(f"test_{uuid4().hex[:12]}", tmp_path)
    first = [
        _document("alpha", "bulk:txt:notes.txt", "notes.txt", 0),
        _document("beta", "bulk:txt:notes.txt", "notes.txt", 10),
    ]
    second = [_document("replacement", "bulk:txt:notes.txt", "notes.txt", 0)]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    replacement_embedding = np.array([[1.0, 0.0]], dtype=np.float32)

    store.add_documents(first, embeddings)
    store.add_documents(first, embeddings)
    assert store.collection.count() == 2
    assert _source_ids(store) == ["bulk:txt:notes.txt"]

    store.replace_documents(second, replacement_embedding)
    assert store.collection.count() == 1
    result = store.collection.get(include=["documents"])
    assert result["documents"] == ["replacement"]
    store.close()


def test_source_manifest_paginates_beyond_one_chroma_page(tmp_path: Path) -> None:
    store = VectorStore(f"test_{uuid4().hex[:12]}", tmp_path)
    documents = [
        _document(f"row-{index}", "bulk:txt:large.txt", "large.txt", index) for index in range(1001)
    ]
    embeddings = np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(documents), 1))

    store.add_documents(documents, embeddings, update_manifest=False)
    store.rebuild_manifest()

    assert store.collection.count() == 1001
    assert _source_ids(store) == ["bulk:txt:large.txt"]
    store.close()


def test_source_manifest_is_persisted_and_reloaded(tmp_path: Path) -> None:
    collection_name = f"test_{uuid4().hex[:12]}"
    store = VectorStore(collection_name, tmp_path)
    store.add_documents(
        [_document("alpha", "bulk:txt:a.txt", "a.txt", 0)],
        np.array([[1.0, 0.0]], dtype=np.float32),
    )

    reloaded = VectorStore(collection_name, tmp_path)

    assert reloaded.get_source_catalog() == [
        {
            "source_id": "bulk:txt:a.txt",
            "source_file": "a.txt",
            "file_type": "txt",
        }
    ]
    reloaded.close()
    store.close()


def test_legacy_l2_collection_is_rejected(tmp_path: Path) -> None:
    import chromadb

    collection_name = f"test_{uuid4().hex[:12]}"
    client = chromadb.PersistentClient(path=str(tmp_path))
    client.create_collection(
        collection_name,
        configuration={"hnsw": {"space": "l2"}},
    )

    with pytest.raises(VectorStoreError, match="expected cosine"):
        VectorStore(collection_name, tmp_path)
