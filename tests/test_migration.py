from pathlib import Path
from uuid import uuid4

import chromadb
import numpy as np

from migrate_legacy_index import migrate
from src.vector_store import VectorStore


def test_legacy_migration_copies_embeddings_and_assigns_stable_source(tmp_path: Path) -> None:
    legacy_name = f"legacy_{uuid4().hex[:12]}"
    target_name = f"target_{uuid4().hex[:12]}"
    pdf_dir = tmp_path / "pdf"
    text_dir = tmp_path / "textfiles"
    pdf_dir.mkdir()
    text_dir.mkdir()
    (text_dir / "notes.txt").write_text("current source", encoding="utf-8")

    client = chromadb.PersistentClient(path=str(tmp_path))
    legacy = client.create_collection(
        legacy_name,
        configuration={"hnsw": {"space": "l2"}},
    )
    legacy.add(
        ids=["old-1", "duplicate"],
        embeddings=[[1.0, 0.0], [1.0, 0.0]],
        documents=["legacy text", "legacy text"],
        metadatas=[
            {"source_file": "notes.txt", "file_type": "txt", "page": 0, "doc_index": 0},
            {"source_file": "notes.txt", "file_type": "txt", "page": 0, "doc_index": 0},
        ],
    )
    client.close()

    seen, copied = migrate(
        persist_directory=tmp_path,
        target_collection=target_name,
        legacy_collection=legacy_name,
        pdf_directory=pdf_dir,
        text_directory=text_dir,
        replace_target=True,
    )

    assert (seen, copied) == (2, 1)
    target = VectorStore(target_name, tmp_path)
    try:
        assert target.collection.count() == 1
        assert [item["source_id"] for item in target.get_source_catalog()] == ["bulk:txt:notes.txt"]
        result = target.collection.get(include=["metadatas", "embeddings"])
        assert result["metadatas"][0]["page"] == 1
        assert np.allclose(result["embeddings"][0], [1.0, 0.0])
    finally:
        target.close()
