"""Embedding model management."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL, EMBEDDING_MODEL_REVISION

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manage a local SentenceTransformer embedding model."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        device: str | None = None,
        model_revision: str | None = EMBEDDING_MODEL_REVISION,
    ) -> None:
        self.model_name = model_name
        self.model_revision = model_revision
        self.device = device
        self.model: SentenceTransformer | None = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            import torch

            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info(
                "Loading embedding model %s (revision=%s) on %s",
                self.model_name,
                self.model_revision or "unpinned",
                self.device,
            )
            kwargs = {"device": self.device}
            if self.model_revision:
                kwargs["revision"] = self.model_revision
            self.model = SentenceTransformer(self.model_name, **kwargs)
        except Exception:
            logger.exception("Unable to load embedding model %s", self.model_name)
            raise

    def generate_embeddings(
        self,
        texts: Sequence[str],
        is_query: bool = False,
        *,
        show_progress_bar: bool | None = None,
    ) -> np.ndarray:
        """Generate normalized embeddings for passages or queries."""
        if self.model is None:
            raise RuntimeError("Embedding model is not loaded")
        if not texts:
            raise ValueError("texts must contain at least one item")
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("texts must contain only strings")

        prefix = "query: " if is_query else "passage: "
        prepared_texts = [f"{prefix}{text}" for text in texts]
        progress = (not is_query) if show_progress_bar is None else show_progress_bar
        logger.info(
            "Generating %s embeddings for %d texts",
            "query" if is_query else "passage",
            len(prepared_texts),
        )
        embeddings = np.asarray(
            self.model.encode(
                prepared_texts,
                normalize_embeddings=True,
                show_progress_bar=progress,
                convert_to_numpy=True,
            )
        )
        # encode(normalize_embeddings=True) already returns unit vectors. The
        # checks below only reject a model that silently failed to do so, which
        # would otherwise corrupt every downstream cosine comparison.
        if embeddings.ndim != 2 or not np.isfinite(embeddings).all():
            raise ValueError("Embedding model returned invalid vectors")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Embedding model returned a zero-length vector")
        logger.info("Generated embeddings with shape %s", embeddings.shape)
        return embeddings
