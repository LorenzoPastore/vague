"""Embedding utilities for the Vague library."""

from __future__ import annotations

import logging
import platform
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict[str, object] = {}


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _mlx_available() -> bool:
    try:
        import mlx  # noqa: F401
        return True
    except ImportError:
        return False


@lru_cache(maxsize=4)
def _load_model(model: str) -> object:
    """Load and cache a SentenceTransformer model."""
    from sentence_transformers import SentenceTransformer

    if _is_apple_silicon() and _mlx_available():
        logger.info("Apple Silicon detected with mlx available; using mps device.")
        device = "mps"
    else:
        device = None  # sentence-transformers will auto-select

    logger.info("Loading sentence-transformer model: %s", model)
    return SentenceTransformer(model, device=device)


class Embedder:
    """Wraps a SentenceTransformer model and returns unit-normalized embeddings.

    On Apple Silicon with mlx installed the model runs on the MPS device for
    hardware acceleration.
    """

    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        """Initialize the Embedder.

        Args:
            model: SentenceTransformer model name or path.
        """
        self.model_name = model
        self._model = _load_model(model)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts and return unit-normalized vectors.

        Args:
            texts: Non-empty list of strings to embed.

        Returns:
            Array of shape (N, D) with L2-normalized rows.

        Raises:
            ValueError: If texts is empty.
        """
        if not texts:
            raise ValueError("texts must be a non-empty list.")

        raw: np.ndarray = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        return self._normalize(raw)

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text and return a unit-normalized vector.

        Args:
            text: String to embed.

        Returns:
            Array of shape (D,).
        """
        return self.embed([text])[0]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return vectors / norms
