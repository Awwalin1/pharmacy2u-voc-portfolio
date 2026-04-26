"""
embedder.py

Sentence-transformers wrapper for the multi-channel contact corpus.

Uses all-MiniLM-L6-v2 — 80MB, fast, well-suited to short customer-contact text.
For production VoC at higher accuracy, consider all-mpnet-base-v2 (420MB).

The model is loaded lazily on first use. Embeddings are L2-normalised so
cosine similarity reduces to a dot product (faster downstream).
"""

from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class ContactEmbedder:
    """Embeds customer contact text into dense vectors for similarity search
    and downstream classification."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None

    def _ensure_loaded(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a list of strings. Returns a (n, dim) float32 numpy array,
        L2-normalised so cosine similarity is a dot product."""
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)

        model = self._ensure_loaded()
        vectors = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vectors.astype(np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single string. Returns a (dim,) vector."""
        return self.embed([text])[0]