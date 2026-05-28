from __future__ import annotations
import logging
from sentence_transformers import SentenceTransformer
from core.config import settings

logger = logging.getLogger(__name__)

_embedder_instance: Embedder | None = None


class Embedder:
    def __init__(self):
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        self.model = SentenceTransformer(
            settings.embedding_model,
            trust_remote_code=True   # required for some HF models
        )
        # run a warmup pass so the first real call isn't slow
        self.model.encode(["warmup"], show_progress_bar=False)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedder ready. Dimension: {self.dimension}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of strings. Returns normalized float vectors.
        Handles empty input gracefully.
        """
        if not texts:
            return []

        vectors = self.model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,   # unit vectors → cosine sim = dot product
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return vectors.tolist()

    def embed_one(self, text: str) -> list[float]:
        """Convenience wrapper for single-string embedding."""
        return self.embed([text])[0]


def get_embedder() -> Embedder:
    """
    Returns the global Embedder instance, creating it on first call.
    All agents call this — the model loads exactly once per process.
    """
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = Embedder()
    return _embedder_instance