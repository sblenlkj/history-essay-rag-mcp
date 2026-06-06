from __future__ import annotations

from typing import Protocol


class EmbeddingModelPort(Protocol):
    """Port for text embedding models."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple document texts."""
        ...