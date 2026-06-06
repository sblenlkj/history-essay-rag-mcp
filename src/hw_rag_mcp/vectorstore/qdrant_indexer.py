from __future__ import annotations

from pathlib import Path
from typing import Any

import re
from collections.abc import Iterable

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from hw_rag_mcp.chunks.schemas import EnrichedChunk, validate_interpretation_type
from hw_rag_mcp.vectorstore.embedding_model_port import EmbeddingModelPort
from hw_rag_mcp.vectorstore.models import QdrantPoint, QdrantSearchResult


class QdrantIndexer:
    def __init__(
        self,
        path: Path | str,
        collection_name: str,
        embedding_model: EmbeddingModelPort,
        distance: Distance = Distance.COSINE,
    ) -> None:
        self.path = Path(path)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.distance = distance
        self.client = QdrantClient(path=str(self.path))

    def collection_exists(self) -> bool:
        return self.client.collection_exists(self.collection_name)

    def recreate_collection(self, vector_size: int) -> None:
        if self.collection_exists():
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=self.distance,
            ),
        )

    def index_chunks(
        self,
        chunks: list[EnrichedChunk],
        batch_size: int = 32,
        recreate: bool = True,
    ) -> None:
        if not chunks:
            raise ValueError("No chunks to index")

        first_vector = self.embedding_model.embed_query(chunks[0].embedding_text)

        if recreate:
            self.recreate_collection(vector_size=len(first_vector))
        elif not self.collection_exists():
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=len(first_vector),
                    distance=self.distance,
                ),
            )

        first_point = self._make_point(
            chunk=chunks[0],
            vector=first_vector,
        )

        self.client.upsert(
            collection_name=self.collection_name,
            points=[first_point],
        )

        for start in range(1, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            texts = [chunk.embedding_text for chunk in batch]
            vectors = self.embedding_model.embed_documents(texts)

            points = [
                self._make_point(chunk=chunk, vector=vector)
                for chunk, vector in zip(batch, vectors, strict=True)
            ]

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

    def search(
        self,
        query: str,
        k: int = 5,
    ) -> list[QdrantSearchResult]:
        if not self.collection_exists():
            raise ValueError(
                f"Collection {self.collection_name!r} not found. "
                "Run N1 ingest notebook first or check QDRANT_PATH."
            )

        query_vector = self.embedding_model.embed_query(query)

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=k,
            with_payload=True,
        )

        return [
            QdrantSearchResult.from_scored_point(rank=rank, point=point)
            for rank, point in enumerate(response.points, start=1)
        ]

    def _make_point(
        self,
        chunk: EnrichedChunk,
        vector: list[float],
    ) -> PointStruct:
        qdrant_point = QdrantPoint.from_chunk(
            chunk=chunk,
            vector=vector,
        )

        return PointStruct(**qdrant_point.to_point_struct_kwargs())

    def count(self) -> int:
        if not self.collection_exists():
            return 0

        info = self.client.get_collection(self.collection_name)
        return int(info.points_count or 0)

    def get_collection_info(self) -> Any:
        if not self.collection_exists():
            raise ValueError(f"Collection {self.collection_name!r} not found")

        return self.client.get_collection(self.collection_name)

    def close(self) -> None:
        self.client.close()


    def search_by_quote_author(
        self,
        author: str,
        k: int = -1,
    ) -> list[QdrantSearchResult]:
        """
        Search chunks by quote author.

        This is metadata search, not vector search.
        It is useful for queries like:
        "Были ли темы на основе цитаты Сталина?"
        """
        normalized_author_query = self._normalize_text(author)

        matched_records = []

        for record in self._iter_all_records():
            payload = record.payload or {}
            quote_author = str(payload.get("quote_author") or "")
            normalized_quote_author = self._normalize_text(quote_author)

            if normalized_author_query in normalized_quote_author:
                matched_records.append(record)

        if k != -1:
            matched_records = matched_records[:k]

        return [
            QdrantSearchResult.from_record(
                rank=1,
                record=record,
                score=1.0,
            )
            for record in matched_records
        ]

    def search_by_interpretation_type(
        self,
        interpretation_type: str,
        k: int = -1,
    ) -> list[QdrantSearchResult]:
        validate_interpretation_type(interpretation_type)

        matched_records = []

        for record in self._iter_all_records():
            payload = record.payload or {}
            current_type = payload.get("interpretation_type")

            if current_type == interpretation_type:
                matched_records.append(record)

        if k != -1:
            matched_records = matched_records[:k]

        return [
            QdrantSearchResult.from_record(
                rank=1,
                record=record,
                score=1.0,
            )
            for record in matched_records
        ]

    def _iter_all_records(
        self,
        batch_size: int = 128,
    ) -> Iterable[Any]:
        if not self.collection_exists():
            raise ValueError(
                f"Collection {self.collection_name!r} not found. "
                "Run N1 ingest notebook first or check QDRANT_PATH."
            )

        offset = None

        while True:
            records, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            yield from records

            if next_offset is None:
                break

            offset = next_offset

    def _normalize_text(
        self,
        text: str,
    ) -> str:
        text = text.lower().replace("ё", "е")
        text = re.sub(r"\s+", " ", text)
        return text.strip()