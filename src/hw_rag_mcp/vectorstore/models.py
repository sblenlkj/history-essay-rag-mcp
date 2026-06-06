from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from hw_rag_mcp.chunks.schemas import EnrichedChunk


class QdrantPayload(BaseModel):
    point_id: int
    document_id: str
    chunk_id: str
    source: str
    year: int
    stage: str
    grades: str
    topic_number: int
    quote_author: str
    historical_period: str
    keywords: list[str] = Field(default_factory=list)
    interpretation_type: str
    interpretation_position: str
    notes: str
    embedding_text: str
    display_text: str

    @classmethod
    def from_chunk(cls, chunk: EnrichedChunk) -> "QdrantPayload":
        return cls(
            **chunk.metadata,
            embedding_text=chunk.embedding_text,
            display_text=chunk.display_text,
        )


class QdrantPoint(BaseModel):
    id: int
    vector: list[float]
    payload: QdrantPayload

    @classmethod
    def from_chunk(
        cls,
        chunk: EnrichedChunk,
        vector: list[float],
    ) -> "QdrantPoint":
        return cls(
            id=chunk.point_id,
            vector=vector,
            payload=QdrantPayload.from_chunk(chunk),
        )

    def to_point_struct_kwargs(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "vector": self.vector,
            "payload": self.payload.model_dump(mode="json"),
        }


class QdrantSearchResult(BaseModel):
    rank: int
    score: float
    point_id: int | str
    document_id: str | None = None
    chunk_id: str | None = None
    source: str | None = None
    year: int | None = None
    stage: str | None = None
    topic_number: int | None = None
    quote_author: str | None = None
    historical_period: str | None = None
    keywords: list[str] = Field(default_factory=list)
    interpretation_type: str | None = None
    display_text: str | None = None

    @classmethod
    def from_scored_point(
        cls,
        rank: int,
        point: Any,
    ) -> "QdrantSearchResult":
        payload = point.payload or {}

        return cls(
            rank=rank,
            score=float(point.score),
            point_id=point.id,
            document_id=payload.get("document_id"),
            chunk_id=payload.get("chunk_id"),
            source=payload.get("source"),
            year=payload.get("year"),
            stage=payload.get("stage"),
            topic_number=payload.get("topic_number"),
            quote_author=payload.get("quote_author"),
            historical_period=payload.get("historical_period"),
            keywords=payload.get("keywords", []),
            interpretation_type=payload.get("interpretation_type"),
            display_text=payload.get("display_text"),
        )
    

    @classmethod
    def from_record(
        cls,
        rank: int,
        record: Any,
        score: float = 1.0,
    ) -> "QdrantSearchResult":
        payload = record.payload or {}

        return cls(
            rank=rank,
            score=score,
            point_id=record.id,
            document_id=payload.get("document_id"),
            chunk_id=payload.get("chunk_id"),
            source=payload.get("source"),
            year=payload.get("year"),
            stage=payload.get("stage"),
            topic_number=payload.get("topic_number"),
            quote_author=payload.get("quote_author"),
            historical_period=payload.get("historical_period"),
            keywords=payload.get("keywords", []),
            interpretation_type=payload.get("interpretation_type"),
            display_text=payload.get("display_text"),
        )