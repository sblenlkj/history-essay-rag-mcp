from __future__ import annotations

from pydantic import BaseModel, Field

from hw_rag_mcp.vectorstore.models import QdrantSearchResult


class BoostedSearchItem(BaseModel):
    final_rank: int
    final_score: float
    dense_rank: int
    dense_score: float
    applied_bonuses: list[dict] = Field(default_factory=list)
    result: QdrantSearchResult


class SearchResponse(BaseModel):
    query: str
    k: int
    results: list[QdrantSearchResult]


class BoostedSearchResponse(BaseModel):
    query: str
    k: int
    results: list[BoostedSearchItem]