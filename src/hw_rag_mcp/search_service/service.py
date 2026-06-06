from __future__ import annotations

from hw_rag_mcp.chunks.schemas import validate_interpretation_type
from hw_rag_mcp.search_service.models import BoostedSearchItem, BoostedSearchResponse, SearchResponse
from hw_rag_mcp.vectorstore.models import QdrantSearchResult
from hw_rag_mcp.vectorstore.qdrant_indexer import QdrantIndexer


class HistoryEssaySearchService:
    """
    Use case layer for history essay topic retrieval.

    This service intentionally stays above QdrantIndexer:
    - QdrantIndexer knows how to talk to Qdrant.
    - HistoryEssaySearchService knows which retrieval scenarios the app supports.
    """

    def __init__(
        self,
        indexer: QdrantIndexer,
        author_bonus: float = 0.20,
        interpretation_type_bonus: float = 0.15,
    ) -> None:
        self.indexer = indexer
        self.author_bonus = author_bonus
        self.interpretation_type_bonus = interpretation_type_bonus

    def semantic_search(
        self,
        query: str,
        k: int = 5,
    ) -> SearchResponse:
        results = self.indexer.search(query=query, k=k)

        return SearchResponse(
            query=query,
            k=k,
            results=results,
        )

    def search_by_quote_author(
        self,
        author: str,
        k: int = 5,
    ) -> SearchResponse:
        results = self.indexer.search_by_quote_author(
            author=author,
            k=k,
        )

        return SearchResponse(
            query=author,
            k=k,
            results=results,
        )

    def search_by_interpretation_type(
        self,
        interpretation_type: str,
        k: int = 5,
    ) -> SearchResponse:
        validate_interpretation_type(interpretation_type)

        results = self.indexer.search_by_interpretation_type(
            interpretation_type=interpretation_type,
            k=k,
        )

        return SearchResponse(
            query=interpretation_type,
            k=k,
            results=results,
        )

    def search_with_metadata_boosts(
        self,
        query: str,
        k: int = 5,
        dense_k: int = 20,
        quote_author: str | None = None,
        interpretation_type: str | None = None,
    ) -> BoostedSearchResponse:
        """
        Dense search + metadata-aware reranking.

        Dense search creates semantic candidates.
        Metadata searches create bonus sets.
        Final score = dense score + metadata bonuses.
        """
        if interpretation_type is not None:
            validate_interpretation_type(interpretation_type)

        dense_results = self.indexer.search(
            query=query,
            k=dense_k,
        )

        bonus_result_lists: dict[str, list[QdrantSearchResult]] = {}
        bonus_weights: dict[str, float] = {}

        if quote_author:
            bonus_result_lists["quote_author"] = self.indexer.search_by_quote_author(
                author=quote_author,
                k=-1,
            )
            bonus_weights["quote_author"] = self.author_bonus

        if interpretation_type:
            bonus_result_lists["interpretation_type"] = self.indexer.search_by_interpretation_type(
                interpretation_type=interpretation_type,
                k=-1,
            )
            bonus_weights["interpretation_type"] = self.interpretation_type_bonus

        boosted_items = self._apply_metadata_boosts(
            dense_results=dense_results,
            bonus_result_lists=bonus_result_lists,
            bonus_weights=bonus_weights,
            top_k=k,
        )

        return BoostedSearchResponse(
            query=query,
            k=k,
            results=boosted_items,
        )

    def _apply_metadata_boosts(
        self,
        dense_results: list[QdrantSearchResult],
        bonus_result_lists: dict[str, list[QdrantSearchResult]],
        bonus_weights: dict[str, float],
        top_k: int,
    ) -> list[BoostedSearchItem]:
        bonus_chunk_ids_by_name = {
            name: {
                result.chunk_id
                for result in results
                if result.chunk_id is not None
            }
            for name, results in bonus_result_lists.items()
        }

        boosted_items: list[BoostedSearchItem] = []

        for result in dense_results:
            final_score = float(result.score)
            applied_bonuses: list[dict] = []

            for bonus_name, bonus_chunk_ids in bonus_chunk_ids_by_name.items():
                if result.chunk_id in bonus_chunk_ids:
                    bonus_value = bonus_weights.get(bonus_name, 0.0)
                    final_score += bonus_value

                    applied_bonuses.append(
                        {
                            "name": bonus_name,
                            "bonus": bonus_value,
                        }
                    )

            boosted_items.append(
                BoostedSearchItem(
                    final_rank=0,
                    final_score=final_score,
                    dense_rank=result.rank,
                    dense_score=float(result.score),
                    applied_bonuses=applied_bonuses,
                    result=result,
                )
            )

        boosted_items.sort(
            key=lambda item: item.final_score,
            reverse=True,
        )

        limited_items = boosted_items if top_k == -1 else boosted_items[:top_k]

        return [
            item.model_copy(update={"final_rank": final_rank})
            for final_rank, item in enumerate(limited_items, start=1)
        ]