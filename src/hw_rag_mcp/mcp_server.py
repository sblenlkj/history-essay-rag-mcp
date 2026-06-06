from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastmcp import FastMCP
from langchain_gigachat import GigaChatEmbeddings

from hw_rag_mcp.search_service.service import HistoryEssaySearchService
from hw_rag_mcp.settings import get_settings
from hw_rag_mcp.vectorstore.qdrant_indexer import QdrantIndexer


SEMANTIC_SEARCH_K = 5
AUTHOR_SEARCH_K = 5
INTERPRETATION_SEARCH_K = 5
INTERPRETATION_SEARCH_DENSE_K = 20


mcp = FastMCP("history-essay-rag")


@lru_cache(maxsize=1)
def get_search_service() -> HistoryEssaySearchService:
    """
    Lazy singleton for MCP tools.

    The MCP server runs as a separate process, so it initializes
    its own settings, embedding model and Qdrant client.

    Important:
    The embedding model used here must be exactly the same as the model
    used during N1 ingest. Otherwise Qdrant will fail with vector size mismatch.
    """
    settings = get_settings()

    embedding_model = GigaChatEmbeddings(
        credentials=settings.require_gigachat_credentials(),
        scope=settings.gigachat_scope,
        model=settings.gigachat_embeddings_model,
        verify_ssl_certs=settings.gigachat_verify_ssl_certs,
    )

    indexer = QdrantIndexer(
        path=settings.qdrant_path,
        collection_name=settings.qdrant_collection,
        embedding_model=embedding_model,
    )

    return HistoryEssaySearchService(indexer=indexer)


def _to_jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")

    return value


def _extract_topic_content(result: dict[str, Any]) -> str:
    """
    Extract only the topic text from display_text.

    EnrichedChunk.display_text usually contains metadata plus a section:

    Текст темы:
    «...»

    For the agent we keep only the actual topic content to reduce noise,
    but still return document_id, chunk_id and source separately.
    """
    display_text = str(result.get("display_text") or "").strip()

    if "Текст темы:" in display_text:
        content = display_text.split("Текст темы:", maxsplit=1)[1].strip()
    else:
        content = display_text

    if "Стабильный идентификатор фрагмента:" in content:
        content = content.split(
            "Стабильный идентификатор фрагмента:",
            maxsplit=1,
        )[0].strip()

    return content.strip()


def _compact_search_response(response: Any) -> dict:
    """
    Convert internal search response to a compact MCP response.

    The returned payload satisfies the homework requirement:
    each result contains document_id, chunk_id, source, author and content.

    We intentionally do not return all internal metadata to reduce noise
    for the agent.
    """
    data = _to_jsonable(response)

    compact_results: list[dict[str, str | None]] = []

    for item in data.get("results", []):
        # Semantic search returns QdrantSearchResult directly.
        # Boosted search returns BoostedSearchItem with nested `result`.
        raw_result = item.get("result", item)

        compact_results.append(
            {
                "document_id": raw_result.get("document_id"),
                "chunk_id": raw_result.get("chunk_id"),
                "source": raw_result.get("source"),
                "author": raw_result.get("quote_author"),
                "content": _extract_topic_content(raw_result),
            }
        )

    return {
        "query": data.get("query"),
        "results": compact_results,
    }


@mcp.tool
def semantic_search(query: str) -> dict:
    """
    Semantic search over history essay topics.

    Use this tool when the user asks for themes by historical meaning:
    period, event, person, process, reform, war, region, institution,
    policy, social phenomenon or broad historical topic.

    Parameters:
    - query:
      A rich semantic search query, not a short keyword.
      Rewrite the user's request into a complete historical search idea.
      Include the relevant period, historical actor, process, event, policy,
      institution, region and conceptual context if they are present or implied.

      Good examples:
      - "Киевская Русь, объединение восточных славян, формирование древнерусского государства, политическая роль Киева в IX-X веках."
      - "Экономическая и денежная политика Древнерусского государства, торговля, городское развитие и монетарные отношения Киевской Руси."
      - "Реформы Петра I, модернизация российского государства, армии, флота, управления и общества в начале XVIII века."
      - "Советская индустриализация, форсированное развитие тяжелой промышленности, коллективизация и социальные последствия политики 1930-х годов."

      Bad examples:
      - "Киевская Русь"
      - "Пётр"
      - "Сталин"
      - "монеты"
      - "реформы"

    Returns compact top-k results with:
    - document_id
    - chunk_id
    - source
    - content
    - author
    """
    service = get_search_service()
    response = service.semantic_search(
        query=query,
        k=SEMANTIC_SEARCH_K,
    )

    return _compact_search_response(response)


@mcp.tool
def search_by_quote_author(author: str) -> dict:
    """
    Search essay topics by quote author.

    Use this tool only when the user asks for topics based on a specific
    quote author.

    Parameters:
    - author:
      Quote author's surname only, without initials.
      Good examples:
      - "Сталин"
      - "Карамзин"
      - "Вернадский"
      - "Ключевский"
      - "Мавродин"

      Bad examples:
      - "И.В. Сталин"
      - "цитата Сталина про индустриализацию"
      - "темы по Сталину"
      - "Сталинская политика"

    This tool does not perform semantic search. It only searches the structured
    quote_author metadata field.

    Returns compact top-k results with:
    - document_id
    - chunk_id
    - source
    - content
    - author
    """
    service = get_search_service()
    response = service.search_by_quote_author(
        author=author,
        k=AUTHOR_SEARCH_K,
    )

    return _compact_search_response(response)


@mcp.tool
def search_by_query_and_interpretation_type(
    query: str,
    interpretation_type: str,
) -> dict:
    """
    Semantic search with interpretation_type metadata boost.

    Use this tool when the user asks for a historical topic together with
    a type of author interpretation or assessment.

    This tool does two things:
    1. finds semantic candidates by query;
    2. boosts candidates that match the requested interpretation_type.

    Parameters:
    - query:
      A rich semantic search query, not a short keyword.
      Rewrite the user's request into a complete historical search idea.
      Include historical object, period, person, event, process, policy,
      institution, region and conceptual context.

      Good examples:
      - User asks: "Найди отрицательную оценку Александра Невского."
        query: "Александр Невский, политика князя в условиях монгольского нашествия, отношения с Ордой, оценка его роли в истории Руси XIII века."
        interpretation_type: "negative_assessment"

      - User asks: "Есть сравнительные темы про Русь и Западную Европу?"
        query: "Киевская Русь и Западная Европа, сравнительное развитие городов, торговли, экономики, политических и социальных институтов в Средние века."
        interpretation_type: "comparative"

      - User asks: "Найди темы про причины реформ Петра I."
        query: "Реформы Петра I, причины и предпосылки модернизации российского государства, армии, флота, управления и общества в начале XVIII века."
        interpretation_type: "causal_explanation"

      Bad query examples:
      - "Александр Невский"
      - "Киевская Русь"
      - "Пётр"
      - "отрицательная оценка"
      - "сравнение"

    - interpretation_type:
      Required interpretation type.

      Allowed values:
      - "positive_assessment"
        Use when the user asks for a positive, approving or favorable assessment.

      - "negative_assessment"
        Use when the user asks for a negative, critical or unfavorable assessment.

      - "mixed_assessment"
        Use when the user asks for an ambiguous, contradictory, mixed or two-sided assessment.

      - "comparative"
        Use when the user asks for comparison between countries, periods, people,
        models of development or historical processes.

      - "causal_explanation"
        Use when the user asks for causes, prerequisites, consequences or explanation.

      - "revisionist"
        Use when the user asks for a revisionist, controversial, non-standard
        or challenging interpretation.

      - "unknown"
        Use only when the user explicitly asks for topics with unknown interpretation type.

    Returns compact top-k results with:
    - document_id
    - chunk_id
    - source
    - content
    - author
    """
    service = get_search_service()
    response = service.search_with_metadata_boosts(
        query=query,
        k=INTERPRETATION_SEARCH_K,
        dense_k=INTERPRETATION_SEARCH_DENSE_K,
        quote_author=None,
        interpretation_type=interpretation_type,
    )

    return _compact_search_response(response)


if __name__ == "__main__":
    mcp.run()