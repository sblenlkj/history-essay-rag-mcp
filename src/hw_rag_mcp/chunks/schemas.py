from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

InterpretationType = Literal[
    "positive_assessment",
    "negative_assessment",
    "mixed_assessment",
    "comparative",
    "causal_explanation",
    "revisionist",
    "unknown",
]

INTERPRETATION_TYPES: tuple[str, ...] = (
    "positive_assessment",
    "negative_assessment",
    "mixed_assessment",
    "comparative",
    "causal_explanation",
    "revisionist",
    "unknown",
)

class InvalidInterpretationTypeError(ValueError):
    def __init__(self, interpretation_type: str) -> None:
        allowed_values = ", ".join(INTERPRETATION_TYPES)
        super().__init__(
            f"Unknown interpretation_type: {interpretation_type!r}. "
            f"Allowed values: {allowed_values}"
        )

def validate_interpretation_type(interpretation_type: str) -> None:
    if interpretation_type not in INTERPRETATION_TYPES:
        raise InvalidInterpretationTypeError(interpretation_type)


Stage = Literal[
    "school",
    "municipal",
    "regional",
    "final",
    "unknown",
]


class EssayTopic(BaseModel):
    topic_number: int
    quote_author: str
    quote_text: str
    historical_period: str
    keywords: list[str] = Field(default_factory=list)
    interpretation_type: InterpretationType
    interpretation_position: str
    notes: str

    @field_validator(
        "quote_author",
        "quote_text",
        "historical_period",
        "interpretation_position",
        "notes",
    )
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return " ".join(value.split())

    @field_validator("keywords")
    @classmethod
    def normalize_keywords(cls, value: list[str]) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()

        for keyword in value:
            normalized = " ".join(keyword.split())

            if not normalized:
                continue

            key = normalized.lower()
            if key in seen:
                continue

            result.append(normalized)
            seen.add(key)

        return result


class EssayDocument(BaseModel):
    document_id: str
    source_pdf: str
    stage: Stage
    year: int
    grades: str
    essay_topics: list[EssayTopic]

    @field_validator("grades")
    @classmethod
    def validate_grades(cls, value: str) -> str:
        if value != "10-11":
            raise ValueError('grades must be "10-11"')
        return value


class EnrichedChunk(BaseModel):
    point_id: int
    document_id: str
    chunk_id: str
    source: str

    year: int
    stage: Stage
    grades: str
    topic_number: int

    quote_author: str
    quote_text: str
    historical_period: str
    keywords: list[str] = Field(default_factory=list)
    interpretation_type: InterpretationType
    interpretation_position: str
    notes: str

    @property
    def id(self) -> int:
        return self.point_id

    @property
    def embedding_text(self) -> str:
        keywords = "; ".join(self.keywords) if self.keywords else "unknown"

        return "\n".join(
            [
                f"Исторический период: {self.historical_period}.",
                f"Автор цитаты: {self.quote_author}.",
                f"Ключевые слова: {keywords}.",
                f"Позиция автора: {self.interpretation_position}",
                "",
                "Текст темы:",
                self.quote_text,
            ]
        )

    @property
    def display_text(self) -> str:
        keywords = "; ".join(self.keywords) if self.keywords else "unknown"

        return "\n".join(
            [
                f"Тема исторического эссе №{self.topic_number}.",
                f"Документ: {self.document_id}.",
                f"Источник: {self.source}.",
                f"Год: {self.year}.",
                f"Этап олимпиады: {self.stage}.",
                f"Классы: {self.grades}.",
                "",
                f"Автор цитаты: {self.quote_author}.",
                f"Исторический период: {self.historical_period}.",
                f"Ключевые слова: {keywords}.",
                f"Тип интерпретации: {self.interpretation_type}.",
                f"Позиция автора: {self.interpretation_position}",
                f"Аннотация: {self.notes}",
                "",
                "Текст темы:",
                f"«{self.quote_text}»",
                "",
                f"Стабильный идентификатор фрагмента: {self.chunk_id}.",
            ]
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "point_id": self.point_id,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "source": self.source,
            "year": self.year,
            "stage": self.stage,
            "grades": self.grades,
            "topic_number": self.topic_number,
            "quote_author": self.quote_author,
            "historical_period": self.historical_period,
            "keywords": self.keywords,
            "interpretation_type": self.interpretation_type,
            "interpretation_position": self.interpretation_position,
            "notes": self.notes,
        }