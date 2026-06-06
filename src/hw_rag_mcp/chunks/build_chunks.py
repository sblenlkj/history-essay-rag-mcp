from __future__ import annotations

import json
from pathlib import Path

from hw_rag_mcp.chunks.schemas import EnrichedChunk, EssayDocument, EssayTopic


class ChunkBuilder:
    def build_from_dir(
        self,
        input_dir: Path | str,
        output_dir: Path | str,
        json_output_name: str = "chunks.json",
        jsonl_output_name: str = "chunks.jsonl",
    ) -> list[EnrichedChunk]:
        documents = self.load_documents_from_dir(Path(input_dir))
        chunks = self.build_from_documents(documents)

        self.validate_unique_chunk_ids(chunks)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.save_json(chunks, output_path / json_output_name)
        self.save_jsonl(chunks, output_path / jsonl_output_name)

        return chunks

    def load_documents_from_dir(self, input_dir: Path) -> list[EssayDocument]:
        paths = sorted(input_dir.glob("*.json"))

        if not paths:
            raise FileNotFoundError(f"No JSON files found in {input_dir}")

        return [self.load_document(path) for path in paths]

    def load_document(self, path: Path) -> EssayDocument:
        with path.open("r", encoding="utf-8") as file:
            raw = json.load(file)

        return EssayDocument.model_validate(raw)

    def build_from_documents(self, documents: list[EssayDocument]) -> list[EnrichedChunk]:
        chunks: list[EnrichedChunk] = []
        point_id = 1

        for document in documents:
            for topic in document.essay_topics:
                chunks.append(
                    self.build_from_topic(
                        document=document,
                        topic=topic,
                        point_id=point_id,
                    )
                )
                point_id += 1

        return chunks

    def build_from_topic(
        self,
        document: EssayDocument,
        topic: EssayTopic,
        point_id: int,
    ) -> EnrichedChunk:
        chunk_id = self.make_chunk_id(
            document_id=document.document_id,
            topic_number=topic.topic_number,
        )

        return EnrichedChunk(
            point_id=point_id,
            document_id=document.document_id,
            chunk_id=chunk_id,
            source=document.source_pdf,
            year=document.year,
            stage=document.stage,
            grades=document.grades,
            topic_number=topic.topic_number,
            quote_author=topic.quote_author,
            quote_text=topic.quote_text,
            historical_period=topic.historical_period,
            keywords=topic.keywords,
            interpretation_type=topic.interpretation_type,
            interpretation_position=topic.interpretation_position,
            notes=topic.notes,
        )

    def make_chunk_id(self, document_id: str, topic_number: int) -> str:
        return f"{document_id}_topic_{topic_number:03d}"

    def validate_unique_chunk_ids(self, chunks: list[EnrichedChunk]) -> None:
        seen: set[str] = set()
        duplicates: set[str] = set()

        for chunk in chunks:
            if chunk.chunk_id in seen:
                duplicates.add(chunk.chunk_id)

            seen.add(chunk.chunk_id)

        if duplicates:
            raise ValueError(f"Duplicated chunk_id values found: {sorted(duplicates)}")

    def save_json(self, chunks: list[EnrichedChunk], path: Path) -> None:
        data = [chunk.model_dump(mode="json") for chunk in chunks]

        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)

    def save_jsonl(self, chunks: list[EnrichedChunk], path: Path) -> None:
        with path.open("w", encoding="utf-8") as file:
            for chunk in chunks:
                file.write(chunk.model_dump_json())
                file.write("\n")