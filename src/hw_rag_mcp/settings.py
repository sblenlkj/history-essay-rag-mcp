from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def find_repo_root(start: Path | None = None) -> Path:
    """
    Find repository root by walking up from the current directory.

    We use `.env.example` and `pyproject.toml` as stable project markers.
    This makes notebooks work both from the repository root and from `notebooks/`.
    """
    current = (start or Path.cwd()).resolve()

    for path in [current, *current.parents]:
        if (path / ".env.example").exists() and (path / "pyproject.toml").exists():
            return path

    for path in [current, *current.parents]:
        if (path / "pyproject.toml").exists():
            return path

    return current


REPO_ROOT = find_repo_root()


class Settings(BaseSettings):
    """
    Application settings for the history essay RAG + MCP project.

    Values are loaded from:
    1. environment variables;
    2. `.env` file in repository root;
    3. defaults defined here.
    """

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    # Project paths
    repo_root: Path = Field(default=REPO_ROOT)
    raw_json_dir: Path = Field(default=Path("data/raw_json"))
    chunks_dir: Path = Field(default=Path("data/chunks"))
    chunks_json_path: Path = Field(default=Path("data/chunks/chunks.json"))
    chunks_jsonl_path: Path = Field(default=Path("data/chunks/chunks.jsonl"))
    eval_dir: Path = Field(default=Path("data/eval"))
    eval_queries_path: Path = Field(default=Path("data/eval/eval_queries.json"))
    eval_results_path: Path = Field(default=Path("data/eval/eval_results.json"))

    # Qdrant
    qdrant_path: Path = Field(
        default=Path("data/vectorstore/qdrant"),
        validation_alias="QDRANT_PATH",
    )
    qdrant_collection: str = Field(
        default="history_essay_chunks",
        validation_alias="QDRANT_COLLECTION",
    )

    # GigaChat
    gigachat_credentials: str | None = Field(
        default=None,
        validation_alias="GIGACHAT_CREDENTIALS",
    )
    gigachat_scope: str = Field(
        default="GIGACHAT_API_PERS",
        validation_alias="GIGACHAT_SCOPE",
    )
    gigachat_model: str = Field(
        default="GigaChat-2",
        validation_alias="GIGACHAT_MODEL",
    )
    gigachat_embeddings_model: str = Field(
        default="EmbeddingsGigaR",
        validation_alias="GIGACHAT_EMBEDDINGS_MODEL",
    )
    gigachat_verify_ssl_certs: bool = Field(
        default=False,
        validation_alias="GIGACHAT_VERIFY_SSL_CERTS",
    )

    # Langfuse
    langfuse_public_key: str | None = Field(
        default=None,
        validation_alias="LANGFUSE_PUBLIC_KEY",
    )
    langfuse_secret_key: str | None = Field(
        default=None,
        validation_alias="LANGFUSE_SECRET_KEY",
    )
    langfuse_base_url: str = Field(
        default="https://cloud.langfuse.com",
        validation_alias="LANGFUSE_BASE_URL",
    )

    @field_validator(
        "raw_json_dir",
        "chunks_dir",
        "chunks_json_path",
        "chunks_jsonl_path",
        "qdrant_path",
        mode="after",
    )
    @classmethod
    def resolve_project_relative_path(cls, value: Path) -> Path:
        if value.is_absolute():
            return value

        return REPO_ROOT / value

    @computed_field
    @property
    def env_path(self) -> Path:
        return self.repo_root / ".env"

    @computed_field
    @property
    def env_status(self) -> str:
        if self.env_path.exists():
            return str(self.env_path)

        return f"не найден ({self.repo_root / '.env.example'} найден)"

    @computed_field
    @property
    def has_gigachat_credentials(self) -> bool:
        return bool(self.gigachat_credentials)

    @computed_field
    @property
    def has_langfuse_credentials(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    def require_gigachat_credentials(self) -> str:
        if not self.gigachat_credentials:
            raise RuntimeError(
                "GIGACHAT_CREDENTIALS is not set. "
                "Add it to `.env` or export it as an environment variable."
            )

        return self.gigachat_credentials

    def require_langfuse_public_key(self) -> str:
        if not self.langfuse_public_key:
            raise RuntimeError(
                "LANGFUSE_PUBLIC_KEY is not set. "
                "Add it to `.env` or export it as an environment variable."
            )

        return self.langfuse_public_key

    def require_langfuse_secret_key(self) -> str:
        if not self.langfuse_secret_key:
            raise RuntimeError(
                "LANGFUSE_SECRET_KEY is not set. "
                "Add it to `.env` or export it as an environment variable."
            )

        return self.langfuse_secret_key

    def summary(self) -> dict[str, Any]:
        """
        Safe configuration summary for notebooks.

        Does not expose secret values.
        """
        return {
            "env_file": self.env_status,
            "repo_root": str(self.repo_root),
            "raw_json_dir": str(self.raw_json_dir),
            "chunks_dir": str(self.chunks_dir),
            "chunks_json_path": str(self.chunks_json_path),
            "chunks_jsonl_path": str(self.chunks_jsonl_path),
            "qdrant_path": str(self.qdrant_path),
            "qdrant_collection": self.qdrant_collection,
            "gigachat_model": self.gigachat_model,
            "gigachat_embeddings_model": self.gigachat_embeddings_model,
            "gigachat_scope": self.gigachat_scope,
            "gigachat_verify_ssl_certs": self.gigachat_verify_ssl_certs,
            "has_gigachat_credentials": self.has_gigachat_credentials,
            "langfuse_base_url": self.langfuse_base_url,
            "has_langfuse_credentials": self.has_langfuse_credentials,
        }

    def print_summary(self) -> None:
        """
        Pretty-print safe settings summary in notebooks.
        """
        print("✓ Конфигурация загружена")
        print(f"  Env file:             {self.env_status}")
        print(f"  Repo root:            {self.repo_root}")
        print(f"  LLM:                  {self.gigachat_model}")
        print(f"  Embeddings:           {self.gigachat_embeddings_model}")
        print(f"  Qdrant path:          {self.qdrant_path}")
        print(f"  Qdrant collection:    {self.qdrant_collection}")
        print(f"  Langfuse base URL:    {self.langfuse_base_url}")
        print(
            "  GigaChat credentials: "
            f"{'заданы' if self.has_gigachat_credentials else 'не заданы'}"
        )
        print(
            "  Langfuse credentials: "
            f"{'заданы' if self.has_langfuse_credentials else 'не заданы'}"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Cached settings singleton.

    Use this function in notebooks, services, MCP server and tests.
    """
    return Settings()