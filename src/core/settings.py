"""Central application settings using environment variables."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, SecretStr, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.core.enums import Environment, LogLevel


class Settings(BaseSettings):
    """Project-wide configuration and environment settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = Field(default="Arabic Intelligent Knowledge Assistant")
    app_env: Environment = Field(default=Environment.DEVELOPMENT)
    app_debug: bool = Field(default=True)
    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=8000)

    secret_key: SecretStr = Field(default=SecretStr("change-me-in-production"))

    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_db: str = Field(default="arabic_rag")
    postgres_user: str = Field(default="postgres")
    postgres_password: SecretStr = Field(default=SecretStr("postgres"))
    postgres_echo: bool = Field(default=False)

    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)
    qdrant_api_key: str | None = Field(default=None)
    qdrant_collection_name: str = Field(default="arabic_documents_e5_small")
    qdrant_timeout: int = Field(default=30)

    max_pdf_size_bytes: int = Field(default=50 * 1024 * 1024, gt=0)
    max_pdf_pages: int = Field(default=500, gt=0)

    openai_api_key: str | None = Field(default=None)
    openai_model: str = Field(default="gpt-4o-mini")

    streamlit_server_port: int = Field(default=8501)
    streamlit_server_address: str = Field(default="0.0.0.0")

    log_level: LogLevel = Field(default=LogLevel.INFO)

    @property
    def postgres_dsn(self) -> PostgresDsn:
        """Constructs database DSN using the configured values."""
        return PostgresDsn.build(
            scheme="postgresql+asyncpg",
            username=self.postgres_user,
            password=self.postgres_password.get_secret_value(),
            host=self.postgres_host,
            port=self.postgres_port,
            path=self.postgres_db,
        )

    @property
    def postgres_sync_dsn(self) -> str:
        """Constructs a sync connection DSN for SQLAlchemy-compatible tools."""
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password.get_secret_value()}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def qdrant_url(self) -> str:
        """Constructs the vector service URL."""
        return f"http://{self.qdrant_host}:{self.qdrant_port}"

    @field_validator("app_env", mode="before")
    @classmethod
    def validate_env(cls, value: str | None) -> str:
        if value is None:
            return Environment.DEVELOPMENT.value
        return value.lower()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Returns a cached application settings instance."""
    return Settings()
