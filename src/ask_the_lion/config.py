from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-sonnet-5-5"
    llm_timeout_seconds: float = 30.0

    embedding_model: str = "intfloat/multilingual-e5-base"
    pdf_url: str = ""
    pdf_path: Path = Path("data/perguntas_respostas.pdf")
    chroma_dir: Path = Path("chroma_db")
    collection_name: str = "irpf_qa"

    top_k: int = 8
    min_relevance: float = 0.84
    chunk_size: int = 2500
    chunk_overlap: int = 250

    max_question_chars: int = 1000
    max_body_bytes: int = 20_000
    rate_limit: str = "10/minute"
    cors_origins: list[str] = []
