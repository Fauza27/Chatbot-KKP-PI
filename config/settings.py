from functools import lru_cache
from pathlib import Path
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import ValidationInfo


def _find_env_file() -> str:
    """Cari file .env dari direktori project root."""
    # Cari dari lokasi file ini ke atas sampai ketemu .env
    current = Path(__file__).resolve().parent
    for _ in range(5):  # max 5 level ke atas
        env_path = current / ".env"
        if env_path.exists():
            return str(env_path)
        current = current.parent
    return ".env"  # fallback


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_find_env_file(),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    APP_NAME: str = "Chatbot KKP/PI Assistant"
    VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"

    # OpenAI
    open_api_key: str
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-large"

    # Supabase
    supabase_url: str
    supabase_service_key: str

    # Database Tables
    table_parent_chunks: str = "parent_documents"
    table_child_chunks: str = "child_documents"

    # Retrieval
    retrieval_top_k: int = Field(default=30)
    rerank_top_n: int = Field(default=8)
    bm25_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    dense_weight: float = Field(default=0.6, ge=0.0, le=1.0)

    # Evaluasi
    ragas_sample_size: int = Field(default=50, ge=10, le=500)

    # Cross-Encoder
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Hugging Face
    hf_token: str | None = None

    # Logging
    log_level: str = "INFO"

    # Telegram
    TELEGRAM_BOT_TOKEN: str
    TELEGRAM_WEBHOOK_URL: str = ""
    TELEGRAM_WEBHOOK_SECRET: str = ""
    TELEGRAM_WEBHOOK_PATH: str = "/api/telegram/webhook"


@field_validator("TELEGRAM_WEBHOOK_SECRET", mode='after')
@classmethod
def validate_webhook_secret(cls, value: str, info: ValidationInfo) -> str:
    env = info.data.get("ENVIRONMENT", "development")
    if env == "production" and info.data.get("TELEGRAM_WEBHOOK_URL"):
        if not value or len(value) < 16:
            raise ValueError("TELEGRAM_WEBHOOK_SECRET is required and must be at least 16 chars in production")
    return value

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings instance (loaded once)."""
    return Settings()