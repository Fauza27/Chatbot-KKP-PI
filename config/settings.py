from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # ── OpenAI ──────────────────────────────────────────────
    open_api_key: str
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-large"

    # ── Supabase ────────────────────────────────────────────
    supabase_url: str
    supabase_service_key: str

    # ── Database Tables ─────────────────────────────────────
    table_parent_chunks: str = "parent_documents"
    table_child_chunks: str = "child_documents"

    # ── Retrieval ───────────────────────────────────────────
    retrieval_top_k: int = Field(default=30, description="Jumlah kandidat dari hybrid search (sudah optimal)")
    rerank_top_n: int = Field(default=8, description="Jumlah dokumen setelah reranking (dinaikkan dari 6 ke 8 untuk lebih banyak konteks)")
    bm25_weight: float = Field(default=0.4, ge=0.0, le=1.0, description="Bobot BM25 dinaikkan untuk keyword matching yang lebih kuat")
    dense_weight: float = Field(default=0.6, ge=0.0, le=1.0, description="Bobot dense search (diturunkan untuk balance dengan BM25)")

    # ── Evaluasi ────────────────────────────────────────────
    ragas_sample_size: int = Field(default=50, ge=10, le=500)

    # ── Cross-Encoder ───────────────────────────────────────
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── Hugging Face ───────────────────────────────────────
    hf_token: str | None = None

    # ── Logging ─────────────────────────────────────────────
    log_level: str = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings instance (loaded once)."""
    return Settings()