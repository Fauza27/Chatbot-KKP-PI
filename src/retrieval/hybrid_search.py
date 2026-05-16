from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from loguru import logger
from supabase import Client, create_client

from config.settings import get_settings
from src.retrieval.query_expansion import expand_query_smart

settings = get_settings()


@dataclass
class HybridSearchResult:
    """
    Hasil pencarian hybrid (BM25 FTS + vector) yang sudah digabung lewat RRF
    di sisi PostgreSQL. Lihat fungsi `hybrid_search` di scripts/supabase.sql.
    """
    document: Document
    hybrid_score: float
    child_id: str
    parent_id: str


class HybridSearcher:
    """
    Hybrid retriever yang melakukan fusion BM25 + vector di PostgreSQL.

    Tokenisasi BM25 menggunakan `to_tsvector('indonesian')` di Postgres
    (memiliki Snowball stemmer untuk bahasa Indonesia). Bobot fusion dan
    parameter RRF dikontrol via settings.
    """

    def __init__(self, supabase_client: Client | None = None):
        self._supabase = supabase_client or create_client(
            settings.supabase_url, settings.supabase_service_key
        )
        self._embedder = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.open_api_key,
            dimensions=2000,
        )

    def search(
        self,
        query: str,
        filters: dict[str, str] | None = None,
        top_k: int | None = None,
        enable_query_expansion: bool = True,
    ) -> list[HybridSearchResult]:
        k = top_k or settings.retrieval_top_k
        filters = filters or {}

        # Apply query expansion for better recall.
        original_query = query
        if enable_query_expansion:
            query = expand_query_smart(query, enable_expansion=True)
            if query != original_query:
                logger.info(
                    f"Query expansion applied: '{original_query}' → '{query[:100]}...'"
                )

        logger.info(f"Hybrid search: '{original_query}' | filters: {filters} | top_k: {k}")

        query_embedding = self._embedder.embed_query(query)

        rpc_params: dict[str, Any] = {
            "query_embedding": query_embedding,
            "query_text": query,
            "match_count": k,
            "fts_weight": settings.bm25_weight,
            "vector_weight": settings.dense_weight,
            "rrf_k": 60,
            "filter_section": filters.get("section"),
        }

        response = self._supabase.rpc("hybrid_search", rpc_params).execute()
        db_results = response.data or []

        # Fallback: kalau hybrid search tidak menemukan apa pun (mis. query
        # tidak match FTS sama sekali dan vector juga lemah), coba dense-only.
        if not db_results:
            logger.warning("Tidak ada hasil dari hybrid_search RPC.")
            logger.info("Fallback ke dense-only via match_child_documents...")
            fallback_response = self._supabase.rpc(
                "match_child_documents",
                {
                    "query_embedding": query_embedding,
                    "match_threshold": 0.0,
                    "match_count": k,
                    "filter_section": filters.get("section"),
                },
            ).execute()

            if not fallback_response.data:
                logger.warning("Dense search juga kosong, tidak ada hasil.")
                return []

            # Normalisasi field: dense-only return `similarity`, kita pakai
            # itu sebagai hybrid_score agar struktur output konsisten.
            db_results = []
            for row in fallback_response.data:
                row = dict(row)
                row["rrf_score"] = row.get("similarity", 0.0)
                db_results.append(row)

        results: list[HybridSearchResult] = []
        for row in db_results:
            doc = Document(
                page_content=row["content"],
                metadata={
                    "child_id": row["id"],
                    "parent_id": row.get("parent_id", ""),
                    "title": row.get("title", ""),
                    "section": row.get("section", ""),
                    "pages": row.get("pages", []),
                    "source": row.get("source", ""),
                },
            )

            results.append(
                HybridSearchResult(
                    document=doc,
                    hybrid_score=float(row.get("rrf_score", 0.0)),
                    child_id=row["id"],
                    parent_id=row.get("parent_id", ""),
                )
            )

        logger.info(f"Hybrid search selesai: {len(results)} results")
        if results:
            logger.info(
                f"  Top: {results[0].child_id} | hybrid={results[0].hybrid_score:.4f}"
            )

        return results
