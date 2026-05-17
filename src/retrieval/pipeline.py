"""
Single source of truth untuk pipeline retrieval RAG:
self-query → hybrid search → parent fetching → reranking.

Fungsi `run_retrieval` dipakai oleh:
- `src/services/ai_services.py::_handle_retrieval`
- `src/generation/chain.py::_fallback_to_retrieval`

Keduanya butuh logic yang sama, jadi diekstrak ke sini agar tidak duplikat.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from config.settings import get_settings


@dataclass
class RetrievalResult:
    """Hasil pipeline retrieval, siap dikonsumsi generator."""
    parent_documents: list[dict]
    is_empty: bool

    @property
    def num_docs(self) -> int:
        return len(self.parent_documents)


def run_retrieval(query: str, rerank_query: str | None = None) -> RetrievalResult:
    """
    Jalankan pipeline retrieval lengkap untuk satu query.

    `query` adalah teks yang dipakai untuk semantic search (sudah
    direformulasi kalau perlu).
    `rerank_query` adalah teks yang dipakai cross-encoder; default ke
    `query` kalau tidak diberikan. Biasanya pakai pertanyaan asli user
    di sini agar reranking konsisten dengan intent original.
    """
    # Lazy import untuk menghindari circular import dengan modul yang
    # sebelumnya juga mengimport pipeline ini.
    from src.retrieval.self_query import extract_query_components
    from src.retrieval.hybrid_search import HybridSearcher
    from src.retrieval.parent_child import ParentChildFetcher
    from src.retrieval.reranker import CrossEncoderReranker

    settings = get_settings()
    rerank_query = rerank_query or query

    parsed = extract_query_components(query)

    searcher = HybridSearcher()
    search_results = searcher.search(
        query=parsed.semantic_query,
        filters=parsed.filters,
    )

    if not search_results:
        return RetrievalResult(parent_documents=[], is_empty=True)

    fetcher = ParentChildFetcher()
    parent_results = fetcher.fetch_parents(search_results)

    try:
        reranker = CrossEncoderReranker()
        reranked = reranker.rerank(query=rerank_query, documents=parent_results)
    except Exception as e:
        logger.warning(f"Reranking failed, using unranked top-N: {e}")
        reranked = parent_results[: settings.rerank_top_n]

    return RetrievalResult(parent_documents=reranked, is_empty=False)
