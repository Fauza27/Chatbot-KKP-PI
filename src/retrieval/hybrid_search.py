from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from loguru import logger
from rank_bm25 import BM25Okapi
from supabase import Client, create_client

from config.settings import get_settings

settings = get_settings()


@dataclass
class HybridSearchResult:
    """One document result of hybrid search with combined score."""
    document: Document
    dense_score: float
    bm25_score: float
    hybrid_score: float
    child_id: str
    parent_id: str


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer for BM25 (Indonesian text)."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)

    stopwords_id = {
        "dan", "atau", "yang", "di", "ke", "dari", "untuk", "pada",
        "dengan", "adalah", "ini", "itu", "dalam", "tidak", "akan",
        "bisa", "ada", "juga", "sudah", "saya", "apa", "bagaimana",
        "tersebut", "oleh", "sebagai", "telah", "dapat", "secara",
        "serta", "bahwa", "maupun", "antara", "setiap", "sesuai",
    }

    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords_id and len(t) > 1]
    return tokens


def _reciprocal_rank_fusion(
    dense_results: list[tuple[str, float]],
    bm25_results: list[tuple[str, float]],
    k: int = 60,
    dense_weight: float | None = None,
    bm25_weight: float | None = None,
) -> dict[str, float]:
    """Reciprocal Rank Fusion (RRF)."""
    w_dense = dense_weight or settings.dense_weight
    w_bm25 = bm25_weight or settings.bm25_weight

    scores: dict[str, float] = {}

    for rank, (doc_id, _) in enumerate(dense_results, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + w_dense / (k + rank)

    for rank, (doc_id, _) in enumerate(bm25_results, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + w_bm25 / (k + rank)

    return scores


class HybridSearcher:
    """Orchestrator for hybrid BM25 + Dense search."""

    def __init__(self, supabase_client: Client | None = None):
        self._supabase = supabase_client or create_client(
            settings.supabase_url, settings.supabase_service_key
        )
        # ✅ BUG 1 FIX: typo "dimmensions" → "dimensions"
        self._embedder = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.open_api_key,
            dimensions=2000,
        )
        self._bm25_corpus: list[str] = []
        self._bm25_doc_ids: list[str] = []
        self._bm25_index: BM25Okapi | None = None

    def _build_bm25_index(self, documents: list[dict]) -> bool:
        """Build BM25 index from documents list."""
        if not documents:
            logger.warning("BM25 index tidak dibangun: documents kosong.")
            self._bm25_corpus = []
            self._bm25_doc_ids = []
            self._bm25_index = None
            return False

        self._bm25_corpus = [doc["content"] for doc in documents]
        self._bm25_doc_ids = [doc["id"] for doc in documents]

        tokenized_corpus = [_tokenize(text) for text in self._bm25_corpus]
        self._bm25_index = BM25Okapi(tokenized_corpus)
        return True

    def search(
        self,
        query: str,
        filters: dict[str, str] | None = None,
        top_k: int | None = None,
    ) -> list[HybridSearchResult]:
        """Run hybrid search: dense → BM25 re-scoring → RRF."""
        k = top_k or settings.retrieval_top_k
        filters = filters or {}

        logger.info(f"Hybrid search: '{query}' | filters: {filters} | top_k: {k}")

        query_embedding = self._embedder.embed_query(query)

        rpc_params: dict[str, Any] = {
            "query_embedding": query_embedding,
            "query_text": query,
            "match_count": k * 2,
            "fts_weight": settings.bm25_weight,
            "vector_weight": settings.dense_weight,
            "rrf_k": 60,
            "filter_section": filters.get("section"),
        }

        response = self._supabase.rpc("hybrid_search", rpc_params).execute()

        if not response.data:
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

            db_results = []
            for row in fallback_response.data:
                row["rrf_score"] = row.get("similarity", 0.0)
                db_results.append(row)
        else:
            db_results = response.data

        bm25_built = self._build_bm25_index(db_results)
        query_tokens = _tokenize(query)

        # ✅ BUG 2 & 3 FIX: logika if/else yang benar tanpa duplikat assignment
        if bm25_built and self._bm25_index is not None:
            bm25_scores_raw = self._bm25_index.get_scores(query_tokens)
            # ✅ BUG 3 FIX: hitung np.max() sekali saja
            max_bm25_val = float(np.max(bm25_scores_raw))
            max_bm25 = max_bm25_val if max_bm25_val > 0 else 1.0
            bm25_scores_normalized = bm25_scores_raw / max_bm25

            bm25_ranked = sorted(
                [
                    (self._bm25_doc_ids[i], float(score))
                    for i, score in enumerate(bm25_scores_normalized)
                ],
                key=lambda x: x[1],
                reverse=True,
            )
            bm25_lookup = {
                self._bm25_doc_ids[i]: float(score)
                for i, score in enumerate(bm25_scores_normalized)
            }
        else:
            logger.warning("BM25 index tidak tersedia, menggunakan dense score saja.")
            bm25_ranked = []
            bm25_lookup = {}
        # ✅ BUG 2 FIX: baris duplikat "bm25_lookup = {...}" dihapus dari sini

        dense_ranked = [
            (row["id"], row.get("rrf_score", 0.0)) for row in db_results
        ]

        final_scores = _reciprocal_rank_fusion(dense_ranked, bm25_ranked)
        db_lookup = {row["id"]: row for row in db_results}

        results: list[HybridSearchResult] = []

        for doc_id, hybrid_score in sorted(
            final_scores.items(), key=lambda x: x[1], reverse=True
        )[:k]:
            if doc_id not in db_lookup:
                continue

            row = db_lookup[doc_id]

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
                    dense_score=row.get("rrf_score", 0.0),
                    bm25_score=bm25_lookup.get(doc_id, 0.0),
                    hybrid_score=hybrid_score,
                    child_id=row["id"],
                    parent_id=row.get("parent_id", ""),
                )
            )

        logger.info(f"Hybrid search selesai: {len(results)} results")
        if results:
            logger.info(
                f"  Top: {results[0].child_id} | "
                f"hybrid={results[0].hybrid_score:.4f} | "
                f"dense={results[0].dense_score:.4f} | "
                f"bm25={results[0].bm25_score:.4f}"
            )

        return results