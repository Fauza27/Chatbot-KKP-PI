from __future__ import annotations

from supabase import create_client, Client
from loguru import logger

from config.settings import get_settings

# Import tipe dari hybrid search
from src.retrieval.hybrid_search import HybridSearchResult


class ParentChildFetcher:
    """
    Take parent chunks from Supabase based on child chunks
    found by hybrid search.
    """

    def __init__(self, supabase_client: Client | None = None):
        settings = get_settings()
        self._supabase = supabase_client or create_client(
            settings.supabase_url, settings.supabase_service_key
        )
        self._parent_table = settings.table_parent_chunks

    def fetch_parents(self, search_results: list[HybridSearchResult]) -> list[dict]:
        """
        Take parent chunks based on HybridSearchResult.
        """
        if not search_results:
            logger.warning("No search results to fetch parent")
            return []

        parent_scores: dict[str, dict] = {}
        for result in search_results:
            pid = result.parent_id
            if not pid:
                logger.warning(f"Child '{result.child_id}' has no parent_id")
                continue

            score = result.hybrid_score

            if pid not in parent_scores:
                parent_scores[pid] = {
                    "best_score": score,
                    "matched_children": [result.child_id],
                }
            else:
                parent_scores[pid]["best_score"] = max(
                    parent_scores[pid]["best_score"], score
                )
                parent_scores[pid]["matched_children"].append(result.child_id)

        unique_parent_ids = list(parent_scores.keys())
        logger.info(
            f"De-duplikasi: {len(search_results)} children → "
            f"{len(unique_parent_ids)} unique parents"
        )

        response = (
            self._supabase.table(self._parent_table)
            .select("*")
            .in_("parent_id", unique_parent_ids)
            .execute()
        )

        parents = response.data or []

        if len(parents) != len(unique_parent_ids):
            found_ids = {p["parent_id"] for p in parents}
            missing = set(unique_parent_ids) - found_ids
            logger.warning(f"Parent IDs not found in DB: {missing}")

        for parent in parents:
            pid = parent["parent_id"]
            info = parent_scores.get(pid, {})
            parent["best_child_score"] = info.get("best_score", 0.0)
            parent["matched_children"] = info.get("matched_children", [])

        parents.sort(key=lambda x: x["best_child_score"], reverse=True)

        if parents:
            logger.info(
                f"Fetched {len(parents)} parent chunks. "
                f"Top parent: '{parents[0]['title']}' "
                f"(score={parents[0]['best_child_score']:.4f})"
            )
        else:
            logger.warning("No parents found")

        return parents

    def format_context(self, parents: list[dict], max_parents: int = 10) -> str:
        """
        Format parent chunks into context string for LLM.
        Dinaikkan dari 5 ke 10 untuk memberikan konteks lebih kaya dan coverage lebih baik.
        """
        if not parents:
            return "No context found."

        context_parts = []
        for i, parent in enumerate(parents[:max_parents], 1):
            section = parent.get("section", "Unknown")
            title = parent.get("title", "Untitled")
            content = parent.get("content", "")
            matched = parent.get("matched_children", [])

            context_parts.append(
                f"── Dokumen {i} ──\n"
                f"Bagian: {section}\n"
                f"Judul: {title}\n"
                f"Child chunks relevan: {', '.join(matched)}\n"
                f"\n{content}\n"
            )

        return "\n".join(context_parts)