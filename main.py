"""
Entry point utama: jalankan RAG pipeline secara interaktif.
 
Mengorkestrasi semua komponen:
Self-Query → Hybrid Search → Parent-Child → Reranking → Generation
 
Penggunaan:
    python main.py
    python main.py --question "Apa syarat untuk mengambil PI?"
    python main.py --ingest
    python main.py --evaluate
    python main.py --debug --question "..."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from loguru import logger

from config.settings import get_settings


# ── Konfigurasi Logger ─────────────────────────────────────────
def setup_logger(debug: bool = False):
    """Setup loguru logger."""
    logger.remove()  # hapus default handler

    if debug:
        logger.add(
            sys.stderr,
            level="DEBUG",
            format=(
                "<green>{time:HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            ),
        )
    else:
        logger.add(
            sys.stderr,
            level="INFO",
            format=(
                "<green>{time:HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<level>{message}</level>"
            ),
        )


# ── Pipeline RAG ──────────────────────────────────────────────
def run_rag_pipeline(question: str, debug: bool = False) -> dict:
    """
    Jalankan seluruh pipeline RAG untuk satu pertanyaan.

    Pipeline:
    1. Self-Query Extraction (metadata filter via heuristik + LLM)
    2. Hybrid Search (BM25 + Dense via pgvector + RRF)
    3. Parent-Child Fetching (child → parent lookup)
    4. Cross-Encoder Reranking (top-N most relevant)
    5. Prompt Engineering + GPT (generate answer)

    Args:
        question: Pertanyaan dari user
        debug: Jika True, tampilkan detail setiap tahap

    Returns:
        Dict berisi:
        - answer: Jawaban dari LLM
        - contexts: List of context strings (untuk evaluasi RAGAS)
        - metadata: Dict detail setiap tahap pipeline
    """
    from src.retrieval.self_query import extract_query_components
    from src.retrieval.hybrid_search import HybridSearcher
    from src.retrieval.parent_child import ParentChildFetcher
    from src.retrieval.reranker import CrossEncoderReranker
    from src.generation.chain import generate_answer

    metadata = {}

    # ── TAHAP 1: Self-Query ─────────────────────────────────
    logger.info("=" * 60)
    logger.info("TAHAP 1: Self-Query Extraction")
    logger.info("=" * 60)

    parsed = extract_query_components(question)

    metadata["self_query"] = {
        "original_query": parsed.original_query,
        "semantic_query": parsed.semantic_query,
        "filters": parsed.filters,
    }

    if debug:
        logger.debug(f"  Query asli    : {parsed.original_query}")
        logger.debug(f"  Query semantik: {parsed.semantic_query}")
        logger.debug(f"  Filters       : {parsed.filters}")

    # ── TAHAP 2: Hybrid Search ──────────────────────────────
    logger.info("=" * 60)
    logger.info("TAHAP 2: Hybrid Search (BM25 + Dense + RRF)")
    logger.info("=" * 60)

    searcher = HybridSearcher()
    search_results = searcher.search(
        query=parsed.semantic_query,
        filters=parsed.filters,
    )

    metadata["hybrid_search"] = {
        "num_results": len(search_results),
        "top_scores": [
            {
                "child_id": r.child_id,
                "hybrid_score": round(r.hybrid_score, 4),
                "dense_score": round(r.dense_score, 4),
                "bm25_score": round(r.bm25_score, 4),
            }
            for r in search_results[:5]
        ],
    }

    if debug:
        for i, r in enumerate(search_results[:5], 1):
            title = r.document.metadata.get("title", "")[:50]
            logger.debug(
                f"  [{i}] {r.child_id} | "
                f"hybrid={r.hybrid_score:.4f} | "
                f"dense={r.dense_score:.4f} | "
                f"bm25={r.bm25_score:.4f} | "
                f"{title}"
            )

    if not search_results:
        return {
            "answer": (
                "Maaf, saya tidak menemukan informasi yang relevan "
                "dalam panduan KKP/PI. Silakan coba pertanyaan lain atau "
                "konsultasikan dengan Dosen Pembimbing."
            ),
            "contexts": [],
            "metadata": metadata,
        }

    # ── TAHAP 3: Parent-Child Fetching ──────────────────────
    logger.info("=" * 60)
    logger.info("TAHAP 3: Parent-Child Fetching")
    logger.info("=" * 60)

    fetcher = ParentChildFetcher()
    parent_results = fetcher.fetch_parents(search_results)

    metadata["parent_child"] = {
        "num_parents": len(parent_results),
        "parents": [
            {
                "parent_id": p["parent_id"],
                "title": p["title"],
                "matched_children": p.get("matched_children", []),
            }
            for p in parent_results
        ],
    }

    if debug:
        for p in parent_results:
            logger.debug(
                f"  Parent: {p['parent_id']} | {p['title']} | "
                f"children={p.get('matched_children', [])}"
            )

    # ── TAHAP 4: Cross-Encoder Reranking ────────────────────
    logger.info("=" * 60)
    logger.info("TAHAP 4: Cross-Encoder Reranking")
    logger.info("=" * 60)

    reranker = CrossEncoderReranker()
    reranked_parents = reranker.rerank(
        query=question,
        documents=parent_results,
    )

    metadata["reranking"] = {
        "num_reranked": len(reranked_parents),
        "reranked": [
            {
                "parent_id": p["parent_id"],
                "title": p["title"],
                "ce_score": round(p.get("cross_encoder_score", 0), 4),
            }
            for p in reranked_parents
        ],
    }

    if debug:
        for i, p in enumerate(reranked_parents, 1):
            logger.debug(
                f"  [{i}] {p['parent_id']} | "
                f"CE={p.get('cross_encoder_score', 0):.4f} | "
                f"{p['title'][:50]}"
            )

    # ── TAHAP 5: Format Context + Generate Answer ───────────
    logger.info("=" * 60)
    logger.info("TAHAP 5: Prompt Engineering + LLM Generation")
    logger.info("=" * 60)

    context_str = fetcher.format_context(reranked_parents)
    contexts_list = [p["content"] for p in reranked_parents]

    if debug:
        logger.debug(f"  Context length: {len(context_str)} chars")
        logger.debug(f"  Num context docs: {len(contexts_list)}")

    answer = generate_answer(question=question, context=context_str)

    metadata["generation"] = {
        "context_length": len(context_str),
        "answer_length": len(answer),
    }

    return {
        "answer": answer,
        "contexts": contexts_list,
        "metadata": metadata,
    }


# ── Mode: Ingestion ──────────────────────────────────────────
def run_ingest(dataset: str = "both"):
    """Jalankan pipeline ingestion: load JSON → embed → upsert ke Supabase."""
    from src.ingestion.embedder import run_ingestion

    project_root = Path(__file__).resolve().parent
    dataset_map = {
        "pi": ("child_chunk_pi.json", "parent_chunk_pi.json"),
        "kkp": ("child_chunk_kkp.json", "parent_chunk_kkp.json"),
    }

    def ingest_one(name: str) -> None:
        child_file, parent_file = dataset_map[name]
        child_path = project_root / "extract-pdf" / child_file
        parent_path = project_root / "extract-pdf" / parent_file

        if not child_path.exists():
            logger.error(f"File tidak ditemukan: {child_path}")
            sys.exit(1)
        if not parent_path.exists():
            logger.error(f"File tidak ditemukan: {parent_path}")
            sys.exit(1)

        stats = run_ingestion(
            child_chunks_path=str(child_path),
            parent_chunks_path=str(parent_path),
        )

        logger.info(f"Ingestion selesai untuk {name.upper()}!")
        logger.info(f"Stats: {stats}")

    if dataset == "both":
        for name in ("pi", "kkp"):
            ingest_one(name)
    else:
        ingest_one(dataset)


# ── Mode: Evaluasi ────────────────────────────────────────────
def run_eval(dataset: str = "pi"):
    from src.evaluation.ragas_eval import (
        run_evaluation,
        EVAL_QUESTIONS_PI,
        EVAL_QUESTIONS_KKP,
    )

    dataset_map = {
        "pi": EVAL_QUESTIONS_PI,
        "kkp": EVAL_QUESTIONS_KKP,
        "both": EVAL_QUESTIONS_PI + EVAL_QUESTIONS_KKP,
    }

    eval_data = dataset_map.get(dataset, EVAL_QUESTIONS_PI)
    logger.info(f"Evaluasi dataset: {dataset.upper()} ({len(eval_data)} pertanyaan)")

    def pipeline_fn(question: str) -> dict:
        result = run_rag_pipeline(question, debug=False)
        return {"answer": result["answer"], "contexts": result["contexts"]}

    scores = run_evaluation(pipeline_fn=pipeline_fn, eval_data=eval_data)
    logger.info(f"Evaluation scores: {scores}")


# ── Mode: Interaktif ─────────────────────────────────────────
# main.py — run_interactive() yang diperbaiki
def run_interactive(debug: bool = False):
    from src.generation.memory import ConversationMemory, IntentType
    from src.generation.intent_classifier import IntentClassifier, reformulate_query
    from src.generation.chain import RAGChain

    print("\n" + "=" * 60)
    print("🎓 Chatbot Panduan KKP/PI")
    print("   STMIK Widya Cipta Dharma")
    print("=" * 60)
    print("Ketik pertanyaan Anda, atau 'quit' untuk keluar.\n")

    memory = ConversationMemory(max_turns=5)
    classifier = IntentClassifier()
    rag_chain = RAGChain()

    while True:
        try:
            question = input("📝 Pertanyaan: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nSampai jumpa! 👋")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q", "keluar"):
            print("\nSampai jumpa! 👋")
            break

        memory.add_user_turn(question)
        print("\n⏳ Sedang mencari jawaban...\n")

        try:
            intent, confidence, reason = classifier.classify(question, memory)

            if debug:
                logger.debug(f"Intent: {intent.value} (conf={confidence:.2f}) | {reason}")

            if intent == IntentType.CONVERSATIONAL:
                result = rag_chain.invoke_conversational(
                    question=question,
                    conversation_history=memory.get_history_for_llm(),
                )
                answer = result["answer"]

            elif intent == IntentType.CLARIFICATION:
                result = rag_chain.invoke_clarification(
                    question=question,
                    conversation_history=memory.get_history_for_llm(),
                    last_context_docs_text=memory.get_last_retrieved_docs(),
                )
                answer = result["answer"]

            else:  # NEEDS_RETRIEVAL
                # Reformulasi jika ada referensi implisit seperti "itu", "tersebut"
                search_query = reformulate_query(question, memory)

                from src.retrieval.self_query import extract_query_components
                from src.retrieval.hybrid_search import HybridSearcher
                from src.retrieval.parent_child import ParentChildFetcher
                from src.retrieval.reranker import CrossEncoderReranker

                parsed = extract_query_components(search_query)
                searcher = HybridSearcher()
                search_results = searcher.search(
                    query=parsed.semantic_query,
                    filters=parsed.filters,
                )

                if not search_results:
                    answer = (
                        "Maaf, informasi tersebut tidak ditemukan dalam panduan KKP/PI. "
                        "Silakan konsultasikan dengan Dosen Pembimbing."
                    )
                else:
                    fetcher = ParentChildFetcher()
                    parent_results = fetcher.fetch_parents(search_results)
                    reranker = CrossEncoderReranker()
                    reranked_parents = reranker.rerank(
                        query=question, documents=parent_results
                    )
                    result = rag_chain.invoke_with_history(
                        question=question,
                        context_documents=reranked_parents,
                        conversation_history=memory.get_history_for_llm(),
                    )
                    answer = result["answer"]
                    # Simpan konteks untuk clarification berikutnya
                    memory.add_assistant_turn(
                        content=answer,
                        retrieved_doc_contents=[p["content"] for p in reranked_parents],
                    )
                    _print_answer(answer, len(reranked_parents))
                    print()
                    continue

            memory.add_assistant_turn(content=answer)
            _print_answer(answer, 0)

        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"\n❌ Terjadi error: {e}")
            print("Silakan coba lagi.\n")

        print()

# main.py — run_interactive() yang diperbaiki
def run_interactive(debug: bool = False):
    from src.generation.memory import ConversationMemory, IntentType
    from src.generation.intent_classifier import IntentClassifier, reformulate_query
    from src.generation.chain import RAGChain

    print("\n" + "=" * 60)
    print("🎓 Chatbot Panduan KKP/PI")
    print("   STMIK Widya Cipta Dharma")
    print("=" * 60)
    print("Ketik pertanyaan Anda, atau 'quit' untuk keluar.\n")

    memory = ConversationMemory(max_turns=5)
    classifier = IntentClassifier()
    rag_chain = RAGChain()

    while True:
        try:
            question = input("📝 Pertanyaan: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nSampai jumpa! 👋")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q", "keluar"):
            print("\nSampai jumpa! 👋")
            break

        memory.add_user_turn(question)
        print("\n⏳ Sedang mencari jawaban...\n")

        try:
            intent, confidence, reason = classifier.classify(question, memory)

            if debug:
                logger.debug(f"Intent: {intent.value} (conf={confidence:.2f}) | {reason}")

            if intent == IntentType.CONVERSATIONAL:
                result = rag_chain.invoke_conversational(
                    question=question,
                    conversation_history=memory.get_history_for_llm(),
                )
                answer = result["answer"]

            elif intent == IntentType.CLARIFICATION:
                result = rag_chain.invoke_clarification(
                    question=question,
                    conversation_history=memory.get_history_for_llm(),
                    last_context_docs_text=memory.get_last_retrieved_docs(),
                )
                answer = result["answer"]

            else:  # NEEDS_RETRIEVAL
                # Reformulasi jika ada referensi implisit seperti "itu", "tersebut"
                search_query = reformulate_query(question, memory)

                from src.retrieval.self_query import extract_query_components
                from src.retrieval.hybrid_search import HybridSearcher
                from src.retrieval.parent_child import ParentChildFetcher
                from src.retrieval.reranker import CrossEncoderReranker

                parsed = extract_query_components(search_query)
                searcher = HybridSearcher()
                search_results = searcher.search(
                    query=parsed.semantic_query,
                    filters=parsed.filters,
                )

                if not search_results:
                    answer = (
                        "Maaf, informasi tersebut tidak ditemukan dalam panduan KKP/PI. "
                        "Silakan konsultasikan dengan Dosen Pembimbing."
                    )
                else:
                    fetcher = ParentChildFetcher()
                    parent_results = fetcher.fetch_parents(search_results)
                    reranker = CrossEncoderReranker()
                    reranked_parents = reranker.rerank(
                        query=question, documents=parent_results
                    )
                    result = rag_chain.invoke_with_history(
                        question=question,
                        context_documents=reranked_parents,
                        conversation_history=memory.get_history_for_llm(),
                    )
                    answer = result["answer"]
                    # Simpan konteks untuk clarification berikutnya
                    memory.add_assistant_turn(
                        content=answer,
                        retrieved_doc_contents=[p["content"] for p in reranked_parents],
                    )
                    _print_answer(answer, len(reranked_parents))
                    print()
                    continue

            memory.add_assistant_turn(content=answer)
            _print_answer(answer, 0)

        except Exception as e:
            logger.error(f"Error: {e}")
            print(f"\n❌ Terjadi error: {e}")
            print("Silakan coba lagi.\n")

        print()


def _print_answer(answer: str, num_docs: int) -> None:
    print("─" * 60)
    print("💡 JAWABAN:")
    print("─" * 60)
    print(answer)
    print("─" * 60)
    if num_docs > 0:
        print(f"📚 Sumber: {num_docs} dokumen digunakan")


# ── Main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="RAG Chatbot - Panduan KKP/PI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh penggunaan:
  python main.py                                    # mode interaktif
  python main.py --question "Apa syarat PI?"        # single question
    python main.py --ingest --dataset both            # ingest data KKP + PI
    python main.py --ingest --dataset pi              # ingest data PI
    python main.py --ingest --dataset kkp             # ingest data KKP
  python main.py --evaluate                         # evaluasi dengan RAGAS
  python main.py --debug --question "..."           # debug mode
        """,
    )

    parser.add_argument(
        "--question", "-q",
        type=str,
        help="Pertanyaan tunggal (tanpa mode interaktif)",
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Jalankan ingestion: embed + upload data ke Supabase",
    )
    parser.add_argument(
        "--dataset",
        choices=["pi", "kkp", "both"],
        default="both",
        help="Dataset ingestion: pi, kkp, atau both",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Jalankan evaluasi RAGAS pada pipeline",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Tampilkan detail setiap tahap pipeline",
    )

    args = parser.parse_args()

    # Setup logger
    setup_logger(debug=args.debug)

    # Validate settings
    try:
        settings = get_settings()
        logger.info(
            f"Settings loaded: LLM={settings.llm_model}, "
            f"Embedding={settings.embedding_model}"
        )
    except Exception as e:
        logger.error(f"Gagal load settings: {e}")
        logger.error("Pastikan file .env sudah dikonfigurasi dengan benar.")
        sys.exit(1)

    # Route ke mode yang dipilih
    if args.ingest:
        run_ingest(args.dataset)
    elif args.evaluate:
        run_eval(dataset=args.dataset)  
    elif args.question:
        result = run_rag_pipeline(args.question, debug=args.debug)
        print("\n" + "─" * 60)
        print("💡 JAWABAN:")
        print("─" * 60)
        print(result["answer"])
        print("─" * 60)
    else:
        run_interactive(debug=args.debug)


if __name__ == "__main__":
    main()