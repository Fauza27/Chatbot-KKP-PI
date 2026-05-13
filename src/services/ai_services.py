from typing import Dict, Any, Optional
from loguru import logger

from src.generation.memory import ConversationMemory, IntentType
from src.generation.intent_classifier import IntentClassifier, reformulate_query
from src.generation.chain import RAGChain
from config.settings import get_settings

settings = get_settings()

# Key: session_id (string), Value: ConversationMemory
_session_store: dict[str, ConversationMemory] = {}

_classifier = IntentClassifier()
_rag_chain = RAGChain()


class ChatError(Exception):
    """Custom exception for chat-related errors"""
    pass


class RetrievalError(ChatError):
    """Exception for retrieval-related errors"""
    pass


def get_or_create_memory(session_id: str) -> ConversationMemory:
    """Get or create conversation memory for a session"""
    if session_id not in _session_store:
        _session_store[session_id] = ConversationMemory(max_turns=5)
    return _session_store[session_id]


def clear_session(session_id: str) -> bool:
    """Clear conversation memory for a session"""
    if session_id in _session_store:
        del _session_store[session_id]
        logger.info(f"Session {session_id} cleared")
        return True
    return False


def get_session_stats() -> Dict[str, Any]:
    """Get statistics about active sessions"""
    return {
        "active_sessions": len(_session_store),
        "total_turns": sum(m.turn_count for m in _session_store.values()),
        "sessions": list(_session_store.keys()),
    }


def chat(query: str, session_id: str) -> Dict[str, Any]:
    """
    Main chat function.

    Returns dict with keys: answer, num_docs, intent, confidence, reasoning, sources
    """
    if not query or not query.strip():
        return {"answer": "Pertanyaan tidak boleh kosong.", "num_docs": 0, "error": "empty_query"}

    if not session_id:
        return {"answer": "Session ID diperlukan.", "num_docs": 0, "error": "missing_session_id"}

    memory = get_or_create_memory(session_id)
    question = query.strip()

    memory.add_user_turn(question)
    logger.info(f"[session={session_id}] Question: {question}")

    try:
        intent, confidence, reason = _classifier.classify(question, memory)
        logger.info(f"[session={session_id}] Intent: {intent.value} (conf={confidence:.2f}) - {reason}")

        if intent == IntentType.CONVERSATIONAL:
            result = _rag_chain.invoke_conversational(
                question=question,
                conversation_history=memory.get_history_for_llm(),
            )
            answer = result["answer"]
            memory.add_assistant_turn(content=answer)
            return {
                "answer": answer,
                "num_docs": 0,
                "intent": intent.value,
                "confidence": confidence,
                "reasoning": reason,
                "sources": [],
            }

        elif intent == IntentType.CLARIFICATION:
            result = _rag_chain.invoke_clarification(
                question=question,
                conversation_history=memory.get_history_for_llm(),
                last_context_docs_text=memory.get_last_retrieved_docs(),
            )
            answer = result["answer"]

            # If clarification triggered a fallback retrieval it may return sources
            sources = result.get("sources", [])
            if sources:
                memory.add_assistant_turn(
                    content=answer,
                    retrieved_doc_contents=[s.get("content", "") for s in sources],
                )
            else:
                memory.add_assistant_turn(content=answer)

            return {
                "answer": answer,
                "num_docs": len(sources),
                "intent": intent.value,
                "confidence": confidence,
                "reasoning": reason,
                "sources": sources,
            }

        else:  # NEEDS_RETRIEVAL
            return _handle_retrieval(question, memory, session_id, intent, confidence, reason)

    except Exception as e:
        logger.error(f"[session={session_id}] Error processing query: {e}", exc_info=True)
        return {
            "answer": (
                "Maaf, terjadi kesalahan saat memproses pertanyaan Anda. "
                "Silakan coba lagi atau hubungi administrator jika masalah berlanjut."
            ),
            "num_docs": 0,
            "error": str(e),
            "error_type": type(e).__name__,
        }


def _handle_retrieval(
    question: str,
    memory: ConversationMemory,
    session_id: str,
    intent: IntentType,
    confidence: float,
    reason: str,
) -> Dict[str, Any]:
    """Handle retrieval-based questions."""
    from src.retrieval.self_query import extract_query_components
    from src.retrieval.hybrid_search import HybridSearcher
    from src.retrieval.parent_child import ParentChildFetcher
    from src.retrieval.reranker import CrossEncoderReranker

    # Step 1: Query reformulation
    try:
        search_query = reformulate_query(question, memory)
        if search_query != question:
            logger.info(f"[session={session_id}] Reformulated: '{question}' → '{search_query}'")
    except Exception as e:
        logger.warning(f"[session={session_id}] Reformulation failed, using original: {e}")
        search_query = question

    # Step 2: Self-query extraction
    parsed = extract_query_components(search_query)

    # Step 3: Hybrid search
    searcher = HybridSearcher()
    search_results = searcher.search(
        query=parsed.semantic_query,
        filters=parsed.filters,
    )

    if not search_results:
        answer = (
            "Maaf, informasi tersebut tidak ditemukan dalam panduan KKP/PI yang tersedia. "
            "Silakan konsultasikan langsung dengan Dosen Pembimbing atau Program Studi Anda."
        )
        memory.add_assistant_turn(content=answer)
        return {
            "answer": answer,
            "num_docs": 0,
            "intent": intent.value,
            "confidence": confidence,
            "reasoning": reason,
            "sources": [],
        }

    # Step 4: Parent-child fetching
    fetcher = ParentChildFetcher()
    parent_results = fetcher.fetch_parents(search_results)

    # Step 5: Reranking
    try:
        reranker = CrossEncoderReranker()
        reranked_parents = reranker.rerank(query=question, documents=parent_results)
    except Exception as e:
        logger.warning(f"[session={session_id}] Reranking failed, using unranked: {e}")
        reranked_parents = parent_results[: settings.rerank_top_n]

    # Step 6: Generation
    result = _rag_chain.invoke_with_history(
        question=question,
        context_documents=reranked_parents,
        conversation_history=memory.get_history_for_llm(),
    )
    answer = result["answer"]

    # Update memory
    memory.add_assistant_turn(
        content=answer,
        retrieved_doc_contents=[p["content"] for p in reranked_parents],
    )

    return {
        "answer": answer,
        "num_docs": len(reranked_parents),
        "intent": intent.value,
        "confidence": confidence,
        "reasoning": reason,
        "sources": [
            {
                "section": p.get("section", ""),
                "title": p.get("title", ""),
                "parent_id": p.get("parent_id", ""),
                "score": p.get("cross_encoder_score", 0.0),
            }
            for p in reranked_parents[:3]
        ],
    }
