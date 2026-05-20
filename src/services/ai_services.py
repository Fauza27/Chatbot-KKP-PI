from typing import Dict, Any, Optional
import time
from loguru import logger

from src.generation.memory import ConversationMemory, IntentType
from src.generation.intent_classifier import IntentClassifier, reformulate_query
from src.generation.chain import RAGChain
from src.services.session_store import get_session_store
from config.settings import get_settings

settings = get_settings()

# Legacy in-memory session store (fallback jika database tidak tersedia)
_legacy_session_store: dict[str, tuple[ConversationMemory, float]] = {}
_legacy_session_lock = None

# Initialize session store berdasarkan konfigurasi
if settings.USE_DATABASE_SESSIONS:
    try:
        _session_store = get_session_store()
        logger.info("Using database-backed session storage")
    except Exception as e:
        logger.error(f"Failed to initialize database session store: {e}")
        logger.warning("Falling back to in-memory session storage")
        settings.USE_DATABASE_SESSIONS = False

if not settings.USE_DATABASE_SESSIONS:
    from threading import Lock
    _legacy_session_lock = Lock()
    logger.info("Using in-memory session storage")

_classifier = IntentClassifier()
_rag_chain = RAGChain()


class ChatError(Exception):
    """Custom exception for chat-related errors"""
    pass


class RetrievalError(ChatError):
    """Exception for retrieval-related errors"""
    pass


def _evict_idle_sessions(now: float) -> int:
    """Hapus session yang idle melebihi SESSION_CLEANUP_INTERVAL detik (legacy in-memory only)."""
    if settings.USE_DATABASE_SESSIONS:
        return 0  # Database handles cleanup
    
    ttl = settings.SESSION_CLEANUP_INTERVAL
    expired = [sid for sid, (_, last_ts) in _legacy_session_store.items() if now - last_ts > ttl]
    for sid in expired:
        _legacy_session_store.pop(sid, None)
    if expired:
        logger.info(f"Evicted {len(expired)} idle session(s)")
    return len(expired)


def _evict_lru_if_full() -> None:
    """Jika store sudah penuh, hapus session paling lama tidak diakses (legacy in-memory only)."""
    if settings.USE_DATABASE_SESSIONS:
        return  # Database handles LRU
    
    cap = settings.MAX_ACTIVE_SESSIONS
    if len(_legacy_session_store) <= cap:
        return
    # Sort by last_access_ts ascending; buang sampai cap.
    overflow = len(_legacy_session_store) - cap
    sorted_items = sorted(_legacy_session_store.items(), key=lambda kv: kv[1][1])
    for sid, _ in sorted_items[:overflow]:
        _legacy_session_store.pop(sid, None)
    logger.info(f"Evicted {overflow} LRU session(s) due to MAX_ACTIVE_SESSIONS cap")


def get_or_create_memory(session_id: str) -> ConversationMemory:
    """Get or create conversation memory for a session."""
    if settings.USE_DATABASE_SESSIONS:
        # Database-backed session storage
        return _session_store.load_memory(session_id)
    
    # Legacy in-memory session storage
    now = time.time()
    with _legacy_session_lock:
        # Lazy cleanup setiap kali ada akses; murah karena cuma scan dict di-memori.
        _evict_idle_sessions(now)

        existing = _legacy_session_store.get(session_id)
        if existing is not None:
            memory, _ = existing
            _legacy_session_store[session_id] = (memory, now)
            return memory

        memory = ConversationMemory(max_turns=5)
        _legacy_session_store[session_id] = (memory, now)
        _evict_lru_if_full()
        return memory


def _save_memory_if_needed(session_id: str, memory: ConversationMemory) -> None:
    """Save memory to persistent storage if using database sessions."""
    if settings.USE_DATABASE_SESSIONS:
        try:
            _session_store.save_memory(session_id, memory)
        except Exception as e:
            logger.error(f"Failed to save session {session_id}: {e}")
            # Don't fail the chat operation if save fails


def clear_session(session_id: str) -> bool:
    """Clear conversation memory for a session"""
    if settings.USE_DATABASE_SESSIONS:
        return _session_store.delete_session(session_id)
    
    # Legacy in-memory
    with _legacy_session_lock:
        if session_id in _legacy_session_store:
            del _legacy_session_store[session_id]
            logger.info(f"Session {session_id} cleared")
            return True
    return False


def get_session_stats() -> Dict[str, Any]:
    """Get statistics about active sessions"""
    if settings.USE_DATABASE_SESSIONS:
        return _session_store.get_session_stats()
    
    # Legacy in-memory
    with _legacy_session_lock:
        return {
            "active_sessions": len(_legacy_session_store),
            "total_turns": sum(m.turn_count for m, _ in _legacy_session_store.values()),
            "sessions": list(_legacy_session_store.keys()),
        }


def cleanup_sessions() -> int:
    """Manually trigger session cleanup. Returns number of sessions cleaned."""
    if settings.USE_DATABASE_SESSIONS:
        return _session_store.cleanup_idle_sessions()
    
    # Legacy in-memory
    now = time.time()
    with _legacy_session_lock:
        return _evict_idle_sessions(now)


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
            
            # Save to persistent storage
            _save_memory_if_needed(session_id, memory)
            
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

            # Save to persistent storage
            _save_memory_if_needed(session_id, memory)

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
    from src.retrieval.pipeline import run_retrieval

    # Step 1: Query reformulation (resolve referensi implisit "itu", "tadi", dst.)
    try:
        search_query = reformulate_query(question, memory)
        if search_query != question:
            logger.info(f"[session={session_id}] Reformulated: '{question}' → '{search_query}'")
    except Exception as e:
        logger.warning(f"[session={session_id}] Reformulation failed, using original: {e}")
        search_query = question

    # Step 2-5: Self-query → hybrid search → parent fetch → rerank
    retrieval = run_retrieval(query=search_query, rerank_query=question)

    if retrieval.is_empty:
        answer = (
            "Maaf, informasi tersebut tidak ditemukan dalam panduan KKP/PI yang tersedia. "
            "Silakan konsultasikan langsung dengan Dosen Pembimbing atau Program Studi Anda."
        )
        memory.add_assistant_turn(content=answer)
        
        # Save to persistent storage
        _save_memory_if_needed(session_id, memory)
        
        return {
            "answer": answer,
            "num_docs": 0,
            "intent": intent.value,
            "confidence": confidence,
            "reasoning": reason,
            "sources": [],
        }

    reranked_parents = retrieval.parent_documents

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

    # Save to persistent storage
    _save_memory_if_needed(session_id, memory)

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
