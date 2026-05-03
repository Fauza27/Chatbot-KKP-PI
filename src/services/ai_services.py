import logging
from src.generation.memory import ConversationMemory, IntentType
from src.generation.intent_classifier import IntentClassifier, reformulate_query
from src.generation.chain import RAGChain

logger = logging.getLogger(__name__)

# Key: session_id (string), Value: ConversationMemory
_session_store: dict[str, ConversationMemory] = {}

_classifier = IntentClassifier()
_rag_chain = RAGChain()


def get_or_create_memory(session_id: str) -> ConversationMemory:
    if session_id not in _session_store:
        _session_store[session_id] = ConversationMemory(max_turns=5)
    return _session_store[session_id]


def chat(query: str, session_id: str) -> dict:
    memory = get_or_create_memory(session_id)
    question = query.strip()

    if not question:
        return {"answer": "Pertanyaan tidak boleh kosong.", "num_docs": 0}

    memory.add_user_turn(question)
    logger.info(f"[session={session_id}] Question: {question}")

    try:
        intent, confidence, reason = _classifier.classify(question, memory)
        logger.info(f"Intent: {intent.value} (conf={confidence:.2f})")

        if intent == IntentType.CONVERSATIONAL:
            result = _rag_chain.invoke_conversational(
                question=question,
                conversation_history=memory.get_history_for_llm(),
            )
            answer = result["answer"]
            memory.add_assistant_turn(content=answer)
            return {"answer": answer, "num_docs": 0}

        elif intent == IntentType.CLARIFICATION:
            result = _rag_chain.invoke_clarification(
                question=question,
                conversation_history=memory.get_history_for_llm(),
                last_context_docs_text=memory.get_last_retrieved_docs(),
            )
            answer = result["answer"]
            memory.add_assistant_turn(content=answer)
            return {"answer": answer, "num_docs": 0}

        else:  # NEEDS_RETRIEVAL
            return _handle_retrieval(question, memory)

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise  


def _handle_retrieval(question: str, memory: ConversationMemory) -> dict:
    from src.retrieval.self_query import extract_query_components
    from src.retrieval.hybrid_search import HybridSearcher
    from src.retrieval.parent_child import ParentChildFetcher
    from src.retrieval.reranker import CrossEncoderReranker

    search_query = reformulate_query(question, memory)
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
        memory.add_assistant_turn(content=answer)
        return {"answer": answer, "num_docs": 0}

    fetcher = ParentChildFetcher()
    parent_results = fetcher.fetch_parents(search_results)

    reranker = CrossEncoderReranker()
    reranked_parents = reranker.rerank(
        query=question,
        documents=parent_results,
    )

    result = _rag_chain.invoke_with_history(
        question=question,
        context_documents=reranked_parents,
        conversation_history=memory.get_history_for_llm(),
    )
    answer = result["answer"]

    memory.add_assistant_turn(
        content=answer,
        retrieved_doc_contents=[p["content"] for p in reranked_parents],
    )

    return {"answer": answer, "num_docs": len(reranked_parents)}