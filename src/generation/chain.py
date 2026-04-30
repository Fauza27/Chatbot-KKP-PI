from __future__ import annotations

from typing import Iterator

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger
from operator import itemgetter

from config.settings import get_settings

settings = get_settings()

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# Menentukan persona, kapabilitas, dan constraint LLM
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Anda adalah asisten akademik STMIK Widya Cipta Dharma. Tugas Anda adalah menjawab pertanyaan seputar panduan Penulisan Ilmiah (PI) dan Kuliah Kerja Praktik (KKP) secara AKURAT, TEGAS, dan RINGKAS.

Aturan WAJIB:
1. Jawab LANGSUNG ke inti pertanyaan. JANGAN gunakan kalimat pembuka seperti "Berdasarkan dokumen..." atau "Menurut panduan...".
2. HANYA gunakan informasi dari konteks yang diberikan. JANGAN menambah informasi dari luar atau membuat asumsi.
3. JANGAN mencantumkan sumber atau nomor BAB di akhir jawaban kecuali diminta secara eksplisit, agar jawaban tetap bersih dan fokus pada informasi.
4. Jika jawaban berupa daftar, gunakan format poin-poin.
5. Jika informasi tidak ada, jawab: "Maaf, informasi tersebut tidak ditemukan dalam panduan PI/KKP yang tersedia."
6. Gunakan Bahasa Indonesia formal yang padat dan jelas.
"""

HUMAN_PROMPT = """KONTEKS DOKUMEN:
{context}

---

PERTANYAAN: {question}

Jawablah secara langsung, padat, dan akurat sesuai konteks di atas."""

HUMAN_PROMPT_WITH_HISTORY = """KONTEKS DOKUMEN:
{context}

---

PERTANYAAN: {question}

Jawablah secara langsung, padat, dan akurat sesuai konteks di atas."""

CONVERSATIONAL_PROMPT = """Riwayat percakapan kita sejauh ini:

{history}

---

Pesan user: {question}

Jawab dengan ramah dan membantu. Jika pertanyaan membutuhkan informasi spesifik \
dari dokumen internal yang tidak ada dalam riwayat, sampaikan bahwa Anda akan \
mencarinya di database dokumen jika diperlukan.
"""

CLARIFICATION_PROMPT = """Riwayat percakapan kita:

{history}

---

Berikut adalah dokumen yang digunakan dalam jawaban sebelumnya:

{context}

---

Pertanyaan lanjutan user: {question}

Berikan elaborasi atau penjelasan tambahan berdasarkan dokumen di atas \
dan riwayat percakapan. Konsisten dengan jawaban sebelumnya."""


def _format_context(documents: list[Document] | list[dict] | str) -> str:
    """
    Format dokumen menjadi string konteks yang terstruktur.

    Args:
        documents: list[Document], list[dict], atau string konteks siap pakai

    Returns:
        String konteks yang siap dimasukkan ke prompt
    """
    if isinstance(documents, str):
        return documents if documents.strip() else "Tidak ada dokumen konteks yang tersedia."

    if not documents:
        return "Tidak ada dokumen konteks yang tersedia."

    formatted_parts: list[str] = []

    for i, doc in enumerate(documents, start=1):
        if isinstance(doc, Document):
            content = doc.page_content
            meta = doc.metadata or {}
        elif isinstance(doc, dict):
            content = doc.get("content", "") or doc.get("page_content", "")
            meta = doc
        else:
            content = str(doc)
            meta = {}

        doc_type = meta.get("doc_type") or meta.get("section") or "dokumen"
        department = meta.get("department", "")
        filename = meta.get("filename", "")
        title = meta.get("title", "")
        matched_children = meta.get("matched_children", [])

        header = f"[Dokumen {i}]"
        if doc_type:
            header += f" Tipe: {str(doc_type).upper()}"
        if title:
            header += f" | Judul: {title}"
        if department:
            header += f" | Departemen: {department}"
        if filename:
            header += f" | Sumber: {filename}"

        score = meta.get("cross_encoder_score")
        if score is not None:
            header += f" | Relevansi: {score:.2f}"

        if matched_children:
            header += f" | Child: {', '.join(matched_children)}"

        formatted_parts.append(f"{header}\n{content}")

    return "\n\n---\n\n".join(formatted_parts)


def _build_sources(context_documents: list[Document] | list[dict] | str, limit: int = 3) -> list[dict]:
    if isinstance(context_documents, str):
        return []

    sources: list[dict] = []

    for doc in context_documents[:limit]:
        if isinstance(doc, Document):
            meta = doc.metadata or {}
            content = doc.page_content
        elif isinstance(doc, dict):
            meta = doc
            content = doc.get("content", "") or doc.get("page_content", "")
        else:
            continue

        sources.append(
            {
                "filename": meta.get("filename", ""),
                "doc_type": meta.get("doc_type", meta.get("section", "")),
                "department": meta.get("department", ""),
                "relevance_score": meta.get("cross_encoder_score"),
                "chunk_preview": (content[:200] + "...") if content else "",
            }
        )

    return sources


def build_rag_chain(streaming: bool = False):
    """
    Bangun RAG chain menggunakan LangChain Expression Language (LCEL).

    Args:
        streaming: Jika True, chain mendukung .stream() untuk streaming output

    Returns:
        LCEL Runnable chain
    """
    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.open_api_key,
        temperature=0,
        max_tokens=1500,
        streaming=streaming,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ])

    output_parser = StrOutputParser()

    chain = (
        {
            "context": lambda x: _format_context(x["context"]),
            "question": itemgetter("question"),
        }
        | prompt
        | llm
        | output_parser
    )

    return chain


class RAGChain:
    """
    High-level wrapper untuk RAG chain.
    Mendukung single-turn (stateless) dan multi-turn (dengan history).
    """

    def __init__(self):
        self._chain = build_rag_chain(streaming=False)
        self._streaming_chain = build_rag_chain(streaming=True)

        self._llm = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.open_api_key,
            temperature=0,
        )

    def invoke(
        self,
        question: str,
        context_documents: list[Document] | list[dict] | str,
        return_sources: bool = True,
    ) -> dict[str, str | list]:
        """
        Single-turn RAG invocation (stateless).

        Args:
            question: Pertanyaan dari user
            context_documents: Dokumen konteks (parent chunks setelah reranking)
            return_sources: Sertakan info sumber dalam response

        Returns:
            {"answer": str, "sources": list[dict]}
        """
        logger.info(f"Generating RAG answer: '{question[:80]}'")

        answer = self._chain.invoke({
            "context": context_documents,
            "question": question,
        })

        result: dict[str, str | list] = {"answer": answer}

        if return_sources:
            result["sources"] = _build_sources(context_documents)

        logger.success(
            "Jawaban: {length} karakter | {docs} dokumen".format(
                length=len(answer),
                docs=len(context_documents) if isinstance(context_documents, list) else 0,
            )
        )
        return result

    def invoke_with_history(
        self,
        question: str,
        context_documents: list[Document] | list[dict] | str,
        conversation_history: list[dict],
        return_sources: bool = True,
    ) -> dict[str, str | list]:
        """
        Multi-turn RAG invocation — menyertakan conversation history.
        """
        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

        logger.info(
            "Generating multi-turn RAG answer: '{question}' (history: {count} pesan)".format(
                question=question[:60],
                count=len(conversation_history),
            )
        )

        context_str = _format_context(context_documents)

        messages = [SystemMessage(content=SYSTEM_PROMPT)]

        for msg in conversation_history:
            if msg.get("role") == "user":
                messages.append(HumanMessage(content=msg.get("content", "")))
            elif msg.get("role") == "assistant":
                messages.append(AIMessage(content=msg.get("content", "")))

        human_content = HUMAN_PROMPT_WITH_HISTORY.format(
            context=context_str,
            question=question,
        )
        messages.append(HumanMessage(content=human_content))

        response = self._llm.invoke(messages)
        answer = response.content

        result: dict[str, str | list] = {"answer": answer}

        if return_sources:
            result["sources"] = _build_sources(context_documents)

        logger.success(f"Multi-turn jawaban: {len(answer)} karakter")
        return result

    def invoke_conversational(
        self,
        question: str,
        conversation_history: list[dict],
    ) -> dict[str, str | list]:
        """
        Jawab pertanyaan conversational TANPA retrieval — hanya dari history.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        logger.info(f"Conversational mode (tanpa retrieval): '{question[:60]}'")

        history_text = "\n".join([
            f"{'User' if m['role'] == 'user' else 'Asisten'}: {m['content'][:150]}"
            for m in conversation_history[-6:]
        ])

        prompt_text = CONVERSATIONAL_PROMPT.format(
            history=history_text or "(Ini adalah pesan pertama)",
            question=question,
        )

        response = self._llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt_text),
        ])
        answer = response.content

        logger.success(f"Conversational jawaban: {len(answer)} karakter")
        return {"answer": answer, "sources": []}

    def invoke_clarification(
        self,
        question: str,
        conversation_history: list[dict],
        last_context_docs_text: list[str],
    ) -> dict[str, str | list]:
        """
        Jawab pertanyaan klarifikasi menggunakan konteks dokumen SEBELUMNYA.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        logger.info(f"Clarification mode (pakai konteks lama): '{question[:60]}'")

        if last_context_docs_text:
            context_str = "\n\n---\n\n".join(
                f"[Dokumen {i + 1}]\n{text}"
                for i, text in enumerate(last_context_docs_text[:3])
            )
        else:
            context_str = "(Tidak ada dokumen konteks dari percakapan sebelumnya)"

        history_text = "\n".join([
            f"{'User' if m['role'] == 'user' else 'Asisten'}: {m['content'][:200]}"
            for m in conversation_history[-6:]
        ])

        prompt_text = CLARIFICATION_PROMPT.format(
            history=history_text or "(Ini adalah percakapan baru)",
            context=context_str,
            question=question,
        )

        response = self._llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt_text),
        ])

        answer = response.content
        logger.success(f"Clarification jawaban: {len(answer)} karakter")

        return {
            "answer": answer,
            "sources": [],
        }

    def stream(
        self,
        question: str,
        context_documents: list[Document] | list[dict] | str,
    ) -> Iterator[str]:
        """Stream jawaban token per token untuk UI real-time."""
        for chunk in self._streaming_chain.stream({
            "context": context_documents,
            "question": question,
        }):
            yield chunk


_rag_chain_instance: object | None = None

def _get_rag_chain():
    """Lazy singleton untuk RAG chain."""
    global _rag_chain_instance
    if _rag_chain_instance is None:
        _rag_chain_instance = build_rag_chain(streaming=False)
    return _rag_chain_instance


def generate_answer(question: str, context: str) -> str:
    """
    Generate jawaban untuk pertanyaan berdasarkan konteks (string).
    """
    chain = _get_rag_chain() 

    logger.info(f"Generating answer untuk: '{question[:80]}...'")
    logger.debug(f"Context length: {len(context)} chars")

    answer = chain.invoke({
        "context": context,
        "question": question,
    })

    logger.info(f"Answer generated: {len(answer)} chars")
    return answer