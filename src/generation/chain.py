"""
LLM Chain untuk generasi jawaban berdasarkan konteks yang sudah di-retrieve.
 
Prompt Engineering
───────────────────
Prompt yang baik untuk RAG memiliki komponen:
1. Persona / role → siapa LLM dalam konteks ini?
2. Task definition → apa yang harus dilakukan?
3. Context injection → dokumen relevan
4. Constraints → apa yang TIDAK boleh dilakukan?
5. Output format → bagaimana format jawaban yang diharapkan?
6. Question → pertanyaan user
 
Anti-hallucination strategy:
- Instruksikan LLM untuk HANYA menjawab berdasarkan konteks yang diberikan
- Minta LLM untuk menyebut "tidak ada informasi" jika tidak tahu
- Minta LLM untuk menyebutkan sumber jika relevan
 
LangChain Expression Language (LCEL)
─────────────────────────────────────
LCEL adalah cara modern (2024+) untuk membangun chain di LangChain.
Pattern: chain = prompt | llm | output_parser
- Operator `|` = pipe, output kiri menjadi input kanan
- Mendukung streaming, async, dan parallel execution otomatis
"""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from loguru import logger

from config.settings import get_settings


# ── System Prompt ──────────────────────────────────────────────
SYSTEM_PROMPT = """Kamu adalah asisten akademik resmi STMIK Widya Cipta Dharma yang membantu mahasiswa memahami panduan Penulisan Ilmiah (PI).

PERAN:
- Kamu menjawab pertanyaan mahasiswa berdasarkan HANYA konteks yang diberikan
- Kamu membantu mahasiswa memahami prosedur, syarat, format, dan ketentuan PI

ATURAN KETAT:
1. Jawab HANYA berdasarkan informasi yang ada dalam konteks di bawah
2. Jika informasi TIDAK ADA dalam konteks, katakan: "Maaf, informasi tersebut tidak ditemukan dalam panduan PI yang tersedia. Silakan konsultasikan dengan Dosen Pembimbing atau Program Studi."
3. JANGAN mengarang atau menambahkan informasi yang tidak ada dalam konteks
4. Jika pertanyaan ambigu, minta klarifikasi
5. Sebutkan bagian/BAB sumber informasi jika relevan (misalnya: "Berdasarkan BAB II panduan PI, ...")

FORMAT JAWABAN:
- Gunakan bahasa Indonesia yang formal dan jelas
- Gunakan poin-poin bernomor untuk daftar/langkah
- Berikan jawaban yang lengkap namun ringkas
- Jika ada banyak informasi relevan, prioritaskan yang paling penting"""


# ── User Prompt Template ──────────────────────────────────────
USER_PROMPT = """Berikut adalah konteks dari Panduan Penyusunan Penulisan Ilmiah (PI) STMIK Widya Cipta Dharma:

{context}

──────────────────────────────────────
PERTANYAAN: {question}

Jawab pertanyaan di atas berdasarkan konteks yang diberikan."""


def build_rag_chain():
    """
    Membangun RAG chain menggunakan LCEL (LangChain Expression Language).

    Pipeline: prompt → ChatOpenAI → StrOutputParser

    Returns:
        LCEL chain yang menerima {"context": str, "question": str}
        dan mengembalikan string jawaban
    """
    settings = get_settings()

    # 1. Prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_PROMPT),
    ])

    # 2. LLM
    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.open_api_key,
        temperature=0.1,  # rendah untuk konsistensi, tapi tidak 0 agar tidak terlalu kaku
        max_tokens=1500,
    )

    # 3. Output parser
    output_parser = StrOutputParser()

    # 4. Chain (LCEL pipe operator)
    chain = prompt | llm | output_parser

    return chain


def generate_answer(question: str, context: str) -> str:
    """
    Generate jawaban untuk pertanyaan berdasarkan konteks.

    Args:
        question: Pertanyaan dari user
        context: Konteks dokumen yang sudah diformat (dari ParentChildFetcher)

    Returns:
        Jawaban string dari LLM
    """
    chain = build_rag_chain()

    logger.info(f"Generating answer untuk: '{question[:80]}...'")
    logger.debug(f"Context length: {len(context)} chars")

    answer = chain.invoke({
        "context": context,
        "question": question,
    })

    logger.info(f"Answer generated: {len(answer)} chars")
    return answer