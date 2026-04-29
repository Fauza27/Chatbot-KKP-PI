import json
 
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from loguru import logger
 
from config.settings import get_settings
from src.generation.memory import ConversationMemory, IntentType

settings = get_settings()

_CLASSIFIER_SYSTEM_PROMPT = """Anda adalah classifier yang menganalisis pesan user \
dalam sistem Q&A Kuliah Kerja Praktek (KKP) dan Penelitian Ilmiah (PI) di STMIK Widya Cipta Dharma.
 
Tugas Anda: tentukan intent pesan user dan kembalikan HANYA JSON, tidak ada penjelasan lain.
 
Tiga kategori intent:
 
1. "needs_retrieval"
   → Pertanyaan spesifik yang butuh informasi dari dokumen pedoman
     (peraturan, angka, prosedur, syarat, ketentuan).
   → Termasuk pertanyaan tentang topik BARU yang belum ada di history.
   Contoh: "Berapa lama minimal KKP?", "Apa syarat maju PI?", "Bagaimana cara daftar KKP?", "Apa saja topik PI yang bisa saya pilih?"
 
2. "conversational"
   → Sapaan, ucapan terima kasih, pertanyaan sangat umum, atau perintah
     yang tidak butuh dokumen pedoman untuk dijawab.
   Contoh: "Halo", "Terima kasih", "Oke mengerti", "Apa itu PI?", "Rangkum percakapan kita"
 
3. "clarification"
   → Meminta elaborasi, contoh, atau penjelasan LEBIH LANJUT dari jawaban sebelumnya.
   → Pertanyaan yang jawabannya sudah bisa dibuat dari konteks history percakapan.
   Contoh: "Bisa jelaskan lebih detail?", "Kasih contohnya dong",
           "Mengapa begitu?", "Maksudnya apa?", "Bagaimana kalau kasusnya berbeda sedikit?"
 
Format output WAJIB (JSON saja, tidak ada teks lain):
{
  "intent": "needs_retrieval" | "conversational" | "clarification",
  "reason": "alasan singkat dalam 1 kalimat",
  "confidence": 0.0-1.0
}"""

def _build_classifier_prompt(
    current_message: str,
    memory: ConversationMemory,
) -> str:
    """build the prompt for intent classifier"""
    parts = []
 
    if not memory.is_empty:
        last_q = memory.get_last_question()
        last_a = memory.get_last_answer()
 
        if last_q and last_a:
            q_short = last_q[:150] + "..." if len(last_q) > 150 else last_q
            a_short = last_a[:200] + "..." if len(last_a) > 200 else last_a
 
            parts.append("=== PERCAKAPAN TERAKHIR ===")
            parts.append(f"User sebelumnya: {q_short}")
            parts.append(f"Asisten menjawab: {a_short}")
            parts.append("")
 
    parts.append(f"=== PESAN USER SEKARANG ===")
    parts.append(current_message)
    parts.append("")
    parts.append("Tentukan intent pesan user sekarang. Output hanya JSON.")
 
    return "\n".join(parts)

class IntentClassifier:
    """
    Classifier LLM based for choosing the intent of each user message.
    """
 
    def __init__(self):
        self._llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=0,
            api_key=settings.open_api_key,  
            max_tokens=150,
        )
        self._cache: dict[str, IntentType] = {}
 
    def classify(
        self,
        message: str,
        memory: ConversationMemory,
    ) -> tuple[IntentType, float, str]:
        """
        clarifying intent of user message.
        """
        if len(message.strip()) <= 9 and not any(
            kw in message.lower()
            for kw in ["apa", "bagaimana", "berapa", "kapan", "siapa", "kenapa", "mengapa"]
        ):
            logger.debug(f"Shortcut → CONVERSATIONAL (pesan sangat pendek: '{message}')")
            return IntentType.CONVERSATIONAL, 0.95, "Pesan terlalu pendek untuk butuh retrieval"
 
        if memory.is_empty:
            logger.debug("Shortcut → NEEDS_RETRIEVAL (tidak ada history)")
            return IntentType.NEEDS_RETRIEVAL, 0.99, "Pertanyaan pertama selalu butuh retrieval"
 
        cache_key = f"{message[:50]}|{memory.turn_count}"
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            logger.debug(f"Cache hit → {cached.value}")
            return cached, 0.9, "Dari cache"
 
        prompt = _build_classifier_prompt(message, memory)
 
        try:
            response = self._llm.invoke([
                SystemMessage(content=_CLASSIFIER_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ])
 
            raw = response.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
 
            parsed = json.loads(raw)
 
            intent_str = parsed.get("intent", "needs_retrieval")
            confidence = float(parsed.get("confidence", 0.8))
            reason = parsed.get("reason", "")
 
            try:
                intent = IntentType(intent_str)
            except ValueError:
                logger.warning(f"Intent tidak dikenal '{intent_str}', fallback ke NEEDS_RETRIEVAL")
                intent = IntentType.NEEDS_RETRIEVAL
 
            self._cache[cache_key] = intent
 
            logger.info(
                f"🎯 Intent: {intent.value} "
                f"(conf: {confidence:.2f}) | {reason}"
            )
 
            return intent, confidence, reason
 
        except (json.JSONDecodeError, KeyError, Exception) as e:
            logger.warning(f"Classifier error: {e} → fallback NEEDS_RETRIEVAL")
            return IntentType.NEEDS_RETRIEVAL, 0.5, f"Fallback karena error: {e}"


_REFORMULATION_PROMPT = """Anda membantu sistem pencarian dokumen internal.
 
Riwayat percakapan:
{history}
 
Pertanyaan terkini user: "{question}"
 
Tugas: Jika pertanyaan terkini menggunakan referensi implisit seperti "itu",
"tersebut", "yang tadi", "lebih detail tentang itu", maka tulis ulang menjadi
pertanyaan yang BERDIRI SENDIRI dan lengkap untuk digunakan sebagai query pencarian.
 
Jika pertanyaan sudah jelas dan mandiri, kembalikan persis sama.
 
Output: HANYA pertanyaan yang sudah ditulis ulang, tanpa penjelasan apapun."""
 
 
def reformulate_query(
    message: str,
    memory: ConversationMemory,
    llm: ChatOpenAI | None = None,
) -> str:
    """
    write again query for standalone use.
    """
    if memory.is_empty:
        return message
 
    implicit_refs = [
        "itu", "tersebut", "tadi", "yang itu", "hal itu",
        "lebih detail", "jelaskan lagi", "elaborasi", "lanjutkan",
        "bagaimana dengan", "kalau untuk", "dan untuk", "gimana kalau",
    ]
 
    has_implicit = any(ref in message.lower() for ref in implicit_refs)
 
    if not has_implicit:
        return message
 
    if llm is None:
        llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=0,
            api_key=settings.open_api_key,  
            max_tokens=100,
        )
 
    history_text = memory.get_conversation_summary()
 
    prompt = _REFORMULATION_PROMPT.format(
        history=history_text,
        question=message,
    )
 
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        reformulated = response.content.strip()
 
        if reformulated and reformulated != message:
            logger.info(f"🔄 Query reformulated: '{message}' → '{reformulated}'")
 
        return reformulated or message
 
    except Exception as e:
        logger.warning(f"Reformulation gagal: {e} → pakai query original")
        return message