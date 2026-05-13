import json
import re
from typing import Set, List, Tuple
 
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from loguru import logger
 
from config.settings import get_settings
from src.generation.memory import ConversationMemory, IntentType

settings = get_settings()

# Topic switching signals
TOPIC_SWITCH_SIGNALS = {
    "explicit": [
        "sekarang", "now", "bagaimana dengan", "how about", "kalau untuk", "what about",
        "lalu untuk", "then for", "selanjutnya", "next", "ganti topik", "change topic",
        "berbeda", "different", "lain", "other", "bukan", "not", "tapi", "but"
    ],
    "domain_keywords": {
        "pi": ["pi", "penulisan ilmiah", "penelitian", "skripsi", "thesis"],
        "kkp": ["kkp", "kuliah kerja praktik", "magang", "internship", "praktik"]
    }
}

# True clarification signals
TRUE_CLARIFICATION_SIGNALS = [
    "lebih detail", "more detail", "jelaskan lagi", "explain again", 
    "elaborasi", "elaborate", "contoh", "example", "maksudnya", "meaning",
    "mengapa", "why", "kenapa", "bagaimana cara", "how to", "bisa dijelaskan",
    "can you explain", "apa maksud", "what does it mean"
]

_CLASSIFIER_SYSTEM_PROMPT = """Anda adalah classifier yang menganalisis pesan user \
dalam sistem Q&A Kuliah Kerja Praktek (KKP) dan Penelitian Ilmiah (PI) di STMIK Widya Cipta Dharma.

PENTING: Perhatikan context switching dan topic switching dengan cermat!

Tugas Anda: tentukan intent pesan user dan kembalikan HANYA JSON, tidak ada penjelasan lain.
 
Tiga kategori intent:
 
1. "needs_retrieval"
   → Pertanyaan spesifik yang butuh informasi dari dokumen pedoman
   → Pertanyaan tentang topik BARU yang berbeda dari history
   → Pertanyaan yang beralih domain (PI ↔ KKP)
   → Pertanyaan yang beralih aspek dalam domain yang sama
   → Mengandung signal switching: "sekarang", "bagaimana dengan", "kalau untuk", "lalu untuk"
   Contoh: "Berapa lama minimal KKP?", "Bagaimana dengan syarat PI?", "Sekarang tentang format laporan"
 
2. "conversational"
   → Sapaan, ucapan terima kasih, pertanyaan sangat umum
   → Perintah yang tidak butuh dokumen pedoman
   Contoh: "Halo", "Terima kasih", "Oke mengerti", "Apa itu PI secara umum?"
 
3. "clarification"
   → HANYA untuk elaborasi/penjelasan lebih lanjut dari jawaban yang SAMA PERSIS
   → Pertanyaan yang jawabannya sudah ada di konteks history
   → TIDAK untuk topic switching atau domain switching
   → Mengandung signal clarification: "lebih detail", "jelaskan lagi", "contoh", "maksudnya"
   Contoh: "Bisa jelaskan lebih detail tentang syarat yang tadi?", "Kasih contoh untuk hal yang sama"

ATURAN KHUSUS:
- Jika ada signal switching ("bagaimana dengan", "kalau untuk", "sekarang") → SELALU "needs_retrieval"
- Jika beralih domain (PI→KKP atau KKP→PI) → SELALU "needs_retrieval"  
- Jika beralih aspek (syarat→format, durasi→prosedur) → SELALU "needs_retrieval"
- Clarification HANYA untuk elaborasi topik yang SAMA PERSIS

Format output WAJIB (JSON saja, tidak ada teks lain):
{
  "intent": "needs_retrieval" | "conversational" | "clarification",
  "reason": "alasan singkat dalam 1 kalimat",
  "confidence": 0.0-1.0,
  "topic_switch_detected": true/false,
  "domain_switch_detected": true/false
}"""


def _detect_topic_switch(current_message: str, memory: ConversationMemory) -> Tuple[bool, bool, str]:
    """
    Detect topic switching and domain switching
    Returns: (topic_switch, domain_switch, reason)
    """
    if not memory.has_prior_context:
        return False, False, "No prior context to compare"
    
    message_lower = current_message.lower()
    
    # Check for explicit switching signals
    explicit_signals = [signal for signal in TOPIC_SWITCH_SIGNALS["explicit"] 
                       if signal in message_lower]
    
    if explicit_signals:
        return True, False, f"Explicit switch signal detected: {explicit_signals[0]}"
    
    # Check for domain switching (PI ↔ KKP)
    last_answer = memory.get_last_answer() or ""
    previous_question = memory.get_previous_question() or ""
    
    # Detect current domain from the new message
    current_domain = None
    for domain, keywords in TOPIC_SWITCH_SIGNALS["domain_keywords"].items():
        if any(keyword in message_lower for keyword in keywords):
            current_domain = domain
            break
    
    # Detect previous domain from the last answer + previous question
    previous_context = (last_answer + " " + previous_question).lower()
    previous_domain = None
    for domain, keywords in TOPIC_SWITCH_SIGNALS["domain_keywords"].items():
        if any(keyword in previous_context for keyword in keywords):
            previous_domain = domain
            break
    
    # Domain switch detection
    if current_domain and previous_domain and current_domain != previous_domain:
        return True, True, f"Domain switch detected: {previous_domain} → {current_domain}"
    
    # Check for aspect switching within same domain
    aspect_keywords = {
        "syarat": ["syarat", "requirement", "persyaratan", "kondisi", "minimal"],
        "format": ["format", "struktur", "template", "bentuk", "susunan", "penulisan"],
        "durasi": ["durasi", "lama", "waktu", "periode", "jangka"],
        "prosedur": ["prosedur", "tahap", "langkah", "proses", "cara", "tahapan"],
        "dosen": ["dosen", "pembimbing", "supervisor", "penguji"],
        "tempat": ["tempat", "lokasi", "instansi", "perusahaan"],
        "ujian": ["ujian", "seminar", "sidang", "presentasi"],
        "laporan": ["laporan", "bab", "halaman", "margin", "font"],
    }
    
    # Detect current aspect
    current_aspect = None
    for aspect, keywords in aspect_keywords.items():
        if any(keyword in message_lower for keyword in keywords):
            current_aspect = aspect
            break
    
    # Detect previous aspect from previous question (not current)
    prev_q_lower = previous_question.lower()
    previous_aspect = None
    for aspect, keywords in aspect_keywords.items():
        if any(keyword in prev_q_lower for keyword in keywords):
            previous_aspect = aspect
            break
    
    # Aspect switch detection — only trigger if the PRIMARY topic changed
    # If the previous answer also contains the current aspect keyword, it's likely elaboration
    if current_aspect and previous_aspect and current_aspect != previous_aspect:
        # Check if the current aspect keyword also appears in the last answer
        # If so, it's likely a clarification about something already mentioned
        last_answer_lower = last_answer.lower()
        current_aspect_in_answer = any(
            keyword in last_answer_lower 
            for keyword in aspect_keywords.get(current_aspect, [])
        )
        if not current_aspect_in_answer:
            return True, False, f"Aspect switch detected: {previous_aspect} → {current_aspect}"
    
    return False, False, "No topic switch detected"


def _detect_true_clarification(current_message: str, memory: ConversationMemory) -> Tuple[bool, str]:
    """
    Detect if this is a true clarification (elaboration of same topic)
    Returns: (is_clarification, reason)
    """
    if not memory.has_prior_context:
        return False, "No prior context for clarification"
    
    message_lower = current_message.lower()
    
    # Check for clarification signals
    clarification_signals = [signal for signal in TRUE_CLARIFICATION_SIGNALS 
                           if signal in message_lower]
    
    if not clarification_signals:
        return False, "No clarification signals found"
    
    # Check if it's asking for elaboration without topic change
    topic_switch, domain_switch, _ = _detect_topic_switch(current_message, memory)
    
    if topic_switch or domain_switch:
        return False, "Topic/domain switch detected, not clarification"
    
    return True, f"True clarification signal: {clarification_signals[0]}"


def _build_classifier_prompt(
    current_message: str,
    memory: ConversationMemory,
    topic_switch: bool = False,
    domain_switch: bool = False,
    switch_reason: str = "",
) -> str:
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
    
    # Add switching detection info
    if topic_switch or domain_switch:
        parts.append("=== ANALISIS SWITCHING ===")
        parts.append(f"Topic switch detected: {topic_switch}")
        parts.append(f"Domain switch detected: {domain_switch}")
        parts.append(f"Reason: {switch_reason}")
        parts.append("")
 
    parts.append(f"=== PESAN USER SEKARANG ===")
    parts.append(current_message)
    parts.append("")
    parts.append("Tentukan intent pesan user sekarang. Output hanya JSON.")
 
    return "\n".join(parts)


class IntentClassifier:
 
    def __init__(self):
        self._llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=0,
            api_key=settings.open_api_key,  
            max_tokens=200,
        )
        self._cache: dict[str, IntentType] = {}
 
    def classify(
        self,
        message: str,
        memory: ConversationMemory,
    ) -> Tuple[IntentType, float, str]:
        message_lower = message.strip().lower()
        
        # Quick shortcut: very short messages without question keywords → CONVERSATIONAL
        if len(message.strip()) <= 9 and not any(
            kw in message_lower
            for kw in ["apa", "bagaimana", "berapa", "kapan", "siapa", "kenapa", "mengapa"]
        ):
            logger.debug(f"Shortcut → CONVERSATIONAL (pesan sangat pendek: '{message}')")
            return IntentType.CONVERSATIONAL, 0.95, "Pesan terlalu pendek untuk butuh retrieval"
        
        # Quick shortcut: obvious conversational patterns (greetings, thanks)
        conversational_patterns = [
            "halo", "hai", "hello", "hi", "hey",
            "selamat pagi", "selamat siang", "selamat sore", "selamat malam",
            "terima kasih", "makasih", "thanks", "thank you",
            "oke", "ok", "baik", "siap", "mengerti", "paham",
            "sampai jumpa", "bye", "dadah",
        ]
        if any(pattern in message_lower for pattern in conversational_patterns):
            # Only if the message is primarily a greeting/thanks (not mixed with a question)
            has_question_word = any(
                kw in message_lower
                for kw in ["apa", "bagaimana", "berapa", "kapan", "siapa", "kenapa", "mengapa", "dimana"]
            )
            if not has_question_word:
                logger.debug(f"Shortcut → CONVERSATIONAL (greeting/thanks pattern)")
                return IntentType.CONVERSATIONAL, 0.95, "Pesan sapaan atau ucapan terima kasih"
 
        # If no prior context (first real question), go straight to retrieval
        if not memory.has_prior_context:
            logger.debug("Shortcut → NEEDS_RETRIEVAL (tidak ada prior context)")
            return IntentType.NEEDS_RETRIEVAL, 0.99, "Pertanyaan pertama selalu butuh retrieval"
 
        # Detect topic and domain switching
        topic_switch, domain_switch, switch_reason = _detect_topic_switch(message, memory)
        
        # If switching detected, force needs_retrieval
        if topic_switch or domain_switch:
            logger.info(f"🔄 Switching detected → NEEDS_RETRIEVAL: {switch_reason}")
            return IntentType.NEEDS_RETRIEVAL, 0.95, switch_reason
        
        # Check for true clarification
        is_clarification, clarification_reason = _detect_true_clarification(message, memory)
        
        if is_clarification:
            logger.info(f"💬 True clarification detected → CLARIFICATION: {clarification_reason}")
            return IntentType.CLARIFICATION, 0.90, clarification_reason
 
        # Use LLM for complex cases
        cache_key = f"{message[:50]}|{memory.turn_count}"
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            logger.debug(f"Cache hit → {cached.value}")
            return cached, 0.9, "Dari cache"
 
        prompt = _build_classifier_prompt(message, memory, topic_switch, domain_switch, switch_reason)
 
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
            
            # Override LLM decision if switching was detected
            if topic_switch or domain_switch and intent_str != "needs_retrieval":
                logger.warning(f"LLM classified as {intent_str} but switching detected, overriding to needs_retrieval")
                intent_str = "needs_retrieval"
                confidence = 0.95
                reason = f"Override: {switch_reason}"
 
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