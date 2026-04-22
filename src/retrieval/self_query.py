from dataclasses import dataclass

from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.query_constructors.supabase import SupabaseVectorTranslator
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from loguru import logger
from supabase import create_client

from config.settings import get_settings

settings = get_settings()


# ── Deskripsi atribut metadata yang bisa difilter ───────────────────────────
# Ini yang LLM gunakan untuk "memahami" field apa saja yang tersedia.
# Semakin detail deskripsi → semakin akurat filter yang dihasilkan.
METADATA_FIELD_INFO = [
    AttributeInfo(
        name="section",
        description=(
            "Bagian/bab dari dokumen Panduan PI. Nilai yang valid: "
            "'Front Matter' (kata pengantar, daftar isi/tabel/gambar), "
            "'Surat Keputusan' (SK pemberlakuan panduan), "
            "'BAB I' (pendahuluan: latar belakang, tujuan panduan), "
            "'BAB II' (ketentuan umum: dosen pembimbing, dosen penguji, "
            "syarat mahasiswa, prosedur pengajuan, pelaksanaan, ujian, penilaian, kelulusan), "
            "'BAB III' (sistematika penyusunan: gambaran umum, struktur laporan), "
            "'BAB IV' (penjelasan sistematika: bagian awal/utama/akhir, isi tiap BAB I-V), "
            "'BAB V' (format penulisan: kertas, margin, huruf, spasi, tabel, gambar, "
            "bahasa, daftar pustaka APA), "
            "'Lampiran' (contoh format dan formulir: sampul, pengesahan, abstrak, "
            "kata pengantar, daftar isi, jadwal, wawancara, bimbingan, ujian, penilaian)"
        ),
        type="string",
    ),
    AttributeInfo(
        name="title",
        description=(
            "Judul spesifik dari bagian dokumen, misalnya: "
            "'BAB II: Dosen Pembimbing (Ketentuan)', "
            "'BAB V: Aturan Penulisan (Margin, Huruf, Spasi, Alinea, Bab)', "
            "'Lampiran 3: Contoh Abstrak'. "
            "Gunakan filter ini jika pengguna menyebut topik spesifik."
        ),
        type="string",
    ),
    AttributeInfo(
        name="source",
        description=(
            "Nama file sumber dokumen. Nilai yang mungkin: "
            "'cleaned_extracted_text.txt' atau "
            "'Panduan Penyusunan Penulisan Imliah (PI) Cetak.md'"
        ),
        type="string",
    ),
]

# Deskripsi koleksi dokumen — konteks untuk LLM saat generate filter
DOCUMENT_CONTENT_DESCRIPTION = (
    "Dokumen panduan penyusunan Penulisan Ilmiah (PI) dari STMIK Widya Cipta Dharma, "
    "mencakup: ketentuan umum (syarat, dosen pembimbing/penguji, prosedur), "
    "sistematika penulisan laporan (BAB I-V), format dan tata cara penulisan "
    "(kertas, margin, huruf, spasi, tabel, gambar, bahasa, daftar pustaka), "
    "sistem penilaian dan kelulusan, serta contoh format dan formulir (lampiran)."
)


@dataclass
class ParsedQuery:
    """result parsing query by Self-Query."""
    semantic_query: str         
    filters: dict               
    original_query: str         


def build_self_query_retriever(
    supabase_client=None,
    top_k: int | None = None,
) -> SelfQueryRetriever:
    """
    Bangun Self-Query Retriever yang terhubung ke Supabase.

    Alur internal retriever ini:
    1. LLM menerima query + deskripsi metadata
    2. LLM menghasilkan structured query (semantic + filters)
    3. Filters diterjemahkan ke SQL WHERE oleh SupabaseVectorTranslator
    4. Vector search dijalankan dengan filter aktif

    Args:
        supabase_client: Supabase client (buat baru jika None)
        top_k: Jumlah dokumen yang dikembalikan (override config)

    Returns:
        SelfQueryRetriever yang siap dipanggil
    """
    if supabase_client is None:
        supabase_client = create_client(settings.supabase_url, settings.supabase_service_key)

    k = top_k or settings.retrieval_top_k

    embedder = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.open_api_key,
    )

    vector_store = SupabaseVectorStore(
        client=supabase_client,
        embedding=embedder,
        table_name=settings.table_child_chunks,
        query_name="match_documents",
    )

    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0,
        api_key=settings.open_api_key,
    )

    retriever = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vector_store,
        document_contents=DOCUMENT_CONTENT_DESCRIPTION,
        metadata_field_info=METADATA_FIELD_INFO,
        structured_query_translator=SupabaseVectorTranslator(),
        search_kwargs={"k": k},
        verbose=True,
    )

    return retriever


def extract_query_components(query: str, llm: ChatOpenAI | None = None) -> ParsedQuery:
    """
    Utilitas: ekstrak semantic query dan filter dari query user.
    Berguna untuk debugging dan logging.

    Ini menganalisis query secara heuristik berdasarkan keyword
    untuk melihat mapping section yang tepat, tanpa menjalankan
    retrieval penuh.

    Args:
        query: Query natural language dari user
        llm: ChatOpenAI instance (opsional, untuk future use)

    Returns:
        ParsedQuery dengan komponen yang terpisah
    """
    logger.debug(f"Menganalisis query: '{query}'")

    filters = {}
    semantic = query
    query_lower = query.lower()

    section_keywords = {
        "BAB I": [
            "latar belakang panduan", "tujuan panduan", "pendahuluan panduan",
        ],
        "BAB II": [
            "dosen pembimbing", "dosen penguji", "syarat", "ketentuan",
            "prosedur", "ujian pi", "penilaian", "kelulusan", "nilai",
            "bimbingan", "sks", "ipk", "plagiarisme", "seminar",
            "pendaftaran", "penggantian dosen", "masa bimbingan",
            "tempat penelitian", "pengajuan judul",
        ],
        "BAB III": [
            "sistematika penulisan", "gambaran umum laporan",
            "struktur laporan", "bagian awal", "bagian utama", "bagian akhir",
        ],
        "BAB IV": [
            "pendahuluan bab", "rumusan masalah", "batasan masalah",
            "tujuan penelitian", "manfaat penelitian", "tinjauan pustaka",
            "kajian empiris", "landasan teori", "metode penelitian",
            "hasil penelitian", "pembahasan", "kesimpulan", "saran",
            "latar belakang masalah",
        ],
        "BAB V": [
            "format", "margin", "huruf", "spasi", "kertas", "font",
            "times new roman", "tabel", "gambar", "bahasa", "eyd",
            "daftar pustaka", "apa", "penomoran halaman", "alinea",
            "angka", "huruf miring",
        ],
        "Lampiran": [
            "contoh", "lampiran", "sampul", "cover", "pengesahan",
            "abstrak", "kata pengantar", "daftar isi", "jadwal",
            "wawancara", "form", "formulir", "berita acara",
            "rekapitulasi", "surat tugas",
        ],
    }

    for section, keywords in section_keywords.items():
        if any(kw in query_lower for kw in keywords):
            filters["section"] = section
            break

    logger.info(f"Query dianalisis — semantic: '{semantic}' | filters: {filters}")

    return ParsedQuery(
        semantic_query=semantic,
        filters=filters,
        original_query=query,
    )