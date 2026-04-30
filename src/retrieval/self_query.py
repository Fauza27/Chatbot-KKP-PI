from __future__ import annotations

from dataclasses import dataclass
from loguru import logger
from config.settings import get_settings
from langchain.chains.query_constructor.base import AttributeInfo

settings = get_settings()

METADATA_FIELD_INFO = [
    AttributeInfo(
        name="section",
        description=(
            "Bagian/bab dari dokumen Panduan PI atau KKP. Nilai yang valid: "
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
            "'Panduan Penyusunan Penulisan Imliah (PI) Cetak' untuk dokumen PI, "
            "'Panduan Penyusunan Kuliah Kerja Praktik (KKP) Cetak' untuk dokumen KKP."
        ),
        type="string",
    ),
]

DOCUMENT_CONTENT_DESCRIPTION = (
    "Dokumen panduan akademik dari STMIK Widya Cipta Dharma yang terdiri dari dua jenis: "
    "(1) Panduan Penulisan Ilmiah (PI) — mencakup ketentuan umum, sistematika laporan PI, "
    "format penulisan, sistem penilaian, dan contoh formulir untuk PI; "
    "(2) Panduan Kuliah Kerja Praktik (KKP) — mencakup ketentuan umum KKP, "
    "sistematika laporan KKP, prosedur pengajuan dan ujian KKP, "
    "sistem penilaian KKP, dan contoh formulir untuk KKP. "
    "Keduanya membahas: dosen pembimbing/penguji, syarat mahasiswa (SKS, IPK), "
    "prosedur pengajuan, pelaksanaan, pendaftaran ujian, penilaian, kelulusan, "
    "format penulisan (margin, huruf, spasi, daftar pustaka APA), dan lampiran formulir."
)


@dataclass
class ParsedQuery:
    """Hasil parsing query oleh Self-Query."""
    semantic_query: str
    filters: dict
    original_query: str



def build_self_query_retriever(
    supabase_client=None,
    top_k: int | None = None,
) -> "SelfQueryRetriever": 
    """
    Bangun Self-Query Retriever yang terhubung ke Supabase.
    """
    from langchain.retrievers.self_query.base import SelfQueryRetriever
    from langchain_community.query_constructors.supabase import SupabaseVectorTranslator
    from langchain_community.vectorstores import SupabaseVectorStore
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from supabase import create_client

    if supabase_client is None:
        supabase_client = create_client(
            settings.supabase_url, settings.supabase_service_key
        )

    k = top_k or settings.retrieval_top_k

    embedder = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.open_api_key,
        dimensions=2000,
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


def extract_query_components(query: str) -> ParsedQuery:
    """Ekstrak semantic query dan filter dari query user."""
    logger.debug(f"Menganalisis query: '{query}'")

    filters: dict = {}
    semantic = query
    query_lower = query.lower()

    section_keywords: dict[str, list[str]] = {
        "BAB I": [
            "latar belakang panduan", "tujuan panduan", "pendahuluan panduan",
        ],
        "BAB II": [
            "dosen pembimbing", "dosen penguji", "syarat", "ketentuan",
            "prosedur", "ujian pi", "ujian kkp", "penilaian", "kelulusan",
            "nilai", "bimbingan", "sks", "ipk", "plagiarisme", "seminar",
            "pendaftaran", "penggantian dosen", "masa bimbingan",
            "tempat penelitian", "tempat kkp", "pengajuan judul",
            "kriteria tempat", "durasi kkp", "lama kkp", "hari kerja",
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
            "narasi kegiatan", "deskripsi kegiatan", "analisis hasil kegiatan",
            "sejarah tempat kkp", "profil tempat",
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
            "rekapitulasi", "surat tugas", "daftar hadir",
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