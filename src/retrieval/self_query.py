from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal
from loguru import logger
from config.settings import get_settings

settings = get_settings()

def _build_metadata_field_info():
    from langchain.chains.query_constructor.base import AttributeInfo

    return [
        AttributeInfo(
            name="section",
            description=(
                "Bagian/bab dari dokumen Panduan PI atau KKP. Nilai yang valid: "
                "'Front Matter' (kata pengantar, daftar isi, daftar tabel, daftar gambar), "
                "'Surat Keputusan' (SK pemberlakuan panduan dari Ketua STMIK), "
                "'BAB I' (pendahuluan: latar belakang panduan, tujuan panduan), "
                "'BAB II' (ketentuan umum: dosen pembimbing, dosen penguji, mahasiswa bimbingan, "
                "syarat PI/KKP, tempat penelitian/KKP, proses penyusunan, prosedur ujian, "
                "tahap akhir, sistem penilaian, kelulusan), "
                "'BAB III' (sistematika penyusunan: gambaran umum laporan, struktur laporan PI/KKP), "
                "'BAB IV' (penjelasan sistematika penulisan: bagian awal, bagian utama, bagian akhir, "
                "isi tiap BAB I-V untuk PI atau BAB I-V untuk KKP), "
                "'BAB V' (format dan tata cara penulisan: kertas, margin, huruf, spasi, alinea, bab, "
                "angka, penomoran halaman, tabel, gambar, bahasa, huruf miring, daftar pustaka APA), "
                "'Lampiran' (contoh format dan formulir: halaman awal, daftar, jadwal, wawancara, "
                "bimbingan, administrasi ujian, persetujuan, perbaikan, berita acara, penilaian, "
                "perubahan pembimbing/judul)"
            ),
            type="string",
        ),
        AttributeInfo(
            name="title",
            description=(
                "Judul spesifik dari bagian dokumen. Contoh untuk PI: "
                "'Front Matter', 'Surat Keputusan', 'BAB I Pendahuluan', "
                "'BAB II Dosen Pembimbing dan Penguji', 'BAB II Mahasiswa Bimbingan', "
                "'BAB II Syarat dan Tempat Penelitian', 'BAB II Proses Penyusunan', "
                "'BAB II Prosedur Ujian dan Tahap Akhir', 'BAB II Sistem Penilaian dan Kelulusan', "
                "'BAB III Sistematika Penulisan PI', 'BAB IV Bagian Awal', 'BAB IV Bagian Utama dan Akhir', "
                "'BAB V Format Penulisan Naskah', 'BAB V Tabel dan Gambar', 'BAB V Bahasa dan Huruf Miring', "
                "'BAB V Daftar Pustaka', 'Lampiran Contoh Halaman Awal', 'Lampiran Contoh Daftar', "
                "'Lampiran Jadwal dan Bimbingan', 'Lampiran Administrasi Ujian', "
                "'Lampiran Persetujuan dan Perbaikan', 'Lampiran Berita Acara dan Penilaian', "
                "'Lampiran Perubahan Pembimbing/Judul'. "
                "Contoh untuk KKP: 'BAB II Dosen Pembimbing dan Penguji', 'BAB II Mahasiswa Bimbingan', "
                "'BAB II Syarat dan Tempat KKP', 'BAB II Proses Penyusunan', "
                "'BAB II Prosedur Ujian dan Tahap Akhir', 'BAB III Sistematika Penulisan KKP', "
                "'Lampiran Tugas dan Bimbingan', 'Lampiran Administrasi Ujian', dll. "
                "Gunakan filter ini jika pengguna menyebut topik spesifik."
            ),
            type="string",
        ),
        AttributeInfo(
            name="source",
            description=(
                "Nama file sumber dokumen. Nilai yang valid: "
                "'Panduan Penyusunan Penulisan Imliah (PI) Cetak' untuk dokumen Penulisan Ilmiah (PI), "
                "'Panduan Penyusunan Kuliah Kerja Praktik (KKP) Cetak' untuk dokumen Kuliah Kerja Praktik (KKP). "
                "Gunakan filter ini jika pengguna secara eksplisit menyebut 'PI' atau 'KKP' atau 'Penulisan Ilmiah' "
                "atau 'Kuliah Kerja Praktik'."
            ),
            type="string",
        ),
    ]

DOCUMENT_CONTENT_DESCRIPTION = (
    "Dokumen panduan akademik dari STMIK Widya Cipta Dharma yang terdiri dari dua jenis: "
    "(1) Panduan Penyusunan Penulisan Ilmiah (PI) — panduan untuk mahasiswa S1 yang telah menyelesaikan "
    "minimal 100 SKS dengan IPK minimal 2.00 untuk melakukan penelitian ilmiah mandiri di instansi/organisasi "
    "atau studi literatur. Mencakup: ketentuan dosen pembimbing dan penguji, hak mahasiswa bimbingan, "
    "mekanisme pembimbingan (minimal 8 kali), syarat dan tempat penelitian (minimal 7 hari kerja), "
    "proses penyusunan (pengajuan judul, pelaksanaan penelitian, konsultasi dengan BKK), "
    "prosedur ujian (berkas administrasi, seminar terbuka, durasi 60 menit, pakaian formal), "
    "sistem penilaian (skala 100: A=80-100, B=70-79, C=60-69, D=40-59, E=0-39), "
    "komponen penilaian (orisinalitas, sistematika, penguasaan materi, argumentasi, penampilan), "
    "capaian pembelajaran per program studi (SI, TI, Bisnis Digital), "
    "sistematika penulisan laporan PI (bagian awal: sampul, pengesahan, abstrak, kata pengantar, daftar; "
    "bagian utama: BAB I Pendahuluan dengan latar belakang masalah/rumusan/batasan/tujuan/manfaat/sistematika, "
    "BAB II Tinjauan Pustaka dengan kajian empiris dan landasan teori, "
    "BAB III Metode Penelitian dengan tempat/waktu/teknik pengumpulan data/metode pengembangan sistem, "
    "BAB IV Hasil Penelitian dan Pembahasan, BAB V Penutup dengan kesimpulan dan saran; "
    "bagian akhir: daftar pustaka dan lampiran), "
    "format penulisan (kertas HVS A4 80 gram, margin atas 3cm bawah 3cm kiri 4cm kanan 3cm, "
    "Times New Roman 12, spasi 1.5, alinea menjorok 1.2cm, penomoran halaman romawi kecil untuk bagian awal "
    "dan angka arab untuk bagian utama), aturan tabel dan gambar (penomoran dua angka, judul tabel di atas "
    "dan judul gambar di bawah, spasi 1, sumber pustaka), bahasa Indonesia baku EYD Edisi Kelima dengan "
    "kalimat pasif, istilah asing dicetak miring, daftar pustaka APA (minimal 15 referensi, 80% dari buku "
    "dan jurnal, maksimal 5 tahun terakhir, hanging indent, DOI/URL untuk referensi elektronik), "
    "serta lampiran formulir (contoh halaman awal, daftar, jadwal penelitian, daftar wawancara, "
    "form bimbingan, daftar hadir menyaksikan ujian, pengecekan syarat administrasi, permohonan ujian, "
    "pendaftaran ujian, persetujuan ujian, persetujuan waktu ujian, perbaikan ujian, berita acara ujian, "
    "penilaian ujian, rekapitulasi penilaian, permohonan penggantian dosen pembimbing, "
    "surat tugas penugasan dosen pengganti, permohonan penggantian judul). "
    "(2) Panduan Penyusunan Kuliah Kerja Praktik (KKP) — panduan untuk mahasiswa S1 yang telah menyelesaikan "
    "minimal 100 SKS dengan IPK minimal 2.00 untuk melakukan praktik kerja di instansi resmi yang relevan "
    "dengan bidang keilmuan (Sistem Informasi Manajemen, Teknik Komputer, Teknik Pemrograman, "
    "Digital Marketing, Entrepreneurship). Mencakup: ketentuan dosen pembimbing dan penguji "
    "(1 pembimbing yaitu Dosen Wali, 2 penguji, jabatan minimal asisten ahli/lektor S2/S3), "
    "hak mahasiswa bimbingan, mekanisme pembimbingan (minimal 8 kali, maksimal 6 bulan, "
    "lembar bimbingan wajib dibawa), syarat dan tempat KKP (instansi resmi dengan aktivitas operasional aktif, "
    "pembimbing lapangan, rekomendasi BKK, kuota maksimal mahasiswa per instansi), "
    "proses penyusunan (pendaftaran mata kuliah, konsultasi BKK, persetujuan instansi, "
    "surat pengantar BAAK, surat balasan instansi, lembar kehadiran, pelaksanaan minimal 30 hari kerja/1 bulan, "
    "lapor ke pembimbing lapangan dan dosen pembimbing, dokumentasi foto, seminar KKP), "
    "prosedur ujian (berkas administrasi lengkap: form pendaftaran, persetujuan ujian, form bimbingan, "
    "transkrip, KRS, kuitansi BPP/SKS, surat keterangan penelitian, presensi dan nilai dari tempat penelitian, "
    "daftar rincian kegiatan, daftar wawancara, draf laporan, persetujuan waktu ujian, "
    "bukti anti-plagiarisme maksimal 30%, penunjukan ketua dan anggota penguji, "
    "seminar terbuka maksimal 15 hari setelah persetujuan, durasi 60 menit, pakaian formal), "
    "sistem penilaian (skala 100: A=80-100, B=70-79, C=60-69, D=40-59, E=0-39), "
    "komponen penilaian (orisinalitas, sistematika, penguasaan materi, argumentasi, penampilan), "
    "penilaian dari dosen internal dan pembimbing lapangan, "
    "capaian pembelajaran per program studi (SI, TI, Bisnis Digital), "
    "kelulusan (karya otentik, nilai minimal C, perbaikan naskah disetujui, jilid KKP diserahkan ke perpustakaan), "
    "sistematika penulisan laporan KKP (bagian awal: sampul depan, halaman pengesahan, kata pengantar, "
    "daftar isi/tabel/gambar/lampiran; bagian utama: BAB I Pendahuluan dengan deskripsi tempat/waktu/lokasi KKP, "
    "BAB II Sejarah dan Profil Tempat KKP, BAB III Narasi Kegiatan, BAB IV Analisis Hasil Kegiatan, "
    "BAB V Penutup dengan kesimpulan dan saran; bagian akhir: lampiran pendukung seperti surat pengantar, "
    "surat balasan, form penilaian), format penulisan (kertas HVS A4 80 gram, "
    "margin atas 3cm bawah 3cm kiri 4cm kanan 3cm, Times New Roman 12, spasi 1.5, alinea menjorok 1.2cm, "
    "penomoran bab dan sub-bab, penomoran halaman romawi kecil untuk bagian awal dan angka arab untuk bagian utama), "
    "aturan tabel dan gambar (penomoran dua angka, judul tabel di atas dan judul gambar di bawah, "
    "tabel tidak terpotong, spasi 1, sumber pustaka), bahasa Indonesia baku EYD Edisi Kelima dengan kalimat pasif, "
    "istilah asing dicetak miring, daftar pustaka APA (minimal 5 referensi, 80% dari buku dan jurnal, "
    "hanging indent, DOI/URL untuk referensi elektronik), serta lampiran formulir "
    "(contoh sampul depan, halaman pengesahan, kata pengantar, daftar isi/tabel/gambar/lampiran, "
    "daftar pustaka, rincian tugas KKP, form daftar hadir KKP, form penilaian tingkat kepuasan pengguna, "
    "form bimbingan laporan KKP, form daftar hadir menyaksikan ujian, form pengecekan syarat administrasi, "
    "form permohonan ujian KKP, form pendaftaran ujian KKP, lembar persetujuan ujian KKP, "
    "lembar persetujuan waktu ujian KKP, lembar perbaikan ujian KKP, berita acara ujian KKP, "
    "form penilaian ujian KKP, form rekapitulasi penilaian ujian KKP, "
    "form permohonan penggantian dosen pembimbing, surat penugasan dosen pengganti, "
    "form permohonan penggantian judul KKP). "
    "Kedua panduan mengikuti Surat Keputusan Ketua STMIK Widya Cipta Dharma yang ditetapkan di Samarinda "
    "pada 20 Januari 2026, mengacu pada Undang-Undang Pendidikan Tinggi, Statuta STMIK WCD tahun 2024, "
    "dan Buku Pedoman Akademik STMIK WCD tahun 2025."
)


@dataclass
class ParsedQuery:
    semantic_query: str
    filters: dict
    original_query: str
    detected_source: str | None = None
    detected_section: str | None = None
    confidence: str = "medium"  # low, medium, high



def build_self_query_retriever(
    supabase_client=None,
    top_k: int | None = None,
) -> "SelfQueryRetriever":
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

    metadata_field_info = _build_metadata_field_info()

    retriever = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vector_store,
        document_contents=DOCUMENT_CONTENT_DESCRIPTION,
        metadata_field_info=metadata_field_info,
        structured_query_translator=SupabaseVectorTranslator(),
        search_kwargs={"k": k},
        verbose=True,
    )

    return retriever


def extract_query_components(query: str) -> ParsedQuery:
    logger.debug(f"Menganalisis query: '{query}'")

    filters: dict = {}
    semantic = query
    query_lower = query.lower()

    # Deteksi source (PI atau KKP) terlebih dahulu
    pi_keywords = [
        "penulisan ilmiah", "penelitian ilmiah", " pi ", "laporan pi",
        "ujian pi", "seminar pi", "panduan pi", "studi literatur",
        "kajian empiris", "landasan teori", "tinjauan pustaka",
        "rumusan masalah", "batasan masalah", "metode penelitian",
        "hasil penelitian", "pembahasan hasil", "untuk pi",
    ]
    
    kkp_keywords = [
        "kuliah kerja praktik", " kkp ", "laporan kkp", "ujian kkp",
        "seminar kkp", "panduan kkp", "tempat kkp", "instansi kkp",
        "pembimbing lapangan", "narasi kegiatan", "analisis hasil kegiatan",
        "sejarah tempat", "profil tempat", "praktik kerja",
        "pengalaman kerja", "kegiatan kkp", "pelaksanaan kkp", "untuk kkp",
    ]

    # Cek apakah query menyebut PI atau KKP secara eksplisit
    is_pi = any(kw in query_lower for kw in pi_keywords)
    is_kkp = any(kw in query_lower for kw in kkp_keywords)

    if is_pi and not is_kkp:
        filters["source"] = "Panduan Penyusunan Penulisan Imliah (PI) Cetak"
    elif is_kkp and not is_pi:
        filters["source"] = "Panduan Penyusunan Kuliah Kerja Praktik (KKP) Cetak"
    # Jika keduanya atau tidak ada yang disebutkan, tidak filter source

    # Mapping keyword ke section yang lebih akurat
    section_keywords: dict[str, list[str]] = {
        "Front Matter": [
            "kata pengantar", "daftar isi", "daftar tabel", "daftar gambar",
            "daftar lampiran", "struktur isi", "struktur panduan",
        ],
        "Surat Keputusan": [
            "surat keputusan", "sk ketua", "menimbang", "mengingat",
            "menetapkan", "pemberlakuan panduan", "statuta",
            "undang-undang pendidikan", "peraturan pemerintah",
        ],
        "BAB I": [
            "latar belakang panduan", "tujuan panduan", "pendahuluan panduan",
            "pentingnya panduan", "fokus panduan", "manfaat panduan",
        ],
        "BAB II": [
            # Dosen
            "dosen pembimbing", "dosen penguji", "ketentuan dosen",
            "syarat dosen", "penggantian dosen", "dosen wali",
            "jabatan fungsional", "asisten ahli", "lektor",
            # Mahasiswa
            "hak mahasiswa", "mahasiswa bimbingan", "mekanisme pembimbingan",
            "lembar bimbingan", "masa bimbingan", "bukti pembimbingan",
            # Syarat
            "syarat pi", "syarat kkp", "sks minimal", "ipk minimal",
            "ip kumulatif", "100 sks", "ipk 2.00", "2,00",
            # Tempat
            "tempat penelitian", "tempat kkp", "instansi", "organisasi",
            "kriteria tempat", "pembimbing lapangan", "bkk",
            "bursa kerja khusus", "mou", "kuota mahasiswa",
            # Proses
            "pengajuan judul", "pengajuan pi", "pengajuan kkp",
            "proses penyusunan", "tahap awal", "pendaftaran mata kuliah",
            "surat pengantar", "surat balasan", "lembar kehadiran",
            "durasi penelitian", "durasi kkp", "30 hari kerja",
            "7 hari kerja", "1 bulan", "minimal 8 kali",
            # Ujian - DIPERLUAS
            "prosedur ujian", "pendaftaran ujian", "berkas administrasi",
            "form pendaftaran", "persetujuan ujian", "seminar",
            "presentasi", "tanya jawab", "pakaian ujian",
            "almamater", "kemeja putih", "durasi ujian", "60 menit",
            "15 hari", "maksimal 15 hari",
            "berkas ujian", "dokumen ujian", "syarat ujian",
            "administrasi ujian", "kelengkapan ujian",
            "pakaian pria", "pakaian wanita", "pakaian mahasiswi",
            "dress code", "ketentuan pakaian", "seragam ujian",
            "celana hitam", "rok hitam", "jilbab hitam",
            "sepatu hitam", "berdasi",
            "halaman minimal", "minimal halaman", "jumlah halaman",
            "40 halaman", "minimal 40", "panjang laporan",
            "tebal laporan", "jumlah minimal",
            "menghadiri seminar", "menyaksikan seminar", "hadir seminar",
            "seminar orang lain", "minimal menghadiri", "wajib menghadiri",
            "daftar hadir", "kehadiran seminar", "3 kali", "tiga kali",
            # Penilaian
            "sistem penilaian", "komponen penilaian", "nilai akhir",
            "skala 100", "predikat nilai", "orisinalitas",
            "sistematika penulisan", "penguasaan materi",
            "kemampuan argumentasi", "presentasi", "penampilan",
            "etika", "nilai a", "nilai b", "nilai c", "nilai d",
            "nilai e", "80-100", "70-79", "60-69",
            # Kelulusan
            "kelulusan", "syarat lulus", "karya otentik",
            "nilai minimal c", "perbaikan naskah", "jilid",
            "perpustakaan", "transkrip", "plagiarisme",
            "anti-plagiarisme", "30%", "kemiripan",
            # Capaian
            "capaian pembelajaran", "capaian profil", "program studi",
            "sistem informasi", "teknik informatika", "bisnis digital",
        ],
        "BAB III": [
            "sistematika penulisan", "sistematika penyusunan",
            "gambaran umum laporan", "struktur laporan",
            "bagian awal laporan", "bagian utama laporan", "bagian akhir laporan",
            "sistematika pi", "sistematika kkp", "sistematika laporan",
        ],
        "BAB IV": [
            # Bagian Awal
            "sampul depan", "cover", "halaman pengesahan",
            "abstrak", "abstract", "kata pengantar",
            "elemen sampul", "isi sampul", "komponen sampul",
            "halaman judul", "judul laporan",
            "isi abstrak", "komponen abstrak", "elemen abstrak",
            "apa yang dimuat abstrak", "isi dari abstrak",
            "abstrak memuat", "abstrak berisi", "abstrak terdiri",
            "dimuat dalam abstrak", "struktur abstrak", "format abstrak",
            "kata kunci", "keywords", "300 kata", "3 alinea",
            "bahasa indonesia dan inggris", "dua bahasa",
            "pengesahan", "form pengesahan", "lembar pengesahan",
            "dimuat dalam pengesahan", "isi pengesahan",
            "elemen pengesahan", "komponen pengesahan",
            "tanda tangan pengesahan", "halaman pengesahan memuat",
            # Bagian Utama PI
            "pendahuluan bab", "latar belakang masalah",
            "rumusan masalah", "batasan masalah",
            "tujuan penelitian", "manfaat penelitian",
            "sistematika penulisan bab", "tinjauan pustaka",
            "kajian empiris", "landasan teori", "metode penelitian",
            "tempat dan waktu penelitian", "teknik pengumpulan data",
            "metode pengembangan sistem", "jadwal penelitian",
            "hasil penelitian", "pembahasan", "penutup",
            "kesimpulan", "saran",
            # Bagian Utama KKP
            "deskripsi tempat kkp", "waktu kkp", "lokasi kkp",
            "sejarah tempat", "profil tempat", "narasi kegiatan",
            "deskripsi kegiatan", "analisis hasil kegiatan",
            "isi bab iii", "bab iii kkp", "bab 3 kkp",
            "isi bab iv", "bab iv kkp", "bab 4 kkp",
            # Bagian Akhir
            "daftar pustaka", "lampiran pendukung",
            "isi lampiran", "dokumen lampiran",
        ],
        "BAB V": [
            # Format Naskah
            "format penulisan", "format naskah", "kertas", "hvs a4",
            "80 gram", "margin", "atas 3 cm", "bawah 3 cm",
            "kiri 4 cm", "kanan 3 cm", "jenis huruf", "font",
            "times new roman", "ukuran 12", "spasi", "1,5 spasi",
            "1.5 spasi", "alinea", "menjorok", "1,2 cm",
            "pengetikan alinea", "bab dan sub-bab", "nama bab",
            "huruf kapital", "romawi", "tepi atas", "tepi kiri",
            "spesifikasi kertas", "jenis kertas", "ukuran kertas",
            "berat kertas", "warna kertas", "kertas hvs",
            # Angka dan Penomoran
            "aturan angka", "bilangan desimal", "koma", "titik",
            "satuan", "singkatan", "penomoran halaman",
            "angka romawi", "angka arab", "tengah bawah",
            "kanan atas", "nomor halaman",
            # Tabel dan Gambar
            "aturan tabel", "format tabel", "nomor tabel",
            "judul tabel", "tabel tidak terpotong", "header tabel",
            "landscape", "sumber tabel", "aturan gambar",
            "format gambar", "nomor gambar", "judul gambar",
            "grafik", "diagram", "bagan", "peta", "foto",
            "ukuran gambar", "letak gambar", "sumber gambar",
            # Bahasa
            "bahasa indonesia", "eyd", "edisi kelima",
            "kalimat pasif", "orang pertama", "orang kedua",
            "istilah asing", "istilah indonesia", "huruf miring",
            "penggunaan huruf miring",
            # Daftar Pustaka
            "daftar pustaka", "referensi", "apa", "american psychological",
            "mendeley", "tools reference", "jumlah referensi",
            "minimal 15", "minimal 5", "80%", "buku dan jurnal",
            "5 tahun terakhir", "nama penulis", "tahun terbit",
            "judul buku", "judul jurnal", "judul artikel",
            "informasi publikasi", "penerbit", "volume", "halaman",
            "doi", "url", "urutan abjad", "hanging indent",
            "no date", "n.d.", "artikel jurnal", "format buku",
            "artikel website", "bab buku", "skripsi", "tesis",
            "disertasi", "media online", "sumber elektronik",
            "pi tidak dipublikasikan", "tidak dipublikasikan",
            "referensi pi", "sumber pi", "format referensi",
            "cara menulis referensi", "penulisan referensi",
        ],
        "Lampiran": [
            # Umum
            "contoh", "lampiran", "formulir", "form",
            "contoh format", "contoh formulir",
            # Halaman Awal
            "contoh sampul", "contoh cover", "contoh pengesahan",
            "contoh abstrak", "contoh kata pengantar",
            "contoh daftar isi", "contoh daftar tabel",
            "contoh daftar gambar", "contoh daftar lampiran",
            "contoh daftar pustaka",
            "elemen sampul", "isi sampul", "komponen sampul",
            "elemen pengesahan", "isi pengesahan", "komponen pengesahan",
            "halaman pengesahan", "lembar pengesahan",
            # Tugas dan Bimbingan
            "rincian tugas", "daftar hadir kkp", "daftar hadir pi",
            "penilaian kepuasan", "tingkat kepuasan pengguna",
            "form bimbingan", "lembar bimbingan",
            "yang harus dibawa", "membawa bimbingan",
            "dokumen bimbingan", "berkas bimbingan",
            # Jadwal dan Wawancara
            "jadwal penelitian", "jadwal kegiatan",
            "daftar wawancara", "pertanyaan wawancara",
            "responden", "pewawancara",
            # Administrasi Ujian
            "daftar hadir menyaksikan", "daftar hadir ujian",
            "pengecekan syarat", "syarat administrasi",
            "form permohonan ujian", "form pendaftaran ujian",
            "lembar persetujuan ujian", "persetujuan waktu ujian",
            "lembar perbaikan", "perbaikan ujian",
            "berkas ujian", "dokumen ujian", "kelengkapan ujian",
            "syarat mendaftar ujian", "persyaratan ujian",
            "administrasi pendaftaran", "dokumen pendaftaran",
            # Berita Acara dan Penilaian
            "berita acara", "berita acara ujian",
            "form penilaian", "penilaian ujian",
            "rekapitulasi penilaian", "komponen penilaian",
            "dewan penguji", "susunan penguji",
            # Perubahan
            "penggantian pembimbing", "penggantian dosen",
            "surat tugas pengganti", "penugasan dosen",
            "penggantian judul", "perubahan judul",
            "revisi judul",
        ],
    }

    def _matches_keyword(text: str, keyword: str) -> bool:
        if " " in keyword:
            # Multi-word keyword: exact substring match
            return keyword in text
        # Single word: use word boundary
        return re.search(rf"\b{re.escape(keyword)}\b", text) is not None

    # Cari section yang paling cocok
    matched_sections = []
    for section, keywords in section_keywords.items():
        matched_keywords = [kw for kw in keywords if _matches_keyword(query_lower, kw)]
        if matched_keywords:
            matched_sections.append((section, len(matched_keywords)))
    
    # Pilih filter section jika confidence HIGH (>= 2 keyword match)
    if matched_sections:
        matched_sections.sort(key=lambda x: x[1], reverse=True)
        best_section = matched_sections[0][0]
        num_matches = matched_sections[0][1]
        
        # Hanya apply filter jika ada minimal 2 keyword match (confidence high)
        if num_matches >= 2:
            filters["section"] = best_section
            confidence = "high"
        else:
            # Jika hanya 1 keyword match, jangan filter section (biarkan search semua BAB)
            confidence = "low"
            logger.debug(f"Section '{best_section}' matched tapi hanya 1 keyword, skip filter untuk coverage lebih luas")
    else:
        confidence = "low"

    # Tentukan detected values untuk logging
    detected_source = filters.get("source")
    detected_section = filters.get("section")

    logger.info(
        f"Query dianalisis — semantic: '{semantic}' | "
        f"filters: {filters} | confidence: {confidence}"
    )

    return ParsedQuery(
        semantic_query=semantic,
        filters=filters,
        original_query=query,
        detected_source=detected_source,
        detected_section=detected_section,
        confidence=confidence,
    )


def get_available_sections(
    source: Literal["PI", "KKP", "both"] = "both"
) -> dict[str, list[str]]:
    sections = {
        "Front Matter": [
            "Kata Pengantar",
            "Daftar Isi, Daftar Tabel, Daftar Gambar",
        ],
        "Surat Keputusan": [
            "SK Pemberlakuan Panduan dari Ketua STMIK",
        ],
        "BAB I": [
            "Pendahuluan: Latar Belakang dan Tujuan Panduan",
        ],
        "BAB II": [
            "Ketentuan Dosen Pembimbing dan Penguji",
            "Hak dan Mekanisme Mahasiswa Bimbingan",
            "Syarat dan Tempat Penelitian/KKP",
            "Proses Penyusunan dan Pengajuan",
            "Prosedur Ujian dan Tahap Akhir",
            "Sistem Penilaian dan Kelulusan",
        ],
        "BAB III": [
            "Sistematika Penyusunan Laporan PI/KKP",
            "Gambaran Umum dan Struktur Laporan",
        ],
        "BAB IV": [
            "Penjelasan Bagian Awal (Sampul, Pengesahan, Abstrak, Kata Pengantar)",
            "Penjelasan Bagian Utama (BAB I-V)",
            "Penjelasan Bagian Akhir (Daftar Pustaka, Lampiran)",
        ],
        "BAB V": [
            "Format Penulisan Naskah (Kertas, Margin, Huruf, Spasi)",
            "Aturan Tabel dan Gambar",
            "Bahasa dan Huruf Miring",
            "Daftar Pustaka (APA Style)",
        ],
        "Lampiran": [
            "Contoh Halaman Awal",
            "Contoh Daftar (Isi, Tabel, Gambar, Pustaka)",
            "Form Jadwal dan Bimbingan",
            "Form Administrasi Ujian",
            "Form Persetujuan dan Perbaikan",
            "Form Berita Acara dan Penilaian",
            "Form Perubahan Pembimbing/Judul",
        ],
    }
    
    return sections


def get_metadata_statistics() -> dict:
    stats = {
        "sources": [
            "Panduan Penyusunan Penulisan Imliah (PI) Cetak",
            "Panduan Penyusunan Kuliah Kerja Praktik (KKP) Cetak",
        ],
        "sections": [
            "Front Matter",
            "Surat Keputusan",
            "BAB I",
            "BAB II",
            "BAB III",
            "BAB IV",
            "BAB V",
            "Lampiran",
        ],
        "pi_parent_chunks": 23,
        "kkp_parent_chunks": 23,
        "total_parent_chunks": 46,
        "key_topics": {
            "PI": [
                "Penelitian ilmiah mandiri",
                "Studi literatur",
                "Minimal 100 SKS, IPK 2.00",
                "Minimal 7 hari kerja penelitian",
                "Minimal 15 referensi (80% buku/jurnal)",
                "Laporan minimal 40 halaman",
                "Anti-plagiarisme maksimal 30%",
            ],
            "KKP": [
                "Praktik kerja di instansi",
                "Minimal 100 SKS, IPK 2.00",
                "Minimal 30 hari kerja (1 bulan)",
                "Minimal 5 referensi (80% buku/jurnal)",
                "Pembimbing lapangan dari instansi",
                "Rekomendasi BKK",
            ],
            "Umum": [
                "1 Dosen Pembimbing (Dosen Wali)",
                "2 Dosen Penguji",
                "Minimal 8 kali bimbingan",
                "Maksimal 6 bulan masa bimbingan",
                "Ujian maksimal 60 menit",
                "Nilai skala 100 (A: 80-100, B: 70-79, C: 60-69)",
                "Format: HVS A4, Times New Roman 12, Spasi 1.5",
                "Margin: Atas 3cm, Bawah 3cm, Kiri 4cm, Kanan 3cm",
                "Daftar Pustaka: APA Style",
            ],
        },
    }
    
    return stats