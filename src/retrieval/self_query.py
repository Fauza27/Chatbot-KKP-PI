from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal
from loguru import logger
from config.settings import get_settings

settings = get_settings()


@dataclass
class ParsedQuery:
    semantic_query: str
    filters: dict
    original_query: str
    detected_source: str | None = None
    detected_section: str | None = None
    confidence: str = "medium"  # low, medium, high


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