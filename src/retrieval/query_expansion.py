"""
Query Expansion untuk meningkatkan recall pada pertanyaan spesifik
"""

from loguru import logger


def expand_query(question: str) -> str:
    """
    Expand query untuk meningkatkan recall pada pertanyaan spesifik.
    
    Menambahkan kata kunci relevan untuk membantu retrieval menemukan
    informasi yang tersembunyi atau tidak eksplisit dalam chunks.
    
    Args:
        question: Pertanyaan asli dari user
        
    Returns:
        Query yang sudah di-expand dengan kata kunci tambahan
    """
    
    question_lower = question.lower()
    expanded = question
    keywords = []
    
    # 1. Pakaian / Dress Code
    if any(word in question_lower for word in ["pakaian", "ketentuan pakaian", "dress code", "berpakaian"]):
        keywords.extend(["ujian", "seminar", "kemeja", "putih", "almamater", "celana", "rok", "hitam", "jilbab"])
        logger.debug("Query expansion: pakaian ujian")
    
    # 2. Abstrak - Maksimal Kata
    if "abstrak" in question_lower and any(word in question_lower for word in ["maksimal", "kata", "panjang"]):
        keywords.extend(["300", "kata", "maksimal", "abstrak", "satu halaman", "ringkas"])
        logger.debug("Query expansion: abstrak maksimal kata")
    
    # 3. Kata Kunci / Keywords - ENHANCED untuk FASE 3A
    if any(word in question_lower for word in ["kata kunci", "keyword"]):
        keywords.extend(["3-5", "kata kunci", "keywords", "abstrak", "minimal", "maksimal", "jumlah", "berapa"])
        logger.debug("Query expansion: kata kunci abstrak (ENHANCED)")
    
    # 4. Sampul / Cover - ENHANCED untuk FASE 3A
    if any(word in question_lower for word in ["sampul", "cover", "halaman depan"]):
        keywords.extend(["judul", "nama", "nim", "logo", "program studi", "tahun", "halaman", "sampul", "elemen", "berisi", "fakultas", "universitas", "STMIK", "Widya Cipta Dharma"])
        logger.debug("Query expansion: elemen sampul (ENHANCED)")
    
    # 5. Minimal Halaman - ENHANCED untuk FASE 3A
    if ("minimal" in question_lower or "berapa" in question_lower) and "halaman" in question_lower:
        keywords.extend(["40", "halaman", "minimal", "laporan", "naskah", "jumlah", "tidak termasuk", "lampiran", "KKP", "PI"])
        logger.debug("Query expansion: minimal halaman (ENHANCED)")
    
    # 6. Jumlah Referensi
    if any(word in question_lower for word in ["jumlah", "minimal"]) and "referensi" in question_lower:
        keywords.extend(["15", "referensi", "daftar pustaka", "minimal", "80%", "buku", "jurnal"])
        logger.debug("Query expansion: jumlah referensi")
    
    # 7. Format Daftar Pustaka
    if "format" in question_lower and any(word in question_lower for word in ["daftar pustaka", "referensi"]):
        keywords.extend(["APA", "American Psychological Association", "format", "penulisan", "sitasi"])
        logger.debug("Query expansion: format daftar pustaka")
    
    # 8. Spasi / Spacing
    if "spasi" in question_lower and any(word in question_lower for word in ["naskah", "penulisan", "laporan"]):
        keywords.extend(["1,5", "spasi", "1.5", "penulisan", "naskah", "utama"])
        logger.debug("Query expansion: spasi penulisan")
    
    # 9. Font / Jenis Huruf
    if any(word in question_lower for word in ["font", "huruf", "jenis huruf"]):
        keywords.extend(["Times New Roman", "12", "font", "ukuran", "jenis"])
        logger.debug("Query expansion: font")
    
    # 10. Margin
    if "margin" in question_lower:
        keywords.extend(["4 cm", "3 cm", "margin", "kiri", "kanan", "atas", "bawah"])
        logger.debug("Query expansion: margin")
    
    # 11. Ukuran Kertas
    if "ukuran kertas" in question_lower or "kertas" in question_lower:
        keywords.extend(["A4", "21 cm", "29.7 cm", "HVS", "80 gram"])
        logger.debug("Query expansion: ukuran kertas")
    
    # 12. Durasi Ujian
    if "durasi" in question_lower or "lama" in question_lower and "ujian" in question_lower:
        keywords.extend(["60 menit", "10 menit", "presentasi", "tanya jawab", "50 menit"])
        logger.debug("Query expansion: durasi ujian")
    
    # 13. Nilai Minimal Lulus
    if "nilai" in question_lower and any(word in question_lower for word in ["minimal", "lulus"]):
        keywords.extend(["C", "60", "nilai", "minimal", "lulus", "predikat"])
        logger.debug("Query expansion: nilai minimal")
    
    # 14. Tingkat Kemiripan / Plagiarisme
    if any(word in question_lower for word in ["kemiripan", "plagiarisme", "plagiasi"]):
        keywords.extend(["30%", "maksimal", "kemiripan", "plagiarisme", "turnitin"])
        logger.debug("Query expansion: plagiarisme")
    
    # 15. Dosen Pembimbing / Penguji
    if "dosen" in question_lower and any(word in question_lower for word in ["pembimbing", "penguji", "jumlah"]):
        keywords.extend(["1 dosen", "2 dosen", "pembimbing", "penguji", "jumlah"])
        logger.debug("Query expansion: dosen pembimbing/penguji")
    
    # 16. Jabatan Fungsional
    if "jabatan" in question_lower and "fungsional" in question_lower:
        keywords.extend(["Asisten Ahli", "Lektor", "jabatan", "fungsional", "minimal"])
        logger.debug("Query expansion: jabatan fungsional")
    
    # 17. Masa Bimbingan
    if "masa" in question_lower and "bimbingan" in question_lower:
        keywords.extend(["1 semester", "maksimal", "masa", "bimbingan", "waktu"])
        logger.debug("Query expansion: masa bimbingan")
    
    # 18. Frekuensi Bimbingan
    if any(word in question_lower for word in ["berapa kali", "minimal"]) and "bimbingan" in question_lower:
        keywords.extend(["8 kali", "minimal", "bimbingan", "bertemu", "konsultasi"])
        logger.debug("Query expansion: frekuensi bimbingan")
    
    # 19. Lama Kegiatan Penelitian
    if "lama" in question_lower and any(word in question_lower for word in ["penelitian", "kegiatan"]):
        keywords.extend(["7 hari", "minimal", "kerja", "penelitian", "instansi"])
        logger.debug("Query expansion: lama penelitian")
    
    # 20. Komponen Penilaian
    if "komponen" in question_lower and "penilaian" in question_lower:
        keywords.extend(["orisinalitas", "sistematika", "penguasaan", "argumentasi", "penampilan", "etika"])
        logger.debug("Query expansion: komponen penilaian")
    
    # Gabungkan query asli dengan keywords
    if keywords:
        # Deduplicate keywords
        unique_keywords = list(dict.fromkeys(keywords))
        expanded = f"{question} {' '.join(unique_keywords)}"
        logger.info(f"Query expanded: '{question}' → added {len(unique_keywords)} keywords")
    
    return expanded


def expand_query_smart(question: str, enable_expansion: bool = True) -> str:
    """
    Smart query expansion dengan toggle.
    
    Args:
        question: Pertanyaan asli
        enable_expansion: Flag untuk enable/disable expansion
        
    Returns:
        Query yang sudah di-expand (jika enabled) atau query asli
    """
    
    if not enable_expansion:
        return question
    
    return expand_query(question)
