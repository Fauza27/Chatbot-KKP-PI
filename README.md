# Chatbot Asisten Virtual Berbasis RAG untuk Panduan PI/KKP

> Sebuah chatbot akademik yang bisa diajak berdiskusi untuk membantu menjawab berbagai pertanyaan seputar KKP/PI, didukung oleh arsitektur RAG yang dirancang cukup matang di balik layar.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-green.svg)](https://openai.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-teal.svg)](https://fastapi.tiangolo.com/)
[![Telegram](https://img.shields.io/badge/Telegram-Bot-blue.svg)](https://core.telegram.org/bots)
[![RAGAS](https://img.shields.io/badge/Evaluation-RAGAS-orange.svg)](https://github.com/explodinggradients/ragas)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success.svg)]()

---

## 🎯 Kenapa Dibuat

Proyek ini berangkat dari masalah yang cukup sering saya lihat di kampus (STMIK Widya Cipta Dharma). Saat memasuki fase penting seperti Kuliah Kerja Praktik (KKP) dan Penulisan Ilmiah (PI) di semester 6, banyak mahasiswa masih merasa bingung harus mulai dari mana. Padahal, buku pedoman sudah tersedia—hanya saja, tidak semua orang benar-benar membacanya atau memahami isinya dengan baik.

Buat saya, proyek ini bukan sekadar tugas atau portofolio. Saya ingin benar-benar memahami bagaimana membangun sistem Retrieval-Augmented Generation (RAG) yang siap digunakan di kondisi nyata (production-grade).

---

## 🏗️ Arsitektur Sistem

Gambaran besar tentang bagaimana infrastruktur ini meramu dan memecahkan setiap pesan dari pengguna:

```text
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Telegram   │  │   REST API   │  │      CLI     │      │
│  │     Bot      │  │   (FastAPI)  │  │  Interface   │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
└─────────┼──────────────────┼──────────────────┼─────────────┘
          │                  │                  │
          └──────────────────┼──────────────────┘
                             │
┌────────────────────────────┼─────────────────────────────────┐
│                    Application Layer                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              AI Services (ai_services.py)            │   │
│  │  • Session Management                                │   │
│  │  • Intent Classification                             │   │
│  │  • Conversation Memory                               │   │
│  └────────┬─────────────────────────────────────────────┘   │
└───────────┼───────────────────────────────────────────────────┘
            │
┌───────────┼───────────────────────────────────────────────────┐
│           │              RAG Pipeline                          │
│  ┌────────▼──────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Self-Query      │  │    Hybrid    │  │   Parent-    │  │
│  │   Extraction      │→ │    Search    │→ │    Child     │  │
│  │  (Metadata Filter)│  │  (BM25+Dense)│  │   Fetching   │  │
│  └───────────────────┘  └──────────────┘  └──────┬───────┘  │
│                                                    │          │
│  ┌───────────────────┐  ┌──────────────┐  ┌──────▼───────┐  │
│  │   LLM Generation  │← │  Cross-Encoder│← │   Reranking  │  │
│  │   (GPT-4o-mini)   │  │   Reranking   │  │  (Top-N)     │  │
│  └───────────────────┘  └──────────────┘  └──────────────┘  │
└───────────────────────────────────────────────────────────────┘
            │
┌───────────▼───────────────────────────────────────────────────┐
│                      Data Layer                                │
│  ┌──────────────────┐  ┌──────────────────┐                  │
│  │    Supabase      │  │   OpenAI API     │                  │
│  │  (Vector Store)  │  │  (Embeddings +   │                  │
│  │  • Parent Docs   │  │      LLM)        │                  │
│  │  • Child Chunks  │  │                  │                  │
│  └──────────────────┘  └──────────────────┘                  │
└───────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Stack Teknologi (dan Alasan Saya Memilihnya)

Saya tidak asal ikut tren atau sekadar memakai teknologi yang sedang hype. Setiap komponen di proyek ini dipilih dengan pertimbangan yang cukup matang, disesuaikan dengan kebutuhan dan keterbatasan yang ada:

- **FastAPI**: Aplikasi ini harus menangani REST API sekaligus menerima trafik rutin dari webhook Telegram. Karena itu, saya butuh framework yang memang dirancang asynchronous sejak awal. FastAPI jadi pilihan karena ringan, cepat, dan tetap stabil meskipun dijalankan di layanan cloud gratis.
- **Supabase (pgvector)**: Mengelola database relasional dan vector database secara terpisah bisa jadi cukup merepotkan. Supabase menawarkan solusi praktis lewat PostgreSQL yang sudah dilengkapi ekstensi pgvector. Dengan ini, saya bisa menyimpan struktur dokumen (misalnya relasi antar bab) sekaligus embedding-nya dalam satu sistem yang rapi dan terintegrasi.
- **Cross-Encoder Model (`ms-marco`)**: Salah satu kendala terbesar penelusuran dokumen murni adalah ketidakmampuan algoritma *vector search* menangkap maksud sebenarnya dari pertanyaan pengguna dan menghasilkan dokumen tidak relevan. Menggunakan LLM seperti GPT-4 untuk reranking sebenarnya efektif, tapi jelas mahal kalau dipakai terus-menerus. Sebagai gantinya, saya menggunakan model Cross-Encoder kecil yang dijalankan secara lokal di CPU. Performanya cukup baik untuk menyaring dan mengurutkan hasil pencarian, tanpa tambahan biaya API.
- **python-telegram-bot**: *Library* ini sudah cukup matang, terutama di bagian ConversationHandler. Ini sangat membantu karena saya tidak perlu membangun sendiri sistem pengelolaan alur percakapan dari nol. Integrasinya dengan FastAPI (via webhook) juga relatif mulus.
- **RAGAS**: Tanpa evaluasi yang jelas, pengembangan RAG rasanya seperti coba-coba tanpa arah. RAGAS saya gunakan untuk mengukur kualitas sistem secara lebih objektif, seperti faithfulness dan context precision. Dengan begitu, setiap perubahan pada pipeline bisa dievaluasi dengan dasar yang lebih terukur, bukan sekadar feeling.

---

## 🧠 Keputusan Desain yang Menarik

Merancang sistem RAG yang benar-benar “paham konteks” ternyata bukan soal menulis kode sebanyak mungkin, tapi soal berani mengambil keputusan dan menerima konsekuensinya. Sepanjang proses ini, saya justru lebih sering dihadapkan pada trade-off dibanding sekadar urusan teknis.

Semua pertimbangan, alasan, dan dilema yang saya temui selama membangun sistem ini saya dokumentasikan dalam Architecture Decision Records (ADR). Berikut gambaran singkat dari beberapa keputusan penting tersebut:

1. **[Arsitektur Monolith Serverless (FastAPI + Webhook)](ADR/ADR-001-Arsitektur-Sistem-Utama.md)** 
   Trade-off: Harus menerima adanya cold start yang bikin respons awal sedikit lebih lambat. Tapi sebagai gantinya, proses deployment jadi jauh lebih sederhana tanpa perlu mengelola banyak microservices.
2. **[Parent-Child Chunking & Hybrid Search](ADR/ADR-002-Strategi-Retrieval-Hierarkis-Hybrid.md)**
   Trade-off: Proses indexing jadi lebih lama dan penggunaan storage meningkat. Namun hasilnya, model bisa memahami konteks dokumen secara lebih utuh,termasuk struktur bab dan referensi—,ehingga jawaban yang dihasilkan jauh lebih akurat.
3. **[Reranking dengan Cross-Encoder Lokal](ADR/ADR-003-Reranking-dengan-Cross-Encoder.md)**
   Trade-off: Waktu respons sedikit bertambah karena ada proses tambahan di CPU. Tapi ini membantu menekan biaya API secara signifikan, tanpa mengorbankan kualitas hasil pencarian.
4. **[Pre-processing Intent Classification](ADR/ADR-004-Klasifikasi-Intent-Percakapan.md)**
   Ada tambahan proses di awal (LLM dipanggil lebih dari sekali), yang berarti ada overhead. Namun ini membuat sistem lebih cerdas dalam memahami konteks percakapan—misalnya membedakan mana pertanyaan serius dan mana sekadar basa-basi seperti “Makasih” sehingga pencarian jadi lebih efisien dan relevan.

---

## 🚀 Cara Menjalankan Secara Lokal

**1. Clone & Setup Environment**
```bash
git clone https://github.com/yourusername/penelitian-ilmiah.git
cd penelitian-ilmiah
python -m venv venv
# Windows: venv\Scripts\activate | Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```
Buka file `.env` dan isi dengan kredensial yang diperlukan: OpenAI API Key, Supabase URL, Supabase Key, dan Telegram Token (opsional).

**2. Setup Database**
Jalankan script SQL di `scripts/supabase.sql` melalui SQL Editor di dashboard Supabase Anda. Script ini akan membuat tabel dan index yang diperlukan untuk vector search.

**3. Ingestion Pipeline**
```bash
python main.py --ingest --dataset both
```
Perintah ini akan memproses PDF panduan, memecahnya menjadi chunks, membuat embeddings menggunakan OpenAI, dan menyimpannya ke Supabase.

**4. Jalankan Aplikasi**
```bash
# Jalankan web server (FastAPI untuk webhook)
python main.py

# Atau jalankan di mode CLI untuk testing
python main.py --cli
```

---

## 📖 Apa yang Saya Pelajari

Proyek ini awalnya kelihatan sederhana, tapi ternyata berkembang jadi eksplorasi yang cukup dalam. Semakin dikerjakan, semakin terasa kalau RAG itu bukan sekadar pipeline Retrieve → Augment → Generate yang bisa langsung jadi.

Justru, bagian tersulitnya ada di keputusan-keputusan kecil yang efeknya baru terasa belakangan.

Ada ssatu hal yang paling membekas buat saya:
  
- **Metrik tidak selalu mencerminkan kenyataan (pengalaman dengan RAGAS)**: 
  Sempat kaget waktu skor faithfulness dari RAGAS turun. Sekilas terlihat seperti ada masalah serius.
  Tapi setelah dicek manual, ternyata jawabannya tetap benar—tidak ada halusinasi.
  Masalahnya ada di cara chatbot menyampaikan jawaban. Saya sengaja membuatnya lebih santai dan tidak terlalu kaku, jadi sering melakukan parafrasa dari teks asli. Di sisi metrik, ini dianggap kurang “faithful”, tapi dari sisi pengguna justru terasa lebih nyaman.
  Dari sini saya belajar: metrik itu penting sebagai panduan, tapi tidak boleh jadi satu-satunya acuan. Tetap perlu validasi manual dan, kalau memungkinkan, feedback langsung dari pengguna. Karena pada akhirnya, yang dinilai bukan dashboard—tapi pengalaman orang yang benar-benar memakai sistemnya.
