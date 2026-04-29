import json
from pathlib import Path
from datetime import datetime

from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from datasets import Dataset
from loguru import logger

from config.settings import get_settings



# ── Evaluation Dataset ──────────────────────────────────────────
# Mendukung evaluasi untuk PI dan KKP
EVAL_QUESTIONS_PI = [
    # ...existing PI questions (copy from above)...
    {"question": "Apa syarat SKS minimal untuk mengambil Penulisan Ilmiah (PI)?", "ground_truth": "Mahasiswa yang berhak mengambil PI telah menyelesaikan mata kuliah dengan jumlah SKS minimal 100 SKS dan IP Kumulatif minimal 2,00."},
    {"question": "Siapa yang menjadi dosen pembimbing PI?", "ground_truth": "Dosen Pembimbing PI adalah Dosen Pembimbing Akademik (Dosen Wali). Dosen Pembimbing harus terdaftar sebagai dosen STMIK Widya Cipta Dharma dan memiliki jabatan fungsional minimal asisten ahli atau lektor dengan kualifikasi pendidikan S2 atau S3."},
    {"question": "Berapa lama maksimal masa bimbingan PI?", "ground_truth": "Maksimal masa bimbingan 6 bulan (1 semester). Perpanjangan dapat diberikan dengan persetujuan Ketua Program Studi setelah rekomendasi Dosen Pembimbing."},
    {"question": "Apa saja komponen penilaian ujian PI?", "ground_truth": "Komponen penilaian meliputi: Orisinalitas Penulisan, Sistematika dan Tata Cara Penulisan Laporan, Penguasaan Materi Sesuai Capaian Pembelajaran Mata Kuliah, Kemampuan Argumentasi dan Presentasi, dan Penampilan/Etika. Setiap komponen diberi nilai dalam rentang 0-100."},
    {"question": "Berapa minimal halaman laporan PI?", "ground_truth": "Laporan PI minimal 40 halaman (di luar cover, daftar isi, daftar tabel, daftar gambar, daftar lampiran, daftar pustaka, dan lampiran)."},
    {"question": "Apa format margin yang digunakan dalam penulisan PI?", "ground_truth": "Margin: atas 3 cm, bawah 3 cm, kiri 4 cm, kanan 3 cm. Naskah rata kiri dan kanan."},
    {"question": "Berapa lama waktu ujian PI?", "ground_truth": "Ujian PI maksimal 60 menit, terdiri dari 10 menit presentasi dan 50 menit tanya jawab."},
    {"question": "Apa skala penilaian PI?", "ground_truth": "Nilai akhir PI menggunakan skala 100 dengan predikat: A (80-100) Sangat Baik - Lulus, B (70-79) Baik - Lulus, C (60-69) Cukup - Lulus, D (40-59) Kurang - Tidak Lulus, E (0-39) Sangat Kurang - Tidak Lulus."},
    {"question": "Jenis huruf apa yang digunakan dalam penulisan PI?", "ground_truth": "Jenis huruf Times New Roman ukuran 12 untuk seluruh naskah. Dalam tabel boleh lebih kecil dari 12."},
    {"question": "Berapa jumlah minimal referensi daftar pustaka?", "ground_truth": "Jumlah referensi minimal 15. 80% berasal dari buku dan jurnal. Disarankan merujuk referensi kurang dari 5 tahun kecuali yang sangat penting."},
    {"question": "Apa saja sistematika penulisan laporan PI?", "ground_truth": "Sistematika penulisan terdiri atas Bagian Awal (Cover, Pengesahan, Abstrak, Kata Pengantar, Daftar Isi/Tabel/Gambar/Lampiran), Bagian Utama (BAB I Pendahuluan, BAB II Tinjauan Pustaka, BAB III Metode Penelitian, BAB IV Hasil dan Pembahasan, BAB V Penutup), dan Bagian Akhir (Daftar Pustaka, Lampiran)."},
    {"question": "Apa syarat kelulusan ujian PI?", "ground_truth": "Mahasiswa dinyatakan lulus apabila: (1) PI merupakan karya otentik, (2) memperoleh nilai minimal C, (3) telah memperbaiki PI sesuai saran dan arahan, dibuktikan dengan penandatanganan halaman pengesahan, (4) telah menyerahkan jilid laporan ke Perpustakaan."},
    {"question": "Berapa batas maksimal tingkat plagiarisme yang diizinkan?", "ground_truth": "Bukti pemeriksaan anti-plagiarisme dengan tingkat kemiripan maksimal 30% dari perangkat lunak/aplikasi pendeteksi plagiarisme yang valid."},
    {"question": "Berapa minimal kehadiran seminar PI sebelum boleh seminar sendiri?", "ground_truth": "Untuk melaksanakan seminar PI, mahasiswa wajib menghadiri seminar laporan PI minimal 3 kali."},
    {"question": "Apa format penulisan daftar pustaka yang digunakan?", "ground_truth": "Menggunakan American Psychological Association (APA). Disarankan menggunakan Mendeley atau tools reference lainnya."},
]

# Contoh: tambahkan pertanyaan KKP sesuai kebutuhan
EVAL_QUESTIONS_KKP = [
    {"question": "Apa syarat SKS minimal untuk mengambil Kuliah Kerja Praktik (KKP)?", "ground_truth": "Mahasiswa yang berhak mengambil KKP telah menyelesaikan mata kuliah dengan jumlah SKS minimal 100 SKS dan IP Kumulatif minimal 2,00."},
    {"question": "Berapa lama pelaksanaan KKP?", "ground_truth": "Pelaksanaan KKP minimal 1 bulan (4 minggu) di instansi/industri yang relevan."},
    {"question": "Apa saja dokumen yang harus dikumpulkan setelah KKP?", "ground_truth": "Laporan KKP, surat keterangan selesai dari instansi, dan logbook kegiatan harian."},
    # ...tambahkan pertanyaan lain sesuai kebutuhan...
]

def get_eval_questions(dataset: str) -> list[dict]:
    """
    Return evaluation questions sesuai dataset (PI/KKP).
    """
    if dataset.lower() == "kkp":
        return EVAL_QUESTIONS_KKP
    return EVAL_QUESTIONS_PI



def create_evaluation_dataset(dataset: str = "pi") -> list[dict]:
    """
    Return evaluation dataset sebagai list of dicts untuk PI atau KKP.
    Args:
        dataset: "pi" atau "kkp"
    Returns:
        List of dicts dengan keys: question, ground_truth
    """
    return get_eval_questions(dataset)


def run_evaluation(
    pipeline_fn,
    eval_data: list[dict] | None = None,
    output_path: str | None = None,
) -> dict:
    """
    Jalankan evaluasi RAGAS pada pipeline RAG.

    Args:
        pipeline_fn: Fungsi yang menerima question (str) dan mengembalikan
                    dict {"answer": str, "contexts": list[str]}
        eval_data: List of evaluation dicts (default: EVAL_QUESTIONS)
        output_path: Path untuk menyimpan hasil (default: auto-generate)

    Returns:
        Dict berisi skor RAGAS per metrik
    """
    settings = get_settings()
    eval_data = eval_data or EVAL_QUESTIONS_PI

    logger.info(f"Memulai evaluasi RAGAS dengan {len(eval_data)} pertanyaan...")

    # 1. Jalankan pipeline untuk setiap pertanyaan
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for i, item in enumerate(eval_data):
        q = item["question"]
        logger.info(f"[{i+1}/{len(eval_data)}] Evaluating: {q[:60]}...")

        try:
            result = pipeline_fn(q)
            questions.append(q)
            answers.append(result["answer"])
            contexts.append(result["contexts"])
            ground_truths.append(item["ground_truth"])
        except Exception as e:
            logger.error(f"Error evaluating question '{q}': {e}")
            questions.append(q)
            answers.append(f"Error: {e}")
            contexts.append([""])
            ground_truths.append(item["ground_truth"])

    # 2. Buat HuggingFace Dataset untuk RAGAS
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    # 3. Setup evaluator LLM dan embeddings
    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.open_api_key,
            temperature=0.0,
        )
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.open_api_key,
        )
    )

    # 4. Jalankan evaluasi RAGAS
    logger.info("Menjalankan RAGAS evaluation...")
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    # 5. Extract scores
    scores = {
        "faithfulness": float(result["faithfulness"]),
        "answer_relevancy": float(result["answer_relevancy"]),
        "context_precision": float(result["context_precision"]),
        "context_recall": float(result["context_recall"]),
    }

    # Hitung rata-rata sebagai overall score
    scores["overall"] = sum(scores.values()) / len(scores)

    logger.info("=" * 60)
    logger.info("HASIL EVALUASI RAGAS")
    logger.info("=" * 60)
    for metric, score in scores.items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        logger.info(f"  {metric:>20s}: {score:.4f} |{bar}|")
    logger.info("=" * 60)

    # 6. Simpan hasil ke file
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"evaluation_results_{timestamp}.json"

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "num_questions": len(eval_data),
        "scores": scores,
        "details": [
            {
                "question": q,
                "answer": a,
                "contexts": c,
                "ground_truth": gt,
            }
            for q, a, c, gt in zip(questions, answers, contexts, ground_truths)
        ],
    }

    output_file = Path(output_path)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Hasil evaluasi disimpan ke: {output_file}")
    return scores