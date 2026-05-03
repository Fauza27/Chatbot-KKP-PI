"""
RAGAS Evaluation WITHOUT Ground Truth
"""

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
)
from ragas.metrics._context_precision import LLMContextPrecisionWithoutReference
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from loguru import logger
import json
from datetime import datetime
from typing import List, Dict

from config.settings import get_settings


#
# EVALUATION QUESTIONS (TANPA GROUND TRUTH)
#

EVAL_QUESTIONS_PI = [
    "Apa syarat SKS minimal untuk mengambil Penulisan Ilmiah (PI)?",
    "Berapa IP Kumulatif minimal untuk mengambil PI?",
    "Apakah PI wajib dilakukan di sebuah instansi atau perusahaan?",
    "Berapa lama minimal kegiatan penelitian PI jika dilakukan di perusahaan atau instansi?",
    "Siapa yang menjadi dosen pembimbing PI?",
    "Berapa jumlah dosen pembimbing dan dosen penguji untuk PI?",
    "Apa yang dimaksud dengan Tempat Penelitian PI?",
    "Apa syarat jabatan fungsional minimal untuk menjadi dosen pembimbing atau penguji PI?",
    "Dalam kondisi apa mahasiswa dapat mengajukan penggantian dosen pembimbing PI?",
    "Apa hak mahasiswa dalam proses bimbingan PI?",
    "Berapa lama maksimal masa bimbingan PI?",
    "Berapa kali minimal mahasiswa harus bertemu dengan dosen pembimbing selama PI?",
    "Di mana bimbingan PI dapat dilaksanakan?",
    "Apa yang harus dibawa mahasiswa setiap kali melakukan bimbingan PI?",
    "Berapa kali minimal mahasiswa harus melapor ke dosen pembimbing selama pelaksanaan penelitian PI?",
    "Berapa kali minimal mahasiswa harus menghadiri seminar PI orang lain sebelum melaksanakan seminar sendiri?",
    "Apa saja berkas yang harus dilampirkan saat mendaftar ujian PI?",
    "Berapa maksimal tingkat kemiripan (plagiarisme) yang diperbolehkan untuk laporan PI?",
    "Berapa lama waktu maksimal pelaksanaan seminar PI setelah persetujuan dosen pembimbing?",
    "Berapa minimal halaman laporan PI?",
    "Apa ketentuan pakaian saat ujian PI untuk mahasiswa pria?",
    "Apa ketentuan pakaian saat ujian PI untuk mahasiswi berjilbab?",
    "Berapa lama durasi ujian PI?",
    "Apa komponen penilaian dalam ujian PI?",
    "Berapa nilai minimal untuk dinyatakan lulus ujian PI?",
    "Apa yang harus dilakukan mahasiswa setelah ujian PI?",
    "Berapa jumlah minimal referensi yang harus ada dalam laporan PI?",
    "Apa format daftar pustaka yang digunakan dalam PI?",
    "Berapa spasi yang digunakan untuk penulisan naskah utama PI?",
    "Apa jenis dan ukuran font yang digunakan untuk naskah PI?",
    "Berapa margin yang digunakan untuk penulisan laporan PI?",
    "Apa ukuran kertas yang digunakan untuk laporan PI?",
    "Bagaimana aturan penulisan judul bab dalam PI?",
    "Bagaimana aturan penulisan judul tabel dalam PI?",
    "Bagaimana aturan penulisan judul gambar dalam PI?",
    "Berapa maksimal kata dalam abstrak PI?",
    "Berapa jumlah kata kunci yang harus ada dalam abstrak PI?",
    "Apa saja elemen yang harus ada di halaman sampul depan PI?",
    "Apa yang dimuat dalam halaman pengesahan PI?",
    "Apa saja bagian yang termasuk dalam Bagian Awal laporan PI?",
    "Apa saja bab yang termasuk dalam Bagian Utama laporan PI?",
    "Apa isi dari BAB I Pendahuluan dalam laporan PI?",
    "Apa isi dari BAB II Tinjauan Pustaka dalam laporan PI?",
    "Apa isi dari BAB III Metode Penelitian dalam laporan PI?",
    "Apa isi dari BAB IV Hasil Penelitian dan Pembahasan dalam laporan PI?",
    "Apa isi dari BAB V Penutup dalam laporan PI?",
    "Apa yang dimaksud dengan Kajian Empiris dalam Tinjauan Pustaka PI?",
    "Apa yang dimaksud dengan Landasan Teori dalam Tinjauan Pustaka PI?",
    "Bagaimana cara menulis referensi buku dalam daftar pustaka PI?",
    "Bagaimana cara menulis referensi jurnal dalam daftar pustaka PI?",
    "Bagaimana cara menulis referensi website dalam daftar pustaka PI?",
    "Bagaimana cara menulis referensi yang tidak dipublikasikan dalam daftar pustaka PI?",
    "Apa yang harus dilakukan jika referensi tidak memiliki nama penulis?",
    "Bagaimana urutan penyusunan daftar pustaka dalam PI?",
]

EVAL_QUESTIONS_KKP = [
    "Apa syarat SKS minimal untuk mengambil Kuliah Kerja Praktik (KKP)?",
    "Berapa IP Kumulatif minimal untuk mengambil KKP?",
    "Berapa jumlah dosen pembimbing dan dosen penguji untuk KKP?",
    "Berapa lama minimal pelaksanaan KKP di instansi?",
    "Apa yang dimaksud dengan Tempat KKP?",
    "Apa kriteria instansi yang dapat menjadi tempat KKP?",
    "Siapa yang menjadi dosen pembimbing KKP?",
    "Apa bidang yang menjadi sasaran pengalaman belajar dalam KKP?",
    "Apa syarat jabatan fungsional minimal untuk menjadi dosen pembimbing atau penguji KKP?",
    "Dalam kondisi apa mahasiswa dapat mengajukan penggantian dosen pembimbing KKP?",
    "Apa hak mahasiswa dalam proses bimbingan KKP?",
    "Berapa lama maksimal masa bimbingan KKP?",
    "Berapa kali minimal mahasiswa harus bertemu dengan dosen pembimbing selama KKP?",
    "Bagaimana mahasiswa diutamakan mencari tempat KKP?",
    "Apa yang harus dilakukan mahasiswa setelah mendapat surat balasan dari instansi tempat KKP?",
    "Berapa kali minimal mahasiswa harus melapor ke dosen pembimbing selama pelaksanaan KKP?",
    "Apa yang harus dibawa mahasiswa setiap kali melakukan bimbingan KKP?",
    "Apa saja berkas yang harus dilampirkan saat mendaftar ujian KKP?",
    "Berapa maksimal tingkat kemiripan (plagiarisme) yang diperbolehkan untuk laporan KKP?",
    "Berapa lama waktu maksimal pelaksanaan seminar KKP setelah persetujuan dosen pembimbing?",
    "Berapa minimal halaman laporan KKP?",
    "Apa ketentuan pakaian saat ujian KKP untuk mahasiswa pria?",
    "Apa ketentuan pakaian saat ujian KKP untuk mahasiswi berjilbab?",
    "Berapa lama durasi ujian KKP?",
    "Apa komponen penilaian dalam ujian KKP?",
    "Berapa nilai minimal untuk dinyatakan lulus ujian KKP?",
    "Apa yang harus dilakukan mahasiswa setelah ujian KKP?",
    "Berapa jumlah minimal referensi yang harus ada dalam laporan KKP?",
    "Apa format daftar pustaka yang digunakan dalam KKP?",
    "Berapa spasi yang digunakan untuk penulisan naskah utama KKP?",
    "Apa jenis dan ukuran font yang digunakan untuk naskah KKP?",
    "Berapa margin yang digunakan untuk penulisan laporan KKP?",
    "Apa ukuran kertas yang digunakan untuk laporan KKP?",
    "Berapa maksimal kata dalam abstrak KKP?",
    "Berapa jumlah kata kunci yang harus ada dalam abstrak KKP?",
    "Apa saja elemen yang harus ada di halaman sampul depan KKP?",
    "Apa yang dimuat dalam halaman pengesahan laporan KKP?",
    "Apa saja bagian yang termasuk dalam Bagian Awal laporan KKP?",
    "Apa saja bab yang termasuk dalam Bagian Utama laporan KKP?",
    "Bagaimana urutan penyusunan daftar pustaka dalam KKP?",
]


#
# EVALUATION FUNCTIONS
#

def evaluate_rag_no_ground_truth(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    dataset_name: str = "both"
) -> Dict:
    
    logger.info(f"Starting RAGAS evaluation (NO GROUND TRUTH) for {dataset_name} dataset...")
    logger.info(f"Number of questions: {len(questions)}")
    
    # Get settings
    settings = get_settings()
    
    # Create dataset
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
    }
    
    dataset = Dataset.from_dict(data)
    
    # Setup evaluator LLM dan embeddings
    logger.info("Setting up evaluator LLM and embeddings...")
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
    
    # Define metrics (NO GROUND TRUTH REQUIRED)
    context_precision_no_ref = LLMContextPrecisionWithoutReference(llm=evaluator_llm)
    
    metrics = [
        faithfulness,                  # Tidak halusinasi - jawaban didukung oleh context
        answer_relevancy,              # Jawaban relevan dengan pertanyaan
        context_precision_no_ref,      # Chunks yang diambil relevan dengan pertanyaan
    ]
    
    logger.info("Metrics to evaluate:")
    for metric in metrics:
        logger.info(f"  - {metric.name}")
    
    # Run evaluation
    logger.info("Running RAGAS evaluation...")
    results = evaluate(
        dataset, 
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )
    
    # Extract scores with safe conversion (handle list results)
    def _safe_score(value) -> float:
        import math
        from statistics import mean
        
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, (list, tuple)):
            # Filter out None and NaN values
            values = [v for v in value if v is not None and not (isinstance(v, float) and math.isnan(v))]
            return mean(values) if values else 0.0
        val = float(value)
        return 0.0 if math.isnan(val) else val
    
    scores = {
        "faithfulness": _safe_score(results["faithfulness"]),
        "answer_relevancy": _safe_score(results["answer_relevancy"]),
        "llm_context_precision_without_reference": _safe_score(results["llm_context_precision_without_reference"]),
    }
    
    # Calculate overall score (average of all metrics)
    overall_score = sum(scores.values()) / len(scores)
    scores["overall"] = overall_score
    
    # Define thresholds
    thresholds = {
        "faithfulness": 0.85,
        "answer_relevancy": 0.85,
        "llm_context_precision_without_reference": 0.80,
    }
    
    # Check if all metrics pass
    all_pass = all(scores[metric] >= thresholds[metric] for metric in thresholds)
    
    # Prepare results
    result = {
        "timestamp": datetime.now().isoformat(),
        "num_questions": len(questions),
        "dataset_type": dataset_name,
        "threshold": thresholds,
        "all_pass": all_pass,
        "scores": scores,
        "details": []
    }
    
    # Add detailed results for each question
    for i in range(len(questions)):
        # Helper function to get score at index
        def _get_score_at_index(metric_result, index):
            if hasattr(metric_result, "tolist"):
                metric_result = metric_result.tolist()
            if isinstance(metric_result, (list, tuple)):
                return metric_result[index] if index < len(metric_result) else None
            return metric_result
        
        detail = {
            "index": i + 1,
            "question": questions[i],
            "answer": answers[i],
            "contexts": contexts[i],
            "metrics": {
                "faithfulness": _get_score_at_index(results["faithfulness"], i),
                "answer_relevancy": _get_score_at_index(results["answer_relevancy"], i),
                "llm_context_precision_without_reference": _get_score_at_index(results["llm_context_precision_without_reference"], i),
            }
        }
        result["details"].append(detail)
    
    # Log results
    logger.info("\n" + "="*80)
    logger.info("EVALUATION RESULTS (NO GROUND TRUTH)")
    logger.info("="*80)
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Questions: {len(questions)}")
    logger.info(f"\nScores:")
    for metric, score in scores.items():
        threshold = thresholds.get(metric, 0.85)
        status = "✅" if metric == "overall" or score >= threshold else "❌"
        logger.info(f"  {status} {metric}: {score:.4f} (threshold: {threshold})")
    logger.info(f"\nOverall Pass: {'✅ YES' if all_pass else '❌ NO'}")
    logger.info("="*80)
    
    return result


def save_evaluation_results(results: Dict, filename: str = None):
    
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_results_no_gt_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Results saved to: {filename}")
    return filename


#
# MAIN EVALUATION FUNCTION
#

def run_full_evaluation_no_gt(rag_pipeline_func, dataset: str = "both"):
    
    # Select questions based on dataset
    if dataset == "pi":
        questions = EVAL_QUESTIONS_PI
    elif dataset == "kkp":
        questions = EVAL_QUESTIONS_KKP
    elif dataset == "both":
        questions = EVAL_QUESTIONS_PI + EVAL_QUESTIONS_KKP
    else:
        raise ValueError(f"Invalid dataset: {dataset}. Must be 'pi', 'kkp', or 'both'")
    
    logger.info(f"Running evaluation on {len(questions)} questions from {dataset} dataset...")
    
    # Generate answers and collect contexts
    answers = []
    contexts = []
    
    for i, question in enumerate(questions, 1):
        logger.info(f"Processing question {i}/{len(questions)}: {question[:60]}...")
        
        try:
            answer, context_list = rag_pipeline_func(question)
            answers.append(answer)
            contexts.append(context_list)
        except Exception as e:
            logger.error(f"Error processing question {i}: {e}")
            answers.append("Error generating answer")
            contexts.append([])
    
    # Run evaluation
    results = evaluate_rag_no_ground_truth(
        questions=questions,
        answers=answers,
        contexts=contexts,
        dataset_name=dataset
    )
    
    # Save results
    filename = save_evaluation_results(results)
    
    return results, filename


if __name__ == "__main__":
    # Example usage
    from main import run_rag_pipeline
    
    logger.info("Starting RAGAS evaluation WITHOUT ground truth...")
    results, filename = run_full_evaluation_no_gt(run_rag_pipeline, dataset="both")
    
    logger.info(f"\n✅ Evaluation complete!")
    logger.info(f"Results saved to: {filename}")
