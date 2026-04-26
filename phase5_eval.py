"""
Phase 5 – Evaluation: So sánh 4 cấu hình A/B/C/D.

Ma trận thực nghiệm:
  |             | LLM gốc     | LLM fine-tuned |
  |-------------|-------------|----------------|
  | Không RAG   | A           | C              |
  | Có RAG      | B           | D              |

Metrics:
  - BLEU, ROUGE-L, BERTScore (generation quality)
  - Recall@5 (retrieval quality, chỉ B & D)
  - Human eval 50 câu (manual scoring)

Cài đặt:
  pip install sacrebleu rouge-score bert-score datasets
"""

import json
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

# ══════════════════════════════════════════════════════════
# CẤU HÌNH
# ══════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).resolve().parent if "__file__" in dir() else Path(".")
PROCESSED_DIR = BASE_DIR / "processed"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════

def compute_bleu(predictions: list[str], references: list[str]) -> float:
    """BLEU score"""
    import sacrebleu
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return bleu.score


def compute_rouge_l(predictions: list[str], references: list[str]) -> float:
    """ROUGE-L F1 score"""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    
    scores = []
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        scores.append(score['rougeL'].fmeasure)
    
    return np.mean(scores) * 100


def compute_bertscore(predictions: list[str], references: list[str]) -> float:
    """BERTScore F1"""
    from bert_score import score as bert_score
    
    P, R, F1 = bert_score(
        predictions, references,
        lang="vi",
        verbose=False
    )
    return F1.mean().item() * 100


def compute_recall_at_k(retrieved_sources: list[list[str]], 
                         gold_sources: list[str], k: int = 5) -> float:
    """
    Recall@K: Tỷ lệ câu hỏi mà gold source nằm trong top-K retrieved.
    """
    hits = 0
    for retrieved, gold in zip(retrieved_sources, gold_sources):
        top_k = retrieved[:k]
        if any(gold.lower() in r.lower() for r in top_k):
            hits += 1
    
    return (hits / len(gold_sources)) * 100 if gold_sources else 0


# ══════════════════════════════════════════════════════════
# EVALUATION RUNNER
# ══════════════════════════════════════════════════════════

def run_evaluation():
    """Chạy evaluation 4 cấu hình"""
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   PHASE 5 – Evaluation (4 Configurations)              ║")
    print("╚══════════════════════════════════════════════════════════╝\n")
    
    # Load test set
    test_path = PROCESSED_DIR / "qa_test.json"
    if not test_path.exists():
        print(f"❌ Chưa có {test_path}!")
        return
    
    with open(test_path, "r", encoding="utf-8") as f:
        test_set = json.load(f)
    
    print(f"📄 Test set: {len(test_set)} câu hỏi\n")
    
    # Import RAG pipeline components
    from phase4_rag import RAGPipeline
    
    configs = {
        "A": {"use_rag": False, "use_finetuned": False, "label": "LLM gốc, không RAG"},
        "B": {"use_rag": True,  "use_finetuned": False, "label": "LLM gốc + RAG"},
        "C": {"use_rag": False, "use_finetuned": True,  "label": "Fine-tuned, không RAG"},
        "D": {"use_rag": True,  "use_finetuned": True,  "label": "Fine-tuned + RAG"},
    }
    
    all_results = {}
    
    for config_name, config in configs.items():
        print(f"\n{'='*60}")
        print(f"⚙️  Config {config_name}: {config['label']}")
        print(f"{'='*60}")
        
        # Init pipeline
        pipeline = RAGPipeline(use_finetuned=config["use_finetuned"])
        
        predictions = []
        references = []
        retrieved_sources = []
        
        for i, qa in enumerate(test_set):
            print(f"  [{i+1}/{len(test_set)}] {qa['question'][:50]}...")
            
            result = pipeline.answer(qa["question"], use_rag=config["use_rag"])
            
            predictions.append(result["answer"])
            references.append(qa["answer"])
            retrieved_sources.append(result.get("sources", []))
            
            time.sleep(0.1)  # Tránh overload GPU
        
        # Compute metrics
        print(f"\n📊 Computing metrics for Config {config_name}...")
        
        bleu = compute_bleu(predictions, references)
        rouge_l = compute_rouge_l(predictions, references)
        bert_f1 = compute_bertscore(predictions, references)
        
        metrics = {
            "config": config_name,
            "label": config["label"],
            "bleu": round(bleu, 2),
            "rouge_l": round(rouge_l, 2),
            "bertscore_f1": round(bert_f1, 2),
        }
        
        # Recall@5 chỉ cho configs có RAG
        if config["use_rag"]:
            gold_sources = [qa.get("source", "") for qa in test_set]
            recall5 = compute_recall_at_k(retrieved_sources, gold_sources, k=5)
            metrics["recall_at_5"] = round(recall5, 2)
        else:
            metrics["recall_at_5"] = "N/A"
        
        all_results[config_name] = metrics
        
        print(f"\n   BLEU:        {metrics['bleu']}")
        print(f"   ROUGE-L:     {metrics['rouge_l']}")
        print(f"   BERTScore:   {metrics['bertscore_f1']}")
        print(f"   Recall@5:    {metrics['recall_at_5']}")
        
        # Lưu predictions
        pred_path = RESULTS_DIR / f"predictions_{config_name}.json"
        preds_data = [
            {"question": qa["question"], "gold": qa["answer"], 
             "prediction": pred, "sources": srcs}
            for qa, pred, srcs in zip(test_set, predictions, retrieved_sources)
        ]
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(preds_data, f, ensure_ascii=False, indent=2)
    
    # ── Tổng kết ──
    print(f"\n\n{'='*70}")
    print("📊 BẢNG TỔNG KẾT KẾT QUẢ")
    print(f"{'='*70}")
    print(f"{'Config':<8} {'Label':<30} {'BLEU':>8} {'ROUGE-L':>9} {'BERT-F1':>9} {'R@5':>8}")
    print("-" * 70)
    
    for name in ["A", "B", "C", "D"]:
        m = all_results[name]
        print(f"{m['config']:<8} {m['label']:<30} {m['bleu']:>8} {m['rouge_l']:>9} "
              f"{m['bertscore_f1']:>9} {str(m['recall_at_5']):>8}")
    
    # Lưu kết quả
    results_path = RESULTS_DIR / "evaluation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Kết quả đã lưu: {results_path}")
    
    # Template cho human eval
    generate_human_eval_template(test_set)


def generate_human_eval_template(test_set):
    """Tạo template CSV cho human evaluation 50 câu"""
    import csv
    
    template_path = RESULTS_DIR / "human_eval_template.csv"
    
    with open(template_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "STT", "Câu hỏi", "Gold Answer",
            "Pred_A", "Score_A_Accuracy", "Score_A_Completeness", "Score_A_Naturalness",
            "Pred_B", "Score_B_Accuracy", "Score_B_Completeness", "Score_B_Naturalness",
            "Pred_C", "Score_C_Accuracy", "Score_C_Completeness", "Score_C_Naturalness",
            "Pred_D", "Score_D_Accuracy", "Score_D_Completeness", "Score_D_Naturalness",
        ])
        
        for i, qa in enumerate(test_set[:50], 1):
            writer.writerow([
                i, qa["question"], qa["answer"],
                "", "", "", "",  # Config A
                "", "", "", "",  # Config B
                "", "", "", "",  # Config C
                "", "", "", "",  # Config D
            ])
    
    print(f"📝 Human eval template: {template_path}")
    print("   Chấm điểm 1-5 cho mỗi tiêu chí: Accuracy, Completeness, Naturalness")


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    run_evaluation()
