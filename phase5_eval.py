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
import csv
import numpy as np
from pathlib import Path
from datetime import datetime
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
        print(f"[ERROR] Chua co {test_path}!")
        return
    
    with open(test_path, "r", encoding="utf-8") as f:
        test_set = json.load(f)
    
    print(f"[INFO] Test set: {len(test_set)} cau hoi\n")
    
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
        print(f"[CONFIG] Config {config_name}: {config['label']}")
        print(f"{'='*60}")
        
        # Init pipeline
        config_start_time = time.time()
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
        
        config_elapsed = time.time() - config_start_time
        
        # Compute metrics
        print(f"\n[STATS] Computing metrics for Config {config_name}...")
        
        bleu = compute_bleu(predictions, references)
        rouge_l = compute_rouge_l(predictions, references)
        bert_f1 = compute_bertscore(predictions, references)
        
        metrics = {
            "config": config_name,
            "label": config["label"],
            "bleu": round(bleu, 2),
            "rouge_l": round(rouge_l, 2),
            "bertscore_f1": round(bert_f1, 2),
            "time_seconds": round(config_elapsed, 1),
            "num_samples": len(test_set),
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
    print("[STATS] BANG TONG KET KET QUA")
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
    
    print(f"\n[OK] Ket qua da luu: {results_path}")

    # --- Lưu CSV bảng so sánh ---
    save_comparison_csv(all_results)

    # --- Lưu metadata ---
    save_run_metadata(all_results, test_set)

    # --- Sinh biểu đồ ---
    generate_charts(all_results)

    # --- Template human eval (auto-fill predictions) ---
    generate_human_eval_template(test_set)


def generate_human_eval_template(test_set):
    """Tạo template CSV cho human evaluation 50 câu, auto-fill predictions nếu có."""
    template_path = RESULTS_DIR / "human_eval_template.csv"

    # Try to load predictions from saved files
    config_preds = {}
    for cfg in ["A", "B", "C", "D"]:
        pred_path = RESULTS_DIR / f"predictions_{cfg}.json"
        if pred_path.exists():
            with open(pred_path, "r", encoding="utf-8") as f:
                config_preds[cfg] = json.load(f)

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
            row = [i, qa["question"], qa["answer"]]
            for cfg in ["A", "B", "C", "D"]:
                if cfg in config_preds and i - 1 < len(config_preds[cfg]):
                    pred_text = config_preds[cfg][i - 1].get("prediction", "")
                else:
                    pred_text = ""
                row.extend([pred_text, "", "", ""])  # pred + 3 empty score columns
            writer.writerow(row)

    print(f"[INFO] Human eval template: {template_path}")
    print("   Chấm điểm 1-5 cho mỗi tiêu chí: Accuracy, Completeness, Naturalness")


def save_comparison_csv(all_results: dict):
    """Export comparison table as CSV for reports/slides."""
    csv_path = RESULTS_DIR / "comparison_table.csv"

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Config", "Label", "BLEU", "ROUGE-L", "BERTScore_F1",
                         "Recall@5", "Time(s)", "Num_Samples"])
        for name in ["A", "B", "C", "D"]:
            m = all_results[name]
            writer.writerow([
                m["config"], m["label"],
                m["bleu"], m["rouge_l"], m["bertscore_f1"],
                m.get("recall_at_5", "N/A"),
                m.get("time_seconds", "N/A"),
                m.get("num_samples", ""),
            ])

    print(f"[OK] Comparison CSV: {csv_path}")


def save_run_metadata(all_results: dict, test_set: list):
    """Save run metadata: timestamps, config, dataset info."""
    meta = {
        "run_timestamp": datetime.now().isoformat(),
        "test_set_size": len(test_set),
        "test_set_path": str(PROCESSED_DIR / "qa_test.json"),
        "configs": {
            name: {
                "label": r["label"],
                "time_seconds": r.get("time_seconds"),
            }
            for name, r in all_results.items()
        },
        "total_time_seconds": sum(
            r.get("time_seconds", 0) for r in all_results.values()
            if isinstance(r.get("time_seconds"), (int, float))
        ),
        "metrics_computed": ["BLEU", "ROUGE-L", "BERTScore_F1", "Recall@5"],
    }

    meta_path = RESULTS_DIR / "run_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] Run metadata: {meta_path}")


def generate_charts(all_results: dict):
    """Generate bar chart comparing 4 configs across metrics."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend (works on server/Colab)
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed. Skip chart generation.")
        print("   Install: pip install matplotlib")
        return

    configs = ["A", "B", "C", "D"]
    labels = [all_results[c]["label"] for c in configs]
    metrics = ["bleu", "rouge_l", "bertscore_f1"]
    metric_labels = ["BLEU", "ROUGE-L", "BERTScore F1"]

    x = np.arange(len(configs))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
        values = [float(all_results[c].get(metric, 0)) for c in configs]
        bars = ax.bar(x + i * width, values, width, label=mlabel)
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Configuration")
    ax.set_ylabel("Score")
    ax.set_title("So s\u00e1nh 4 C\u1ea5u h\u00ecnh RAG - BLEU / ROUGE-L / BERTScore")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"{c}\n{l}" for c, l in zip(configs, labels)],
                        fontsize=8)
    ax.legend()
    ax.set_ylim(0, max(
        max(float(all_results[c].get(m, 0)) for c in configs)
        for m in metrics
    ) * 1.2 + 1)
    plt.tight_layout()

    chart_path = RESULTS_DIR / "comparison_chart.png"
    fig.savefig(str(chart_path), dpi=150)
    plt.close(fig)
    print(f"[OK] Chart saved: {chart_path}")

    # --- Recall@5 chart (only B & D) ---
    rag_configs = [c for c in configs if all_results[c].get("recall_at_5") != "N/A"]
    if rag_configs:
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        recall_vals = [float(all_results[c]["recall_at_5"]) for c in rag_configs]
        recall_labels = [f"{c}: {all_results[c]['label']}" for c in rag_configs]
        bars = ax2.bar(recall_labels, recall_vals, color=["#4C78A8", "#E45756"])
        for bar, val in zip(bars, recall_vals):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=10)
        ax2.set_ylabel("Recall@5 (%)")
        ax2.set_title("Retrieval Quality: Recall@5")
        ax2.set_ylim(0, 105)
        plt.tight_layout()

        recall_path = RESULTS_DIR / "recall_chart.png"
        fig2.savefig(str(recall_path), dpi=150)
        plt.close(fig2)
        print(f"[OK] Recall chart: {recall_path}")


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    run_evaluation()
