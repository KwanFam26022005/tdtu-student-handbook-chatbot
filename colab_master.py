"""
🎓 MASTER NOTEBOOK — Chatbot Sổ Tay Sinh Viên TDTU
===================================================
RAG + Fine-tuned LLM Pipeline

Hướng dẫn:
  1. Upload toàn bộ thư mục project lên Google Drive
  2. Mount Drive trong Colab
  3. Chạy từng cell theo thứ tự

Yêu cầu: Google Colab với GPU T4 (Free tier OK)
"""

# ══════════════════════════════════════════════════════════
# CELL 1: SETUP & MOUNT DRIVE
# ══════════════════════════════════════════════════════════
"""
# Chạy cell này đầu tiên trên Colab:

from google.colab import drive
drive.mount('/content/drive')

# Đường dẫn project trên Drive (SỬA LẠI cho đúng)
PROJECT_DIR = "/content/drive/MyDrive/datamining"

import sys
sys.path.insert(0, PROJECT_DIR)

%cd {PROJECT_DIR}

# Kiểm tra GPU
!nvidia-smi
"""

# ══════════════════════════════════════════════════════════
# CELL 2: CÀI ĐẶT DEPENDENCIES
# ══════════════════════════════════════════════════════════
"""
# Chạy trên Colab:

# Core
!pip install -q pymupdf easyocr Pillow

# NLP & RAG
!pip install -q sentence-transformers faiss-cpu rank-bm25 
!pip install -q langchain langchain-community

# Fine-tuning (Unsloth)
!pip install -Uq "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install -Uq --no-deps trl peft accelerate bitsandbytes

# Evaluation
!pip install -q sacrebleu rouge-score bert-score

# Demo
!pip install -q gradio

# QA Generation
!pip install -q google-generativeai

print("✅ Tất cả dependencies đã cài xong!")
"""

# ══════════════════════════════════════════════════════════
# CELL 3: PHASE 1 — OCR
# ══════════════════════════════════════════════════════════
"""
# Chạy trên Colab:

%run phase1_ocr.py

# Kiểm tra output
!ls -la raw_text/
!echo "---"
!head -50 raw_text/*.txt | head -100
"""

# ══════════════════════════════════════════════════════════
# CELL 4: PHASE 2A+2B — CLEANING & CHUNKING
# ══════════════════════════════════════════════════════════
"""
# Chạy trên Colab:

from phase2_process import process_cleaning, process_chunking

process_cleaning()
process_chunking()

# Kiểm tra chunks
import json
with open("processed/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)
print(f"\\nTổng chunks: {len(chunks)}")
print(f"Ví dụ chunk đầu tiên:\\n{chunks[0]['text_with_context'][:300]}")
"""

# ══════════════════════════════════════════════════════════
# CELL 5: PHASE 2C — BUILD VECTOR STORE
# ══════════════════════════════════════════════════════════
"""
# Chạy trên Colab:

from phase2_process import build_vector_store
build_vector_store()
"""

# ══════════════════════════════════════════════════════════
# CELL 6: PHASE 2D — GENERATE QA PAIRS
# ══════════════════════════════════════════════════════════
"""
# Đặt API key (chọn 1 trong 2 cách):

# Cách 1: Colab Secrets (khuyến nghị)
from google.colab import userdata
import os
os.environ["GOOGLE_API_KEY"] = userdata.get('GOOGLE_API_KEY')

# Cách 2: Nhập trực tiếp (KHÔNG commit lên GitHub!)
# os.environ["GOOGLE_API_KEY"] = "AIza..."

from phase2_process import process_qa_generation
process_qa_generation()

# Kiểm tra
import json
with open("processed/qa_train.json", "r", encoding="utf-8") as f:
    train = json.load(f)
with open("processed/qa_test.json", "r", encoding="utf-8") as f:
    test = json.load(f)
print(f"Train: {len(train)} | Test: {len(test)}")
print(f"\\nVí dụ QA:")
print(f"  Q: {train[0]['question']}")
print(f"  A: {train[0]['answer'][:200]}")
"""

# ══════════════════════════════════════════════════════════
# CELL 7: PHASE 3 — FINE-TUNING
# ══════════════════════════════════════════════════════════
"""
# ⚠️ Cell này mất ~30-60 phút trên T4

%run phase3_finetune.py

# Kiểm tra output
!ls -la outputs/finetune/lora_adapter/
"""

# ══════════════════════════════════════════════════════════
# CELL 8: PHASE 4 — TEST RAG PIPELINE
# ══════════════════════════════════════════════════════════
"""
# Test nhanh RAG pipeline

from phase4_rag import RAGPipeline

# Test với base model + RAG (Config B)
pipeline_b = RAGPipeline(use_finetuned=False)
result = pipeline_b.answer("Sinh viên cần bao nhiêu tín chỉ để tốt nghiệp?", use_rag=True)
print(f"Config B: {result['answer']}")
print(f"Sources: {result['sources']}")

# Test với fine-tuned model + RAG (Config D)
pipeline_d = RAGPipeline(use_finetuned=True)
result = pipeline_d.answer("Sinh viên cần bao nhiêu tín chỉ để tốt nghiệp?", use_rag=True)
print(f"\\nConfig D: {result['answer']}")
print(f"Sources: {result['sources']}")
"""

# ══════════════════════════════════════════════════════════
# CELL 9: PHASE 5 — FULL EVALUATION
# ══════════════════════════════════════════════════════════
"""
# ⚠️ Cell này mất ~1-2 giờ (chạy 4 configs × 50 câu)

%run phase5_eval.py

# Xem kết quả
import json
with open("results/evaluation_results.json", "r", encoding="utf-8") as f:
    results = json.load(f)

import pandas as pd
df = pd.DataFrame(results.values())
print(df.to_markdown(index=False))
"""

# ══════════════════════════════════════════════════════════
# CELL 10: PHASE 6 — GRADIO DEMO
# ══════════════════════════════════════════════════════════
"""
# Chạy demo (Colab sẽ tạo public link)

%run phase6_demo.py
"""

# ══════════════════════════════════════════════════════════
# CELL 11: UPLOAD ARTIFACTS
# ══════════════════════════════════════════════════════════
"""
# Push lên HuggingFace Hub

import os
os.environ["HF_TOKEN"] = "hf_..."  # Token của bạn

# Upload LoRA adapter
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="outputs/finetune/lora_adapter",
    repo_id="your-username/tdtu-qwen2.5-3b-lora",
    repo_type="model",
    token=os.environ["HF_TOKEN"]
)

# Upload dataset
api.upload_folder(
    folder_path="processed",
    repo_id="your-username/tdtu-student-handbook-qa",
    repo_type="dataset",
    token=os.environ["HF_TOKEN"]
)

print("✅ Đã upload lên HuggingFace Hub!")
"""

print("📋 Đây là file hướng dẫn. Chạy từng cell trên Google Colab.")
print("   Xem các comment block trong file để biết cell nào chạy cell nào.")
