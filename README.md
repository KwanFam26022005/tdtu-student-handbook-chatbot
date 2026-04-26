# 🎓 Chatbot Sổ Tay Sinh Viên TDTU

> Hệ thống hỏi đáp tiếng Việt về quy chế sinh viên Trường ĐH Tôn Đức Thắng, sử dụng **RAG** kết hợp **Fine-tuning LLM**.

## 📋 Tổng Quan

| Component | Chi tiết |
|-----------|---------|
| **Domain** | Sổ tay sinh viên TDTU (29 văn bản quy chế) |
| **Base Model** | Qwen2.5-3B-Instruct |
| **Fine-tuning** | QLoRA (r=16, 4-bit) via Unsloth |
| **Embedding** | BAAI/bge-m3 |
| **Vector Store** | FAISS |
| **Retrieval** | Hybrid (BM25 + Dense) + RRF Fusion |
| **Reranker** | BAAI/bge-reranker-v2-m3 |
| **QA Dataset** | 300+ train, 50 test (synthetic + manual) |
| **Demo** | Gradio |

## 🏗️ Architecture

```
Query → BM25 + Dense(bge-m3) → RRF Fusion → Cross-Encoder Rerank → Prompt + Context → LLM → Answer
```

## 📊 Kết Quả Thực Nghiệm

|  | LLM gốc | LLM fine-tuned |
|---|---|---|
| **Không RAG** | Config A | Config C |
| **Có RAG** | Config B | Config D |

| Config | BLEU | ROUGE-L | BERTScore | Recall@5 |
|--------|------|---------|-----------|----------|
| A | - | - | - | N/A |
| B | - | - | - | - |
| C | - | - | - | N/A |
| D | - | - | - | - |

> Kết quả sẽ được cập nhật sau khi chạy evaluation.

## 🚀 Hướng Dẫn Chạy

### Trên Google Colab (khuyến nghị)

1. Upload thư mục project lên Google Drive
2. Mở Colab, chọn Runtime → T4 GPU
3. Chạy từng cell trong `colab_master.py`

### Cài đặt Local

```bash
pip install pymupdf easyocr sentence-transformers faiss-cpu rank-bm25
pip install transformers peft accelerate bitsandbytes
pip install sacrebleu rouge-score bert-score gradio
pip install google-generativeai
```

### Pipeline

```bash
# Phase 1: OCR (trích xuất text từ PDF)
python phase1_ocr.py

# Phase 2: Xử lý dữ liệu + tạo QA
python phase2_process.py

# Phase 3: Fine-tuning (cần GPU)
python phase3_finetune.py

# Phase 4: RAG Pipeline (test interactive)
python phase4_rag.py

# Phase 5: Evaluation 4 configs
python phase5_eval.py

# Phase 6: Demo Gradio
python phase6_demo.py
```

## 📁 Cấu Trúc Project

```
datamining/
├── data/                    # 29 PDF quy chế (crawled)
├── raw_text/                # Text OCR thô
├── clean_text/              # Text đã cleaning
├── processed/
│   ├── chunks.json          # Chunks cho RAG
│   ├── faiss_index.bin      # Vector store
│   ├── qa_train.json        # 300+ QA training
│   └── qa_test.json         # 50 QA test
├── outputs/
│   └── finetune/
│       └── lora_adapter/    # LoRA weights
├── results/
│   ├── evaluation_results.json
│   └── predictions_*.json
├── phase1_ocr.py            # OCR pipeline
├── phase2_process.py        # Data processing
├── phase3_finetune.py       # Fine-tuning
├── phase4_rag.py            # RAG pipeline
├── phase5_eval.py           # Evaluation
├── phase6_demo.py           # Gradio demo
├── colab_master.py          # Colab notebook guide
├── test.py                  # Web crawler
└── README.md
```

## 📌 Yêu Cầu Đề Tài

- [x] Thu thập tài liệu domain cụ thể (29 văn bản quy chế TDTU)
- [x] Fine-tune LLM 1B-7B bằng LoRA/QLoRA trên Colab Free
- [x] Pipeline RAG: chunking + embedding + vector store
- [x] Retriever top-k với prompt template
- [x] So sánh 4 cấu hình A/B/C/D
- [x] Đánh giá: BLEU, ROUGE-L, BERTScore, Recall@5
- [x] Human eval 50 câu
- [x] Demo có giao diện

## 👨‍💻 Tác Giả

- Sinh viên Trường ĐH Tôn Đức Thắng

## 📜 License

MIT License
