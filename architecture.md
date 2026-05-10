# Chatbot Hỏi Đáp Sổ Tay Sinh Viên TDTU — RAG + Fine-tuning Qwen2.5-3B (QLoRA)

**Sinh viên:** Phạm Hồng Đăng Khoa  
**GVHD:** PGS.TS Lê Anh Cường  
**Trường:** Đại học Tôn Đức Thắng (TDTU)  
**Học kỳ:** 2025–2026

---

## 1. Tổng quan

### 1.1 Bài toán

Xây dựng hệ thống chatbot hỏi đáp tiếng Việt chuyên biệt cho domain **Sổ tay Sinh viên TDTU** (29 văn bản quy chế), kết hợp hai kỹ thuật:

- **Retrieval-Augmented Generation (RAG):** truy xuất đoạn quy chế liên quan trước khi sinh câu trả lời, giảm hallucination.
- **Fine-tuning LLM bằng QLoRA:** tinh chỉnh mô hình ngôn ngữ lớn trên dữ liệu domain-specific, cải thiện chất lượng sinh ngôn ngữ.

### 1.2 Stack kỹ thuật tổng hợp

| Thành phần | Công nghệ |
|---|---|
| OCR | GPT-4o-mini Vision (primary) / Surya OCR (fallback) |
| Embedding | BAAI/bge-m3 (multilingual, 1024-dim) |
| Vector Store | FAISS (IndexFlatIP — cosine similarity) |
| Sparse Retriever | BM25 (rank_bm25) |
| Fusion | Reciprocal Rank Fusion (RRF, k=60) |
| Reranker | BAAI/bge-reranker-v2-m3 (Cross-Encoder) |
| Base LLM | Qwen2.5-3B-Instruct |
| Fine-tuning | QLoRA 4-bit (r=16, α=32, NF4) |
| Training | SFTTrainer (trl) + AdamW 8-bit |
| Evaluation | BLEU, ROUGE-L, BERTScore, Recall@5, Human Eval |
| Demo UI | Gradio (brand TDTU) |

---

## 2. Kiến trúc tổng thể

### 2.1 Sơ đồ End-to-End 6 Phase

```mermaid
flowchart TB
    subgraph Phase1["Phase 1 — OCR Pipeline"]
        PDF["📄 29 PDF quy chế TDTU"]
        OCR["GPT-4o-mini Vision\n+ Surya fallback"]
        RAW["raw_text/*.txt"]
        PDF --> OCR --> RAW
    end

    subgraph Phase2["Phase 2 — Data Processing (phase2_process.py)"]
        CLEAN["2A: Cleaning &\nNormalization"]
        CHUNK["2B: Semantic Chunking\n(Điều/Khoản aware)"]
        NORM["2B+: Chunk Normalize\n(merge tiny / split large)"]
        FAISS["2C: FAISS Index\n(bge-m3 embeddings)"]
        QA["2D: QA Generation\n(API + Template)"]
        RAW --> CLEAN --> CHUNK --> NORM --> FAISS
        CHUNK --> QA
    end

    subgraph Phase3["Phase 3 — Fine-tuning"]
        BASE["Qwen2.5-3B-Instruct"]
        QLORA["QLoRA 4-bit\nr=16, α=32, 3 epochs"]
        LORA["LoRA Adapter"]
        QA --> QLORA
        BASE --> QLORA --> LORA
    end

    subgraph Phase4["Phase 4 — RAG Pipeline"]
        QUERY["User Query"]
        REWRITE["Query Rewriter\n(synonym mapping)"]
        FILTER["Metadata Filter\n(topic → source)"]
        HYBRID["Hybrid Retrieval\nBM25 + Dense → RRF"]
        RERANK["Cross-Encoder Reranker\nbge-reranker-v2-m3"]
        EXPAND["Parent Context\nExpander"]
        COMPRESS["Contextual\nCompressor"]
        GEN["LLM Generator\n(base or fine-tuned)"]
        ANS["Answer + Sources"]
        QUERY --> REWRITE --> FILTER
        FAISS --> HYBRID
        FILTER --> HYBRID --> RERANK --> EXPAND --> COMPRESS --> GEN --> ANS
        LORA -.-> GEN
    end

    subgraph Phase5["Phase 5 — Evaluation"]
        EVAL["4 Configs: A/B/C/D\nBLEU, ROUGE-L,\nBERTScore, Recall@5"]
        ANS --> EVAL
    end

    subgraph Phase6["Phase 6 — Gradio Demo"]
        UI["Gradio Chat UI\nBrand TDTU"]
        ANS --> UI
    end
```

### 2.2 Sơ đồ RAG Pipeline chi tiết (Phase 4)

```mermaid
flowchart LR
    Q["Query"] --> QR["QueryRewriter"]
    QR --> MF["MetadataFilter"]
    MF --> DR["DenseRetriever\nFAISS + bge-m3"]
    MF --> SR["SparseRetriever\nBM25"]
    DR --> RRF["RRF Fusion\nk=60"]
    SR --> RRF
    RRF --> RE["CrossEncoder\nReranker\ntop-5"]
    RE --> PE["ParentContext\nExpander\n+2 siblings"]
    PE --> CC["Contextual\nCompressor"]
    CC --> LLM["Qwen2.5-3B\n+ LoRA"]
    LLM --> A["Answer"]
```

---

## 3. Chi tiết từng Phase

---

### Phase 1 — OCR Pipeline (`phase1_ocr.py`)

**Mục tiêu:** Trích xuất text từ 29 file PDF quy chế TDTU (dạng ảnh scan) thành plain text.

**Kiến trúc engine (fallback chain):**

| Thứ tự | Engine | Ưu điểm | Điều kiện |
|---|---|---|---|
| 1 | GPT-4o-mini Vision | Hiểu bảng, tiếng Việt có dấu chính xác, giữ layout Markdown | Cần `OPENAI_API_KEY` |
| 2 | Surya OCR | Offline, GPU-based, chất lượng tốt | Cần `pip install surya-ocr` |
| 3 | EasyOCR | Legacy fallback, chất lượng thấp nhất | Cần `pip install easyocr` |

**Chi tiết kỹ thuật:**

- **Render:** PyMuPDF (fitz) → PNG 300 DPI
- **Structured prompt:** Yêu cầu OCR giữ nguyên bảng Markdown, ký hiệu toán học (≥, ≤), cấu trúc Điều/Khoản, lọc con dấu/chữ ký → tag `[Con dấu]`, `[Chữ ký]`
- **Post-processing:** Chuẩn hóa Unicode NFC, sửa lỗi OCR chữ Đ (`Ðiều` → `Điều`), xóa ký tự điều khiển, gộp khoảng trắng thừa
- **Progress tracking:** Resume-capable — lưu `ocr_progress.json`, skip file đã OCR
- **Rate limiting:** 1s delay giữa request, retry 3 lần, xử lý HTTP 429

**Input/Output:**
- Input: `data/*.pdf` (29 file)
- Output: `raw_text/*.txt`

**Kết quả thực nghiệm Phase 1:**

| Thống kê | Giá trị |
|---|---|
| Số file PDF đã OCR | 29 |
| Tổng số từ | 137,166 |
| Tổng ký tự | 624,667 |
| Trung bình/file | 4,729 từ |
| File dài nhất | quydinhhoanphi.txt (28,664 từ) |
| File ngắn nhất | Nhiệm vụ thực hiện 3 nội dung đạo đức.txt (286 từ) |

**Mức độ bảo toàn cấu trúc Markdown sau OCR:**

| Loại cấu trúc | Số lượng nhận diện |
|---|---|
| Headings (Chương/Điều) | 55 |
| Table rows (dòng bảng `\|`) | 2,975 |
| List items (gạch đầu dòng) | 1,341 |

**Phân bố từ khóa cốt lõi trong corpus:**

| Từ khóa | Tần suất |
|---|---|
| sinh viên | 1,307 |
| học bổng | 787 |
| điều | 696 |
| tốt nghiệp | 327 |
| chứng chỉ | 128 |
| kỷ luật | 123 |
| khoản | 118 |
| ielts | 55 |
| ưu tú | 42 |

> **Nhận xét:** OCR giữ được gần 3,000 dòng bảng Markdown — chứng minh GPT-4o-mini Vision hiệu quả cho tài liệu scan tiếng Việt có bảng biểu. Phân bố từ khóa cho thấy corpus tập trung vào chủ đề sinh viên (1,307 lần) và học bổng (787 lần), phù hợp cho domain chatbot sổ tay sinh viên.

> **SLIDE NOTE:**  
> "Phase 1 dùng GPT-4o-mini Vision để OCR 29 văn bản PDF quy chế TDTU thành 137,166 từ text, giữ nguyên 2,975 dòng bảng Markdown và 55 heading. Engine có fallback chain: OpenAI → Surya → EasyOCR, kèm resume capability. Post-processing sửa lỗi OCR tiếng Việt, chuẩn hóa Unicode NFC."

---

### Phase 2 — Data Processing (`phase2_process.py`)

**Mục tiêu:** Biến raw text thành knowledge base có cấu trúc: clean text → chunks → normalize → FAISS vector store → QA pairs cho fine-tuning. Toàn bộ pipeline nằm trong **một file duy nhất** `phase2_process.py` để dễ quản lý và chạy tuần tự.

**5 bước con:**

#### 2A — Cleaning & Normalization

- Unicode NFC, xóa ký tự điều khiển
- Loại tag `[Con dấu]`, `[Chữ ký]` từ Phase 1
- Sửa lỗi OCR chữ Đ (8 patterns)
- Gộp dòng bị ngắt giữa chừng (regex tiếng Việt, bảo toàn dòng bảng Markdown `|`)
- Gộp khoảng trắng thừa, xóa dòng trống liên tiếp

#### 2B — Semantic Chunking

- **Structure-aware mode** (≥3 section headers): Tách theo Chương/Điều/Khoản/Mục bằng regex
- **Paragraph-based mode** (fallback): Tách theo paragraph + overlap
- Config: `CHUNK_SIZE=512` tokens, `CHUNK_OVERLAP=64`, `MIN_CHUNK_LENGTH=50`
- Bảo toàn Markdown table (merge contiguous table lines thành 1 block)
- Thêm context header: `[source] - Chương X - Điều Y`

#### 2B+ — Chunk Normalization (tích hợp trong `phase2_process.py`)

- **Merge tiny chunks** (<80 chars): gộp vào chunk liền kề cùng source (`merge_tiny_chunks()`)
- **Split oversized chunks** (>3000 chars): tách tại paragraph boundary (`split_large_chunks()`)
- **Parent-child mapping:** build `parent_map.json` — liên kết chunk với siblings cùng section/chapter (`build_parent_map()`)
- Được gọi tự động giữa bước 2B và 2C trong pipeline

#### 2C — Build Vector Store

- Embedding model: **BAAI/bge-m3** (multilingual, 1024 dimensions)
- Index: **FAISS IndexFlatIP** (inner product = cosine vì đã normalize)
- Embed `text_with_context` (có context header)
- Batch size: 32, normalize_embeddings=True

#### 2D — Generate QA Pairs

- **API-based** (GPT-4o-mini): Sinh 3-5 QA/chunk, đa dạng loại (factual, conditional, procedural, reasoning)
- **Template-based** (fallback): Regex extraction — 2-4 QA/chunk, không cần API
- Mục tiêu: ≥300 QA pairs cho training
- Split: train/test (seed=42), test_size = min(50, 20%)
- Resume-capable: `qa_progress.json`

**Thứ tự thực thi khi chạy `python phase2_process.py`:**

```
2A (Cleaning) → 2B (Chunking) → 2B+ (Normalize) → 2C (FAISS) → 2D (QA Gen)
```

**Input/Output:**
- Input: `raw_text/*.txt`
- Output: `processed/chunks.json`, `processed/faiss_index.bin`, `processed/parent_map.json`, `processed/qa_train.json`, `processed/qa_test.json`

**Kết quả thực nghiệm Phase 2:**

*Chunking Statistics:*

| Thống kê | Giá trị |
|---|---|
| Tổng chunks | 768 |
| Từ 29 source files | 29 |
| Chunk có metadata Điều/Khoản | 351 (45.7%) |
| Chunk có metadata Chương | 160 (20.8%) |
| Unique chapters | 4 |
| Unique sections (Điều) | 36 |
| Chunks có parent_id (từ split) | 43 |

| Thống kê độ dài chunk | Giá trị (ký tự) |
|---|---|
| Min | 12 |
| Max | 14,241 |
| Trung bình | 918 |
| Median | 685 |
| Tiny chunks (<80 chars) | 11 |
| Oversized chunks (>3000 chars) | 19 |

*Top 5 văn bản có nhiều chunk nhất:*

| Văn bản | Số chunk |
|---|---|
| Hướng dẫn học bổng khen thưởng hỗ trợ người học và hỗ trợ khác | 106 |
| Hướng dẫn học bổng hỗ trợ cho cộng đồng (2025-2026) | 94 |
| Quy chế tổ chức và quản lý đào tạo (2021) | 88 |
| Quy định hoàn phí | 87 |
| Quy chế Công tác sinh viên | 50 |

*QA Dataset:*

| Thống kê | Giá trị |
|---|---|
| Tổng QA pairs | 3,226 |
| Train set | 3,176 |
| Test set | 50 |
| Nguồn sinh | 100% API (GPT-4o-mini) |
| Avg question length | 12 từ |
| Avg answer length | 40 từ |

| Loại câu hỏi | Số lượng | Tỷ lệ |
|---|---|---|
| Factual | 890 | 27.6% |
| Conditional | 795 | 24.6% |
| Procedural | 774 | 24.0% |
| Reasoning | 767 | 23.8% |

*Vector Store:*

| Thống kê | Giá trị |
|---|---|
| FAISS index size | 3.0 MB |
| Dimension | 1,024 (bge-m3) |
| Số vectors | 768 |
| Index type | IndexFlatIP (cosine) |
| Chunks chứa keyword "sinh viên" | 382/768 (49.7%) |

> **Nhận xét:** Chunking structure-aware nhận diện được 36 Điều khác nhau trên 4 Chương, bao phủ 45.7% tổng chunks — cho thấy phần lớn văn bản quy chế TDTU có cấu trúc pháp lý rõ ràng. QA dataset đạt 3,226 cặp (gấp 10× mục tiêu 300), phân bố đều 4 loại câu hỏi (~25% mỗi loại). Gần 50% chunks chứa keyword "sinh viên", khẳng định corpus phù hợp cho domain chatbot sổ tay sinh viên.

> **SLIDE NOTE:**  
> "Phase 2 xử lý 137K từ thành 768 chunks (median 685 ký tự), nhận diện 36 Điều pháp lý. FAISS index 3MB với bge-m3 1024-dim. Sinh 3,226 QA pairs bằng API, phân bố đều 4 loại: factual/conditional/procedural/reasoning. Train 3,176, test 50."

---

### Phase 3 — Fine-tuning Qwen2.5-3B QLoRA (`phase3_finetune.py`)

**Mục tiêu:** Fine-tune Qwen2.5-3B-Instruct trên dữ liệu domain TDTU bằng QLoRA để cải thiện chất lượng trả lời.

**Cấu hình model:**

| Tham số | Giá trị |
|---|---|
| Base model | `Qwen/Qwen2.5-3B-Instruct` |
| Quantization | NF4 4-bit (BitsAndBytesConfig) |
| LoRA rank (r) | 16 |
| LoRA alpha (α) | 32 |
| LoRA dropout | 0 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Trainable params | ~0.5% tổng params |

**Cấu hình training:**

| Tham số | Giá trị |
|---|---|
| Batch size (per device) | 4 |
| Gradient accumulation | 4 steps (effective batch = 16) |
| Learning rate | 2e-4 |
| Scheduler | Cosine |
| Epochs | 3 |
| Warmup steps | 10 |
| Precision | bf16 (auto-detect) |
| Optimizer | AdamW 8-bit |
| Max sequence length | 2048 tokens |
| Eval split | 5% |
| Save strategy | Mỗi 50 steps, giữ 2 checkpoint gần nhất |
| Best model | load_best_model_at_end (by eval_loss) |

**Features bổ sung:**
- **Google Drive backup callback:** tự động copy checkpoint lên Drive sau mỗi save
- **Resume from checkpoint:** tìm checkpoint local → Drive → train from scratch
- **Chat format:** System prompt + User question + Assistant answer (Qwen chat template)
- **HuggingFace Hub push:** tự động push LoRA adapter nếu có `HF_TOKEN`

**Input/Output:**
- Input: `processed/qa_train.json` (300+ QA pairs)
- Output: `outputs/finetune/lora_adapter/`

> **SLIDE NOTE:**  
> "Phase 3 fine-tune Qwen2.5-3B-Instruct bằng QLoRA 4-bit trên Google Colab T4. Chỉ train ~0.5% tham số (LoRA r=16, α=32) trên 7 attention/MLP modules. Training 3 epochs với AdamW 8-bit, cosine scheduler, effective batch size 16. Có auto-backup lên Google Drive và resume capability."

---

### Phase 4 — RAG Pipeline (`phase4_rag.py`)

**Mục tiêu:** Hệ thống truy xuất và sinh câu trả lời end-to-end với 5 cải tiến so với RAG cơ bản.

**Kiến trúc 7 bước:**

| Bước | Component | Mô tả |
|---|---|---|
| 1 | QueryRewriter | Dictionary-based synonym mapping (24 cặp slang → formal) |
| 2 | MetadataFilter | Pre-filter chunks theo topic (8 categories) → source mapping |
| 3 | HybridRetriever | BM25 (sparse) + FAISS/bge-m3 (dense), top-20 mỗi retriever |
| 4 | RRF Fusion | Reciprocal Rank Fusion (k=60), merge results từ multiple query variants |
| 5 | CrossEncoderReranker | bge-reranker-v2-m3, max_length=512, chọn top-5 |
| 6 | ParentContextExpander | Thêm tối đa 2 sibling chunks cùng section (score × 0.5) |
| 7 | ContextualCompressor | Keyword overlap scoring, giữ top-5 sentences/chunk + boost Điều/Khoản |

**5 cải tiến so với RAG cơ bản:**

1. **Hybrid Retrieval:** BM25 + Dense thay vì chỉ Dense → bắt được cả exact match lẫn semantic similarity
2. **Metadata Pre-filtering:** Thu hẹp search space theo topic trước khi retrieve
3. **Query Rewriting:** Chuyển slang sinh viên thành ngôn ngữ pháp lý (vd: "đuổi học" → "buộc thôi học")
4. **Parent Context Expansion:** Khi retrieve được Khoản 2, tự động kéo thêm Khoản 1, 3 cùng Điều
5. **Contextual Compression:** Chỉ giữ câu liên quan trong chunk, giảm noise cho LLM

**LLM Generator config:**
- 4-bit quantization, float16
- max_new_tokens=256, temperature=0.3, top_p=0.9
- repetition_penalty=1.1
- System prompt yêu cầu trích dẫn Điều/Khoản cụ thể

**Input/Output:**
- Input: User query + FAISS index + chunks + LoRA adapter (optional)
- Output: Answer + sources + sections + retrieval scores

> **SLIDE NOTE:**  
> "Phase 4 là RAG pipeline enhanced với 5 cải tiến: hybrid retrieval BM25+Dense, metadata pre-filtering, query rewriting cho slang sinh viên, parent context expansion kéo sibling chunks, và contextual compression giảm noise. Pipeline 7 bước từ query đến answer, rerank bằng cross-encoder bge-reranker-v2-m3."

---

### Phase 5 — Evaluation (`phase5_eval.py`)

**Mục tiêu:** So sánh 4 cấu hình A/B/C/D trên tập test 50 câu, đo lường cả generation quality lẫn retrieval quality.

**Ma trận thực nghiệm:**

|  | LLM gốc (Qwen2.5-3B) | LLM fine-tuned (+LoRA) |
|---|---|---|
| **Không RAG** | Config A | Config C |
| **Có RAG** | Config B | Config D |

**Metrics đo lường:**

| Metric | Loại | Mô tả |
|---|---|---|
| BLEU | Generation | Corpus-level BLEU (sacrebleu) — đo n-gram overlap |
| ROUGE-L | Generation | Longest common subsequence F1 — đo recall nội dung |
| BERTScore | Generation | Cosine similarity of contextual embeddings — đo semantic match |
| Recall@5 | Retrieval | % câu hỏi có gold source trong top-5 retrieved (chỉ B & D) |
| Human Eval | Manual | 50 câu × 3 tiêu chí (Accuracy, Completeness, Naturalness) × 4 configs |

**Output artifacts:**
- `results/evaluation_results.json` — kết quả tổng hợp
- `results/predictions_[A/B/C/D].json` — predictions từng config
- `results/comparison_table.csv` — bảng CSV cho báo cáo
- `results/comparison_chart.png` — bar chart so sánh 3 metrics
- `results/recall_chart.png` — Recall@5 chart (B vs D)
- `results/human_eval_template.csv` — template chấm điểm thủ công (auto-fill predictions)

> **SLIDE NOTE:**  
> "Phase 5 đánh giá 4 cấu hình theo ma trận 2×2: có/không RAG × gốc/fine-tuned. Dùng 4 metrics tự động (BLEU, ROUGE-L, BERTScore, Recall@5) và human evaluation 50 câu. Kết quả xuất sẵn CSV, chart, và template human eval."

---

### Phase 6 — Gradio Demo (`phase6_demo.py`)

**Mục tiêu:** Giao diện chatbot cho demo thuyết trình, brand identity TDTU.

**Features:**
- Chat interface với 8 câu hỏi mẫu
- Toggle RAG on/off và base/fine-tuned model
- Hiển thị nguồn trích dẫn (source + section)
- Brand TDTU: header gradient #1565C0, font Be Vietnam Pro, badge đỏ #E8293A
- Public link qua `share=True` (Colab compatible)

**Brand palette:**
- Primary: `#1565C0` (TDTU Blue) — 60%+ UI
- Accent: `#E8293A` (TDTU Red) — chỉ badge, divider
- Surface: `#E8F0FB` — input, card backgrounds
- Text: `#0A2A5E` (body), `#5A7AB0` (muted)

> **SLIDE NOTE:**  
> "Phase 6 là Gradio demo với brand identity TDTU: header xanh dương gradient, font Be Vietnam Pro, chat bubbles có style khác nhau cho bot/user. Hỗ trợ toggle RAG on/off và fine-tuned model để demo so sánh trực tiếp trên cùng câu hỏi."

---

## 4. Kết quả thực nghiệm

### 4.1 Bảng so sánh 4 cấu hình

| Config | Mô tả | BLEU | ROUGE-L | BERTScore F1 | Recall@5 | Thời gian |
|---|---|---|---|---|---|---|
| A | LLM gốc, không RAG | 4.06 | 26.37 | 68.51 | N/A | 859s |
| B | LLM gốc + RAG | 6.46 | 27.38 | 69.68 | 86.0% | 1,219s |
| C | Fine-tuned, không RAG | **17.38** | **43.68** | **77.31** | N/A | 481s |
| D | Fine-tuned + RAG (Full) | **18.53** | **46.25** | **77.93** | **86.0%** | 844s |

> **Phân tích kết quả:**
>
> 1. **Fine-tuning là yếu tố quyết định chất lượng sinh ngôn ngữ:**
>    - Config C (FT, no RAG) đạt BLEU 17.38, vượt xa Config A (gốc, no RAG) 4.06 — cải thiện **+328%**.
>    - ROUGE-L tăng từ 26.37 → 43.68 (+65.7%), BERTScore tăng 68.51 → 77.31 (+12.9%).
>    - Điều này cho thấy QLoRA fine-tuning trên 3,176 QA pairs đã dạy model hiểu đúng phong cách trả lời domain quy chế TDTU.
>
> 2. **RAG bổ sung thêm nhưng không phải yếu tố chính:**
>    - Config B (gốc + RAG) chỉ cải thiện nhẹ so với A: BLEU 4.06 → 6.46 (+59%), ROUGE-L 26.37 → 27.38 (+3.8%).
>    - Config D (FT + RAG) cải thiện nhẹ so với C: BLEU 17.38 → 18.53 (+6.6%), ROUGE-L 43.68 → 46.25 (+5.9%).
>    - RAG có tác dụng rõ hơn khi kết hợp với fine-tuned model (D > C > B > A).
>
> 3. **Retrieval đạt chất lượng cao:**
>    - Recall@5 = 86% ở cả Config B và D — tức 43/50 câu hỏi có đúng nguồn quy chế trong top-5 kết quả truy xuất.
>
> 4. **Fine-tuned model nhanh hơn:**
>    - Config C (481s) nhanh gần 2× so với Config A (859s) cho cùng 50 câu, có thể do model fine-tuned sinh câu trả lời ngắn gọn, đúng trọng tâm hơn.
>
> 5. **Config D là cấu hình tốt nhất tổng thể:** đạt điểm cao nhất trên tất cả metrics.

### 4.2 Phân tích retrieval

| Metric | Config B (gốc+RAG) | Config D (FT+RAG) |
|---|---|---|
| Recall@5 | 86.0% | 86.0% |

> **Nhận xét:** Recall@5 đồng nhất giữa B và D vì cả hai dùng chung retrieval pipeline (FAISS + BM25 + RRF + Reranker). Sự khác biệt chất lượng câu trả lời đến từ LLM generator, không phải retriever. Recall 86% nghĩa là hybrid retrieval + reranking hoạt động hiệu quả trên corpus 768 chunks.

### 4.3 Training metrics (Phase 3)

| Metric | Giá trị |
|---|---|
| Total steps | 567 |
| Epochs | 3 |
| Training loss (initial) | 2.5233 |
| Training loss (final) | 0.2768 |
| Eval loss (best, step 550) | **0.3836** |
| Eval loss (final, step 567) | 0.3836 |
| Best checkpoint | `checkpoint-550` |
| Learning rate (peak) | 2e-4 |
| Learning rate (final) | ~0 (cosine decay) |
| GPU | Google Colab T4 (15GB VRAM) |
| QA Train samples | 3,176 |

**Diễn biến training:**

| Step | Train Loss | Eval Loss | Ghi chú |
|---|---|---|---|
| 5 | 2.5233 | — | Khởi đầu |
| 50 | 0.4929 | 0.4758 | Hội tụ nhanh |
| 100 | 0.4622 | 0.4414 | — |
| 200 | 0.3358 | 0.4116 | — |
| 300 | 0.3612 | 0.3907 | — |
| 400 | 0.2917 | 0.3880 | — |
| 500 | 0.2757 | 0.3836 | — |
| 550 | 0.2859 | **0.3835** | ✅ Best checkpoint |
| 567 | — | 0.3836 | End of training |

> **Nhận xét:** Training loss giảm mạnh từ 2.52 → 0.28 (giảm ~89%) trong 3 epochs, cho thấy model học tốt trên domain quy chế TDTU. Eval loss ổn định quanh 0.383-0.388 từ step 350 trở đi, **không có dấu hiệu overfitting** (eval loss không tăng ngược). Gap train-eval nhỏ (0.28 vs 0.38) xác nhận model generalize tốt.

### 4.4 Tổng hợp thời gian chạy evaluation

| Thống kê | Giá trị |
|---|---|
| Tổng thời gian eval 4 configs | 3,403s (~57 phút) |
| Số câu hỏi test | 50 |
| Platform | Google Colab T4 |
| Ngày chạy | 2026-05-09 |

---

## 5. Kết luận & Hướng phát triển

### 5.1 Kết luận

- **Fine-tuning QLoRA là yếu tố then chốt:** Config D (FT+RAG) đạt BLEU 18.53, ROUGE-L 46.25, BERTScore 77.93 — vượt trội Config A (gốc) lần lượt +356%, +75%, +13.8%. QLoRA fine-tuning trên 3,176 QA pairs domain-specific đã tạo ra sự khác biệt lớn nhất.
- **RAG cải thiện thêm khi kết hợp fine-tuning:** RAG đơn lẻ (Config B) chỉ cải thiện nhẹ so với baseline, nhưng khi kết hợp FT (Config D vs C), RAG đóng góp +6.6% BLEU và +5.9% ROUGE-L.
- **Retrieval pipeline đạt 86% Recall@5:** Hybrid retrieval (BM25 + Dense + RRF) kết hợp cross-encoder reranking cho phép tìm đúng nguồn quy chế trong 43/50 câu hỏi.
- **5 cải tiến RAG** (query rewriting, metadata filter, hybrid retrieval, parent expansion, contextual compression) đảm bảo context chất lượng cho LLM.
- **QLoRA cho phép fine-tune trên Colab Free:** Mô hình 3B parameters, chỉ train ~0.5% tham số, chạy trên T4 GPU (~15GB VRAM).
- **Pipeline 6 phase end-to-end** từ PDF scan → chatbot demo, có thể tái sử dụng cho domain khác.

### 5.2 Hướng phát triển

1. **Mở rộng nguồn dữ liệu:** Thêm thông báo mới, FAQ từ phòng Công tác Sinh viên, cập nhật realtime.
2. **Multi-turn conversation:** Hỗ trợ hội thoại nhiều lượt, nhớ context câu hỏi trước.
3. **Feedback loop:** Cho phép sinh viên đánh giá câu trả lời → retrain model.
4. **Triển khai production:** Deploy lên server TDTU hoặc tích hợp vào app sinh viên.
5. **Nâng cấp LLM:** Thử nghiệm với Qwen2.5-7B hoặc Gemma-2-9B khi có GPU mạnh hơn.
6. **Đa ngôn ngữ:** Hỗ trợ tiếng Anh cho sinh viên quốc tế.
7. **Cải thiện chunk normalization:** Hiện còn 11 tiny chunks (<80 chars) và 19 oversized chunks (>3000 chars) cần xử lý triệt để hơn.

---

## 6. Tài liệu tham khảo

⚠️ TODO: Bổ sung danh sách tài liệu tham khảo theo format APA/IEEE.

---

## 7. SLIDE OUTLINE (12 slides)

| Slide | Tiêu đề | Nội dung chính | Thời lượng |
|---|---|---|---|
| 1 | Trang bìa | Tên đề tài, SV, GVHD, logo TDTU | 30s |
| 2 | Tổng quan & Bài toán | Motivation, 29 văn bản quy chế, RAG + Fine-tuning | 2 phút |
| 3 | Kiến trúc tổng thể | Mermaid diagram 6 phase, stack kỹ thuật | 2 phút |
| 4 | Phase 1 — OCR | GPT-4o-mini Vision, 137K từ, 2,975 dòng bảng | 1.5 phút |
| 5 | Phase 2 — Data Processing | 768 chunks, 3,226 QA, FAISS 3MB | 2 phút |
| 6 | Phase 3 — Fine-tuning | QLoRA config, training curves, Colab setup | 2 phút |
| 7 | Phase 4 — RAG Pipeline | 7-step pipeline, 5 cải tiến, sơ đồ chi tiết | 3 phút |
| 8 | Phase 5 — Evaluation | Ma trận A/B/C/D, Config D tốt nhất | 2 phút |
| 9 | Kết quả & Phân tích | BLEU 18.53, ROUGE-L 46.25, FT +328% vs baseline | 2 phút |
| 10 | Phase 6 — Demo LIVE | Chạy demo Gradio trực tiếp, hỏi 2-3 câu mẫu | 3 phút |
| 11 | Kết luận & Hướng phát triển | Đóng góp chính, 7 hướng phát triển | 1.5 phút |
| 12 | Q&A | Cảm ơn, mở phần hỏi đáp | — |

**Tổng thời gian ước tính:** ~20 phút (không tính Q&A)

---

## ✅ Checklist trước khi nộp

- [x] **Kết quả evaluation:** Chạy `python phase5_eval.py` ✅ — Config D đạt BLEU 18.53, ROUGE-L 46.25, BERTScore 77.93, Recall@5 86%
- [x] **Training metrics:** ✅ — 567 steps, 3 epochs, train loss 2.52→0.28, eval loss best 0.3836 (step 550), no overfitting
- [ ] **Tài liệu tham khảo:** Bổ sung mục 6 theo format yêu cầu của trường
- [ ] **Demo sẵn sàng:** Test `python phase6_demo.py` trên Colab, đảm bảo public link hoạt động trước buổi thuyết trình

