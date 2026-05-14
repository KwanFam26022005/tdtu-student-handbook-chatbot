# Đề tài 1: Xây dựng hệ thống hỏi đáp tiếng Việt trên một miền tri thức cụ thể sử dụng RAG kết hợp Fine-tuning LLM

## Yêu cầu bắt buộc

### 1. Dữ liệu
- Chọn **1 domain**:
  - Quy chế trường học
  - Luật giao thông
  - Y tế phổ thông
  - Hành chính công
  - ...

- Thu thập tài liệu / đoạn văn để xây dựng **Knowledge Base**.

- Tạo:
  - ≥ 300 cặp Question-Answer dùng để fine-tune
  - ≥ 50 cặp Question-Answer test được chuẩn bị thủ công

---

### 2. Fine-tuning
- Fine-tune 1 mô hình LLM có kích thước từ **1B – 7B parameters** bằng:
  - LoRA
  - hoặc QLoRA

- Huấn luyện trên **Google Colab Free**.

- Mô hình gợi ý:
  - Tự chọn

---

### 3. Pipeline RAG
Bao gồm các thành phần:
- Chunking
- Embedding
- Vector Store:
  - Chroma
  - hoặc FAISS

- Retriever:
  - Top-k retrieval
  - Có prompt template rõ ràng

---

### 4. Thực nghiệm so sánh 4 cấu hình

|                     | LLM gốc | LLM Fine-tuned |
|---------------------|----------|----------------|
| Không dùng RAG      | A        | C              |
| Có dùng RAG         | B        | D              |

---

### 5. Đánh giá

#### Đánh giá định lượng
- BLEU
- ROUGE-L
- BERTScore

#### Đánh giá Retrieval
- Recall@5

#### Human Evaluation
- 50 câu hỏi đánh giá thủ công

---

### 6. Demo
- Có demo hệ thống
- Khuyến khích xây dựng giao diện người dùng (UI)

---

# Sản phẩm cần nộp

- Mã nguồn GitHub (có README)
- Báo cáo từ 15–20 trang
- Slide thuyết trình
- Video demo 3–5 phút
- Dataset và checkpoint:
  - HuggingFace Hub
  - Google Drive