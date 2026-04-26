# BÁO CÁO TIẾN ĐỘ TUẦN

- **Tuần báo cáo:** Tuần 4 / Tháng 4 / 2026
- **Tên dự án:** Xây dựng hệ thống hỏi đáp tiếng Việt về Sổ tay Sinh viên TDTU — RAG kết hợp Fine-tuning LLM
- **Phase 1:** Thu thập dữ liệu & OCR
- **Phase 2:** Xử lý dữ liệu & Chunking

---

## Phase 1 — Thu thập dữ liệu & OCR

### 1. Dữ liệu (Data)

| Thông số | Chi tiết |
|----------|---------|
| Nguồn dữ liệu | Cổng quy chế học vụ TDTU (`quychehocvu.tdtu.edu.vn`) |
| Hệ thống backend | **Syncfusion PDF Viewer** – hiển thị PDF dạng ảnh qua REST API |
| Số lượng văn bản | **29 file PDF** (từ 32 ID hợp lệ, 3 ID trùng lặp tài liệu) |
| Tổng dung lượng | **~91 MB** |
| Định dạng gốc | PDF dạng **ảnh render** (không phải PDF text gốc) |
| Tình trạng | ✅ Thu thập đầy đủ, lưu tại `data/` |

Danh sách văn bản tiêu biểu:

| # | Tên văn bản | Dung lượng |
|---|------------|-----------|
| 1 | Quy chế tổ chức và quản lý đào tạo (2021) | 7.27 MB |
| 2 | Nội quy phòng thi (2020) | 7.22 MB |
| 3 | Quy chế Công tác sinh viên | 5.18 MB |
| 4 | Quy định kiểm soát và xử lý hành vi đạo văn | 4.18 MB |
| 5 | Quy định hoàn phí | 20.24 MB |
| ... | *24 văn bản khác* | ... |

### 2. Hướng tiếp cận (Approach)

#### 2.1. Kỹ thuật Web Scraping – Reverse-engineering Syncfusion PDF Viewer API

Website TDTU không cho tải trực tiếp file PDF. Tài liệu được hiển thị thông qua component **Syncfusion PDF Viewer** (JavaScript), render từng trang PDF thành ảnh base64 qua REST API. Để thu thập dữ liệu, nhóm đã reverse-engineer luồng API gồm 3 bước:

**Bước 1 – Quét danh sách ID hợp lệ:**
- Gửi HTTP GET đến `/QuyChe/Index?page={1..5}`
- Parse HTML bằng **BeautifulSoup** (`html.parser`)
- Trích xuất ID từ các thẻ `<a href="/QuyChe/Detail/{id}">`
- Kết quả: 32 ID hợp lệ `[9, 13, 19, 47, 55, ..., 150]`

**Bước 2 – Load document metadata:**
- Gửi POST đến `{serviceUrl}/Load` với payload JSON:
```json
{
  "action": "Load",
  "document": "9/11.2457-QD cap chung nhan.pdf",
  "isFileName": true,
  "zoomFactor": 1
}
```
- API trả về `hashId` và `pageCount` nhưng **không trả về `documentId`**
- **Kỹ thuật xử lý:** Tự sinh `documentId` phía client theo format Syncfusion: `Sync_PdfViewer_{UUID4}`

**Bước 3 – Render từng trang:**
- Gửi POST đến `{serviceUrl}/RenderPdfPages` cho mỗi trang (`pageNumber: 0..N`)
- API trả về ảnh dạng **base64 string**
- Decode base64 → PIL Image → ghép thành file PDF hoàn chỉnh

#### 2.2. Kỹ thuật xác thực – Session Cookie Management

- Sử dụng `requests.Session()` để duy trì cookie xuyên suốt các request
- Cookie `.AspNetCore.Cookies` và `.AspNetCore.Session` được parse vào **cookie jar** (thay vì hardcode trong header — tránh xung đột `Content-Type`)
- Tự động phát hiện `serviceUrl` từ HTML bằng regex: `serviceUrl\s*[:=]\s*['"]([^'"]+)['"]`

#### 2.3. Kỹ thuật OCR – EasyOCR + PyMuPDF

Do PDF output là ảnh (không có text layer), cần OCR để trích xuất văn bản:

| Kỹ thuật | Công cụ | Vai trò |
|----------|---------|---------|
| PDF → Image | **PyMuPDF (fitz)** | Render mỗi trang PDF thành pixmap ở 300 DPI |
| Image → Text | **EasyOCR** (model `vi`) | Nhận dạng ký tự tiếng Việt từ ảnh PNG |

Pipeline OCR chi tiết:
```
PDF file
  → fitz.open(pdf_path)
  → page.get_pixmap(matrix=Matrix(300/72, 300/72))   # Render 300 DPI
  → pixmap.tobytes("png")                             # Chuyển sang PNG bytes
  → reader.readtext(img_bytes, detail=0, paragraph=True)  # OCR
  → "\n".join(results)                                 # Ghép text theo trang
  → write_text(output.txt, encoding="utf-8")           # Lưu file
```

**Tại sao chọn EasyOCR thay vì Tesseract/PaddleOCR?**
- EasyOCR dựa trên **CRAFT** (text detection) + **CRNN** (text recognition) — kiến trúc Deep Learning
- Hỗ trợ **tiếng Việt native** với dấu thanh chính xác (ă, ơ, ư, ễ, ự...)
- Pure Python — không cần cài đặt system dependency (poppler, Tesseract binary)
- Tự động fallback CPU khi không có GPU CUDA

**Kết quả test OCR (1 file mẫu):**

| File test | Trang | Kết quả |
|-----------|-------|---------|
| `Nhiệm vụ thực hiện 3 nội dung đạo đức.pdf` | 1 | ✅ Trích xuất chính xác, dấu tiếng Việt đầy đủ |

### 3. Xử lý dữ liệu (Data Processing)

#### Pipeline tổng thể Phase 1:

```
┌─────────────────────────────────────────────────────────────┐
│                    WEB SCRAPING PIPELINE                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [1] GET /QuyChe/Index?page=1..5                            │
│       │                                                     │
│       ▼                                                     │
│  BeautifulSoup parse HTML → Trích xuất 32 ID                │
│       │                                                     │
│       ▼                                                     │
│  [2] POST {serviceUrl}/Load                                 │
│       │  payload: {document, isFileName, zoomFactor}         │
│       │  response: {hashId, pageCount}                       │
│       │  + Client-generated: documentId = Sync_PdfViewer_*   │
│       │                                                     │
│       ▼                                                     │
│  [3] POST {serviceUrl}/RenderPdfPages  (×N trang)           │
│       │  payload: {documentId, hashId, pageNumber}           │
│       │  response: {images: [base64_string]}                 │
│       │  + Retry logic: 3 lần, sleep 2s giữa mỗi lần       │
│       │                                                     │
│       ▼                                                     │
│  [4] base64.decode → PIL.Image → save PDF                   │
│       │                                                     │
│       ▼                                                     │
│  data/*.pdf  (29 files, ~91 MB)                             │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                      OCR PIPELINE                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  data/*.pdf                                                 │
│       │                                                     │
│       ▼                                                     │
│  PyMuPDF render (300 DPI) → PNG bytes                       │
│       │                                                     │
│       ▼                                                     │
│  EasyOCR (lang='vi', CRAFT+CRNN) → text blocks              │
│       │                                                     │
│       ▼                                                     │
│  raw_text/*.txt  (29 files)                                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### Công cụ & thư viện:

| Công cụ | Kỹ thuật cốt lõi | Vai trò |
|---------|-------------------|---------|
| `requests.Session` | HTTP persistent connection + cookie jar | Duy trì phiên xác thực |
| `BeautifulSoup` | HTML DOM parser (`html.parser`) | Parse trang Index lấy ID |
| `Pillow (PIL)` | Image processing | Decode base64 → Image → PDF |
| `uuid` | UUID4 generation | Tạo `documentId` phía client |
| `PyMuPDF (fitz)` | PDF rendering engine | PDF → PNG (300 DPI pixmap) |
| `EasyOCR` | CRAFT (detection) + CRNN (recognition) | Ảnh → text tiếng Việt |

### 4. Cải thiện & Vướng mắc (Improvements & Blockers)

**Cải thiện (các bug đã fix trong quá trình phát triển):**

| Bug | Nguyên nhân | Kỹ thuật fix |
|-----|------------|-------------|
| API không trả về `documentId` | Syncfusion tạo ID phía client, không phải server | Tự sinh `Sync_PdfViewer_{uuid4()}` |
| `TypeError: NoneType is not iterable` | API trả về `images: null` ở một số trang | Kiểm tra None-safe: `isinstance(images_val, list)` |
| Request bị reject | Hardcode `Content-Type: application/json` trong `session.headers` | Chuyển sang `session.cookies` jar, để `requests` tự xử lý header |
| Mất kết nối giữa chừng | Network timeout | Retry logic 3 lần + `time.sleep(2)` giữa mỗi lần |

**Vướng mắc hiện tại:**

| Vấn đề | Mức độ | Hướng xử lý |
|--------|--------|-------------|
| OCR chạy CPU rất chậm (~5 phút/file) | Trung bình | Chuyển lên Colab GPU (tăng tốc 5-10x) |
| Cookie xác thực có thời hạn | Thấp | Lấy lại thủ công từ browser DevTools |

### 5. Kế hoạch tuần tới

- [ ] Chạy OCR batch toàn bộ 29 PDF trên Colab GPU → `raw_text/`
- [ ] Kiểm tra chất lượng OCR trên các file lớn (bảng biểu, nhiều cột)
- [ ] Cleaning & normalization text → `clean_text/`
- [ ] Semantic chunking (theo Điều/Khoản/Mục) → `chunks.json`

---

## Phase 2 — Xử lý dữ liệu & Chunking

### 1. Dữ liệu (Data)

| Thông số | Chi tiết |
|----------|---------|
| Dữ liệu đầu vào | Text thô từ OCR (Phase 1) — `raw_text/*.txt` |
| Số lượng dự kiến | 29 file `.txt` |
| Ước tính tổng | ~200–400 trang, ~100,000–200,000 từ |
| Tình trạng | ⏳ Chờ Phase 1 OCR hoàn tất, **code pipeline đã sẵn sàng** |

### 2. Hướng tiếp cận (Approach)

Phase 2 chia thành **4 sub-phase** (2A → 2D), mỗi sub-phase áp dụng kỹ thuật riêng:

#### 2A. Text Cleaning — Regex-based Normalization Pipeline

Văn bản sau OCR chứa nhiều nhiễu cần xử lý. Pipeline cleaning gồm **8 bước tuần tự**:

| Bước | Kỹ thuật | Mô tả | Ví dụ |
|------|----------|-------|-------|
| 1 | `unicodedata.normalize("NFC")` | Chuẩn hóa Unicode về dạng Composed | `e + ̂ + ́` → `ế` (1 codepoint) |
| 2 | Regex `[\x00-\x08\x0b...]` | Xóa ký tự điều khiển (giữ `\n`, `\t`) | Loại bỏ null bytes |
| 3 | Regex `---\s*Trang\s+\d+/\d+\s*---` | Xóa marker trang từ OCR output | `--- Trang 3/10 ---` → `\n` |
| 4 | Dictionary replacement | Sửa lỗi OCR phổ biến tiếng Việt | `Ðiều` → `Điều`, `Ðại` → `Đại` |
| 5 | Regex Vietnamese char class | Gộp dòng bị ngắt giữa chừng | `sinh viên\ncần` → `sinh viên cần` |
| 6 | Regex `[^\S\n]+` | Gộp khoảng trắng thừa | `sinh   viên` → `sinh viên` |
| 7 | Regex `\n{3,}` | Xóa dòng trống liên tiếp (giữ tối đa 2) | 5 dòng trống → 2 |
| 8 | Line-by-line strip | Cắt bỏ trailing whitespace mỗi dòng | — |

**Kỹ thuật đặc biệt — Gộp dòng tiếng Việt (Bước 5):**

OCR thường ngắt dòng giữa một câu. Regex sử dụng **full Vietnamese character class** để phát hiện pattern "dòng kết thúc bằng chữ thường Việt + dòng sau bắt đầu bằng chữ thường Việt":

```python
# Character class bao phủ toàn bộ 134 ký tự Việt có dấu
VIET_LOWER = r'[a-zàáạảãăắằặẳẵâấầậẩẫđèéẹẻẽêếềệểễìíịỉĩòóọỏõôốồộổỗơớờợởỡùúụủũưứừựửữỳýỵỷỹ]'
pattern = rf'({VIET_LOWER},?)\n({VIET_LOWER})'
# Match: "sinh viên,\ncần" → replace: "sinh viên, cần"
```

#### 2B. Semantic Chunking — Structure-aware Document Splitting

Thay vì cắt cơ học (fixed-size), sử dụng **2 chiến lược chunking** tùy thuộc cấu trúc văn bản:

**Chiến lược 1: Structure-aware (cho văn bản quy chế có cấu trúc)**

Áp dụng khi phát hiện ≥3 section headers trong văn bản. Regex pattern phát hiện cấu trúc pháp lý:

```python
section_pattern = re.compile(
    r'^(Chương\s+[IVXLCDM\d]+|'    # Chương I, Chương II, Chương 1
    r'Điều\s+\d+|'                  # Điều 1, Điều 15
    r'Mục\s+\d+|'                   # Mục 1, Mục 3
    r'Phần\s+[IVXLCDM\d]+)',        # Phần I, Phần II
    re.MULTILINE
)
```

Logic phân tách:
1. Tìm tất cả vị trí section headers bằng `finditer()`
2. Cắt text giữa 2 headers liên tiếp thành 1 chunk
3. Nếu section quá dài (> `CHUNK_SIZE × 2` = 1024 ký tự) → chia nhỏ tiếp theo paragraph
4. Track `current_chapter` để gán metadata Chương cho mỗi Điều

**Chiến lược 2: Paragraph-based with overlap (cho văn bản không có cấu trúc)**

Fallback khi văn bản < 3 section headers:
- Tách theo paragraph (`\n\n`)
- Buffer accumulation: gộp paragraph cho đến khi đạt `CHUNK_SIZE × 1.5`
- **Sentence-level overlap**: giữ lại 2 câu cuối của chunk trước làm prefix cho chunk sau

**Contextual Header Injection:**

Mỗi chunk được gắn nhãn ngữ cảnh ở đầu để tránh mất thông tin khi trích dẫn rời:

```
[Quy chế đào tạo] - Chương III - Điều 15
Sinh viên được phép đăng ký tối đa 25 tín chỉ mỗi học kỳ...
```

Cấu hình chunking:

```python
CHUNK_SIZE = 512       # tokens (~768 ký tự Việt)
CHUNK_OVERLAP = 64     # tokens overlap
MIN_CHUNK_LENGTH = 50  # Bỏ chunk quá ngắn (header rỗng, v.v.)
```

#### 2C. Vector Store — FAISS + bge-m3 Embedding

| Component | Kỹ thuật | Chi tiết |
|-----------|----------|---------|
| Embedding Model | `BAAI/bge-m3` | Multilingual SOTA, hỗ trợ Việt, output 1024-dim |
| Normalization | L2 normalize | `normalize_embeddings=True` → cosine similarity |
| Index type | `IndexFlatIP` | Inner Product (= cosine khi vector đã normalize) |
| Batch size | 32 | Cân bằng giữa tốc độ và VRAM |

Pipeline:
```
chunks.json
  → SentenceTransformer("BAAI/bge-m3").encode(texts)
  → numpy array (N × 1024), float32, L2-normalized
  → faiss.IndexFlatIP(1024).add(embeddings)
  → faiss.write_index → faiss_index.bin
```

#### 2D. Synthetic QA Generation — Gemini API + Structured Prompting

**Kỹ thuật: LLM-as-Data-Generator**

Sử dụng **Gemini 2.0 Flash** (free tier, 15 RPM) để sinh QA từ mỗi chunk:

- **Prompt engineering:** System prompt yêu cầu đa dạng 4 loại câu hỏi (Factual, Conditional, Procedural, Reasoning)
- **Output parsing:** Regex `\[.*\]` (DOTALL) để trích JSON array từ response
- **Metadata enrichment:** Mỗi QA pair được gắn `source`, `section`, `chunk_id`
- **Rate limiting:** `time.sleep(4)` giữa mỗi request (15 RPM = 1 request/4s)
- **Train/Test split:** Random shuffle (seed=42), lấy 50 đầu làm test, còn lại làm train

Ví dụ prompt:
```
Dựa vào đoạn văn bản quy chế sau, hãy tạo 3-5 cặp hỏi-đáp bằng tiếng Việt.
Yêu cầu:
- Câu hỏi phải TỰ NHIÊN, như sinh viên thực sự hỏi
- Câu trả lời phải CHÍNH XÁC, trích dẫn cụ thể số điều, khoản
- Đa dạng loại: factual, conditional, procedural, reasoning

Đoạn văn: {chunk_text_with_context}
Trả về JSON array: [{"question": "...", "answer": "...", "type": "..."}]
```

### 3. Xử lý dữ liệu (Data Processing)

#### Pipeline tổng thể Phase 2:

```
raw_text/*.txt
    │
    ▼
 [2A] CLEANING ─────────────────────────────────────────────
    │  • Unicode NFC normalization
    │  • OCR error correction (Ð→Đ, |→l)
    │  • Vietnamese line-joining (regex)
    │  • Whitespace & blank line normalization
    │
    ▼
 clean_text/*.txt
    │
    ▼
 [2B] CHUNKING ─────────────────────────────────────────────
    │  • Regex detect: Chương/Điều/Khoản/Mục
    │  • Structure-aware split (≥3 headers)
    │  • Paragraph-based fallback + overlap
    │  • Context header injection
    │
    ▼
 processed/chunks.json
    │
    ├──────────────────────────┐
    ▼                          ▼
 [2C] VECTOR STORE          [2D] QA GENERATION
    │  • bge-m3 encode          │  • Gemini 2.0 Flash API
    │  • L2 normalize           │  • 3-5 QA/chunk
    │  • FAISS IndexFlatIP      │  • 4 types: F/C/P/R
    │                           │  • JSON parse + metadata
    ▼                          ▼
 faiss_index.bin            qa_train.json (≥300)
 chunks_metadata.json       qa_test.json (50)
```

#### Công cụ & thư viện:

| Công cụ | Kỹ thuật cốt lõi | Vai trò |
|---------|-------------------|---------|
| `unicodedata` | NFC normalization | Chuẩn hóa encoding tiếng Việt |
| `re` (regex) | Pattern matching + substitution | Cleaning, section detection |
| `sentence-transformers` | Transformer encoder | Load & run bge-m3 embedding |
| `faiss-cpu` | Approximate Nearest Neighbor | Vector store + similarity search |
| `google-generativeai` | Gemini API client | Sinh synthetic QA pairs |

### 4. Cải thiện & Vướng mắc (Improvements & Blockers)

**Cải thiện so với approach naive:**

| Aspect | Naive approach | Approach của nhóm |
|--------|---------------|-------------------|
| Chunking | Fixed-size 500 chars | Structure-aware (Điều/Khoản/Mục) |
| Context | Chunk rời, mất ngữ cảnh | Context header injection |
| Embedding | Generic multilingual model | bge-m3 (SOTA multilingual, 1024-dim) |
| QA Generation | Manual labeling | LLM-as-Generator (Gemini) + metadata |

**Vướng mắc:**

| Vấn đề | Mức độ | Hướng xử lý |
|--------|--------|-------------|
| Chưa chạy được (phụ thuộc Phase 1 OCR) | Cao | Chạy Phase 1 trước trên Colab |
| Gemini free tier giới hạn 15 RPM | Trung bình | Sleep 4s/request, ước tính ~30-60 phút |
| Chất lượng QA phụ thuộc chất lượng OCR | Trung bình | Kiểm tra OCR thủ công trước khi sinh QA |

### 5. Kế hoạch tuần tới

- [ ] Chạy Phase 2 hoàn chỉnh (2A → 2D) sau khi OCR xong
- [ ] Thống kê: số chunks, phân bố chunk size, coverage
- [ ] Review chất lượng QA, loại bỏ câu kém chất lượng
- [ ] Curate thủ công 50 cặp QA test (gold standard)
- [ ] Bắt đầu Phase 3: Fine-tuning Qwen2.5-3B-Instruct (QLoRA trên Colab T4)
