"""
Phase 2 – Data Processing: Cleaning, Chunking, Vector Store, QA Generation.

Workflow:
  2A. Cleaning & Normalization (raw_text → clean_text)
  2B. Semantic Chunking (clean_text → chunks.json)
  2C. Build Vector Store (chunks → FAISS index)
  2D. Generate QA pairs (chunks → qa_train.json + qa_test.json)

Cài đặt:
  pip install sentence-transformers faiss-cpu langchain langchain-community
  pip install openai   # Cho QA generation bằng OpenAI
  
Chạy:
  python phase2_process.py
"""

import os
import re
import json
import time
import sys
import math
import unicodedata
from pathlib import Path

# ══════════════════════════════════════════════════════════
# CẤU HÌNH
# ══════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).resolve().parent if "__file__" in dir() else Path(".")
RAW_TEXT_DIR = BASE_DIR / "raw_text"
CLEAN_TEXT_DIR = BASE_DIR / "clean_text"
OUTPUT_DIR = BASE_DIR / "processed"

CLEAN_TEXT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Chunking config
CHUNK_SIZE = 512       # tokens (ước tính ~1 token ≈ 1.5 ký tự Việt)
CHUNK_OVERLAP = 64     # tokens overlap
MIN_CHUNK_LENGTH = 50  # Bỏ chunk quá ngắn

# Embedding model
EMBEDDING_MODEL = "BAAI/bge-m3"

# ══════════════════════════════════════════════════════════
# 2A. CLEANING & NORMALIZATION
# ══════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """
    Làm sạch text OCR:
    - Chuẩn hóa Unicode NFC
    - Sửa lỗi OCR phổ biến (giữ nguyên ký tự pipe cho Markdown table)
    - Loại bỏ tag [Con dấu], [Chữ ký] từ Phase 1 v2
    - Gộp dòng bị ngắt giữa chừng
    - Xóa header/footer lặp
    - Giữ cấu trúc Điều/Khoản và bảng Markdown
    """
    # 1. Chuẩn hóa Unicode
    text = unicodedata.normalize("NFC", text)

    # 2. Xóa ký tự điều khiển (giữ \n, \t)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # 3. Xóa marker trang từ OCR output
    text = re.sub(r'---\s*Trang\s+\d+/\d+\s*---', '\n', text)

    # 4. Loại bỏ tag con dấu / chữ ký (sinh bởi OpenAI Vision)
    text = re.sub(r'\[Con dấu\]', '', text)
    text = re.sub(r'\[Chữ ký\]', '', text)

    # 5. Sửa lỗi OCR phổ biến cho tiếng Việt
    #    LƯU Ý: KHÔNG thay '|' → 'l' vì Phase 1 v2 xuất bảng Markdown dùng '|'
    ocr_fixes = {
        'Ðiều': 'Điều',
        'Ðại': 'Đại',
        'Ðào': 'Đào',
        'Ðánh': 'Đánh',
        'Ðể': 'Để',
        'Ðược': 'Được',
        'Ðối': 'Đối',
        'Ðơn': 'Đơn',
    }
    for wrong, right in ocr_fixes.items():
        text = text.replace(wrong, right)

    # 6. Gộp dòng bị ngắt giữa chừng (CHỈ cho dòng text thường, BỎ QUA dòng bảng)
    #    Dòng bảng Markdown bắt đầu bằng '|' — không gộp
    lines = text.split('\n')
    merged_lines = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Không gộp nếu dòng hiện tại hoặc dòng sau là dòng bảng Markdown
        if (stripped.startswith('|') or
                (i + 1 < len(lines) and lines[i + 1].strip().startswith('|'))):
            merged_lines.append(line)
        else:
            merged_lines.append(line)
    text = '\n'.join(merged_lines)
    # Gộp dòng text thường bị ngắt giữa chừng
    text = re.sub(
        r'([a-zàáạảãăắằặẳẵâấầậẩẫđèéẹẻẽêếềệểễìíịỉĩòóọỏõôốồộổỗơớờợởỡùúụủũưứừựửữỳýỵỷỹ,])'
        r'\n'
        r'(?!\|)'
        r'([a-zàáạảãăắằặẳẵâấầậẩẫđèéẹẻẽêếềệểễìíịỉĩòóọỏõôốồộổỗơớờợởỡùúụủũưứừựửữỳýỵỷỹ])',
        r'\1 \2', text
    )

    # 7. Gộp khoảng trắng thừa (giữ \n)
    text = re.sub(r'[^\S\n]+', ' ', text)

    # 8. Xóa dòng trống liên tiếp (giữ tối đa 2)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 9. Strip mỗi dòng
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    return text.strip()


def process_cleaning():
    """Cleaning tất cả file raw_text → clean_text"""
    print("\n[PHASE 2A] Cleaning & Normalization")
    print("=" * 60)

    txt_files = sorted(RAW_TEXT_DIR.glob("*.txt"))
    if not txt_files:
        print(f"[ERROR] Không có file .txt nào trong {RAW_TEXT_DIR}")
        print("   Hãy chạy phase1_ocr.py trước!")
        return False

    for idx, txt_path in enumerate(txt_files, 1):
        raw = txt_path.read_text(encoding="utf-8")
        cleaned = clean_text(raw)

        out_path = CLEAN_TEXT_DIR / txt_path.name
        out_path.write_text(cleaned, encoding="utf-8")

        print(f"  [{idx}/{len(txt_files)}] {txt_path.name}: "
              f"{len(raw):,} -> {len(cleaned):,} ky tu")

    print(f"\n[OK] Da clean {len(txt_files)} file -> {CLEAN_TEXT_DIR}")
    return True


# ══════════════════════════════════════════════════════════
# 2B. SEMANTIC CHUNKING
# ══════════════════════════════════════════════════════════

def _merge_table_paragraphs(paragraphs: list[str]) -> list[str]:
    """
    Merge contiguous Markdown table lines into a single paragraph block.
    This prevents table rows from being split across different chunks.
    """
    merged = []
    table_buffer = []

    for para in paragraphs:
        lines = para.strip().split('\n')
        is_table = any(line.strip().startswith('|') for line in lines)

        if is_table:
            table_buffer.append(para)
        else:
            if table_buffer:
                # Flush table block as single paragraph
                merged.append('\n'.join(table_buffer))
                table_buffer = []
            merged.append(para)

    if table_buffer:
        merged.append('\n'.join(table_buffer))

    return merged


def semantic_chunk(text: str, source_name: str) -> list[dict]:
    """
    Tach text thanh chunks thong minh:
    - Uu tien tach theo Dieu/Khoan/Muc/Chuong
    - Bao toan Markdown table (khong cat doi bang)
    - Fallback: tach theo paragraph + overlap
    - Them context header vao moi chunk
    """
    chunks = []
    
    # Tìm cấu trúc pháp lý: Chương, Điều, Khoản, Mục
    # Pattern: "Điều X.", "Chương I", "Mục 1", v.v.
    section_pattern = re.compile(
        r'^(Chương\s+[IVXLCDM\d]+|'
        r'Điều\s+\d+|'
        r'Mục\s+\d+|'
        r'Phần\s+[IVXLCDM\d]+)',
        re.MULTILINE
    )
    
    # Tìm tất cả vị trí section headers
    matches = list(section_pattern.finditer(text))
    
    if len(matches) >= 3:
        # === Mode: Structure-aware chunking ===
        current_chapter = ""
        
        for i, match in enumerate(matches):
            header = match.group(0).strip()
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            
            section_text = text[start:end].strip()
            
            # Track chương hiện tại
            if header.startswith("Chương"):
                current_chapter = header
            
            # Nếu section quá dài, chia nhỏ theo paragraph
            if len(section_text) > CHUNK_SIZE * 2:
                paragraphs = _merge_table_paragraphs(section_text.split('\n\n'))
                buffer = ""
                for para in paragraphs:
                    if len(buffer) + len(para) > CHUNK_SIZE * 1.5:
                        if buffer.strip():
                            chunks.append({
                                "text": buffer.strip(),
                                "source": source_name,
                                "section": header,
                                "chapter": current_chapter,
                            })
                        buffer = para + "\n\n"
                    else:
                        buffer += para + "\n\n"
                if buffer.strip():
                    chunks.append({
                        "text": buffer.strip(),
                        "source": source_name,
                        "section": header,
                        "chapter": current_chapter,
                    })
            else:
                if len(section_text) >= MIN_CHUNK_LENGTH:
                    chunks.append({
                        "text": section_text,
                        "source": source_name,
                        "section": header,
                        "chapter": current_chapter,
                    })
        
        # Thêm phần trước section đầu tiên (nếu có)
        if matches and matches[0].start() > MIN_CHUNK_LENGTH:
            preamble = text[:matches[0].start()].strip()
            if len(preamble) >= MIN_CHUNK_LENGTH:
                chunks.insert(0, {
                    "text": preamble,
                    "source": source_name,
                    "section": "Phần mở đầu",
                    "chapter": "",
                })
    else:
        # === Mode: Paragraph-based chunking with overlap ===
        paragraphs = _merge_table_paragraphs(text.split('\n\n'))
        buffer = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if len(buffer) + len(para) > CHUNK_SIZE * 1.5:
                if buffer.strip() and len(buffer.strip()) >= MIN_CHUNK_LENGTH:
                    chunks.append({
                        "text": buffer.strip(),
                        "source": source_name,
                        "section": "",
                        "chapter": "",
                    })
                # Overlap: giữ lại câu cuối của buffer
                last_sentences = buffer.strip().split('.')
                overlap_text = '. '.join(last_sentences[-2:]) if len(last_sentences) > 1 else ""
                buffer = overlap_text + "\n\n" + para + "\n\n"
            else:
                buffer += para + "\n\n"
        
        if buffer.strip() and len(buffer.strip()) >= MIN_CHUNK_LENGTH:
            chunks.append({
                "text": buffer.strip(),
                "source": source_name,
                "section": "",
                "chapter": "",
            })
    
    # Thêm context header vào mỗi chunk
    for chunk in chunks:
        header_parts = [f"[{chunk['source']}]"]
        if chunk['chapter']:
            header_parts.append(chunk['chapter'])
        if chunk['section']:
            header_parts.append(chunk['section'])
        
        context_header = " - ".join(header_parts)
        chunk["text_with_context"] = f"{context_header}\n{chunk['text']}"
        chunk["id"] = f"{chunk['source']}_{chunks.index(chunk)}"
    
    return chunks


def process_chunking():
    """Chunking tat ca clean text -> chunks.json"""
    print("\n[PHASE 2B] Semantic Chunking")
    print("=" * 60)
    
    txt_files = sorted(CLEAN_TEXT_DIR.glob("*.txt"))
    if not txt_files:
        print("[ERROR] Khong co file clean text. Chay cleaning truoc!")
        return False
    
    all_chunks = []
    
    for idx, txt_path in enumerate(txt_files, 1):
        text = txt_path.read_text(encoding="utf-8")
        source_name = txt_path.stem
        
        chunks = semantic_chunk(text, source_name)
        all_chunks.extend(chunks)
        
        print(f"  [{idx}/{len(txt_files)}] {source_name}: {len(chunks)} chunks")
    
    # Gán ID duy nhất
    for i, chunk in enumerate(all_chunks):
        chunk["id"] = f"chunk_{i:04d}"
    
    # Lưu
    chunks_path = OUTPUT_DIR / "chunks.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    
    print(f"\n[OK] Tong: {len(all_chunks)} chunks -> {chunks_path}")
    
    # Thống kê
    lengths = [len(c["text"]) for c in all_chunks]
    print(f"   Min: {min(lengths)} ký tự | Max: {max(lengths)} ký tự | "
          f"Trung bình: {sum(lengths)//len(lengths)} ký tự")
    
    return True


# ══════════════════════════════════════════════════════════
# 2C. BUILD VECTOR STORE
# ══════════════════════════════════════════════════════════

def build_vector_store():
    """Embed chunks + luu FAISS index"""
    print("\n[PHASE 2C] Build Vector Store (FAISS + bge-m3)")
    print("=" * 60)

    chunks_path = OUTPUT_DIR / "chunks.json"
    if not chunks_path.exists():
        print("[ERROR] Chua co chunks.json. Chay chunking truoc!")
        return False

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"  Loaded {len(chunks)} chunks")

    # Embedding
    from sentence_transformers import SentenceTransformer
    import numpy as np

    print(f"  Loading embedding model: {EMBEDDING_MODEL}...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    # Embed tat ca chunks (dung text_with_context de co ngu canh)
    texts = [c["text_with_context"] for c in chunks]

    print(f"  Dang embed {len(texts)} chunks...")
    embeddings = embed_model.encode(
        texts,
        show_progress_bar=True,
        batch_size=32,
        normalize_embeddings=True
    )
    # Build FAISS index
    import faiss

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product (cosine vi da normalize)
    index.add(np.array(embeddings).astype('float32'))

    # Luu
    faiss_path = OUTPUT_DIR / "faiss_index.bin"
    faiss.write_index(index, str(faiss_path))

    # Luu metadata mapping
    metadata = [{"id": c["id"], "source": c["source"], "section": c["section"],
                 "chapter": c["chapter"]} for c in chunks]
    meta_path = OUTPUT_DIR / "chunks_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] FAISS index: {faiss_path} (dim={dim}, n={index.ntotal})")
    print(f"[OK] Metadata: {meta_path}")
    return True


# ══════════════════════════════════════════════════════════
# 2D. GENERATE QA PAIRS
# ══════════════════════════════════════════════════════════

QA_SYSTEM_PROMPT = """Bạn là chuyên gia tạo dữ liệu huấn luyện cho chatbot sổ tay sinh viên.
Dựa vào đoạn quy chế/quy định được cung cấp, hãy tạo các cặp câu hỏi - trả lời bằng tiếng Việt.

Yêu cầu:
1. Câu hỏi phải TỰ NHIÊN, như sinh viên thực sự hỏi (không formal quá)
2. Câu trả lời phải CHÍNH XÁC, trích dẫn cụ thể từ văn bản (số điều, khoản)
3. Đa dạng loại câu hỏi:
   - Factual: "Bao nhiêu tín chỉ để tốt nghiệp?"
   - Conditional: "Nếu bị cảnh báo học vụ thì sao?"
   - Procedural: "Thủ tục xin bảo lưu như thế nào?"
   - Reasoning: "Tại sao sinh viên bị buộc thôi học?"
4. Mỗi câu trả lời nên 2-5 câu, đầy đủ nhưng gọn

Trả về JSON array:
[{"question": "...", "answer": "...", "type": "factual|conditional|procedural|reasoning"}]"""


def generate_qa_template(chunk: dict) -> list[dict]:
    """
    Sinh QA pairs từ chunk bằng template — KHÔNG cần API.
    Trích xuất thông tin từ văn bản quy chế tiếng Việt bằng regex.
    Đảm bảo 2-4 QA/chunk cho chunks có đủ nội dung.
    """
    text = chunk['text']
    section = chunk.get('section', '')
    chapter = chunk.get('chapter', '')
    source = chunk.get('source', '').replace('_', ' ')
    chunk_id = chunk.get('id', '')
    
    qa_pairs = []
    
    # Split thành sentences
    sentences = [s.strip() for s in re.split(r'(?<=[.;])\s+', text) if len(s.strip()) > 25]
    if len(sentences) < 1:
        return []
    
    def make_answer(sents, n=3):
        ans = '. '.join(sents[:n])
        return ans if ans.endswith(('.', ';', ':')) else ans + '.'
    
    def src_prefix():
        if section and section != "Phần mở đầu":
            return f"Theo {section}, "
        return "Theo quy chế, "
    
    def make_qa(q, a, qtype):
        return {"question": q, "answer": a, "type": qtype,
                "source": chunk.get("source", ""), "section": section,
                "chunk_id": chunk_id, "generated_by": "template"}
    
    # 1. FACTUAL: nội dung chính
    if section and section != "Phần mở đầu" and len(sentences) >= 2:
        qa_pairs.append(make_qa(
            f"{section} quy định những nội dung gì?",
            make_answer(sentences), "factual"))
    elif len(sentences) >= 2:
        topic = sentences[0][:80].rstrip(' ,;:')
        qa_pairs.append(make_qa(
            f"Quy chế quy định gì về: \"{topic}\"?",
            make_answer(sentences), "factual"))
    
    # 2. FACTUAL: số liệu
    for pat, desc in [(r'\d+\s*tín chỉ', 'số tín chỉ'), (r'\d+\s*học kỳ', 'số học kỳ'),
                      (r'\d+\s*ngày', 'số ngày'), (r'\d+\s*tháng', 'số tháng'),
                      (r'\d+(?:[.,]\d+)?\s*điểm', 'mức điểm'), (r'\d+\s*%', 'tỷ lệ'),
                      (r'\d+\s*năm', 'thời gian'), (r'\d+\s*lần', 'số lần')]:
        m = re.search(pat, text)
        if m:
            for s in sentences:
                if m.group(0) in s:
                    qa_pairs.append(make_qa(
                        f"{src_prefix()}{desc} được quy định như thế nào?",
                        s.strip() + ('.' if not s.strip().endswith('.') else ''),
                        "factual"))
                    break
            break
    
    # 3. CONDITIONAL
    for pat in [r'[Nn]ếu\s+.{15,}?(?:thì|sẽ|phải).{10,}?[.;]',
                r'[Tt]rường hợp\s+.{15,}?[.;]',
                r'[Kk]hông được\s+.{10,}?[.;]']:
        cm = re.search(pat, text)
        if cm:
            matched = cm.group(0).strip()
            if 'không được' in matched.lower():
                qa_pairs.append(make_qa(
                    f"{src_prefix()}sinh viên không được làm gì?",
                    matched, "conditional"))
            else:
                qa_pairs.append(make_qa(
                    f"Điều gì xảy ra trong trường hợp được nêu tại {section or 'quy chế'}?",
                    matched, "conditional"))
            break
    
    # 4. PROCEDURAL
    for kw in ['thủ tục', 'quy trình', 'hồ sơ', 'trình tự', 'bao gồm', 'cần có']:
        if kw in text.lower():
            for s in sentences:
                if kw in s.lower():
                    qa_pairs.append(make_qa(
                        f"{src_prefix()}{kw} được quy định thế nào?",
                        s.strip() + ('.' if not s.strip().endswith('.') else ''),
                        "procedural"))
                    break
            break
    
    # 5. REASONING
    for kw in ['nhằm', 'mục đích', 'với mục tiêu', 'để đảm bảo']:
        if kw in text.lower():
            for s in sentences:
                if kw in s.lower():
                    qa_pairs.append(make_qa(
                        f"Mục đích của {section if section else 'quy định này'} là gì?",
                        s.strip(), "reasoning"))
                    break
            break
    
    # Đảm bảo ít nhất 2 QA
    if len(qa_pairs) < 2 and len(sentences) >= 3:
        qa_pairs.append(make_qa(
            f"Hãy tóm tắt nội dung chính của {section if section else 'đoạn quy chế này'}.",
            make_answer(sentences, 4), "factual"))
    
    return qa_pairs[:4]


def generate_qa_api(chunks: list[dict]) -> tuple[list[dict], set]:
    """
    Sinh QA pairs bang OpenAI API (best-effort).
    Returns: (qa_pairs, done_chunk_ids)
    """
    import openai
    
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("   [WARNING] Khong co OPENAI_API_KEY -> chi dung template")
        return [], set()

    client = openai.OpenAI(api_key=api_key)

    MODEL_CHAIN = [
        "gpt-4o-mini",
        "gpt-4o",
    ]

    current_model_idx = 0
    print(f"   [MODEL] {MODEL_CHAIN[current_model_idx]}")

    # Resume logic
    progress_path = OUTPUT_DIR / "qa_progress.json"
    if progress_path.exists():
        with open(progress_path, "r", encoding="utf-8") as f:
            progress_data = json.load(f)
        all_qa = progress_data.get("qa_pairs", [])
        done_ids = set(progress_data.get("done_chunk_ids", []))
        print(f"   [RESUME] {len(all_qa)} QA tu {len(done_ids)} chunks")
    else:
        all_qa = []
        done_ids = set()

    pending = [c for c in chunks if c["id"] not in done_ids]
    print(f"   [PENDING] Con {len(pending)} chunks can API")
    
    if not pending:
        return all_qa, done_ids
    
    consecutive_429 = 0
    
    for chunk in pending:
        prompt = f"""{QA_SYSTEM_PROMPT}

Đoạn văn bản quy chế:
---
{chunk['text_with_context']}
---

Hãy tạo 3-5 cặp câu hỏi-trả lời. CHỈ trả về JSON array, KHÔNG markdown."""
        
        success = False
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=MODEL_CHAIN[current_model_idx],
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                )
                resp = response.choices[0].message.content
                
                # Robust JSON parse
                resp_clean = re.sub(r'```json\s*', '', resp)
                resp_clean = re.sub(r'```\s*', '', resp_clean)
                json_match = re.search(r'\[.*\]', resp_clean, re.DOTALL)
                
                if json_match:
                    qa_pairs = json.loads(json_match.group(0))
                    valid = []
                    for qa in qa_pairs:
                        if not isinstance(qa, dict):
                            continue
                        q = str(qa.get("question", "")).strip()
                        a = str(qa.get("answer", "")).strip()
                        t = str(qa.get("type", "factual")).strip().lower()
                        if not q or not a:
                            continue
                        if t not in {"factual", "conditional", "procedural", "reasoning"}:
                            t = "factual"
                        valid.append({
                            "question": q, "answer": a, "type": t,
                            "source": chunk["source"],
                            "section": chunk.get("section", ""),
                            "chunk_id": chunk["id"],
                            "generated_by": "api"
                        })
                    if valid:
                        all_qa.extend(valid)
                        done_ids.add(chunk["id"])
                        consecutive_429 = 0
                        print(f"   [OK] {chunk['id']}: {len(valid)} QA (tong: {len(all_qa)})")
                        success = True
                        break

                print(f"   [WARNING] {chunk['id']}: JSON parse fail (lan {attempt+1})")
                done_ids.add(chunk["id"])
                success = True
                break
                
            except Exception as e:
                err = str(e)
                if "429" in err or "quota" in err.lower() or "rate" in err.lower():
                    consecutive_429 += 1
                    wait = 60
                    m = re.search(r'retry.*?(\d+\.?\d*)\s*s', err, re.IGNORECASE)
                    if m:
                        wait = float(m.group(1)) + 5
                    
                    if consecutive_429 >= 5:
                        current_model_idx += 1
                        if current_model_idx < len(MODEL_CHAIN):
                            nm = MODEL_CHAIN[current_model_idx]
                            print(f"\n   [SWITCH] Chuyen model: {nm}")
                            consecutive_429 = 0
                            wait = 10
                        else:
                            print(f"\n   [PAUSE] Het quota. Luu {len(all_qa)} QA tu API.")
                            with open(progress_path, "w", encoding="utf-8") as f:
                                json.dump({"qa_pairs": all_qa,
                                          "done_chunk_ids": list(done_ids)},
                                         f, ensure_ascii=False, indent=2)
                            return all_qa, done_ids
                    
                    print(f"   [WAIT] 429 (lan {attempt+1}). Cho {wait:.0f}s...")
                    time.sleep(wait)
                else:
                    print(f"   [ERROR] {chunk['id']}: {err[:80]}")
                    time.sleep(5)
        
        if not success:
            print(f"   [SKIP] Skip {chunk['id']}")
        
        # Save progress mỗi 10 chunks
        if len(done_ids) % 10 == 0 and len(done_ids) > 0:
            with open(progress_path, "w", encoding="utf-8") as f:
                json.dump({"qa_pairs": all_qa, "done_chunk_ids": list(done_ids)},
                         f, ensure_ascii=False, indent=2)
            print(f"   [SAVE] Saved: {len(all_qa)} QA / {len(done_ids)} chunks")
        
        time.sleep(1)
    
    # Save cuoi
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump({"qa_pairs": all_qa, "done_chunk_ids": list(done_ids)},
                 f, ensure_ascii=False, indent=2)
    
    return all_qa, done_ids


def process_qa_generation():
    """
    Sinh QA pairs: API truoc, template bo sung.
    Dam bao >=300 QA pairs cho training.
    """
    print("\n[PHASE 2D] Generate QA Pairs")
    print("=" * 60)

    chunks_path = OUTPUT_DIR / "chunks.json"
    if not chunks_path.exists():
        print("[ERROR] Chua co chunks.json!")
        return False

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"  {len(chunks)} chunks")
    print(f"  Muc tieu: >=300 QA pairs cho training")
    print(f"  Chien luoc: API (best-effort) + Template (fallback)\n")

    # -- Buoc 1: Thu API --
    print("  -- Buoc 1: Gemini API --")
    api_qa, api_done_ids = generate_qa_api(chunks)
    print(f"\n  [STATS] API: {len(api_qa)} QA tu {len(api_done_ids)} chunks")

    # -- Buoc 2: Template cho chunks con lai --
    print("\n  -- Buoc 2: Template-based generation --")
    template_qa = []
    template_count = 0
    for chunk in chunks:
        if chunk["id"] in api_done_ids:
            continue
        tqa = generate_qa_template(chunk)
        if tqa:
            template_qa.extend(tqa)
            template_count += 1

    print(f"  [STATS] Template: {len(template_qa)} QA tu {template_count} chunks")

    # -- Buoc 3: Neu van thieu, sinh template cho TAT CA chunks --
    all_qa = api_qa + template_qa
    if len(all_qa) < 300:
        print(f"\n  [WARNING] Moi co {len(all_qa)} QA, can them. Sinh template cho chunks da co API...")
        for chunk in chunks:
            if chunk["id"] not in api_done_ids:
                continue
            tqa = generate_qa_template(chunk)
            existing_questions = {q["question"] for q in all_qa}
            for qa in tqa:
                if qa["question"] not in existing_questions:
                    all_qa.append(qa)
                    existing_questions.add(qa["question"])
            if len(all_qa) >= 350:
                break

    if not all_qa:
        print("[ERROR] Khong sinh duoc QA nao!")
        return False

    # -- Luu ket qua --
    qa_all_path = OUTPUT_DIR / "qa_all.json"
    with open(qa_all_path, "w", encoding="utf-8") as f:
        json.dump(all_qa, f, ensure_ascii=False, indent=2)

    # Split train/test
    import random
    random.seed(42)
    random.shuffle(all_qa)

    test_size = min(50, len(all_qa) // 5)
    test_set = all_qa[:test_size]
    train_set = all_qa[test_size:]

    train_path = OUTPUT_DIR / "qa_train.json"
    test_path = OUTPUT_DIR / "qa_test.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_set, f, ensure_ascii=False, indent=2)
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_set, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Tong QA: {len(all_qa)}")
    print(f"   Train: {len(train_set)} -> {train_path}")
    print(f"   Test:  {len(test_set)} -> {test_path}")

    # Thong ke
    from collections import Counter
    types = Counter(qa.get("type", "unknown") for qa in all_qa)
    sources = Counter(qa.get("generated_by", "unknown") for qa in all_qa)
    print(f"   Phan bo type: {dict(types)}")
    print(f"   Nguon: {dict(sources)}")

    return True



# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("+" + "=" * 58 + "+")
    print("|   PHASE 2 - Data Processing & QA Generation            |")
    print("+" + "=" * 58 + "+")

    # 2A: Clean
    if not process_cleaning():
        sys.exit(1)

    # 2B: Chunk
    if not process_chunking():
        sys.exit(1)

    # 2C: Vector Store (can GPU cho embedding nhanh)
    build_vector_store()

    # 2D: QA Generation (can OPENAI_API_KEY)
    process_qa_generation()

    print("\n[OK] Phase 2 hoan tat! Tiep theo: chay phase3_finetune.py")

