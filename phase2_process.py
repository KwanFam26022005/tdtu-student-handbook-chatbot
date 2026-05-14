"""
Phase 2 – Data Processing: Cleaning, Chunking, Normalization, Vector Store, QA Generation.

Workflow:
  2A. Cleaning & Normalization (raw_text → clean_text)
  2B. Semantic Chunking (clean_text → chunks.json)
  2B+. Chunk Normalization (merge tiny / split large → parent_map.json)
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
import numpy as np
from pathlib import Path
from collections import defaultdict

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
CHUNK_SIZE = 768       # ký tự (≈512 tokens, 1 token ≈ 1.5 ký tự Việt)
CHUNK_OVERLAP = 96     # ký tự overlap
MIN_CHUNK_LENGTH = 50  # Bỏ chunk quá ngắn

# Chunk normalization config (Phase 2B+)
MIN_CHUNK_CHARS = 80   # Gộp chunk < 80 ký tự
MAX_CHUNK_CHARS = 3000 # Tách chunk > 3000 ký tự

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

    # 6. Gộp dòng bị ngắt giữa chừng (skip dòng bảng Markdown bắt đầu bằng '|')
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

    # 10. Chuẩn hóa viết tắt TDTU (word boundary, không thay trong URL/code)  # [NEW]
    TDTU_ABBREVS = {                                                           # [NEW]
        "SV": "sinh viên", "ĐH": "đại học", "P.ĐH": "Phòng đại học",          # [NEW]
        "HK": "học kỳ", "TC": "tín chỉ", "GPA": "điểm trung bình tích lũy",  # [NEW]
    }                                                                          # [NEW]
    for abbr, full in TDTU_ABBREVS.items():                                    # [NEW]
        # Chỉ thay khi đứng riêng (word boundary), skip nếu nằm trong URL     # [NEW]
        pattern = r'(?<![:/\w])\b' + re.escape(abbr) + r'\b(?![:/\w])'         # [NEW]
        text = re.sub(pattern, full, text)                                     # [NEW]

    # 11. Loại bỏ dòng footer/header lặp (≥3 lần trong cùng text)             # [NEW]
    line_list = text.split('\n')                                               # [NEW]
    from collections import Counter as _Counter                                # [NEW]
    line_counts = _Counter(line.strip() for line in line_list if line.strip())  # [NEW]
    repeated = {ln for ln, cnt in line_counts.items() if cnt >= 3}             # [NEW]
    if repeated:                                                               # [NEW]
        seen_repeated = set()                                                  # [NEW]
        filtered_lines = []                                                    # [NEW]
        for line in line_list:                                                 # [NEW]
            stripped = line.strip()                                            # [NEW]
            if stripped in repeated:                                           # [NEW]
                if stripped not in seen_repeated:                               # [NEW]
                    seen_repeated.add(stripped)                                 # [NEW]
                    filtered_lines.append(line)  # Giữ lần đầu tiên           # [NEW]
                # else: bỏ qua (xóa occurrence lặp)                            # [NEW]
            else:                                                              # [NEW]
                filtered_lines.append(line)                                    # [NEW]
        text = '\n'.join(filtered_lines)                                       # [NEW]

    return text.strip()


def process_cleaning():
    """Cleaning tất cả file raw_text → clean_text"""
    print("\n[PHASE 2A] Cleaning & Normalization")
    print("=" * 60)

    txt_files = sorted(f for f in RAW_TEXT_DIR.glob("*.txt")
                       if f.name != "ocr_progress.json")
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
            if len(section_text) > CHUNK_SIZE * 3:
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
        chunk["id"] = f"temp_{id(chunk)}"  # Temp ID, overwritten in process_chunking
    
    return chunks


def process_chunking():
    """Chunking: ưu tiên clean_text/, fallback raw_text/ nếu thiếu."""
    print("\n[PHASE 2B] Semantic Chunking")
    print("=" * 60)

    # Master list từ raw_text (bỏ ocr_progress.json)
    raw_files = sorted(f for f in RAW_TEXT_DIR.glob("*.txt")
                       if f.name != "ocr_progress.json")
    if not raw_files:
        print("[ERROR] Khong co file text nao!")
        return False

    all_chunks = []

    for idx, raw_path in enumerate(raw_files, 1):
        source_name = raw_path.stem
        clean_path = CLEAN_TEXT_DIR / raw_path.name

        if clean_path.exists():
            text = clean_path.read_text(encoding="utf-8")
        else:
            # Fallback: clean in-memory và lưu
            raw = raw_path.read_text(encoding="utf-8")
            text = clean_text(raw)
            CLEAN_TEXT_DIR.mkdir(parents=True, exist_ok=True)
            clean_path.write_text(text, encoding="utf-8")

        chunks = semantic_chunk(text, source_name)
        all_chunks.extend(chunks)
        print(f"  [{idx}/{len(raw_files)}] {source_name[:55]}: {len(chunks)} chunks")

    if not all_chunks:
        print("[ERROR] Khong tao duoc chunk nao!")
        return False

    # Gán ID duy nhất
    for i, chunk in enumerate(all_chunks):
        chunk["id"] = f"chunk_{i:04d}"

    # Lưu
    chunks_path = OUTPUT_DIR / "chunks.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Tong: {len(all_chunks)} chunks tu {len(raw_files)} files")

    lengths = [len(c["text"]) for c in all_chunks]
    print(f"   Min: {min(lengths)} | Max: {max(lengths)} | "
          f"Avg: {sum(lengths)//len(lengths)} ky tu")

    # Coverage check
    chunked = {c["source"] for c in all_chunks}
    missing = {f.stem for f in raw_files} - chunked
    if missing:
        print(f"\n  [WARNING] {len(missing)} files co 0 chunks:")
        for s in sorted(missing):
            print(f"    - {s}")

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

    # [MOD] Embed dùng text_for_retrieval (decoupled RAG), fallback text_with_context
    texts = [c.get("text_for_retrieval", c["text_with_context"]) for c in chunks]  # [MOD]

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
# 2B+. CHUNK NORMALIZATION & PARENT-CHILD MAPPING
# ══════════════════════════════════════════════════════════

def merge_tiny_chunks(chunks: list[dict]) -> list[dict]:
    """
    Merge chunks shorter than MIN_CHUNK_CHARS into their nearest neighbor.
    Only merges within the same source file to avoid cross-document contamination.
    """
    if not chunks:
        return chunks

    from itertools import groupby
    groups = []
    for key, group in groupby(chunks, key=lambda c: c.get("source", "")):
        groups.append((key, list(group)))

    merged_all = []
    merge_count = 0

    for source, group_chunks in groups:
        merged = []
        buffer = None

        for chunk in group_chunks:
            if len(chunk["text"]) < MIN_CHUNK_CHARS:
                merge_count += 1
                if buffer is None:
                    buffer = chunk.copy()
                else:
                    buffer["text"] += "\n" + chunk["text"]
                    buffer["text_with_context"] += "\n" + chunk["text"]
            else:
                if buffer is not None:
                    chunk = chunk.copy()
                    merged_text = buffer["text"] + "\n" + chunk["text"]
                    chunk["text"] = merged_text
                    # Rebuild text_with_context properly
                    hdr = [f"[{chunk['source']}]"]
                    if chunk.get('chapter'): hdr.append(chunk['chapter'])
                    if chunk.get('section'): hdr.append(chunk['section'])
                    chunk["text_with_context"] = f"{' - '.join(hdr)}\n{merged_text}"
                    buffer = None
                merged.append(chunk)

        if buffer is not None:
            if merged:
                last = merged[-1].copy()
                last["text"] += "\n" + buffer["text"]
                last["text_with_context"] += "\n" + buffer["text_with_context"]
                merged[-1] = last
            else:
                merged.append(buffer)

        merged_all.extend(merged)

    print(f"  Merged {merge_count} tiny chunks")
    return merged_all


def split_large_chunks(chunks: list[dict]) -> list[dict]:
    """
    Split chunks longer than MAX_CHUNK_CHARS at paragraph boundaries.
    Each sub-chunk inherits metadata and gets a parent_id link.
    Table-aware: nếu chunk chứa Markdown table, prepend header+divider vào mỗi sub-chunk.
    """
    result = []
    split_count = 0

    for chunk in chunks:
        if len(chunk["text"]) <= MAX_CHUNK_CHARS:
            result.append(chunk)
            continue

        split_count += 1
        text_body = chunk["text"]

        # [NEW] Detect Markdown table: chunk chứa ký tự '|'
        is_table = '|' in text_body                                            # [NEW]
        table_header = ""                                                      # [NEW]
        if is_table:                                                           # [NEW]
            body_lines = text_body.split('\n')                                 # [NEW]
            # Tách 2 dòng đầu (header row + divider row)                       # [NEW]
            if len(body_lines) >= 2:                                           # [NEW]
                table_header = body_lines[0] + '\n' + body_lines[1] + '\n'     # [NEW]
                text_body = '\n'.join(body_lines[2:])                          # [NEW]

        paragraphs = text_body.split("\n\n")
        sub_texts = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) > MAX_CHUNK_CHARS and current.strip():
                sub_texts.append(current.strip())
                current = para + "\n\n"
            else:
                current += para + "\n\n"

        if current.strip():
            sub_texts.append(current.strip())

        header_parts = [f"[{chunk['source']}]"]
        if chunk.get("chapter"):
            header_parts.append(chunk["chapter"])
        if chunk.get("section"):
            header_parts.append(chunk["section"])
        context_header = " - ".join(header_parts)

        for i, sub_text in enumerate(sub_texts):
            new_chunk = chunk.copy()
            # [NEW] Nếu là bảng, prepend header+divider vào mỗi sub-chunk
            if is_table and table_header:                                      # [NEW]
                sub_text = table_header + sub_text                             # [NEW]
            new_chunk["text"] = sub_text
            new_chunk["text_with_context"] = f"{context_header}\n{sub_text}"
            new_chunk["parent_id"] = chunk["id"]
            new_chunk["sub_index"] = i
            result.append(new_chunk)

    print(f"  Split {split_count} oversized chunks")
    return result


def build_parent_map(chunks: list[dict]) -> dict:
    """
    Build a mapping: chunk_id -> {chapter, section, source, siblings}.
    Used by ParentContextExpander in Phase 4 to pull related chunks.
    """
    chapter_groups = defaultdict(list)
    for chunk in chunks:
        key = (chunk.get("source", ""), chunk.get("chapter", ""))
        chapter_groups[key].append(chunk["id"])

    section_groups = defaultdict(list)
    for chunk in chunks:
        key = (chunk.get("source", ""), chunk.get("section", ""))
        section_groups[key].append(chunk["id"])

    parent_map = {}
    for chunk in chunks:
        ch_key = (chunk.get("source", ""), chunk.get("chapter", ""))
        sec_key = (chunk.get("source", ""), chunk.get("section", ""))

        parent_map[chunk["id"]] = {
            "chapter": chunk.get("chapter", ""),
            "section": chunk.get("section", ""),
            "source": chunk.get("source", ""),
            "parent_id": chunk.get("parent_id", ""),
            "section_siblings": [
                cid for cid in section_groups.get(sec_key, [])
                if cid != chunk["id"]
            ],
            "chapter_siblings": [
                cid for cid in chapter_groups.get(ch_key, [])
                if cid != chunk["id"]
            ],
        }

    return parent_map


def process_chunk_normalization():
    """Phase 2B+: Normalize chunks va build parent-child mapping."""
    print("\n[PHASE 2B+] Chunk Normalization & Parent-Child Mapping")
    print("=" * 60)

    chunks_path = OUTPUT_DIR / "chunks.json"
    if not chunks_path.exists():
        print(f"[ERROR] {chunks_path} not found. Chay chunking truoc!")
        return False

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    orig_count = len(chunks)
    orig_lens = [len(c["text"]) for c in chunks]
    print(f"  Loaded {orig_count} chunks")
    print(f"  min={min(orig_lens)}, max={max(orig_lens)}, "
          f"avg={sum(orig_lens) // len(orig_lens)}")

    # Step 1: Merge tiny
    print(f"\n  [STEP 1] Merging tiny chunks (<{MIN_CHUNK_CHARS} chars)...")
    chunks = merge_tiny_chunks(chunks)
    print(f"  Result: {len(chunks)} chunks")

    # Step 2: Split oversized
    print(f"\n  [STEP 2] Splitting oversized chunks (>{MAX_CHUNK_CHARS} chars)...")
    chunks = split_large_chunks(chunks)
    print(f"  Result: {len(chunks)} chunks")

    # Re-assign sequential IDs
    for i, chunk in enumerate(chunks):
        chunk["id"] = f"chunk_{i:04d}"

    # Step 3: Parent map
    print("\n  [STEP 3] Building parent-child mapping...")
    parent_map = build_parent_map(chunks)

    # Save chunks (overwrite)
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    parent_map_path = OUTPUT_DIR / "parent_map.json"
    with open(parent_map_path, "w", encoding="utf-8") as f:
        json.dump(parent_map, f, ensure_ascii=False, indent=2)

    metadata = [
        {"id": c["id"], "source": c["source"],
         "section": c.get("section", ""), "chapter": c.get("chapter", "")}
        for c in chunks
    ]
    meta_path = OUTPUT_DIR / "chunks_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # Stats
    new_lens = [len(c["text"]) for c in chunks]
    print(f"\n  [STATS] Normalization Results:")
    print(f"  Before: {orig_count} chunks | "
          f"min={min(orig_lens)} max={max(orig_lens)}")
    print(f"  After:  {len(chunks)} chunks | "
          f"min={min(new_lens)} max={max(new_lens)}")
    print(f"  Avg: {sum(new_lens) // len(new_lens)} chars")

    print(f"\n[OK] Chunk normalization complete!")
    print(f"  chunks.json:   {chunks_path}")
    print(f"  parent_map:    {parent_map_path}")
    return True


# ══════════════════════════════════════════════════════════
# 2B++ SEMANTIC TAGGING (regex only, no API)                  # [NEW]
# ══════════════════════════════════════════════════════════

def add_semantic_tags(chunks: list[dict]) -> list[dict]:                        # [NEW]
    """
    Gán semantic_tags dict vào mỗi chunk bằng regex.
    Không dùng API.
    Tags: doi_tuong, loai_van_ban, do_quan_trong.
    """
    # Keyword maps                                                             # [NEW]
    DOI_TUONG_KW = {                                                           # [NEW]
        "sinh viên": ["sinh viên", "sv", "người học", "học viên"],              # [NEW]
        "giảng viên": ["giảng viên", "gv", "người dạy", "giáo viên"],          # [NEW]
        "cán bộ": ["cán bộ", "nhân viên", "viên chức"],                        # [NEW]
    }                                                                          # [NEW]
    LOAI_VB_KW = {                                                             # [NEW]
        "quy định": ["quy định", "điều", "khoản"],                             # [NEW]
        "hướng dẫn": ["hướng dẫn", "thủ tục", "quy trình", "trình tự"],       # [NEW]
        "quy chế": ["quy chế"],                                                # [NEW]
        "thông báo": ["thông báo", "thông tin"],                               # [NEW]
    }                                                                          # [NEW]
    HIGH_IMPORTANCE_KW = [                                                     # [NEW]
        "bắt buộc", "cấm", "kỷ luật", "buộc thôi học", "học bổng"             # [NEW]
    ]                                                                          # [NEW]

    print("\n[PHASE 2B++] Semantic Tagging (regex)")                           # [NEW]
    print("=" * 60)                                                            # [NEW]

    for chunk in chunks:                                                       # [NEW]
        text_lower = chunk["text"].lower()                                     # [NEW]

        # --- doi_tuong ---                                                    # [NEW]
        detected_dt = []                                                       # [NEW]
        for label, keywords in DOI_TUONG_KW.items():                           # [NEW]
            if any(kw in text_lower for kw in keywords):                        # [NEW]
                detected_dt.append(label)                                      # [NEW]
        if not detected_dt:                                                    # [NEW]
            detected_dt = ["tất cả"]                                           # [NEW]

        # --- loai_van_ban ---                                                 # [NEW]
        detected_lvb = []                                                      # [NEW]
        for label, keywords in LOAI_VB_KW.items():                             # [NEW]
            if any(kw in text_lower for kw in keywords):                        # [NEW]
                detected_lvb.append(label)                                     # [NEW]
        if not detected_lvb:                                                   # [NEW]
            detected_lvb = ["quy định"]  # default                             # [NEW]

        # --- do_quan_trong ---                                                # [NEW]
        if any(kw in text_lower for kw in HIGH_IMPORTANCE_KW):                 # [NEW]
            importance = "cao"                                                  # [NEW]
        else:                                                                  # [NEW]
            importance = "trung bình"  # mặc định                              # [NEW]

        chunk["semantic_tags"] = {                                             # [NEW]
            "doi_tuong": detected_dt,                                          # [NEW]
            "loai_van_ban": detected_lvb,                                      # [NEW]
            "do_quan_trong": importance,                                       # [NEW]
        }                                                                      # [NEW]

    # Stats                                                                    # [NEW]
    from collections import Counter as _StatsCounter                           # [NEW]
    imp_counts = _StatsCounter(c["semantic_tags"]["do_quan_trong"] for c in chunks)  # [NEW]
    print(f"  Tagged {len(chunks)} chunks")                                    # [NEW]
    print(f"  do_quan_trong: {dict(imp_counts)}")                              # [NEW]
    print(f"[OK] Semantic tagging complete!")                                  # [NEW]

    return chunks                                                              # [NEW]


# ══════════════════════════════════════════════════════════
# 2B+++ DECOUPLED RAG — Retrieval Summaries (GPT-4o-mini)     # [NEW]
# ══════════════════════════════════════════════════════════

SUMMARY_SYSTEM_PROMPT = """Bạn là trợ lý tóm tắt văn bản pháp lý tiếng Việt cho hệ thống RAG.
Nhiệm vụ: Tạo bản tóm tắt ngắn để tối ưu retrieval (tìm kiếm).

Yêu cầu:
1. Tóm tắt nội dung chính ≤60 từ
2. Liệt kê 5-8 từ khóa quan trọng nhất (keyword)
3. Ghi rõ section header nếu có

Trả về JSON object DUY NHẤT (KHÔNG markdown):
{"summary": "...", "keywords": ["kw1", "kw2", ...]}"""                         # [NEW]


def generate_retrieval_summaries(chunks: list[dict]) -> list[dict]:            # [NEW]
    """
    Gọi GPT-4o-mini để sinh text_for_retrieval và text_for_generation cho mỗi chunk.
    Resume logic tương tự generate_qa_api() — lưu progress vào summary_progress.json.
    """
    import openai                                                              # [NEW]

    api_key = os.environ.get("OPENAI_API_KEY", "")                             # [NEW]
    if not api_key:                                                            # [NEW]
        print("  [WARNING] Khong co OPENAI_API_KEY -> fallback text_with_context")  # [NEW]
        for chunk in chunks:                                                   # [NEW]
            chunk["text_for_retrieval"] = chunk.get("text_with_context", chunk["text"])  # [NEW]
            chunk["text_for_generation"] = chunk["text"]                        # [NEW]
        return chunks                                                          # [NEW]

    client = openai.OpenAI(api_key=api_key)                                    # [NEW]

    print("\n[PHASE 2B+++] Generate Retrieval Summaries (GPT-4o-mini)")        # [NEW]
    print("=" * 60)                                                            # [NEW]

    # Resume logic (cùng pattern với generate_qa_api)                          # [NEW]
    progress_path = OUTPUT_DIR / "summary_progress.json"                       # [NEW]
    if progress_path.exists():                                                 # [NEW]
        with open(progress_path, "r", encoding="utf-8") as f:                  # [NEW]
            progress_data = json.load(f)                                       # [NEW]
        done_summaries = {item["chunk_id"]: item for item in progress_data.get("summaries", [])}  # [NEW]
        print(f"  [RESUME] {len(done_summaries)} summaries đã có")             # [NEW]
    else:                                                                      # [NEW]
        done_summaries = {}                                                    # [NEW]

    # Áp dụng summaries đã có                                                 # [NEW]
    for chunk in chunks:                                                       # [NEW]
        if chunk["id"] in done_summaries:                                      # [NEW]
            chunk["text_for_retrieval"] = done_summaries[chunk["id"]]["text_for_retrieval"]  # [NEW]
            chunk["text_for_generation"] = chunk["text"]                        # [NEW]

    pending = [c for c in chunks if c["id"] not in done_summaries]             # [NEW]
    print(f"  [PENDING] Còn {len(pending)} chunks cần API")                    # [NEW]

    if not pending:                                                            # [NEW]
        # Đảm bảo tất cả chunks đều có field                                  # [NEW]
        for chunk in chunks:                                                   # [NEW]
            chunk.setdefault("text_for_retrieval", chunk.get("text_with_context", chunk["text"]))  # [NEW]
            chunk.setdefault("text_for_generation", chunk["text"])              # [NEW]
        return chunks                                                          # [NEW]

    consecutive_429 = 0                                                        # [NEW]
    new_summaries = list(done_summaries.values())                               # [NEW]

    for chunk in pending:                                                      # [NEW]
        section = chunk.get("section", "")                                     # [NEW]
        prompt = f"""{SUMMARY_SYSTEM_PROMPT}

Section: {section}
Đoạn văn bản:
---
{chunk['text']}
---

Trả về JSON object. KHÔNG markdown."""                                         # [NEW]

        success = False                                                        # [NEW]
        for attempt in range(3):                                               # [NEW]
            try:                                                               # [NEW]
                response = client.chat.completions.create(                      # [NEW]
                    model="gpt-4o-mini",                                        # [NEW]
                    messages=[{"role": "user", "content": prompt}],              # [NEW]
                    temperature=0.1,                                           # [NEW]
                )                                                              # [NEW]
                resp = response.choices[0].message.content                      # [NEW]

                # Parse JSON                                                   # [NEW]
                resp_clean = re.sub(r'```json\s*', '', resp)                    # [NEW]
                resp_clean = re.sub(r'```\s*', '', resp_clean)                  # [NEW]
                json_match = re.search(r'\{.*\}', resp_clean, re.DOTALL)        # [NEW]

                if json_match:                                                 # [NEW]
                    data = json.loads(json_match.group(0))                      # [NEW]
                    summary = str(data.get("summary", "")).strip()              # [NEW]
                    keywords = data.get("keywords", [])                         # [NEW]
                    if isinstance(keywords, list):                              # [NEW]
                        keywords = [str(k).strip() for k in keywords]          # [NEW]
                    else:                                                      # [NEW]
                        keywords = []                                          # [NEW]

                    # Build text_for_retrieval                                 # [NEW]
                    kw_str = ", ".join(keywords) if keywords else ""            # [NEW]
                    header_part = f"[{section}] " if section else ""            # [NEW]
                    text_for_retrieval = f"{header_part}{summary}\nTừ khóa: {kw_str}"  # [NEW]

                    chunk["text_for_retrieval"] = text_for_retrieval            # [NEW]
                    chunk["text_for_generation"] = chunk["text"]                # [NEW]

                    new_summaries.append({                                     # [NEW]
                        "chunk_id": chunk["id"],                                # [NEW]
                        "text_for_retrieval": text_for_retrieval,               # [NEW]
                    })                                                         # [NEW]
                    done_summaries[chunk["id"]] = {                             # [NEW]
                        "chunk_id": chunk["id"],                                # [NEW]
                        "text_for_retrieval": text_for_retrieval,               # [NEW]
                    }                                                          # [NEW]
                    consecutive_429 = 0                                        # [NEW]
                    success = True                                             # [NEW]
                    print(f"  [OK] {chunk['id']}: summary {len(summary)} chars, {len(keywords)} kw")  # [NEW]
                    break                                                      # [NEW]
                else:                                                          # [NEW]
                    print(f"  [WARNING] {chunk['id']}: JSON parse fail (lần {attempt+1})")  # [NEW]

            except Exception as e:                                             # [NEW]
                err = str(e)                                                   # [NEW]
                if "429" in err or "quota" in err.lower() or "rate" in err.lower():  # [NEW]
                    consecutive_429 += 1                                       # [NEW]
                    wait = 60                                                  # [NEW]
                    m = re.search(r'retry.*?(\d+\.?\d*)\s*s', err, re.IGNORECASE)  # [NEW]
                    if m:                                                      # [NEW]
                        wait = float(m.group(1)) + 5                           # [NEW]

                    if consecutive_429 >= 5:                                    # [NEW]
                        print(f"\n  [PAUSE] Hết quota. Lưu {len(new_summaries)} summaries.")  # [NEW]
                        with open(progress_path, "w", encoding="utf-8") as f:  # [NEW]
                            json.dump({"summaries": new_summaries},            # [NEW]
                                     f, ensure_ascii=False, indent=2)          # [NEW]
                        # Fallback cho chunks chưa có summary                  # [NEW]
                        for c in chunks:                                        # [NEW]
                            c.setdefault("text_for_retrieval", c.get("text_with_context", c["text"]))  # [NEW]
                            c.setdefault("text_for_generation", c["text"])      # [NEW]
                        return chunks                                          # [NEW]

                    print(f"  [WAIT] 429 (lần {attempt+1}). Chờ {wait:.0f}s...")  # [NEW]
                    time.sleep(wait)                                           # [NEW]
                else:                                                          # [NEW]
                    print(f"  [ERROR] {chunk['id']}: {err[:80]}")              # [NEW]
                    time.sleep(5)                                              # [NEW]

        if not success:                                                        # [NEW]
            # Fallback: dùng text_with_context                                 # [NEW]
            chunk["text_for_retrieval"] = chunk.get("text_with_context", chunk["text"])  # [NEW]
            chunk["text_for_generation"] = chunk["text"]                        # [NEW]
            print(f"  [FALLBACK] {chunk['id']}: dùng text_with_context")       # [NEW]

        # Save progress mỗi 10 chunks                                         # [NEW]
        if len(done_summaries) % 10 == 0 and len(done_summaries) > 0:          # [NEW]
            with open(progress_path, "w", encoding="utf-8") as f:              # [NEW]
                json.dump({"summaries": new_summaries},                        # [NEW]
                         f, ensure_ascii=False, indent=2)                      # [NEW]
            print(f"  [SAVE] Saved: {len(new_summaries)} summaries")           # [NEW]

        time.sleep(1)                                                          # [NEW]

    # Save cuối                                                                # [NEW]
    with open(progress_path, "w", encoding="utf-8") as f:                      # [NEW]
        json.dump({"summaries": new_summaries},                                # [NEW]
                 f, ensure_ascii=False, indent=2)                              # [NEW]

    # Đảm bảo tất cả chunks đều có field                                      # [NEW]
    for chunk in chunks:                                                       # [NEW]
        chunk.setdefault("text_for_retrieval", chunk.get("text_with_context", chunk["text"]))  # [NEW]
        chunk.setdefault("text_for_generation", chunk["text"])                  # [NEW]

    print(f"\n[OK] Generated {len(new_summaries)} retrieval summaries")        # [NEW]
    return chunks                                                              # [NEW]


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

    # 2B+: Normalize chunks & build parent map
    if not process_chunk_normalization():
        sys.exit(1)

    # 2B++: Semantic tagging (regex, no API)                                   # [NEW]
    chunks_path = OUTPUT_DIR / "chunks.json"                                   # [NEW]
    with open(chunks_path, "r", encoding="utf-8") as f:                        # [NEW]
        chunks = json.load(f)                                                  # [NEW]
    chunks = add_semantic_tags(chunks)                                         # [NEW]

    # 2B+++: Generate retrieval summaries (GPT-4o-mini)                        # [NEW]
    chunks = generate_retrieval_summaries(chunks)                              # [NEW]

    # Save chunks with new fields                                             # [NEW]
    with open(chunks_path, "w", encoding="utf-8") as f:                        # [NEW]
        json.dump(chunks, f, ensure_ascii=False, indent=2)                     # [NEW]

    # 2C: Vector Store (can GPU cho embedding nhanh)
    build_vector_store()

    # 2D: QA Generation (can OPENAI_API_KEY)
    process_qa_generation()

    print("\n[OK] Phase 2 hoan tat! Tiep theo: chay phase3_finetune.py")
