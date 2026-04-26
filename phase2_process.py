"""
Phase 2 – Data Processing: Cleaning, Chunking, Vector Store, QA Generation.

Workflow:
  2A. Cleaning & Normalization (raw_text → clean_text)
  2B. Semantic Chunking (clean_text → chunks.json)
  2C. Build Vector Store (chunks → FAISS index)
  2D. Generate QA pairs (chunks → qa_train.json + qa_test.json)

Cài đặt:
  pip install sentence-transformers faiss-cpu langchain langchain-community
  pip install google-generativeai   # Cho QA generation bằng Gemini
  
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
    - Sửa lỗi OCR phổ biến
    - Gộp dòng bị ngắt giữa chừng
    - Xóa header/footer lặp
    - Giữ cấu trúc Điều/Khoản
    """
    # 1. Chuẩn hóa Unicode
    text = unicodedata.normalize("NFC", text)
    
    # 2. Xóa ký tự điều khiển (giữ \n, \t)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    
    # 3. Xóa marker trang từ OCR output
    text = re.sub(r'---\s*Trang\s+\d+/\d+\s*---', '\n', text)
    
    # 4. Sửa lỗi OCR phổ biến cho tiếng Việt
    ocr_fixes = {
        'Ðiều': 'Điều',
        'Ðại': 'Đại',
        'Ðào': 'Đào',
        'Ðánh': 'Đánh',
        'Ðể': 'Để',
        'Ðược': 'Được',
        'Ðối': 'Đối',
        'Ðơn': 'Đơn',
        'l1': 'l1',      # Không sửa số
        '|': 'l',        # | thường bị nhầm với l
    }
    for wrong, right in ocr_fixes.items():
        text = text.replace(wrong, right)
    
    # 5. Gộp dòng bị ngắt giữa chừng
    # (dòng kết thúc bằng chữ thường + dòng sau bắt đầu bằng chữ thường)
    text = re.sub(r'([a-zàáạảãăắằặẳẵâấầậẩẫđèéẹẻẽêếềệểễìíịỉĩòóọỏõôốồộổỗơớờợởỡùúụủũưứừựửữỳýỵỷỹ,])\n([a-zàáạảãăắằặẳẵâấầậẩẫđèéẹẻẽêếềệểễìíịỉĩòóọỏõôốồộổỗơớờợởỡùúụủũưứừựửữỳýỵỷỹ])', r'\1 \2', text)
    
    # 6. Gộp khoảng trắng thừa (giữ \n)
    text = re.sub(r'[^\S\n]+', ' ', text)
    
    # 7. Xóa dòng trống liên tiếp (giữ tối đa 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 8. Strip mỗi dòng
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    
    return text.strip()


def process_cleaning():
    """Cleaning tất cả file raw_text → clean_text"""
    print("\n📝 PHASE 2A: Cleaning & Normalization")
    print("=" * 60)
    
    txt_files = sorted(RAW_TEXT_DIR.glob("*.txt"))
    if not txt_files:
        print(f"❌ Không có file .txt nào trong {RAW_TEXT_DIR}")
        print("   → Hãy chạy phase1_ocr.py trước!")
        return False
    
    for idx, txt_path in enumerate(txt_files, 1):
        raw = txt_path.read_text(encoding="utf-8")
        cleaned = clean_text(raw)
        
        out_path = CLEAN_TEXT_DIR / txt_path.name
        out_path.write_text(cleaned, encoding="utf-8")
        
        print(f"  [{idx}/{len(txt_files)}] {txt_path.name}: "
              f"{len(raw):,} → {len(cleaned):,} ký tự")
    
    print(f"\n✅ Đã clean {len(txt_files)} file → {CLEAN_TEXT_DIR}")
    return True


# ══════════════════════════════════════════════════════════
# 2B. SEMANTIC CHUNKING
# ══════════════════════════════════════════════════════════

def semantic_chunk(text: str, source_name: str) -> list[dict]:
    """
    Tách text thành chunks thông minh:
    - Ưu tiên tách theo Điều/Khoản/Mục/Chương
    - Fallback: tách theo paragraph + overlap
    - Thêm context header vào mỗi chunk
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
                paragraphs = section_text.split('\n\n')
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
        paragraphs = text.split('\n\n')
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
    """Chunking tất cả clean text → chunks.json"""
    print("\n📝 PHASE 2B: Semantic Chunking")
    print("=" * 60)
    
    txt_files = sorted(CLEAN_TEXT_DIR.glob("*.txt"))
    if not txt_files:
        print("❌ Không có file clean text. Chạy cleaning trước!")
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
    
    print(f"\n✅ Tổng: {len(all_chunks)} chunks → {chunks_path}")
    
    # Thống kê
    lengths = [len(c["text"]) for c in all_chunks]
    print(f"   Min: {min(lengths)} ký tự | Max: {max(lengths)} ký tự | "
          f"Trung bình: {sum(lengths)//len(lengths)} ký tự")
    
    return True


# ══════════════════════════════════════════════════════════
# 2C. BUILD VECTOR STORE
# ══════════════════════════════════════════════════════════

def build_vector_store():
    """Embed chunks + lưu FAISS index"""
    print("\n📝 PHASE 2C: Build Vector Store (FAISS + bge-m3)")
    print("=" * 60)
    
    chunks_path = OUTPUT_DIR / "chunks.json"
    if not chunks_path.exists():
        print("❌ Chưa có chunks.json. Chạy chunking trước!")
        return False
    
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    print(f"  📄 Loaded {len(chunks)} chunks")
    
    # Embedding
    from sentence_transformers import SentenceTransformer
    import numpy as np
    
    print(f"  🔧 Loading embedding model: {EMBEDDING_MODEL}...")
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    
    # Embed tất cả chunks (dùng text_with_context để có ngữ cảnh)
    texts = [c["text_with_context"] for c in chunks]
    
    print(f"  ⚡ Đang embed {len(texts)} chunks...")
    embeddings = embed_model.encode(
        texts, 
        show_progress_bar=True, 
        batch_size=32,
        normalize_embeddings=True
    )
    
    # Build FAISS index
    import faiss
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product (cosine vì đã normalize)
    index.add(np.array(embeddings).astype('float32'))
    
    # Lưu
    faiss_path = OUTPUT_DIR / "faiss_index.bin"
    faiss.write_index(index, str(faiss_path))
    
    # Lưu metadata mapping
    metadata = [{"id": c["id"], "source": c["source"], "section": c["section"], 
                 "chapter": c["chapter"]} for c in chunks]
    meta_path = OUTPUT_DIR / "chunks_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ FAISS index: {faiss_path} (dim={dim}, n={index.ntotal})")
    print(f"✅ Metadata: {meta_path}")
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


def generate_qa_batch(chunks: list[dict], batch_size: int = 5) -> list[dict]:
    """
    Sinh QA pairs từ chunks bằng Gemini API.
    
    Features:
      - Exponential backoff khi bị 429 (quota exceeded)
      - Parse retry-after time từ error message
      - Resume từ file progress (không chạy lại chunk đã xong)
      - Fallback model chain khi model chính bị quota
      - Lưu progress sau mỗi chunk thành công
    
    Cần set: GOOGLE_API_KEY environment variable
    """
    import google.generativeai as genai
    
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        print("⚠️  Chưa set GOOGLE_API_KEY!")
        print("   export GOOGLE_API_KEY='your-key-here'")
        print("   Hoặc trên Colab: from google.colab import userdata; key = userdata.get('GOOGLE_API_KEY')")
        return []
    
    genai.configure(api_key=api_key)
    
    # Danh sách model fallback (thử lần lượt khi bị quota)
    # gemini-2.0-flash và 1.5-flash đã deprecated/xóa → dùng 2.5 series
    MODEL_CHAIN = [
        "gemini-2.5-flash",       # Stable, best price-performance
        "gemini-2.5-flash-lite",  # Budget, fastest
        "gemini-2.0-flash",       # Deprecated nhưng vẫn chạy được (fallback cuối)
    ]
    
    current_model_idx = 0
    model = genai.GenerativeModel(MODEL_CHAIN[current_model_idx])
    print(f"   🤖 Model hiện tại: {MODEL_CHAIN[current_model_idx]}")
    
    # ── Resume logic: load progress nếu có ──
    progress_path = OUTPUT_DIR / "qa_progress.json"
    if progress_path.exists():
        with open(progress_path, "r", encoding="utf-8") as f:
            progress_data = json.load(f)
        all_qa = progress_data.get("qa_pairs", [])
        done_ids = set(progress_data.get("done_chunk_ids", []))
        print(f"   📂 Resume: đã có {len(all_qa)} QA từ {len(done_ids)} chunks trước đó")
    else:
        all_qa = []
        done_ids = set()
    
    # ── Hàm parse retry-after từ error message ──
    def parse_retry_seconds(error_msg: str) -> float:
        """Trích xuất số giây chờ từ 'Please retry in X.XXs'"""
        match = re.search(r'retry in (\d+\.?\d*)', str(error_msg))
        if match:
            return float(match.group(1)) + 2  # +2s buffer
        return 60  # Default 60s nếu không parse được

    def parse_json_array(response_text: str) -> list[dict] | None:
        """Parse JSON array an toàn từ output model (có thể kèm markdown/code-fence)."""
        if not response_text:
            return None

        cleaned = response_text.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

        # Thử parse trực tiếp trước
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass

        # Fallback: lấy JSON array lớn nhất trong text
        start_idx = cleaned.find("[")
        end_idx = cleaned.rfind("]")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            candidate = cleaned[start_idx:end_idx + 1]
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                return None

        return None
    
    consecutive_429 = 0
    MAX_CONSECUTIVE_429 = 5  # Sau 5 lần 429 liên tiếp → chuyển model
    base_sleep_seconds = max(4.0, 60.0 / max(batch_size * 3, 15))
    min_qa_per_chunk = 3
    max_qa_per_chunk = 5
    
    pending_chunks = [c for c in chunks if c["id"] not in done_ids]
    total_pending = len(pending_chunks)
    if total_pending:
        print(f"   🧩 Còn {total_pending} chunks cần xử lý")

    # Chạy theo nhiều vòng để tránh bỏ sót chunk khi gặp quota tạm thời
    max_global_passes = 4
    for pass_idx in range(1, max_global_passes + 1):
        if not pending_chunks:
            break

        print(f"\n   🔁 Pass {pass_idx}/{max_global_passes} - pending: {len(pending_chunks)} chunks")
        next_round_pending = []

        for chunk in pending_chunks:
            # Skip chunk đã xử lý
            if chunk["id"] in done_ids:
                continue
        
            prompt = f"""{QA_SYSTEM_PROMPT}

Đoạn văn bản quy chế:
---
{chunk['text_with_context']}
---

Hãy tạo từ {min_qa_per_chunk} đến {max_qa_per_chunk} cặp câu hỏi-trả lời từ đoạn trên.
Chỉ trả về JSON array hợp lệ, không thêm markdown/code fence."""
        
            # Retry logic cho mỗi chunk
            max_retries = 3
            success = False

            for attempt in range(max_retries):
                try:
                    response = model.generate_content(prompt)
                    response_text = getattr(response, "text", "")
                    qa_pairs = parse_json_array(response_text)

                    if qa_pairs:
                        # Lọc output để tránh format không hợp lệ
                        valid_pairs = []
                        for qa in qa_pairs:
                            if not isinstance(qa, dict):
                                continue
                            question = str(qa.get("question", "")).strip()
                            answer = str(qa.get("answer", "")).strip()
                            qa_type = str(qa.get("type", "factual")).strip().lower()
                            if not question or not answer:
                                continue
                            if qa_type not in {"factual", "conditional", "procedural", "reasoning"}:
                                qa_type = "factual"
                            valid_pairs.append({
                                "question": question,
                                "answer": answer,
                                "type": qa_type,
                                "source": chunk["source"],
                                "section": chunk.get("section", ""),
                                "chunk_id": chunk["id"],
                            })

                        if valid_pairs:
                            all_qa.extend(valid_pairs)
                            done_ids.add(chunk["id"])
                            consecutive_429 = 0  # Reset counter
                            base_sleep_seconds = max(3.5, base_sleep_seconds * 0.95)
                            print(
                                f"   ✅ Chunk {chunk['id']}: {len(valid_pairs)} QA pairs "
                                f"(tổng: {len(all_qa)})"
                            )
                            success = True
                            break

                    print(f"   ⚠️  Chunk {chunk['id']}: JSON không hợp lệ (lần {attempt+1})")
                    time.sleep(3 + attempt * 2)
                except Exception as e:
                    error_msg = str(e)

                    if "429" in error_msg or "quota" in error_msg.lower():
                        consecutive_429 += 1
                        wait_time = max(parse_retry_seconds(error_msg), base_sleep_seconds * 1.5)
                        base_sleep_seconds = min(65.0, max(base_sleep_seconds * 1.25, wait_time / 2))

                        # Nếu 429 liên tiếp quá nhiều → thử model khác
                        if consecutive_429 >= MAX_CONSECUTIVE_429:
                            current_model_idx += 1
                            if current_model_idx < len(MODEL_CHAIN):
                                new_model = MODEL_CHAIN[current_model_idx]
                                print(f"\n   🔄 Chuyển sang model: {new_model}")
                                model = genai.GenerativeModel(new_model)
                                consecutive_429 = 0
                                wait_time = 5  # Thử sớm với model mới
                            else:
                                print(
                                    f"\n   ⏸️  Tất cả model đều bị quota. "
                                    f"Đã lưu {len(all_qa)} QA. Chờ reset quota rồi chạy lại."
                                )
                                with open(progress_path, "w", encoding="utf-8") as f:
                                    json.dump(
                                        {"qa_pairs": all_qa, "done_chunk_ids": list(done_ids)},
                                        f,
                                        ensure_ascii=False,
                                        indent=2,
                                    )
                                return all_qa

                        print(f"   ⏳ 429 Rate limit (lần {attempt+1}). Chờ {wait_time:.0f}s...")
                        time.sleep(wait_time)
                    elif "503" in error_msg or "unavailable" in error_msg.lower():
                        wait_time = min(40.0, 8.0 * (attempt + 1))
                        print(f"   ⏳ 503 Service unavailable (lần {attempt+1}). Chờ {wait_time:.0f}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"   ❌ Chunk {chunk['id']} (lần {attempt+1}): {error_msg[:120]}")
                        time.sleep(5 + attempt * 2)

            if not success:
                # Không đánh dấu done; retry lại ở pass sau
                next_round_pending.append(chunk)
                print(f"   🔁 Sẽ retry chunk {chunk['id']} ở pass sau")

            # Lưu progress sau mỗi 10 chunks thành công
            if len(done_ids) % 10 == 0 and len(done_ids) > 0:
                with open(progress_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {"qa_pairs": all_qa, "done_chunk_ids": list(done_ids)},
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )
                print(f"   💾 Progress saved: {len(all_qa)} QA / {len(done_ids)} chunks")

            time.sleep(base_sleep_seconds)  # Adaptive pacing

        pending_chunks = next_round_pending

    if pending_chunks:
        print(f"\n   ⚠️  Còn {len(pending_chunks)} chunks chưa xử lý sau {max_global_passes} passes.")
    
    # Lưu progress cuối cùng
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump({"qa_pairs": all_qa, "done_chunk_ids": list(done_ids)}, 
                 f, ensure_ascii=False, indent=2)
    
    return all_qa


def process_qa_generation():
    """Sinh QA pairs từ chunks"""
    print("\n📝 PHASE 2D: Generate QA Pairs")
    print("=" * 60)
    
    chunks_path = OUTPUT_DIR / "chunks.json"
    if not chunks_path.exists():
        print("❌ Chưa có chunks.json!")
        return False
    
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    
    print(f"  📄 {len(chunks)} chunks, mỗi chunk sinh ~3-5 QA")
    print(f"  🎯 Mục tiêu: ≥300 QA pairs cho training")
    print(f"  ⏱️  Ước tính: {len(chunks) * 4 // 60} phút (rate limit 15 RPM)\n")
    
    # Sinh QA
    qa_pairs = generate_qa_batch(chunks, batch_size=5)
    
    if not qa_pairs:
        print("❌ Không sinh được QA nào. Kiểm tra API key!")
        return False
    
    # Lưu tất cả
    qa_all_path = OUTPUT_DIR / "qa_all.json"
    with open(qa_all_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    
    # Split train/test an toàn khi dataset còn ít
    # Ưu tiên đa dạng type cho test set
    import random
    random.seed(42)
    random.shuffle(qa_pairs)
    
    test_size = min(50, max(1, math.floor(len(qa_pairs) * 0.25)))
    test_set = qa_pairs[:test_size]
    train_set = qa_pairs[test_size:]
    
    # Lưu train/test
    train_path = OUTPUT_DIR / "qa_train.json"
    test_path = OUTPUT_DIR / "qa_test.json"
    
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_set, f, ensure_ascii=False, indent=2)
    with open(test_path, "w", encoding="utf-8") as f:
        json.dump(test_set, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Tổng QA: {len(qa_pairs)}")
    print(f"   Train: {len(train_set)} → {train_path}")
    print(f"   Test:  {len(test_set)} → {test_path}")
    
    # Thống kê type
    from collections import Counter
    types = Counter(qa.get("type", "unknown") for qa in qa_pairs)
    print(f"   Phân bố: {dict(types)}")
    
    return True


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   PHASE 2 – Data Processing & QA Generation            ║")
    print("╚══════════════════════════════════════════════════════════╝")
    
    # 2A: Clean
    if not process_cleaning():
        sys.exit(1)
    
    # 2B: Chunk
    if not process_chunking():
        sys.exit(1)
    
    # 2C: Vector Store (cần GPU cho embedding nhanh)
    build_vector_store()
    
    # 2D: QA Generation (cần GOOGLE_API_KEY)
    process_qa_generation()
    
    print("\n✅ Phase 2 hoàn tất! Tiếp theo: chạy phase3_finetune.py")
