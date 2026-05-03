"""
Phase 1 – OCR Pipeline (v2): Trích xuất text từ PDF ảnh.

Cải tiến so với v1 (EasyOCR):
  - OpenAI GPT-4o-mini Vision – hiểu cấu trúc bảng, tiếng Việt có dấu chính xác
  - Giữ nguyên layout bảng dưới dạng Markdown table
  - Nhận diện ký tự đặc biệt (≥, ≤) chính xác
  - Lọc nhiễu con dấu/chữ ký → tag [Con dấu], [Chữ ký]
  - Fallback Surya OCR khi không có API key
  - Progress tracking & resume

Pipeline:
  1. Đọc từng trang PDF bằng PyMuPDF (fitz) → ảnh PNG 300 DPI
  2. OCR bằng OpenAI Vision (primary) hoặc Surya (fallback)
  3. Post-processing: sửa lỗi OCR, chuẩn hóa Unicode
  4. Lưu text vào raw_text/<tên_file>.txt

Cài đặt:
  pip install pymupdf Pillow openai

  # (Tùy chọn) Fallback Surya OCR:
  pip install surya-ocr

Chạy:
  # Set API key (lấy từ https://platform.openai.com/)
  set OPENAI_API_KEY=your_key_here    # Windows
  export OPENAI_API_KEY=your_key_here # Linux/Mac/Colab

  python phase1_ocr.py
"""

import os
import sys
import re
import io
import json
import time
import base64
import unicodedata
from pathlib import Path

# PyMuPDF
import fitz

# PIL
from PIL import Image

# ══════════════════════════════════════════════════════════
# CẤU HÌNH
# ══════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).resolve().parent if "__file__" in dir() else Path(".")
PDF_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "raw_text"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Progress tracking
PROGRESS_FILE = OUTPUT_DIR / "ocr_progress.json"

# DPI cho render ảnh từ PDF
RENDER_DPI = 300

# API key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# OpenAI model
OPENAI_MODEL = "gpt-4o-mini"

# Rate limiting
API_DELAY = 1.0  # giây giữa các request
MAX_RETRIES = 3


# ══════════════════════════════════════════════════════════
# STRUCTURED PROMPT CHO OPENAI VISION
# ══════════════════════════════════════════════════════════
OCR_SYSTEM_PROMPT = "Bạn là chuyên gia OCR văn bản hành chính tiếng Việt. Chỉ trả về nội dung text thuần."

OCR_USER_PROMPT = """Trích xuất TOÀN BỘ nội dung text từ ảnh này.

QUY TẮC BẮT BUỘC:
1. Giữ CHÍNH XÁC tất cả dấu tiếng Việt (sắc, huyền, hỏi, ngã, nặng, mũ, móc).
2. Nếu trang có bảng, PHẢI xuất dưới dạng Markdown table (| col1 | col2 |).
3. Giữ nguyên ký hiệu toán học: ≥, ≤, >, <.
4. Con dấu mộc → ghi [Con dấu]. Chữ ký tay → ghi [Chữ ký]. KHÔNG cố đọc text bên trong.
5. Giữ nguyên cấu trúc: Tiêu đề IN HOA, Điều/Khoản/Mục, header cơ quan, số hiệu văn bản.
6. CHỈ trả về plain text + Markdown table. KHÔNG thêm giải thích, KHÔNG bọc code block."""


# ══════════════════════════════════════════════════════════
# OPENAI VISION OCR ENGINE
# ══════════════════════════════════════════════════════════
_openai_client = None


def init_openai():
    """Khởi tạo OpenAI client."""
    global _openai_client
    if not OPENAI_API_KEY:
        return False

    try:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print("[OK] OpenAI Vision sẵn sàng!")
        return True
    except ImportError:
        print("[WARNING] Chưa cài openai. Chạy: pip install openai")
        return False
    except Exception as e:
        print(f"[ERROR] Lỗi khởi tạo OpenAI: {e}")
        return False


def ocr_page_openai(img_bytes: bytes, page_info: str = "") -> str:
    """
    OCR một trang bằng OpenAI GPT-4o-mini Vision API.

    Args:
        img_bytes: Ảnh PNG dạng bytes
        page_info: Thông tin trang (vd: "Trang 1/3") để debug

    Returns:
        Text trích xuất được
    """
    b64_image = base64.b64encode(img_bytes).decode("utf-8")

    for attempt in range(MAX_RETRIES):
        try:
            response = _openai_client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": OCR_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": OCR_USER_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{b64_image}",
                                    "detail": "high",
                                },
                            },
                        ],
                    },
                ],
                temperature=0.1,
                max_tokens=8192,
            )

            text = response.choices[0].message.content.strip() if response.choices else ""

            if text:
                return text
            else:
                print(f"      [WARNING] {page_info}: Response rỗng (lần {attempt + 1})")

        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                wait = 30 * (attempt + 1)
                m = re.search(r'retry.*?(\d+\.?\d*)\s*s', err, re.IGNORECASE)
                if m:
                    wait = float(m.group(1)) + 5
                print(f"      [WAIT] {page_info}: Rate limit. Chờ {wait:.0f}s... (lần {attempt + 1})")
                time.sleep(wait)
            else:
                print(f"      [ERROR] {page_info}: {err[:120]} (lần {attempt + 1})")
                time.sleep(5)

    return ""


# ══════════════════════════════════════════════════════════
# SURYA OCR ENGINE (FALLBACK)
# ══════════════════════════════════════════════════════════
_surya_rec_predictor = None
_surya_det_predictor = None
_surya_available = None


def check_surya_available() -> bool:
    """Kiểm tra Surya OCR có cài đặt không."""
    global _surya_available
    if _surya_available is not None:
        return _surya_available
    try:
        import surya
        _surya_available = True
        return True
    except ImportError:
        _surya_available = False
        return False


def init_surya():
    """Khởi tạo Surya OCR predictors."""
    global _surya_rec_predictor, _surya_det_predictor

    if not check_surya_available():
        return False

    try:
        from surya.recognition import RecognitionPredictor
        from surya.detection import DetectionPredictor

        print("[INIT] Đang tải Surya OCR models (lần đầu sẽ tải ~2GB)...")
        _surya_rec_predictor = RecognitionPredictor()
        _surya_det_predictor = DetectionPredictor()
        print("[OK] Surya OCR sẵn sàng!")
        return True
    except Exception as e:
        print(f"[ERROR] Lỗi khởi tạo Surya: {e}")
        return False


def ocr_page_surya(pil_image: Image.Image, page_info: str = "") -> str:
    """
    OCR một trang bằng Surya OCR.

    Args:
        pil_image: Ảnh PIL.Image (RGB)
        page_info: Thông tin trang để debug

    Returns:
        Text trích xuất được
    """
    try:
        from surya.recognition import run_recognition
        from surya.detection import run_detection

        # Detect text regions
        det_results = run_detection(
            images=[pil_image],
            predictor=_surya_det_predictor,
            languages=[["vi"]],
        )

        # Recognize text
        rec_results = run_recognition(
            images=[pil_image],
            predictor=_surya_rec_predictor,
            det_results=det_results,
            languages=[["vi"]],
        )

        if rec_results and rec_results[0]:
            lines = []
            for text_line in rec_results[0].text_lines:
                lines.append(text_line.text)
            return "\n".join(lines)

    except Exception as e:
        print(f"      [ERROR] {page_info} Surya lỗi: {e}")

    return ""


# ══════════════════════════════════════════════════════════
# EASYOCR ENGINE (LEGACY FALLBACK)
# ══════════════════════════════════════════════════════════
_easyocr_reader = None


def init_easyocr():
    """Khởi tạo EasyOCR (fallback cuối cùng)."""
    global _easyocr_reader
    try:
        import easyocr
        print("Đang khởi tạo EasyOCR (fallback cuối cùng)...")
        _easyocr_reader = easyocr.Reader(['vi'], gpu=True, verbose=False)
        print("EasyOCR sẵn sàng (fallback)!")
        return True
    except ImportError:
        print("EasyOCR không có. Chạy: pip install easyocr")
        return False


def ocr_page_easyocr(img_bytes: bytes, page_info: str = "") -> str:
    """OCR bằng EasyOCR (legacy fallback)."""
    try:
        results = _easyocr_reader.readtext(img_bytes, detail=0, paragraph=True)
        return "\n".join(results)
    except Exception as e:
        print(f"      [ERROR] {page_info} EasyOCR lỗi: {e}")
        return ""


# ══════════════════════════════════════════════════════════
# POST-PROCESSING
# ══════════════════════════════════════════════════════════

# Từ điển sửa lỗi OCR phổ biến tiếng Việt hành chính
OCR_FIX_MAP = {
    # Chữ Đ bị nhận sai
    'Ðiều': 'Điều', 'Ðại': 'Đại', 'Ðào': 'Đào',
    'Ðánh': 'Đánh', 'Ðể': 'Để', 'Ðược': 'Được',
    'Ðối': 'Đối', 'Ðơn': 'Đơn', 'Ðình': 'Đình',
    'Ðiểm': 'Điểm', 'Ðăng': 'Đăng', 'Ðồng': 'Đồng',
    # Số La Mã / ký tự nhầm
    'lI': 'II', 'Il': 'II', 'Ill': 'III',
}


def fix_ocr_errors(text: str) -> str:
    """Sửa lỗi OCR phổ biến."""
    # Chuẩn hóa Unicode NFC
    text = unicodedata.normalize("NFC", text)

    # Áp dụng từ điển sửa lỗi
    for wrong, right in OCR_FIX_MAP.items():
        text = text.replace(wrong, right)

    # Xóa ký tự điều khiển (giữ \n, \t, \r)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # Gộp khoảng trắng thừa (giữ \n)
    text = re.sub(r'[^\S\n]+', ' ', text)

    # Xóa dòng trống liên tiếp (giữ tối đa 2)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Strip mỗi dòng
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    return text.strip()


# ══════════════════════════════════════════════════════════
# PROGRESS TRACKING
# ══════════════════════════════════════════════════════════

def load_progress() -> dict:
    """Load tiến trình OCR đã thực hiện."""
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"completed_files": [], "stats": {}}


def save_progress(progress: dict):
    """Lưu tiến trình OCR."""
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════
# HÀM OCR CHÍNH
# ══════════════════════════════════════════════════════════

def render_page_to_image(page, dpi: int = RENDER_DPI) -> tuple:
    """
    Render một trang PDF thành ảnh.

    Returns:
        (pil_image, png_bytes)
    """
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    img_bytes = pix.tobytes("png")

    # Cũng tạo PIL Image cho Surya
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    return pil_img, img_bytes


def ocr_pdf(pdf_path: Path, engine: str = "openai") -> str:
    """
    OCR toàn bộ một file PDF.

    Args:
        pdf_path: Đường dẫn file PDF
        engine: "openai", "surya", hoặc "easyocr"

    Returns:
        Chuỗi text đã OCR và post-process
    """
    doc = fitz.open(str(pdf_path))
    all_text = []
    total_pages = len(doc)

    for page_idx in range(total_pages):
        page = doc[page_idx]
        page_info = f"Trang {page_idx + 1}/{total_pages}"

        # Render ảnh
        pil_img, png_bytes = render_page_to_image(page)

        # OCR theo engine
        page_text = ""

        if engine == "openai" and _openai_client:
            page_text = ocr_page_openai(png_bytes, page_info)
            time.sleep(API_DELAY)

        if not page_text and engine in ("surya", "openai") and _surya_rec_predictor:
            if engine == "openai":
                print(f"      [RETRY] {page_info}: OpenAI thất bại, chuyển sang Surya...")
            page_text = ocr_page_surya(pil_img, page_info)

        if not page_text and _easyocr_reader:
            if engine != "easyocr":
                print(f"      [RETRY] {page_info}: Chuyển sang EasyOCR (fallback cuối)...")
            page_text = ocr_page_easyocr(png_bytes, page_info)

        if not page_text:
            print(f"      [ERROR] {page_info}: Không OCR được!")
            page_text = f"[Không trích xuất được text trang {page_idx + 1}]"

        # Post-processing
        page_text = fix_ocr_errors(page_text)

        all_text.append(f"--- Trang {page_idx + 1}/{total_pages} ---")
        all_text.append(page_text)

        # Progress indicator
        text_preview = page_text[:80].replace('\n', ' ')
        print(f"      {page_info} [OK] ({len(page_text):,} ký tự) | {text_preview}...")

    doc.close()
    return "\n\n".join(all_text)


# ══════════════════════════════════════════════════════════
# MAIN PROCESSING
# ══════════════════════════════════════════════════════════

def select_engine() -> str:
    """Chọn OCR engine tốt nhất có sẵn."""
    # 1. OpenAI Vision (best)
    if OPENAI_API_KEY and init_openai():
        print(f"[PRIMARY] Engine chính: OpenAI {OPENAI_MODEL} Vision")
        # Cũng init Surya làm fallback nếu có
        if check_surya_available():
            init_surya()
            print("   + Surya OCR (fallback)")
        return "openai"

    # 2. Surya OCR (good)
    if check_surya_available() and init_surya():
        print("[SECONDARY] Engine: Surya OCR (không có OPENAI_API_KEY)")
        return "surya"

    # 3. EasyOCR (legacy)
    if init_easyocr():
        print("[LEGACY] Engine: EasyOCR (legacy fallback)")
        print("   [WARNING] Chất lượng sẽ thấp hơn. Khuyến nghị đặt OPENAI_API_KEY.")
        return "easyocr"

    print("[ERROR] Không có OCR engine nào! Cài ít nhất một trong:")
    print("   pip install openai          # Cần OPENAI_API_KEY")
    print("   pip install surya-ocr       # Offline, cần GPU")
    print("   pip install easyocr         # Chất lượng thấp nhất")
    sys.exit(1)


def process_all_pdfs():
    """OCR tất cả PDF trong thư mục data/ → lưu vào raw_text/"""
    # Chọn engine
    engine = select_engine()

    # Load progress
    progress = load_progress()
    completed = set(progress.get("completed_files", []))

    # Tìm PDF
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"[ERROR] Không tìm thấy file PDF nào trong {PDF_DIR}")
        return

    print(f"\n[INFO] Tìm thấy {len(pdf_files)} file PDF")
    print(f"[DIR] Output: {OUTPUT_DIR}")
    if completed:
        print(f"[SKIP] Đã OCR trước đó: {len(completed)} file")
    print("=" * 60)

    total_chars = 0
    success_count = 0
    skip_count = 0
    error_count = 0

    for idx, pdf_path in enumerate(pdf_files, 1):
        # Tên file output
        txt_name = pdf_path.stem + ".txt"
        txt_path = OUTPUT_DIR / txt_name

        # Skip nếu đã OCR rồi (check cả progress file và file tồn tại)
        if pdf_path.name in completed and txt_path.exists() and txt_path.stat().st_size > 100:
            file_size = txt_path.stat().st_size
            print(f"[{idx}/{len(pdf_files)}] [SKIP] Đã có: {txt_name} ({file_size:,} bytes)")
            skip_count += 1
            total_chars += file_size
            success_count += 1
            continue

        print(f"\n[{idx}/{len(pdf_files)}] [PROCESS] OCR: {pdf_path.name}")

        try:
            start_time = time.time()
            text = ocr_pdf(pdf_path, engine=engine)
            elapsed = time.time() - start_time

            # Lưu file
            txt_path.write_text(text, encoding="utf-8")

            char_count = len(text)
            total_chars += char_count
            success_count += 1

            # Update progress
            completed.add(pdf_path.name)
            progress["completed_files"] = list(completed)
            progress["stats"][pdf_path.name] = {
                "chars": char_count,
                "time_s": round(elapsed, 1),
                "engine": engine,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            save_progress(progress)

            print(f"   [OK] Lưu: {txt_name} ({char_count:,} ký tự, {elapsed:.1f}s, engine={engine})")

        except Exception as e:
            error_count += 1
            print(f"   [ERROR] Lỗi: {e}")
            import traceback
            traceback.print_exc()

    # ── Tổng kết ──
    print("\n" + "=" * 60)
    print("[STATS] TỔNG KẾT OCR")
    print("=" * 60)
    print(f"   Engine:      {engine}")
    print(f"   Thành công:  {success_count}/{len(pdf_files)}")
    print(f"   Đã bỏ qua:  {skip_count} (đã OCR trước đó)")
    print(f"   Lỗi:        {error_count}")
    print(f"   Tổng ký tự:  {total_chars:,}")
    print(f"   Ước tính:    ~{total_chars // 5:,} từ")
    print(f"   Output:      {OUTPUT_DIR}")


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  PHASE 1 – OCR Pipeline v2 (OpenAI Vision + Fallback)  ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    if not OPENAI_API_KEY:
        print("[WARNING] OPENAI_API_KEY không được đặt!")
        print("   Set biến môi trường: set OPENAI_API_KEY=your_key")
        print("   Hoặc lấy key tại: https://platform.openai.com/")
        print("   Sẽ thử dùng Surya/EasyOCR nếu có.\n")

    process_all_pdfs()
