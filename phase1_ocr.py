"""
Phase 1 – OCR Pipeline: Trích xuất text từ PDF ảnh.

Workflow:
  1. Đọc từng trang PDF bằng PyMuPDF (fitz) → ảnh PNG
  2. OCR từng ảnh bằng EasyOCR (hỗ trợ tiếng Việt tốt)
  3. Lưu text mỗi file vào raw_text/<tên_file>.txt

Cài đặt:
  pip install pymupdf easyocr Pillow
  
Chạy:
  python phase1_ocr.py
  
Trên Colab:
  !pip install pymupdf easyocr
  %run phase1_ocr.py
"""

import os
import sys
import time
import fitz  # PyMuPDF
import io
from pathlib import Path

# ══════════════════════════════════════════════════════════
# CẤU HÌNH
# ══════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).resolve().parent if "__file__" in dir() else Path(".")
PDF_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "raw_text"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# DPI cho render ảnh từ PDF (cao hơn = chính xác hơn nhưng chậm hơn)
RENDER_DPI = 300

# ══════════════════════════════════════════════════════════
# KHỞI TẠO OCR ENGINE
# ══════════════════════════════════════════════════════════
print("🔧 Đang khởi tạo EasyOCR (lần đầu sẽ tải model ~100MB)...")
import easyocr
reader = easyocr.Reader(
    ['vi'],          # Tiếng Việt
    gpu=True,        # Dùng GPU nếu có (CUDA), tự fallback CPU
    verbose=False
)
print("✅ EasyOCR sẵn sàng!\n")


# ══════════════════════════════════════════════════════════
# HÀM OCR
# ══════════════════════════════════════════════════════════
def ocr_pdf(pdf_path: Path) -> str:
    """
    OCR toàn bộ một file PDF, trả về chuỗi text.
    
    Pipeline: PDF → PyMuPDF render → PNG bytes → EasyOCR → text
    """
    doc = fitz.open(str(pdf_path))
    all_text = []
    total_pages = len(doc)
    
    for page_idx in range(total_pages):
        page = doc[page_idx]
        
        # Render page thành ảnh (pixmap) ở DPI cao
        mat = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)  # 72 DPI là mặc định
        pix = page.get_pixmap(matrix=mat)
        
        # Chuyển pixmap thành bytes PNG
        img_bytes = pix.tobytes("png")
        
        # OCR bằng EasyOCR
        results = reader.readtext(img_bytes, detail=0, paragraph=True)
        page_text = "\n".join(results)
        
        all_text.append(f"--- Trang {page_idx + 1}/{total_pages} ---")
        all_text.append(page_text)
        
        # Progress
        print(f"      Trang {page_idx + 1}/{total_pages} ✓ ({len(results)} đoạn)")
    
    doc.close()
    return "\n\n".join(all_text)


def process_all_pdfs():
    """OCR tất cả PDF trong thư mục data/ → lưu vào raw_text/"""
    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    
    if not pdf_files:
        print(f"❌ Không tìm thấy file PDF nào trong {PDF_DIR}")
        return
    
    print(f"📄 Tìm thấy {len(pdf_files)} file PDF")
    print(f"📁 Output: {OUTPUT_DIR}")
    print("=" * 60)
    
    total_chars = 0
    success_count = 0
    skip_count = 0
    
    for idx, pdf_path in enumerate(pdf_files, 1):
        # Tên file output
        txt_name = pdf_path.stem + ".txt"
        txt_path = OUTPUT_DIR / txt_name
        
        # Skip nếu đã OCR rồi
        if txt_path.exists() and txt_path.stat().st_size > 100:
            print(f"[{idx}/{len(pdf_files)}] ⏭️  Đã có: {txt_name}")
            skip_count += 1
            
            # Đếm chars đã có
            total_chars += txt_path.stat().st_size
            success_count += 1
            continue
        
        print(f"\n[{idx}/{len(pdf_files)}] 📖 OCR: {pdf_path.name}")
        
        try:
            start_time = time.time()
            text = ocr_pdf(pdf_path)
            elapsed = time.time() - start_time
            
            # Lưu file
            txt_path.write_text(text, encoding="utf-8")
            
            char_count = len(text)
            total_chars += char_count
            success_count += 1
            
            print(f"   ✅ Lưu: {txt_name} ({char_count:,} ký tự, {elapsed:.1f}s)")
            
        except Exception as e:
            print(f"   ❌ Lỗi: {e}")
    
    # ── Tổng kết ──
    print("\n" + "=" * 60)
    print("📊 TỔNG KẾT OCR")
    print("=" * 60)
    print(f"   Thành công:  {success_count}/{len(pdf_files)}")
    print(f"   Đã bỏ qua:  {skip_count} (đã OCR trước đó)")
    print(f"   Tổng ký tự:  {total_chars:,}")
    print(f"   Ước tính:    ~{total_chars // 5:,} từ")
    print(f"   Output:      {OUTPUT_DIR}")


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   PHASE 1 – OCR Pipeline (PDF ảnh → Text tiếng Việt)   ║")
    print("╚══════════════════════════════════════════════════════════╝\n")
    
    process_all_pdfs()
    print("\n✅ Phase 1 hoàn tất! Tiếp theo: chạy phase2_process.py")
