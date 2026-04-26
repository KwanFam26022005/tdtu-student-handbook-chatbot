"""
Pipeline Phase 1 – Thu thập và tiền xử lý 50 bài báo VNExpress.

Workflow:
  1. Crawl 10 bài mới nhất × 5 chuyên mục (thể thao, giải trí, kinh doanh, khoa học, pháp luật).
  2. Lưu raw text vào Dataset/{chu_de}/doc_{n}.txt + labels.csv.
  3. Làm sạch (xoá HTML, URL, email, SĐT, chữ thường, gộp khoảng trắng).
  4. Tách từ tiếng Việt (underthesea → fallback pyvi).
  5. Xoá stopwords tiếng Việt.
  6. Ghi kết quả sạch vào Dataset_clean/{chu_de}/doc_{n}.txt.
"""

import csv
import os
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# ──────────────────────────────────────────────
# CẤU HÌNH
# ──────────────────────────────────────────────
CATEGORIES = {
    "the_thao":    "https://vnexpress.net/the-thao",
    "giai_tri":    "https://vnexpress.net/giai-tri",
    "kinh_doanh":  "https://vnexpress.net/kinh-doanh",
    "khoa_hoc":    "https://vnexpress.net/khoa-hoc",
    "phap_luat":   "https://vnexpress.net/phap-luat",
}
ARTICLES_PER_CAT = 10
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "Dataset"
CLEAN_DIR = BASE_DIR / "Dataset_clean"
LABELS_CSV = RAW_DIR / "labels.csv"
STOPWORDS_URL = (
    "https://raw.githubusercontent.com/stopwords-iso/stopwords-vi/master/stopwords-vi.txt"
)
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}
REQUEST_DELAY = 1  # giây


# ──────────────────────────────────────────────
# HÀM TIỆN ÍCH
# ──────────────────────────────────────────────

def fetch_html(url: str) -> BeautifulSoup | None:
    """Tải HTML từ *url*, trả về đối tượng BeautifulSoup hoặc None nếu lỗi."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        print(f"  [LỖI] Không tải được {url}: {e}")
        return None


def get_article_links(category_url: str, n: int = ARTICLES_PER_CAT) -> list[str]:
    """Lấy tối đa *n* link bài viết từ trang chuyên mục VNExpress."""
    soup = fetch_html(category_url)
    if soup is None:
        return []

    links: list[str] = []
    # VNExpress liệt kê bài trong thẻ <article> với class chứa 'title-news'
    for title_tag in soup.select("h2.title-news a, h3.title-news a"):
        href = title_tag.get("href", "")
        if href and href.startswith("https://vnexpress.net/") and href.endswith(".html"):
            if href not in links:
                links.append(href)
        if len(links) >= n:
            break
    return links[:n]


def extract_article(url: str) -> tuple[str, str] | None:
    """Trích xuất (tiêu_đề, nội_dung) từ 1 bài báo VNExpress.

    Bỏ qua quảng cáo, related articles, caption ảnh.
    Trả về None nếu không parse được.
    """
    soup = fetch_html(url)
    if soup is None:
        return None

    # Tiêu đề
    title_tag = soup.select_one("h1.title-detail") or soup.select_one("h1.title_detail")
    title = title_tag.get_text(strip=True) if title_tag else ""

    # Mô tả (sapo)
    desc_tag = soup.select_one("p.description")
    description = desc_tag.get_text(strip=True) if desc_tag else ""

    # Nội dung chính
    body = soup.select_one("article.fck_detail")
    if body is None:
        # Thử selector khác cho một số dạng bài
        body = soup.select_one("div.fck_detail")
    if body is None:
        return (title, description) if title else None

    # Loại bỏ quảng cáo, related, box video embed, caption
    for unwanted in body.select(
        ".box-related-content, .box_category, .box_brief_info, "
        ".banner-ads, .box_ads, .box-tinlienquan, "
        "figcaption, .Image, script, style, .box_comment"
    ):
        unwanted.decompose()

    paragraphs = body.find_all("p", class_=lambda c: c != "author_mail")
    content_parts = [p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)]
    content = " ".join(content_parts)

    full_text = f"{title}. {description} {content}".strip()
    return (title, full_text) if full_text else None


# ──────────────────────────────────────────────
# BƯỚC 1–2: CRAWL & LƯU RAW
# ──────────────────────────────────────────────

def crawl_and_save() -> list[dict]:
    """Cào 50 bài VNExpress, lưu raw text vào Dataset/ và labels.csv.

    Trả về list[dict] với keys: filename, true_label, category, doc_id.
    """
    records: list[dict] = []
    total_done = 0

    for cat_name, cat_url in CATEGORIES.items():
        cat_dir = RAW_DIR / cat_name
        cat_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Chuyên mục: {cat_name.upper()} – {cat_url}")
        print(f"{'='*60}")

        links = get_article_links(cat_url)
        if not links:
            print(f"  ⚠ Không lấy được link nào cho chuyên mục {cat_name}.")
            continue

        doc_id = 0
        for idx, link in enumerate(links, 1):
            print(f"  Đang cào: {link}")
            try:
                result = extract_article(link)
                if result is None:
                    print(f"    → Bỏ qua (không trích xuất được nội dung).")
                    continue

                _title, text = result
                doc_id += 1
                filename = f"doc_{doc_id}.txt"
                filepath = cat_dir / filename

                filepath.write_text(text, encoding="utf-8")
                records.append(
                    {
                        "filename": f"{cat_name}/{filename}",
                        "true_label": cat_name,
                        "category": cat_name,
                        "doc_id": doc_id,
                    }
                )
                total_done += 1
                print(f"    ✓ Lưu {filepath.relative_to(BASE_DIR)}  |  Hoàn tất: {total_done}/50 bài")
            except Exception as e:
                print(f"    ✗ Lỗi bài {link}: {e}")
            finally:
                time.sleep(REQUEST_DELAY)

    # Ghi labels.csv
    LABELS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(LABELS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "true_label"])
        writer.writeheader()
        for r in records:
            writer.writerow({"filename": r["filename"], "true_label": r["true_label"]})
    print(f"\n✓ Đã ghi labels.csv ({len(records)} dòng)")

    return records


# ──────────────────────────────────────────────
# BƯỚC 3: LÀM SẠCH VĂN BẢN
# ──────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Xoá HTML thừa, URL, email, SĐT; chữ thường; gộp khoảng trắng."""
    # Xoá thẻ HTML còn sót
    text = re.sub(r"<[^>]+>", " ", text)
    # Xoá URL
    text = re.sub(r"https?://\S+", " ", text)
    # Xoá email
    text = re.sub(r"\S+@\S+\.\S+", " ", text)
    # Xoá SĐT (VN: 09x, 03x, … hoặc +84…)
    text = re.sub(r"(\+84|0)\d{8,10}", " ", text)
    # Chữ thường
    text = text.lower()
    # Gộp khoảng trắng & strip
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ──────────────────────────────────────────────
# BƯỚC 4: TÁCH TỪ TIẾNG VIỆT
# ──────────────────────────────────────────────

def tokenize_vi(text: str) -> str:
    """Tách từ tiếng Việt bằng underthesea; fallback sang pyvi nếu lỗi.

    Các từ ghép được nối bằng dấu gạch dưới (ví dụ: học_sinh).
    """
    # Thử underthesea trước
    try:
        from underthesea import word_tokenize  # type: ignore
        tokens = word_tokenize(text, format="list")
        return " ".join(t.replace(" ", "_") for t in tokens)
    except Exception:
        pass

    # Fallback sang pyvi
    try:
        from pyvi import ViTokenizer  # type: ignore
        return ViTokenizer.tokenize(text)
    except Exception as e:
        print(f"    ⚠ Không tách từ được (underthesea & pyvi đều lỗi): {e}")
        return text  # trả nguyên bản


# ──────────────────────────────────────────────
# BƯỚC 5: XOÁ STOPWORDS
# ──────────────────────────────────────────────

def load_stopwords() -> set[str]:
    """Tải danh sách stopwords tiếng Việt từ GitHub."""
    try:
        resp = requests.get(STOPWORDS_URL, timeout=10)
        resp.raise_for_status()
        words = set(resp.text.strip().splitlines())
        print(f"✓ Tải {len(words)} stopwords tiếng Việt.")
        return words
    except Exception as e:
        print(f"⚠ Không tải được stopwords ({e}). Dùng danh sách rỗng.")
        return set()


def remove_stopwords(text: str, stopwords: set[str]) -> str:
    """Loại bỏ stopwords khỏi văn bản đã tách từ."""
    tokens = text.split()
    return " ".join(t for t in tokens if t not in stopwords)


# ──────────────────────────────────────────────
# BƯỚC 6: XỬ LÝ & GHI FILE SẠCH
# ──────────────────────────────────────────────

def process_and_save_clean(records: list[dict], stopwords: set[str]) -> None:
    """Đọc raw → clean → tokenize → remove stopwords → ghi vào Dataset_clean/."""
    print(f"\n{'='*60}")
    print("TIỀN XỬ LÝ VĂN BẢN")
    print(f"{'='*60}")

    for rec in records:
        cat = rec["category"]
        doc_id = rec["doc_id"]
        raw_path = RAW_DIR / cat / f"doc_{doc_id}.txt"
        clean_path = CLEAN_DIR / cat / f"doc_{doc_id}.txt"
        clean_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            raw_text = raw_path.read_text(encoding="utf-8")

            # Bước 3: Làm sạch
            text = clean_text(raw_text)

            # Bước 4: Tách từ
            text = tokenize_vi(text)

            # Bước 5: Xoá stopwords
            text = remove_stopwords(text, stopwords)

            clean_path.write_text(text, encoding="utf-8")
            print(f"  ✓ {clean_path.relative_to(BASE_DIR)}")

        except Exception as e:
            print(f"  ✗ Lỗi xử lý {raw_path.relative_to(BASE_DIR)}: {e}")


# ──────────────────────────────────────────────
# BÁO CÁO TỔNG KẾT
# ──────────────────────────────────────────────

def print_summary(records: list[dict]) -> None:
    """In tổng số bài và phân bố theo chủ đề."""
    print(f"\n{'='*60}")
    print("TỔNG KẾT")
    print(f"{'='*60}")
    print(f"Tổng số bài thu thập: {len(records)}/50")
    print(f"\nPhân bố theo chủ đề:")

    from collections import Counter
    dist = Counter(r["true_label"] for r in records)
    for cat in CATEGORIES:
        count = dist.get(cat, 0)
        bar = "█" * count + "░" * (ARTICLES_PER_CAT - count)
        print(f"  {cat:<15} {bar}  {count}/{ARTICLES_PER_CAT}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main() -> None:
    """Chạy toàn bộ pipeline Phase 1."""
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   PIPELINE PHASE 1 – Thu thập & Tiền xử lý VNExpress   ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Bước 1-2: Crawl & lưu raw
    records = crawl_and_save()
    if not records:
        print("Không thu thập được bài nào. Kết thúc.")
        return

    # Bước 5 (tải trước): Stopwords
    stopwords = load_stopwords()

    # Bước 3-4-5-6: Tiền xử lý & ghi file sạch
    process_and_save_clean(records, stopwords)

    # Báo cáo
    print_summary(records)
    print("\n✅ Pipeline Phase 1 hoàn tất.")


if __name__ == "__main__":
    main()
