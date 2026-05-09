import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# ---------------------------------------------------------
# 1. CẤU HÌNH ĐƯỜNG DẪN & TỪ KHÓA
# ---------------------------------------------------------
INPUT_DIR = "raw_text"
OUTPUT_DIR = "eda_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Các từ khóa/thực thể quan trọng thường xuất hiện trong quy chế
TARGET_KEYWORDS = ["điều", "khoản", "sinh viên", "tốt nghiệp", "chứng chỉ", 
                   "ielts", "toeic", "mos", "gpa", "kỷ luật", "học bổng", "ưu tú"]

# ---------------------------------------------------------
# 2. HÀM PHÂN TÍCH TỪNG FILE TEXT
# ---------------------------------------------------------
def analyze_text_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
        
    filename = os.path.basename(filepath)
    
    # 2.1 Đếm từ và ký tự
    words = text.split()
    num_words = len(words)
    num_chars = len(text)
    
    # 2.2 Phân tích cấu trúc Markdown
    # Đếm số dòng bảng (dòng có chứa ký tự | đặc trưng của Markdown table)
    num_table_rows = len(re.findall(r'\|.*\|', text))
    # Đếm thẻ Heading (#, ##, ###)
    num_headings = len(re.findall(r'^#{1,6}\s', text, re.MULTILINE))
    # Đếm danh sách (bắt đầu bằng - hoặc 1., 2.)
    num_list_items = len(re.findall(r'^[\-\*]\s|^\d+\.\s', text, re.MULTILINE))
    
    # 2.3 Phân tích thực thể/Từ khóa quan trọng
    text_lower = text.lower()
    keyword_counts = {kw: text_lower.count(kw) for kw in TARGET_KEYWORDS}
    
    # 2.4 Chất lượng / Nhiễu
    blank_lines = len(re.findall(r'^\s*$', text, re.MULTILINE))
    
    return {
        "Filename": filename,
        "Words": num_words,
        "Characters": num_chars,
        "Headings": num_headings,
        "Table_Rows": num_table_rows,
        "List_Items": num_list_items,
        "Blank_Lines": blank_lines,
        **keyword_counts # Mở rộng dictionary từ khóa vào kết quả
    }

# ---------------------------------------------------------
# 3. CHẠY PIPELINE VÀ TỔNG HỢP KẾT QUẢ
# ---------------------------------------------------------
print(f"Bắt đầu phân tích các file trong thư mục: {INPUT_DIR}/...")
file_paths = glob.glob(os.path.join(INPUT_DIR, "*.txt"))

if not file_paths:
    print("Không tìm thấy file .txt nào. Vui lòng kiểm tra lại đường dẫn!")
    exit()

results = [analyze_text_file(fp) for fp in file_paths]
df = pd.DataFrame(results)

# Lưu kết quả thô ra CSV
csv_path = os.path.join(OUTPUT_DIR, "phase1_eda_summary.csv")
df.to_csv(csv_path, index=False, encoding='utf-8')
print(f"Đã lưu bảng thống kê chi tiết tại: {csv_path}")

# ---------------------------------------------------------
# 4. IN BÁO CÁO NHANH RA CONSOLE (Cho Slide)
# ---------------------------------------------------------
print("\n" + "="*40)
print("BÁO CÁO TỔNG QUAN DỮ LIỆU PHASE 1")
print("="*40)
print(f"Tổng số file PDF đã OCR: {len(df)}")
print(f"Tổng số từ (Total Words): {df['Words'].sum():,}")
print(f"Độ dài trung bình: {df['Words'].mean():.0f} từ/văn bản")
print(f"Độ dài lớn nhất: {df['Words'].max():,} từ ({df.loc[df['Words'].idxmax(), 'Filename']})")
print(f"Tổng số thẻ cấu trúc (Headings): {df['Headings'].sum():,}")
print(f"Tổng số dòng bảng (Table Rows): {df['Table_Rows'].sum():,}")
print("="*40)

# ---------------------------------------------------------
# 5. VẼ BIỂU ĐỒ (VISUALIZATION)
# ---------------------------------------------------------
sns.set_theme(style="whitegrid", font="Be Vietnam Pro" if os.path.exists("Be Vietnam Pro") else "sans-serif")

# Biểu đồ 1: Phân bố độ dài văn bản (Giúp giải thích lý do cần Chunking)
plt.figure(figsize=(10, 6))
sns.histplot(df['Words'], bins=10, kde=True, color='#1565C0')
plt.title('Phân bố độ dài văn bản (Số từ)', fontsize=14, fontweight='bold')
plt.xlabel('Số lượng từ')
plt.ylabel('Số lượng file')
plt.savefig(os.path.join(OUTPUT_DIR, 'word_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()

# Biểu đồ 2: Tần suất các từ khóa cốt lõi
keyword_sums = df[TARGET_KEYWORDS].sum().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x=keyword_sums.values, y=keyword_sums.index, palette="viridis")
plt.title('Tần suất xuất hiện các Keyword cốt lõi trong Quy chế', fontsize=14, fontweight='bold')
plt.xlabel('Số lần xuất hiện')
plt.ylabel('Từ khóa')
plt.savefig(os.path.join(OUTPUT_DIR, 'keyword_frequencies.png'), dpi=300, bbox_inches='tight')
plt.close()

# Biểu đồ 3: Khối lượng cấu trúc trích xuất được (Chứng minh chất lượng OCR)
structure_sums = [df['Headings'].sum(), df['Table_Rows'].sum(), df['List_Items'].sum()]
structure_labels = ['Headings (Chương/Điều)', 'Table Rows (Dòng bảng)', 'List Items (Gạch đầu dòng)']
plt.figure(figsize=(8, 5))
sns.barplot(x=structure_sums, y=structure_labels, palette=['#1565C0', '#E8293A', '#f39c12'])
plt.title('Mức độ bảo toàn Cấu trúc Markdown sau OCR', fontsize=14, fontweight='bold')
plt.xlabel('Tổng số lượng nhận diện được')
for i, v in enumerate(structure_sums):
    plt.text(v + sum(structure_sums)*0.01, i, str(v), color='black', va='center')
plt.savefig(os.path.join(OUTPUT_DIR, 'structure_counts.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"Đã lưu các biểu đồ báo cáo tại thư mục: {OUTPUT_DIR}/")