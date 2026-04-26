import requests
import re
import os
import time
import json
import base64
import io
import uuid
from bs4 import BeautifulSoup
from PIL import Image

# ══════════════════════════════════════════════════════════
# 1. Khởi tạo session
# ══════════════════════════════════════════════════════════
session = requests.Session()

# ĐÂY CHÍNH LÀ CHUỖI COOKIE ĐƯỢC LẤY TỪ JSON CỦA BẠN
MY_COOKIE = ".AspNetCore.Antiforgery.025QBzPfV24=CfDJ8OaxiPJsFKBNjni2-whJ9b9IsbUZz7ixlfcOpcR39UfXhB0GUwS64xygRM33nj4fFeCUjML2CK8gKLkZyDKyD4OxjKSs4LbVuk7xhSfuFOxJjZ_o4dTosqMUO5nNSyoPvCXpl-GFxvHxXOZxkuURYfY; .AspNetCore.Cookies=CfDJ8OaxiPJsFKBNjni2-whJ9b_0tSlT3fIC6xGqKOj4khJXEapbdBvCJftX7zusj-TgVqAWwiz21iTwpZWuAVsY-Nr_8WWFDzZtsi8ogM8tVpC1BFIpmP7j3_29IGZ5N3Tr6oKUdHViObq7dvC2tZz_aIgZv_9NqnqrbxAynSDQckK3t9uZs7XwjByLE3Lz8xcsYCkNelnw8FbwYcKHXxnzmieV9LrEJ8z8ITjk43N-CmtyFXVbrwEHUHrz0xbyduF3TMtSGJ8Y4BGMtrzhVh4mmIKqH2ja3ANs6Y1cPtijnTYBDUXOeVGNuap_4mzqXWUGU4J4SL6OjCc8lHHWdk1BGNGBb4UzxrIh2S9ULxtkO7CkRitppgjZsMnzBm0HKUc6suMoi0GnWEvrbn_qF_oNBtz7Yj89L4t9cowqlG4DUAjjJ3fqStu3EYQDShglBm2357GUZLkkJl7oZ8GIXzXGf8wU2PtxUyyeDdSl1e78iMUrx7G87XGP0gmBCuYV2tCLKF_gK1Xo6azKkRpaRAOhAKrJTB8poqWCHcULNmlTqZxMyxJcXP9zF6PaqpJnb5K_VOCLdw4__6uhW-zLUTnhqOd8iCb9RipRaanVzKPODhD9mKGvwGgE-3HIdUTQofHPyv9e4cqGQb_JXbFCt-cI2aJaJNotJB7O2mfFu8aEj0Bv1Xp_im_LHkkkr1MxDKz0-8bVI0CxG2bB5ofPx2ndNlPgSs4fkS9p8gpekyGZJ86QpKjf-w5TAnvhvUtpkOyZ7yqOfPNAHJYjUkSHP01sL3JAA92uxDMaz1UH4fsIS26A_7NtVYLAhV-kKAaj1rKPwybjMUnLo447KjGenkK65-Lwm_xiZegEZDP1SXVvVQLJGtab0aKUIUOz7ZW-anH_-7O11Y_TGSNF3UjkfUrRhO1J8f_el5oVb6V63v9-mLxKfeHcu4914BoNAXKBvOZCpMvN2mQFMh7PCRustjD7vWVyFOYF3lSY819zp8zzxJ7NK9eOMQD1Dskjc_BJBdGpox9ZW6p5hyEL4lKfPupwmCzk9gsA2tG0qbZTX6GcVoibPUbLuN3v0CZyDMEr_4R0dy0HHjbCTn9pC9XhHhe3GWHd1DlxXqybJb1NkdTyHHLwGd2ZFNN43yIfSOvTVeXuWPUZ2EHOHhmdFxI6rAxMAlZeWJDs3MO6DqxOIuHYw8h04RlNOuEnjCWKw0af4mJa2mZcdd9h9NSIvVP6Wul9HIS0WmaQGLj3DHPXILOHnWTRz6-EFxRAQCCWDKfQNqr-WaV5isMv0b2-N3A_-GQfJX6fJs3NHJV87682Gxyh4fvrt0ohW_qMJ3sFxTLT78sod51NVS4go_DZF42Fq6QGNcM4fNK8mX1zfLHj9ucxjbB45hNkmG6ShxBCU1tM3hMVttFX390jUpPaGEaZS38pYVYL-nVK60eljPGsgmw6jHjHLvNxEturebKgnkUI4vrvwuokDWMJ3WfrykUf4oNrZSdTZxvq5wn1uWZiBCBzxIyD-r7QhaYLTqV-7idmWDIY6a857kGmUc-ck9yYIQfLcuHf21F6ik9vAsM30Es0vQADsFlpTwBKWx_UotkZgjfKSreXjgPoRuWXBQLLxAjMP-cyG2Ia9KAcdRfiK8N_QZg2eI3RKKSsMSzg9bV0A5TzDY7nexzeHaUG6ccmE6_0uj_linzXb6jVRuluXsRuYKi38zdvCRgJqWiGPDRrg3WkUUeh4P1U87ImtiP4VGmv0AUvgQJRYotLl9K6U_6mFg8dwfyIxegvufvrumzidaf0GbPDHhGZHOb4XGeQViELM0E; .AspNetCore.Session=CfDJ8OaxiPJsFKBNjni2%2BwhJ9b9QO7%2FFLnecB%2FKUOyVSA1kpgnRFwCS9AsFMpWSOfTUXyUIXbFzE961wpZX31tdtcD0Fjs165bySnSjB7e6QW5MOuhmzMltITh4x%2BYhmL1RgkRpw1sLIxB%2F0pNw5UqYiVlo4aLzjw%2FlaDORJQm%2BKhAeD"

# Đặt cookie qua jar thay vì header thô – tránh xung đột với requests tự xử lý
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
})

# Parse cookie string thành cookie jar
for cookie_pair in MY_COOKIE.split("; "):
    if "=" in cookie_pair:
        name, value = cookie_pair.split("=", 1)
        session.cookies.set(name, value, domain="quychehocvu.tdtu.edu.vn")

os.makedirs('data', exist_ok=True)


# ══════════════════════════════════════════════════════════
# 2. Quét danh sách ID hợp lệ từ trang Index
# ══════════════════════════════════════════════════════════
def get_valid_ids_from_page(page_number):
    index_url = f"https://quychehocvu.tdtu.edu.vn/QuyChe/Index?page={page_number}" 
    response = session.get(index_url)
    
    if "Login" in response.url or "Đăng nhập" in response.text:
        print("❌ Cảnh báo: Cookie đã hết hạn hoặc không hợp lệ. Trình duyệt bị đẩy về trang đăng nhập!")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    valid_ids = []
    
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        match = re.search(r'/QuyChe/Detail/(\d+)', href)
        if match:
            valid_ids.append(int(match.group(1)))
            
    return list(set(valid_ids)) 


# ══════════════════════════════════════════════════════════
# 3. Tải PDF theo ID
# ══════════════════════════════════════════════════════════
def download_pdf_by_id(quyche_id):
    detail_url = f"https://quychehocvu.tdtu.edu.vn/QuyChe/Detail/{quyche_id}"
    response = session.get(detail_url)
    
    # ── Trích xuất Antiforgery Token từ HTML (nếu có) ──
    antiforgery_token = None
    token_match = re.search(r'name="__RequestVerificationToken"\s+value="([^"]+)"', response.text)
    if token_match:
        antiforgery_token = token_match.group(1)
    
    # ── Trích xuất serviceUrl (base API endpoint thực tế) ──
    service_url_match = re.search(r'["\']?serviceUrl["\']?\s*[:=]\s*["\']([^"\']+)["\']', response.text)
    if service_url_match:
        service_url = service_url_match.group(1)
        # Đảm bảo URL đầy đủ
        if service_url.startswith("/"):
            service_url = "https://quychehocvu.tdtu.edu.vn" + service_url
        print(f"   -> Phát hiện serviceUrl: {service_url}")
    else:
        service_url = "https://quychehocvu.tdtu.edu.vn/api/QuyChe"
    
    path_match = re.search(r'"documentPath":\s*"([^"]+)"', response.text)
    
    if path_match:
        document_path = path_match.group(1)
        
        # Dùng Content-Type chuẩn cho JSON; KHÔNG đặt cứng trong session
        ajax_headers = {
            "Accept": "application/json, text/plain, */*",
            "X-Requested-With": "XMLHttpRequest",
            "Origin": "https://quychehocvu.tdtu.edu.vn",
            "Referer": detail_url,
        }
        
        # Thêm antiforgery token nếu tìm được
        if antiforgery_token:
            ajax_headers["RequestVerificationToken"] = antiforgery_token
        
        try:
            print(f"[ID: {quyche_id}] Đang cấp quyền khởi tạo (Load)...")
            print(f"   -> documentPath: {document_path}")
            print(f"   -> API endpoint: {service_url}/Load")
            
            # ─── BƯỚC 1: LOAD DOCUMENT ───
            load_payload = {
                "action": "Load", 
                "document": document_path,
                "isFileName": True,
                "zoomFactor": 1
            }
            
            load_res = session.post(
                f"{service_url}/Load", 
                json=load_payload, 
                headers=ajax_headers,
                timeout=30
            )
            
            print(f"   -> HTTP Status: {load_res.status_code}")
            print(f"   -> Response Content-Type: {load_res.headers.get('Content-Type', 'N/A')}")
            
            # Debug: In ra response thô (cắt ngắn)
            raw_text = load_res.text[:500]
            print(f"   -> Response (500 ký tự đầu): {raw_text}")
            
            if load_res.status_code != 200:
                print(f"[ID: {quyche_id}] ❌ Lỗi Load API: {load_res.status_code}")
                print(f"   -> Full response: {load_res.text[:1000]}")
                return
            
            # ─── Parse response linh hoạt ───
            try:
                load_data = load_res.json()
                # Nếu server trả về JSON lồng (string JSON bên trong)
                if isinstance(load_data, str):
                    load_data = json.loads(load_data)
            except Exception as json_err:
                print(f"[ID: {quyche_id}] ❌ Lỗi parse JSON: {json_err}")
                print(f"   -> Server trả về: {load_res.text[:300]}")
                return

            if not isinstance(load_data, dict):
                print(f"[ID: {quyche_id}] ❌ Dữ liệu không phải dict: type={type(load_data).__name__}, value={str(load_data)[:300]}")
                return

            # Debug: In toàn bộ keys của response
            print(f"   -> Response keys: {list(load_data.keys())}")
            
            # Server KHÔNG trả về documentId — nó được tạo phía client bởi Syncfusion
            hash_id = load_data.get("hashId", "")
            page_count = int(load_data.get("pageCount", 0))
            
            # Tạo documentId phía client giống Syncfusion PDF Viewer
            doc_id = f"Sync_PdfViewer_{uuid.uuid4()}"
            
            print(f"   -> documentId (client-generated): {doc_id}")
            print(f"   -> hashId: {hash_id}")
            print(f"   -> pageCount: {page_count}")

            if page_count == 0:
                print(f"[ID: {quyche_id}] ❌ Không khởi tạo được tài liệu (pageCount=0).")
                print(f"   -> Full load_data: {json.dumps(load_data, indent=2, ensure_ascii=False)[:1000]}")
                return
                
            print(f"   -> ✅ Chấp thuận! Tài liệu có {page_count} trang. Đang render PDF...")
            
            # ─── BƯỚC 2: RENDER ẢNH TỪNG TRANG ───
            page_images = []
            for page_idx in range(page_count):
                render_payload = {
                    "action": "RenderPdfPages",
                    "documentId": doc_id,
                    "hashId": hash_id,
                    "pageNumber": page_idx,
                    "xCoordinate": 0,
                    "yCoordinate": 0,
                    "zoomFactor": 1
                }
                
                # Retry logic cho render request
                max_retries = 3
                render_data = None
                for attempt in range(max_retries):
                    try:
                        render_res = session.post(
                            f"{service_url}/RenderPdfPages", 
                            json=render_payload, 
                            headers=ajax_headers,
                            timeout=60
                        )
                        
                        if render_res.status_code != 200:
                            print(f"   -> ⚠ Render page {page_idx} lỗi: HTTP {render_res.status_code} (lần {attempt+1})")
                            time.sleep(2)
                            continue
                        
                        render_data = render_res.json()
                        if render_data is not None:
                            break
                    except Exception as render_err:
                        print(f"   -> ⚠ Render page {page_idx} exception (lần {attempt+1}): {render_err}")
                        time.sleep(2)
                
                if render_data is None:
                    print(f"   -> ❌ Bỏ qua trang {page_idx} sau {max_retries} lần thử.")
                    continue
                
                base64_img = None
                # Kiểm tra None-safe: render_data["images"] có thể là None
                images_val = render_data.get("images")
                if images_val is not None and isinstance(images_val, list) and len(images_val) > 0:
                    base64_img = images_val[0]
                elif render_data.get("image") is not None:
                    base64_img = render_data["image"]
                elif render_data.get("documentText") is not None: 
                    base64_img = render_data["documentText"]
                     
                if base64_img and isinstance(base64_img, str):
                    if "base64," in base64_img:
                        base64_img = base64_img.split("base64,")[1]
                        
                    image_data = base64.b64decode(base64_img)
                    img = Image.open(io.BytesIO(image_data)).convert('RGB')
                    page_images.append(img)
                else:
                    print(f"   -> ⚠ Trang {page_idx}: Không tìm thấy ảnh. Keys: {list(render_data.keys())}")
                
            # ─── BƯỚC 3: ĐÓNG GÓI THÀNH FILE PDF ───
            if page_images:
                raw_name = document_path.split("/")[-1]
                safe_name = "".join([c for c in raw_name if c.isalpha() or c.isdigit() or c in ' .-']).rstrip()
                if not safe_name.endswith(".pdf"):
                    safe_name += ".pdf"
                    
                pdf_path = os.path.join('data', safe_name)
                page_images[0].save(pdf_path, save_all=True, append_images=page_images[1:])
                print(f"[ID: {quyche_id}] ✅ Hoàn tất lưu file: {safe_name}\n")
            else:
                 print(f"[ID: {quyche_id}] ❌ Không kéo được ảnh nào.\n")
                
        except Exception as e:
            import traceback
            print(f"[ID: {quyche_id}] ❌ Lỗi quá trình Render: {e}")
            traceback.print_exc()
            print()
    else:
        # Code vét nội dung text tĩnh
        soup = BeautifulSoup(response.text, 'html.parser')
        content_div = soup.find('div', class_='portlet-body') 
        if content_div:
            text_content = content_div.get_text(separator='\n', strip=True)
            if len(text_content) > 100: 
                with open(os.path.join('data', f"QuyChe_{quyche_id}.txt"), "w", encoding="utf-8") as f:
                    f.write(text_content)
                print(f"[ID: {quyche_id}] ✅ Đã lưu dạng Text thuần.\n")


# ══════════════════════════════════════════════════════════
# 4. THỰC THI
# ══════════════════════════════════════════════════════════

# ── Bước kiểm tra cookie trước ──
print("🔐 KIỂM TRA COOKIE...")
test_resp = session.get("https://quychehocvu.tdtu.edu.vn/QuyChe/Index?page=1")
if "Login" in test_resp.url or test_resp.status_code != 200:
    print("❌ COOKIE ĐÃ HẾT HẠN! Hãy cập nhật cookie mới từ trình duyệt.")
    print(f"   -> Redirect URL: {test_resp.url}")
    print(f"   -> Status: {test_resp.status_code}")
    exit(1)
else:
    print(f"✅ Cookie hợp lệ. Status: {test_resp.status_code}\n")

# ══════════════════════════════════════════════════════════
# CHẾ ĐỘ: RETRY CÁC FILE BỊ LỖI (ID 13 & 134)
# 30/32 file đã tải OK ở lần chạy trước
# ══════════════════════════════════════════════════════════
RETRY_MODE = True  # Đổi thành False để tải lại toàn bộ

if RETRY_MODE:
    failed_ids = [13, 134]
    print(f"🔄 CHẾ ĐỘ RETRY: Chỉ tải lại {len(failed_ids)} file bị lỗi: {failed_ids}\n")
    for q_id in failed_ids:
        download_pdf_by_id(q_id)
        time.sleep(1)
    print("\n✅ Hoàn tất retry!")
else:
    print("🔍 BẮT ĐẦU QUÉT ID HỢP LỆ...")
    all_valid_ids = []

    for page in range(1, 6):
        ids_in_page = get_valid_ids_from_page(page)
        all_valid_ids.extend(ids_in_page)
        time.sleep(0.5)

    all_valid_ids = sorted(list(set(all_valid_ids)))
    print(f"🎯 Tìm thấy {len(all_valid_ids)} ID hợp lệ: {all_valid_ids}\n")

    if len(all_valid_ids) > 0:
        print("🚀 BẮT ĐẦU TẢI DỮ LIỆU...")
        for q_id in all_valid_ids:
            download_pdf_by_id(q_id)
            time.sleep(1)

print("\n📊 KIỂM TRA THƯ MỤC data/:")
data_files = os.listdir('data')
print(f"   Tổng số file: {len(data_files)}")
for f in sorted(data_files):
    size = os.path.getsize(os.path.join('data', f))
    print(f"   📄 {f} ({size:,} bytes)")