# BÁO CÁO PHÂN TÍCH FINE-TUNING (PHASE 3)

## 1. Thông số tổng quan
* **Epochs hoàn thành:** 3.00
* **Checkpoint tốt nhất:** `/content/tdtu-student-handbook-chatbot/outputs/finetune/checkpoint-550`
* **Step đạt cấu hình tối ưu:** 550
* **Validation Loss thấp nhất:** 0.3835

## 2. Đánh giá chất lượng mô hình
✅ **Đánh giá:** Quá trình hội tụ tốt, không có dấu hiệu Overfitting nghiêm trọng. Validation loss duy trì ổn định sau khi chạm đáy.

## 3. Cấu trúc thư mục Output
Các tệp kết quả phân tích được lưu tại:
* Ảnh biểu đồ Loss & LR: `loss_lr_curves.png`
* Dữ liệu logs chi tiết: `training_logs.csv`

> **Lưu ý cho Phase 4:** Khi load mô hình để phục vụ hệ thống RAG quy chế sinh viên, hãy trỏ đường dẫn adapter vào checkpoint tối ưu nhất được liệt kê ở phần 1 để có độ chính xác cao nhất.
