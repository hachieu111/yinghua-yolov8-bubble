# yinghua-yolov8-bubble

Huấn luyện **YOLOv8-seg** phát hiện và tách viền **bong bóng thoại** trong trang truyện tranh.
Model dùng cho bước đầu tiên của pipeline dịch truyện (phát hiện bong bóng → OCR → dịch → xóa chữ → đắp chữ)
trong ứng dụng học tiếng Trung qua truyện tranh *YingHua*.

## Kết quả chính (tập test, 211 ảnh, 1 350 bong bóng)

| Model | Box P | Box R | Box mAP50 | Box mAP50-95 | Mask mAP50 | Mask mAP50-95 |
|---|---|---|---|---|---|---|
| Cũ | 0.939 | 0.873 | 0.878 | 0.787 | 0.893 | 0.794 |
| **Mới** | **0.946** | **0.959** | **0.973** | **0.910** | **0.971** | **0.884** |

## Cấu trúc

| Thư mục | Nội dung | Phụ trách |
|---|---|---|
| `data/` | Hướng dẫn tải và chuẩn bị dữ liệu (ảnh không đẩy lên Git) | Người 2 |
| `scripts/` | `prepare_dataset`, `check_labels`, `eda`, `gen_seg_labels_sam` | Người 2 |
| `configs/`, `scripts/train.py`, `notebooks/` | Cấu hình và huấn luyện (Colab/Kaggle) | Người 1 |
| `scripts/evaluate.py`, `scripts/predict.py`, `weights/` | Đánh giá, dự đoán, tải model | Người 3 |
| `results/` | Biểu đồ và bảng số liệu | cả nhóm |
| `docs/` | Các chương báo cáo | cả nhóm |

## Chạy nhanh

```bash
pip install -r requirements.txt

# 1. Chuẩn bị dữ liệu (đã tải bubbles_dataset.zip, xem data/README.md)
python scripts/prepare_dataset.py --src path/to/bubbles_dataset.zip
python scripts/check_labels.py
python scripts/gen_seg_labels_sam.py          # sinh viền bằng SAM, nên chạy trên GPU

# 2. Huấn luyện (khuyên dùng notebooks/train_colab_kaggle.ipynb trên Colab/Kaggle)
python scripts/train.py --weights yolov8m-seg.pt

# 3. Đánh giá và dự đoán
python scripts/evaluate.py --model moi=weights/best.pt --model cu=weights/model_seg_old.pt
python scripts/predict.py --weights weights/best.pt --source path/to/anh_truyen
```

## Dữ liệu và giấy phép
Dataset *manga_bubbles_detect* (tổng hợp từ Roboflow manga-6puie và DLS_Manga_Translator), giấy phép
**CC BY-NC 4.0**: chỉ dùng phi thương mại, phải ghi nguồn. Repo không chứa ảnh gốc.

## Thành viên
| Thành viên | Vai trò |
|---|---|
| Hoàng Hạc Hiếu (trưởng nhóm) | Huấn luyện model, quản lý repo |
| _Nguyễn Thanh Nam_ | Dữ liệu và gán nhãn |
| _Đinh Thị Quỳnh Anh_ | Đánh giá và ứng dụng |
