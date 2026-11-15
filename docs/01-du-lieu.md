# Chương 1. Dữ liệu và tiền xử lý

> Người viết: Người 2. Dàn ý kèm số liệu thật (chạy `scripts/check_labels.py`, `scripts/eda.py`);
> viết thành văn và chèn hình từ `results/data/`.

## 1.1. Nguồn dữ liệu
- Dataset *manga_bubbles_detect*: tổng hợp từ Roboflow *manga-6puie* và *DLS_Manga_Translator*.
- Giấy phép CC BY-NC 4.0 (phi thương mại, ghi nguồn).
- Nội dung: trang manga Nhật, chủ yếu đen trắng. Một lớp: bong bóng thoại.

## 1.2. Thống kê (bảng `results/data/summary.csv`)
| Tập | Số ảnh | Số bong bóng | Ảnh không có bong bóng | TB bong bóng/trang | Nhiều nhất/trang |
|---|---|---|---|---|---|
| train | 2 056 | 12 775 | 259 (12,6%) | 6,21 | 28 |
| val | 404 | 2 632 | 44 (10,9%) | 6,51 | 26 |
| test | 211 | 1 351 | 19 (9,0%) | 6,40 | 23 |
| **Tổng** | **2 671** | **16 758** | 322 | | |

- Tỉ lệ chia khoảng 77% / 15% / 8%.
- Ảnh không có bong bóng (ảnh nền) giúp model học **không** báo nhầm trên trang chỉ có tranh.
- Chèn hình: `bubbles_per_image.png` (phân bố số bong bóng), `box_size.png` (kích thước bong bóng),
  `image_size.png` (kích thước ảnh), `samples.png` (ảnh mẫu kèm nhãn).
- `label_distribution.jpg` (YOLO sinh khi train): số nhãn, phân bố **vị trí tâm** và **kích thước** bong bóng trên
  tập train (12 775 nhãn). Bong bóng có mặt khắp trang nhưng dày hơn ở hai bên (x ≈ 0,2 và 0,8) và mép trên, dưới,
đúng với cách vẽ thoại sát góc khung tranh. Phần lớn bong bóng nhỏ: rộng và cao khoảng 0,05–0,25 kích thước trang,
tỉ lệ rộng/cao gần 1 → lý do cần ảnh đầu vào lớn (1024px).

## 1.3. Kiểm tra chất lượng nhãn (`results/data/label_issues.csv`)
- Không có file nhãn thiếu, dòng sai định dạng hay tọa độ ngoài [0, 1].
- Phát hiện **3 nhãn trùng** (hai khung gần như chồng khít, IoU > 0,9): `val/img_000136`, `test/img_000057`, `test/img_000105`.
  YOLO cũng tự bỏ các nhãn này khi đọc dữ liệu ("duplicate labels removed").

## 1.4. Định dạng nhãn YOLO
- Khung: `class cx cy w h`, chuẩn hóa về 0..1 theo kích thước ảnh.
- Viền: `class x1 y1 x2 y2 … xn yn`, các đỉnh đa giác chuẩn hóa.

## 1.5. Sinh nhãn viền bằng SAM 2.1
- Vấn đề: dataset chỉ có khung, nhưng bài toán cần viền bong bóng.
- Giải pháp: dùng **Segment Anything 2.1** (Meta) với khung gốc làm gợi ý (box prompt) → mặt nạ đúng hình bong bóng.
- Hậu xử lý: cắt mặt nạ trong khung gốc → lấy đường viền lớn nhất → đơn giản hóa đa giác (`approxPolyDP`, 0,2% chu vi).
- Lọc viền hỏng: phủ < 35% khung hoặc khung bao < 60% khung → dùng khung chữ nhật làm đa giác thay thế.
- Kết quả: chạy thử trên 5% dữ liệu (Colab, GPU T4), 446/446 bong bóng có viền SAM hợp lệ (100%), không phải
  dùng khung thay thế. Sau đó sinh viền cho toàn bộ 16 758 bong bóng và lưu thành `seg_labels.zip` để dùng lại khi train.
- Hạn chế: viền là nhãn **tự động**, không phải người vẽ; điểm "mask" khi đánh giá đo độ giống viền SAM.

## 1.6. Chuẩn bị cho huấn luyện (`scripts/prepare_dataset.py`)
- Tạo hai bộ `data/det` (khung) và `data/seg` (viền) dùng chung ảnh bằng **hard link** (không tốn thêm dung lượng).
- Vì sao không dùng symlink thư mục: ultralytics lần theo symlink về thư mục gốc và đọc nhầm nhãn khung thay vì nhãn viền.
- Mỗi bộ có `data.yaml` riêng khai báo đường dẫn và tên lớp `speech_bubble`.
