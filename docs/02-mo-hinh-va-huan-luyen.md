# Chương 2. Mô hình và huấn luyện

> Người viết: Người 1 (trưởng nhóm). Dàn ý kèm số liệu thật; viết thành văn và chèn hình từ `results/training/`.

## 2.1. Bài toán
- Đầu vào: một trang truyện. Đầu ra: vị trí (khung) và **viền** (mặt nạ) của từng bong bóng thoại.
- Một lớp duy nhất: `speech_bubble`. Đây là bài toán **instance segmentation**.
- Vì sao cần viền chứ không chỉ khung: bước xóa chữ (inpaint) và đắp chữ dịch cần biết đúng vùng bên trong bong bóng.

## 2.2. Mô hình YOLOv8-seg
- Kiến trúc: backbone (trích đặc trưng) → neck (trộn đặc trưng nhiều tỉ lệ) → head phát hiện + head mặt nạ (prototype mask).
- Chọn bản **m** (khoảng 27,2 triệu tham số, 104 GFLOPs): cân bằng giữa độ chính xác và tốc độ (khoảng 86 ms/ảnh trên T4 ở 1024px).
- Khởi đầu từ model cũ của dự án (yolov8m-seg đã học bong bóng) thay vì học lại từ đầu → hội tụ nhanh hơn.

## 2.3. Hàm mất mát (giải thích từng cột trong log)
| Loss | Ý nghĩa |
|---|---|
| `box_loss` | Sai số vị trí, kích thước khung (CIoU) |
| `seg_loss` | Sai số mặt nạ viền theo từng điểm ảnh |
| `cls_loss` | Sai số phân loại "có phải bong bóng không" |
| `dfl_loss` | Distribution Focal Loss, giúp đặt từng cạnh khung chính xác |

## 2.4. Cấu hình huấn luyện (`configs/train.yaml`)
| Tham số | Giá trị | Lý do |
|---|---|---|
| Môi trường | Google Colab, GPU Tesla T4 15 GB | miễn phí |
| imgsz | 1024 | chữ trong bong bóng nhỏ, cần ảnh lớn |
| batch | 8 | batch 16 tràn bộ nhớ; batch 8 dùng khoảng 9,3 GB |
| epochs / patience | 60 / 20 | |
| cos_lr | bật | giảm tốc độ học mượt ở cuối |
| close_mosaic | 10 | 10 epoch cuối học trên ảnh thật, không ghép |
| fliplr | 0 | lật ngang làm sai chiều đọc truyện |
| Thời gian | khoảng 3 phút 50 giây/epoch (gồm chấm val), tổng **3,7 giờ** | |
| Kết thúc | **dừng sớm ở epoch 57/60**: epoch tốt nhất là **37**, 20 epoch sau không cải thiện | |

Chỉ số val của `best.pt` (epoch 37, theo `results/training/results.csv`):
| | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Box | 0.952 | 0.946 | 0.972 | 0.902 |
| Mask | 0.959 | 0.937 | 0.971 | 0.885 |

## 2.4b. Tăng cường dữ liệu (chèn `train_batch*.jpg`)
- `train_batch0.jpg`, `train_batch1.jpg`, `train_batch2.jpg`: 3 lô đầu tiên. Mỗi ô ghép 4 trang truyện (**mosaic**),
  có co giãn (scale 0,5), dịch (0,1), đổi màu HSV, xóa ngẫu nhiên vùng nhỏ (erasing 0,4); viền bong bóng vẽ đè lên ảnh.
- `train_batch12850.jpg`, `train_batch12851.jpg`, `train_batch12852.jpg`: lô ở 10 epoch cuối, **đã tắt mosaic**
  (`close_mosaic=10`), mỗi ô là một trang thật → giải thích vì sao loss train rơi gấp ở khoảng epoch 50.
- Không dùng lật ngang (`fliplr=0`) và lật dọc (`flipud=0`) vì làm sai chiều đọc của truyện.

## 2.5. Quá trình huấn luyện (chèn `results.png`)
- Loss train giảm đều: box 0,50 → 0,24; seg 0,90 → 0,40. Rơi gấp ở khoảng epoch 50 do tắt mosaic.
- mAP50 trên val đạt khoảng 0,965 từ epoch khoảng 20, sau đó đi ngang → model hội tụ sớm nhờ khởi đầu từ model cũ.
- mAP50-95 cao nhất khoảng 0,90 (khung) và 0,88 (viền) trong khoảng epoch 25–45.
- `val/cls_loss` thấp nhất ở epoch 25 rồi tăng nhẹ, `val/seg_loss` tăng nhẹ sau epoch 35 → **overfitting nhẹ** ở cuối.
  `best.pt` được chọn ở epoch có điểm cao nhất nên không bị ảnh hưởng.
- Precision giảm nhẹ (0,96 → 0,94) trong khi Recall tăng (0,89 → 0,92): model ít sót bong bóng hơn, phù hợp mục tiêu.

## 2.6. Khó khăn và cách xử lý
| Vấn đề | Cách xử lý |
|---|---|
| Colab ngắt phiên giữa chừng | Lưu kết quả lên Google Drive, checkpoint mỗi 5 epoch, tự train tiếp |
| `CUDA out of memory`, AutoBatch chọn batch 1 | Khởi động lại phiên, đặt cứng batch 8 |
| Nhãn viền không được đọc (segments = 0) | Ultralytics lần theo symlink về thư mục nhãn khung; đổi sang hard link từng file |
| Colab bị nhận nhầm là Kaggle | Nhận Kaggle bằng biến môi trường `KAGGLE_KERNEL_RUN_TYPE` |

## 2.7. Hướng cải thiện
- Model đạt đỉnh ở epoch 37; train khoảng 40 epoch là đủ, tiết kiệm khoảng 1 giờ GPU.
- Thêm ảnh manhua (truyện Trung) để model quen phong cách màu, khổ dài.
