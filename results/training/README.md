# Kết quả huấn luyện

Lần train: `bubbles_seg_1024` (Google Colab, GPU Tesla T4). Dừng sớm ở epoch 57, `best.pt` là epoch 37.

| File | Nội dung |
|---|---|
| `results.csv` | Loss và chỉ số của từng epoch |
| `results.png` | Biểu đồ của `results.csv` |
| `args.yaml` | Toàn bộ tham số đã dùng |
| `train_batch0.jpg`, `train_batch1.jpg`, `train_batch2.jpg` | Lô ảnh đầu tiên: tăng cường dữ liệu bằng mosaic |
| `train_batch12850.jpg` … `train_batch12852.jpg` | Lô ảnh ở 10 epoch cuối, đã tắt mosaic |

Biểu đồ đánh giá (PR, F1, ma trận nhầm lẫn) ở `results/evaluation/`; phân bố nhãn ở `results/data/`.
Model (`best.pt`) không nằm ở đây: tải ở trang Releases của repo.
