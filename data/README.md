# Dữ liệu

Ảnh và nhãn **không** đưa lên Git (khoảng 736 MB). Thư mục này chỉ chứa hướng dẫn.

## Nguồn
- Dataset: **manga_bubbles_detect**, tổng hợp từ Roboflow *manga-6puie* và *DLS_Manga_Translator*.
- Giấy phép: **CC BY-NC 4.0**. Chỉ dùng phi thương mại (học tập, nghiên cứu) và phải ghi nguồn.
- Một lớp: `speech_bubble` (bong bóng thoại). Nhãn gốc là **khung chữ nhật** định dạng YOLO.

| Tập | Số ảnh |
|---|---|
| train | 2 056 |
| val | 404 |
| test | 211 |

## Tải và chuẩn bị
1. Tải `bubbles_dataset.zip` (link Drive do trưởng nhóm gửi).
2. Chạy lần lượt:
   ```bash
   python scripts/prepare_dataset.py --src duong/dan/bubbles_dataset.zip
   python scripts/check_labels.py
   python scripts/eda.py
   python scripts/gen_seg_labels_sam.py      # nên chạy trên GPU
   ```

## Cấu trúc sau khi chuẩn bị
```
data/
├── raw/           ảnh + nhãn gốc giải nén từ zip
├── det/           nhãn khung (data.yaml)
└── seg/           nhãn viền do SAM sinh (data.yaml) ← dùng để train
```

## Định dạng nhãn
- Khung (`det/labels`): `0 cx cy w h`, chuẩn hóa về 0..1.
- Viền (`seg/labels`): `0 x1 y1 x2 y2 ... xn yn`, các đỉnh đa giác chuẩn hóa về 0..1.
