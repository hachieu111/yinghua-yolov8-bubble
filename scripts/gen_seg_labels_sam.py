"""Sinh nhãn viền (đa giác) từ nhãn khung bằng SAM 2.1.

Dataset gốc chỉ có khung chữ nhật. Với mỗi khung, SAM trả về mặt nạ đúng hình bong bóng. Mặt nạ bị cắt
trong khung gốc (bong bóng không thể vượt khung đã gán nhãn), lấy đường viền lớn nhất, rồi lọc:
    - diện tích viền < MIN_FILL × diện tích khung, hoặc
    - khung bao của viền < MIN_SPAN × diện tích khung
→ coi là hỏng (thường gặp khi chữ nằm thẳng trên tranh), dùng chính khung chữ nhật làm đa giác để không mất nhãn.

Nên chạy trên GPU (Colab/Kaggle): khoảng 15–30 phút cho 2 671 ảnh trên T4; CPU chậm hơn nhiều lần.

Ví dụ:
    python scripts/gen_seg_labels_sam.py
    python scripts/gen_seg_labels_sam.py --limit 10      # chạy thử 10 ảnh mỗi tập
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import SAM

SPLITS = ("train", "val", "test")
MIN_FILL = 0.35
MIN_SPAN = 0.60


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sinh nhãn viền bằng SAM 2.1")
    p.add_argument("--det", default="data/det", help="bộ dữ liệu có nhãn khung")
    p.add_argument("--seg", default="data/seg", help="bộ dữ liệu nhận nhãn viền")
    p.add_argument("--sam", default="sam2.1_b.pt", help="trọng số SAM (tự tải lần đầu)")
    p.add_argument("--limit", type=int, default=0, help="chỉ xử lý N ảnh đầu mỗi tập (0 = tất cả)")
    p.add_argument("--report", default="results/data/sam_stats.csv")
    return p.parse_args()


def read_boxes(label: Path, w: int, h: int) -> list[list[int]]:
    boxes = []
    if not label.exists():
        return boxes
    for line in label.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cx, cy, bw, bh = (float(v) for v in parts[1:])
        x1, y1 = max(0, int((cx - bw / 2) * w)), max(0, int((cy - bh / 2) * h))
        x2, y2 = min(w, int((cx + bw / 2) * w)), min(h, int((cy + bh / 2) * h))
        if x2 - x1 > 1 and y2 - y1 > 1:
            boxes.append([x1, y1, x2, y2])
    return boxes


def mask_to_polygon(mask: np.ndarray, box: list[int]) -> np.ndarray | None:
    x1, y1, x2, y2 = box
    clipped = np.zeros_like(mask)
    clipped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    contours, _ = cv2.findContours(clipped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    box_area = (x2 - x1) * (y2 - y1)
    _, _, cw, ch = cv2.boundingRect(c)
    if cv2.contourArea(c) / box_area < MIN_FILL or cw * ch / box_area < MIN_SPAN:
        return None
    c = cv2.approxPolyDP(c, 0.002 * cv2.arcLength(c, True), True).reshape(-1, 2)
    return c if len(c) >= 3 else None


def box_polygon(box: list[int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])


def to_yolo_line(poly: np.ndarray, w: int, h: int) -> str:
    return "0 " + " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in poly)


def process_image(sam: SAM, img_path: Path, label: Path, stats: Counter) -> list[str]:
    h, w = cv2.imread(str(img_path)).shape[:2]
    boxes = read_boxes(label, w, h)
    if not boxes:
        return []
    r = sam(str(img_path), bboxes=boxes, verbose=False)[0]
    masks = r.masks.data.cpu().numpy() if r.masks is not None else []
    lines = []
    for i, box in enumerate(boxes):
        poly = None
        if i < len(masks):
            m = (masks[i] > 0.5).astype(np.uint8)
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            poly = mask_to_polygon(m, box)
        stats["viền SAM" if poly is not None else "dùng khung"] += 1
        lines.append(to_yolo_line(poly if poly is not None else box_polygon(box), w, h))
    return lines


def main() -> None:
    args = parse_args()
    det, seg = Path(args.det), Path(args.seg)
    sam = SAM(args.sam)
    rows = []
    for split in SPLITS:
        stats: Counter = Counter()
        paths = sorted((det / "images" / split).glob("*.jpg"))
        if args.limit:
            paths = paths[: args.limit]
        out_dir = seg / "labels" / split
        out_dir.mkdir(parents=True, exist_ok=True)
        for img_path in tqdm(paths, desc=f"SAM {split}"):
            lines = process_image(sam, img_path, det / "labels" / split / f"{img_path.stem}.txt", stats)
            (out_dir / f"{img_path.stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        total = sum(stats.values()) or 1
        rows.append({"tập": split, "viền SAM": stats["viền SAM"], "dùng khung": stats["dùng khung"],
                     "tỉ lệ viền SAM": f"{stats['viền SAM'] / total:.1%}"})
        print(rows[-1])
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("Đã ghi nhãn viền vào", seg / "labels", "— thống kê:", report)


if __name__ == "__main__":
    main()
