"""Kiểm tra chất lượng nhãn khung (định dạng YOLO: class cx cy w h, chuẩn hóa 0..1).

Phát hiện: ảnh thiếu file nhãn, ảnh không có bong bóng, dòng sai định dạng, tọa độ ngoài [0, 1],
khung quá nhỏ, nhãn trùng lặp. Lưu danh sách lỗi ra results/data/label_issues.csv.

Ví dụ:
    python scripts/check_labels.py
    python scripts/check_labels.py --root data/det --min-size 0.005
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

SPLITS = ("train", "val", "test")
DUP_IOU = 0.9


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Kiểm tra nhãn khung YOLO")
    p.add_argument("--root", default="data/det", help="thư mục có images/ và labels/")
    p.add_argument("--min-size", type=float, default=0.003, help="cạnh khung nhỏ hơn mức này (tỉ lệ ảnh) bị coi là quá nhỏ")
    p.add_argument("--out", default="results/data/label_issues.csv")
    return p.parse_args()


def check_line(parts: list[str], min_size: float) -> str | None:
    if len(parts) != 5:
        return f"sai định dạng ({len(parts)} số)"
    try:
        cls, cx, cy, w, h = int(parts[0]), *(float(v) for v in parts[1:])
    except ValueError:
        return "không phải số"
    if cls != 0:
        return f"lớp lạ {cls}"
    if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)):
        return "tọa độ ngoài [0, 1]"
    if w < min_size or h < min_size:
        return "khung quá nhỏ"
    return None


def check_split(root: Path, split: str, min_size: float, issues: list[dict]) -> Counter:
    stats: Counter = Counter()
    for img in sorted((root / "images" / split).glob("*.jpg")):
        stats["ảnh"] += 1
        label = root / "labels" / split / f"{img.stem}.txt"
        if not label.exists():
            stats["thiếu file nhãn"] += 1
            issues.append({"split": split, "file": img.name, "line": "", "issue": "thiếu file nhãn"})
            continue
        lines = [ln.strip() for ln in label.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if not lines:
            stats["ảnh không có bong bóng"] += 1
        valid: list[str] = []
        for line in lines:
            stats["bong bóng"] += 1
            problem = check_line(line.split(), min_size)
            if problem:
                stats[problem.split(" (")[0]] += 1
                issues.append({"split": split, "file": label.name, "line": line, "issue": problem})
            else:
                valid.append(line)
        for line in find_duplicates(valid):
            stats["nhãn trùng"] += 1
            issues.append({"split": split, "file": label.name, "line": line, "issue": f"trùng (IoU > {DUP_IOU})"})
    return stats


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


def find_duplicates(lines: list[str]) -> list[str]:
    """Khung gần như trùng khít một khung trước nó (người gán nhãn vẽ hai lần)."""
    boxes = [tuple(float(v) for v in ln.split()[1:]) for ln in lines]
    return [lines[j] for j in range(len(boxes)) if any(iou(boxes[i], boxes[j]) > DUP_IOU for i in range(j))]


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    issues: list[dict] = []
    for split in SPLITS:
        stats = check_split(root, split, args.min_size, issues)
        print(f"[{split}] " + ", ".join(f"{k}: {v}" for k, v in stats.items()))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "file", "line", "issue"])
        writer.writeheader()
        writer.writerows(issues)
    print(f"{len(issues)} vấn đề, chi tiết trong {out}")


if __name__ == "__main__":
    main()
