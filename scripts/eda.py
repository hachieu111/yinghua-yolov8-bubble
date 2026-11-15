"""Thống kê khám phá dữ liệu (EDA) và vẽ biểu đồ cho báo cáo.

Sinh vào results/data/:
    summary.csv               số ảnh, số bong bóng, ảnh không có bong bóng theo từng tập
    bubbles_per_image.png     phân bố số bong bóng mỗi trang
    box_size.png              phân bố kích thước bong bóng (tỉ lệ so với ảnh)
    image_size.png            phân bố kích thước ảnh
    samples.png               4 ảnh mẫu có vẽ khung nhãn

Ví dụ:
    python scripts/eda.py
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SPLITS = ("train", "val", "test")
COLOR = (164, 48, 115)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EDA dữ liệu bong bóng thoại")
    p.add_argument("--root", default="data/det")
    p.add_argument("--out", default="results/data")
    return p.parse_args()


def read_boxes(label: Path) -> list[tuple[float, float, float, float]]:
    if not label.exists():
        return []
    boxes = []
    for line in label.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 5:
            boxes.append(tuple(float(v) for v in parts[1:]))
    return boxes


def collect(root: Path) -> dict[str, dict]:
    data = {}
    for split in SPLITS:
        counts, widths, heights, sizes = [], [], [], []
        for img in sorted((root / "images" / split).glob("*.jpg")):
            boxes = read_boxes(root / "labels" / split / f"{img.stem}.txt")
            counts.append(len(boxes))
            widths += [b[2] for b in boxes]
            heights += [b[3] for b in boxes]
            h, w = cv2.imread(str(img), cv2.IMREAD_REDUCED_GRAYSCALE_8).shape[:2]
            sizes.append((w * 8, h * 8))  # đọc ảnh thu nhỏ 1/8 cho nhanh, chỉ cần kích thước gần đúng
        data[split] = {"counts": counts, "widths": widths, "heights": heights, "sizes": sizes}
    return data


def save_summary(data: dict, out: Path) -> None:
    with (out / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["tập", "số ảnh", "số bong bóng", "ảnh không có bong bóng", "TB bong bóng/ảnh", "nhiều nhất/ảnh"])
        for split, d in data.items():
            c = d["counts"]
            row = [split, len(c), sum(c), sum(1 for x in c if x == 0), round(sum(c) / max(1, len(c)), 2), max(c, default=0)]
            w.writerow(row)
            print(*row, sep="\t")


def plot_charts(data: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist([data[s]["counts"] for s in SPLITS], bins=range(0, 30), label=SPLITS)
    ax.set(xlabel="Số bong bóng trên một trang", ylabel="Số trang", title="Phân bố số bong bóng mỗi trang")
    ax.legend()
    fig.tight_layout(); fig.savefig(out / "bubbles_per_image.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    w = data["train"]["widths"]; h = data["train"]["heights"]
    ax.scatter(w, h, s=3, alpha=0.3, color="#a43073")
    ax.set(xlabel="Rộng / rộng ảnh", ylabel="Cao / cao ảnh", title="Kích thước bong bóng (tập train)")
    fig.tight_layout(); fig.savefig(out / "box_size.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    sizes = [s for split in SPLITS for s in data[split]["sizes"]]
    ax.scatter([s[0] for s in sizes], [s[1] for s in sizes], s=4, alpha=0.4)
    ax.set(xlabel="Rộng (px)", ylabel="Cao (px)", title="Kích thước ảnh")
    fig.tight_layout(); fig.savefig(out / "image_size.png", dpi=120); plt.close(fig)


def plot_samples(root: Path, out: Path, k: int = 4) -> None:
    imgs = [p for p in sorted((root / "images" / "train").glob("*.jpg"))
            if read_boxes(root / "labels" / "train" / f"{p.stem}.txt")]
    fig, axes = plt.subplots(1, k, figsize=(5 * k, 8))
    for ax, p in zip(axes, random.Random(0).sample(imgs, k)):
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        boxes = read_boxes(root / "labels" / "train" / f"{p.stem}.txt")
        for cx, cy, bw, bh in boxes:
            pt1 = (int((cx - bw / 2) * W), int((cy - bh / 2) * H))
            pt2 = (int((cx + bw / 2) * W), int((cy + bh / 2) * H))
            cv2.rectangle(img, pt1, pt2, COLOR, max(2, W // 300))
        ax.imshow(img); ax.set_title(f"{p.name}: {len(boxes)} bong bóng"); ax.axis("off")
    fig.tight_layout(); fig.savefig(out / "samples.png", dpi=100); plt.close(fig)


def main() -> None:
    args = parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = collect(root)
    save_summary(data, out)
    plot_charts(data, out)
    plot_samples(root, out)
    print("Đã lưu biểu đồ vào", out)


if __name__ == "__main__":
    main()
