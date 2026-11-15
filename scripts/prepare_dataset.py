"""Chuẩn bị dữ liệu cho YOLO từ bubbles_dataset.zip (hoặc thư mục có sẵn images/ và labels/).

Tạo hai bộ dữ liệu dùng chung ảnh:
    data/det/  nhãn khung chữ nhật gốc (5 số mỗi dòng)
    data/seg/  nhãn viền đa giác, sinh bằng scripts/gen_seg_labels_sam.py
mỗi bộ có data.yaml riêng.

Ví dụ:
    python scripts/prepare_dataset.py --src D:/Downloads/bubbles_dataset.zip
"""
from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from pathlib import Path

SPLITS = ("train", "val", "test")
CLASS_NAME = "speech_bubble"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chuẩn bị dữ liệu bong bóng thoại cho YOLO")
    p.add_argument("--src", required=True, help="file .zip hoặc thư mục chứa images/ và labels/")
    p.add_argument("--out", default="data", help="thư mục đích")
    return p.parse_args()


def find_root(base: Path) -> Path:
    """Tìm thư mục có images/train và labels/train (zip có thể bọc thêm một cấp)."""
    for p in [base, *base.rglob("images")]:
        root = p if p == base else p.parent
        if (root / "images" / "train").is_dir() and (root / "labels" / "train").is_dir():
            return root
    raise SystemExit(f"Không thấy images/train và labels/train trong {base}")


def extract_if_zip(src: Path, out: Path) -> Path:
    if src.is_dir():
        return find_root(src)
    raw = out / "raw"
    if not (raw / "images").exists():
        print("Giải nén", src, "→", raw)
        with zipfile.ZipFile(src) as zf:
            zf.extractall(raw)
    return find_root(raw)


def link_or_copy(src: Path, dst: Path) -> None:
    # YOLO tìm nhãn bằng cách đổi /images/ thành /labels/ trong đường dẫn ảnh, nên mỗi bộ nhãn cần
    # thư mục gốc riêng. Không dùng symlink thư mục: ultralytics lần theo symlink về thư mục gốc và
    # đọc nhầm nhãn khung. Hard link từng file không tốn dung lượng; khác ổ đĩa thì chép.
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        target = dst / f.name
        if f.is_file() and f.suffix != ".cache" and not target.exists():
            try:
                os.link(f, target)
            except OSError:
                shutil.copy2(f, target)


def write_yaml(root: Path) -> Path:
    path = root / "data.yaml"
    path.write_text(
        f"path: {root.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n  0: {CLASS_NAME}\n",
        encoding="utf-8",
    )
    return path


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    raw = extract_if_zip(Path(args.src), out)
    det, seg = out / "det", out / "seg"
    for split in SPLITS:
        link_or_copy(raw / "images" / split, det / "images" / split)
        link_or_copy(raw / "labels" / split, det / "labels" / split)
        link_or_copy(raw / "images" / split, seg / "images" / split)
        (seg / "labels" / split).mkdir(parents=True, exist_ok=True)
        n_img = len(list((raw / "images" / split).glob("*.jpg")))
        n_lbl = len(list((raw / "labels" / split).glob("*.txt")))
        print(f"{split:5s}: {n_img:5d} ảnh, {n_lbl:5d} file nhãn")
    print("Đã tạo:", write_yaml(det), "và", write_yaml(seg))
    print("Bước tiếp: python scripts/check_labels.py, rồi python scripts/gen_seg_labels_sam.py")


if __name__ == "__main__":
    main()
