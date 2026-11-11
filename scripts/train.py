"""Huấn luyện YOLOv8-seg phát hiện và tách viền bong bóng thoại.

Đọc tham số từ configs/train.yaml; tham số dòng lệnh ghi đè file cấu hình.
Nếu lần chạy trước bị ngắt giữa chừng (còn last.pt chưa xong) thì tự train tiếp.

Ví dụ:
    python scripts/train.py --weights yolov8m-seg.pt
    python scripts/train.py --weights weights/model_seg_old.pt --epochs 40 --name finetune
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train YOLOv8-seg bong bóng thoại")
    p.add_argument("--data", default="data/seg/data.yaml", help="file data.yaml do prepare_dataset.py sinh")
    p.add_argument("--cfg", default="configs/train.yaml", help="file tham số train")
    p.add_argument("--weights", default="yolov8m-seg.pt", help="model khởi đầu (COCO hoặc model cũ)")
    p.add_argument("--project", default="runs", help="thư mục lưu kết quả")
    p.add_argument("--name", default="bubbles_seg", help="tên lần chạy")
    p.add_argument("--epochs", type=int, help="ghi đè epochs trong cấu hình")
    p.add_argument("--imgsz", type=int, help="ghi đè imgsz trong cấu hình")
    p.add_argument("--batch", type=int, help="ghi đè batch trong cấu hình")
    p.add_argument("--fraction", type=float, default=1.0, help="tỉ lệ ảnh train dùng (0.05 để chạy thử)")
    return p.parse_args()


def is_finished(last_ckpt: Path) -> bool:
    # Ultralytics ghi epoch = -1 vào last.pt khi đã train xong; resume lúc đó sẽ báo lỗi.
    ckpt = torch.load(last_ckpt, map_location="cpu", weights_only=False)
    return ckpt.get("epoch", -1) < 0


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.cfg).read_text(encoding="utf-8"))
    for key in ("epochs", "imgsz", "batch"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)

    # Đường dẫn tuyệt đối: với đường tương đối, ultralytics 8.4 tự chèn thêm thư mục con theo loại task.
    project = Path(args.project).resolve()
    last_ckpt = project / args.name / "weights" / "last.pt"

    if last_ckpt.exists() and is_finished(last_ckpt):
        print(f"{args.name} đã train xong. Đổi --name để train lần mới.")
        return
    if last_ckpt.exists():
        print("Train tiếp từ", last_ckpt)
        YOLO(str(last_ckpt)).train(resume=True)
        return

    print("Train từ", args.weights, "với", cfg)
    YOLO(args.weights).train(
        data=str(Path(args.data).resolve()),
        project=str(project),
        name=args.name,
        exist_ok=True,
        fraction=args.fraction,
        **cfg,
    )
    print("Xong. Model tốt nhất:", project / args.name / "weights" / "best.pt")


if __name__ == "__main__":
    main()
