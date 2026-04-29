from __future__ import annotations

import argparse

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Adidas/Puma YOLO model.")
    parser.add_argument("--model", default="yolo26n.pt", help="Base YOLO model")
    parser.add_argument(
        "--data",
        default="datasets/adidas_puma_balanced/data.yaml",
        help="Path to dataset data.yaml",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="mps")

    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
    )


if __name__ == "__main__":
    main()
