from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .video import run_video_inference
from .video import is_url


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Adidas/Puma logo detection on a video.")
    parser.add_argument(
        "--weights",
        default="best.pt",
        help="Path to trained YOLO weights, e.g. runs/detect/train/weights/best.pt",
    )
    parser.add_argument(
        "--source",
        default="https://youtu.be/NNb_-fUyDTo?si=Vz6NUwr8FN3-C3mt",
        help="Local video path or YouTube URL",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/logo_inference",
        help="Output directory",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=960, help="Inference image size")
    parser.add_argument(
        "--device",
        default=None,
        help="Device: cpu, mps, 0, etc. Leave empty for auto.",
    )
    parser.add_argument(
        "--sqlite-db",
        default="outputs/logo_inference/video_detections.sqlite3",
        help="SQLite database path for detections.",
    )

    args = parser.parse_args()

    source_type = "youtube_url" if is_url(args.source) else "local_path"

    try:
        run_video_inference(
            weights=Path(args.weights),
            source=args.source,
            out_dir=Path(args.out_dir),
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            sqlite_db_path=Path(args.sqlite_db),
            source_type=source_type,
            source_value=args.source,
        )
    except Exception as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
