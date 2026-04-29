from __future__ import annotations

import csv
import subprocess
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import cv2
from ultralytics import YOLO

from .storage import store_detections_to_postgres


def is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"}


def download_youtube_video(url: str, out_dir: Path, run_id: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    output_template = str(out_dir / f"youtube_video_{run_id}.%(ext)s")

    cmd = [
        "yt-dlp",
        "-f",
        "mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format",
        "mp4",
        "-o",
        output_template,
        url,
    ]

    print("Downloading YouTube video...")
    subprocess.run(cmd, check=True)

    preferred = out_dir / f"youtube_video_{run_id}.mp4"
    if preferred.exists():
        return preferred

    downloaded_files = [
        path
        for path in sorted(out_dir.glob(f"youtube_video_{run_id}.*"))
        if path.suffix.lower() != ".json"
    ]

    if not downloaded_files:
        raise FileNotFoundError("yt-dlp finished but no video file was found.")

    return downloaded_files[0]


def get_brand_name(class_id: int, model_names: dict) -> str:
    raw_name = str(model_names.get(class_id, class_id)).lower().strip()

    if "adidas" in raw_name:
        return "Adidas"

    if "puma" in raw_name:
        return "Puma"

    if class_id == 0:
        return "Adidas"

    if class_id == 1:
        return "Puma"

    return raw_name.title()


def draw_detection(frame, brand: str, conf: float, x1: int, y1: int, x2: int, y2: int):
    color = (255, 0, 0) if brand.lower() == "adidas" else (0, 255, 0)
    label = f"{brand} {conf:.2f}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    text_y = max(25, y1 - 8)
    cv2.putText(
        frame,
        label,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


def run_video_inference(
    weights: Path,
    source: str,
    out_dir: Path,
    conf_threshold: float,
    iou_threshold: float,
    imgsz: int,
    device: str | None,
    progress_callback: Callable[[int, int, int], None] | None = None,
    postgres_dsn: str | None = None,
    postgres_table: str = "video_detections",
):
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    if is_url(source):
        video_path = download_youtube_video(source, out_dir / "downloads", run_id)
    else:
        video_path = Path(source)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not weights.exists():
        raise FileNotFoundError(f"Model weights not found: {weights}")

    print(f"Loading model: {weights}")
    model = YOLO(str(weights))
    print("Model class names:", model.names)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not fps or fps <= 0:
        fps = 25.0

    output_video_path = out_dir / f"labeled_logo_video_{run_id}.mp4"
    output_csv_path = out_dir / f"detections_{run_id}.csv"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    csv_file = output_csv_path.open("w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "frame_index",
            "timestamp_sec",
            "brand",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
            "box_width",
            "box_height",
            "video_path",
        ],
    )
    csv_writer.writeheader()

    frame_index = 0
    detection_count = 0
    postgres_detections: list[dict[str, object]] = []

    print(f"Processing video: {video_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps:.2f}, frames: {total_frames}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_sec = frame_index / fps

        predict_kwargs = {
            "source": frame,
            "conf": conf_threshold,
            "iou": iou_threshold,
            "imgsz": imgsz,
            "verbose": False,
        }
        if device:
            predict_kwargs["device"] = device

        result = model.predict(**predict_kwargs)[0]

        if result.boxes is not None and len(result.boxes) > 0:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            for box, conf, class_id in zip(boxes_xyxy, confidences, class_ids):
                brand = get_brand_name(class_id, model.names)
                if brand not in {"Adidas", "Puma"}:
                    continue

                x1, y1, x2, y2 = box
                x1 = max(0, min(width - 1, int(round(x1))))
                y1 = max(0, min(height - 1, int(round(y1))))
                x2 = max(0, min(width - 1, int(round(x2))))
                y2 = max(0, min(height - 1, int(round(y2))))

                box_width = x2 - x1
                box_height = y2 - y1
                if box_width <= 0 or box_height <= 0:
                    continue

                draw_detection(frame, brand, float(conf), x1, y1, x2, y2)

                row = {
                    "frame_index": frame_index,
                    "timestamp_sec": round(timestamp_sec, 3),
                    "brand": brand,
                    "confidence": round(float(conf), 6),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "box_width": box_width,
                    "box_height": box_height,
                    "video_path": str(video_path),
                }
                csv_writer.writerow(row)
                if postgres_dsn:
                    postgres_detections.append(row)
                detection_count += 1

        writer.write(frame)
        frame_index += 1

        if frame_index % 100 == 0:
            print(f"Processed {frame_index}/{total_frames} frames... detections: {detection_count}")

        if progress_callback and (frame_index % 25 == 0 or frame_index == total_frames):
            progress_callback(frame_index, total_frames, detection_count)

    cap.release()
    writer.release()
    csv_file.close()

    print("\nDone.")
    print(f"Annotated video: {output_video_path}")
    print(f"CSV file:        {output_csv_path}")
    print(f"Total detections: {detection_count}")

    if postgres_dsn:
        store_detections_to_postgres(
            dsn=postgres_dsn,
            table_name=postgres_table,
            run_id=run_id,
            output_video_path=output_video_path,
            output_csv_path=output_csv_path,
            detections=postgres_detections,
        )
        print(f"Postgres table:   {postgres_table}")

    return output_video_path, output_csv_path, detection_count, video_path
