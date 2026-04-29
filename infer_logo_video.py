from __future__ import annotations

import argparse
import csv
from datetime import datetime
import subprocess
import sys
import re
from pathlib import Path
from collections.abc import Callable
from urllib.parse import urlparse

import cv2
from ultralytics import YOLO


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

    # Fallback for your cleaned dataset:
    # 0 = adidas, 1 = puma
    if class_id == 0:
        return "Adidas"

    if class_id == 1:
        return "Puma"

    return raw_name.title()


def draw_detection(frame, brand: str, conf: float, x1: int, y1: int, x2: int, y2: int):
    if brand.lower() == "adidas":
        color = (255, 0, 0)
    else:
        color = (0, 255, 0)

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


def _validate_sql_identifier(name: str) -> None:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(
            f"Invalid Postgres table name: {name!r}. Use letters, numbers, and underscores only."
        )


def store_detections_to_postgres(
    dsn: str,
    table_name: str,
    run_id: str,
    output_video_path: Path,
    output_csv_path: Path,
    detections: list[dict[str, object]],
) -> None:
    if not detections:
        return

    _validate_sql_identifier(table_name)

    import psycopg

    create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id bigserial PRIMARY KEY,
            run_id text NOT NULL,
            frame_index integer NOT NULL,
            timestamp_sec double precision NOT NULL,
            brand text NOT NULL,
            confidence double precision NOT NULL,
            x1 integer NOT NULL,
            y1 integer NOT NULL,
            x2 integer NOT NULL,
            y2 integer NOT NULL,
            box_width integer NOT NULL,
            box_height integer NOT NULL,
            video_path text NOT NULL,
            output_video_path text NOT NULL,
            output_csv_path text NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now()
        )
    """

    insert_sql = f"""
        INSERT INTO {table_name} (
            run_id,
            frame_index,
            timestamp_sec,
            brand,
            confidence,
            x1,
            y1,
            x2,
            y2,
            box_width,
            box_height,
            video_path,
            output_video_path,
            output_csv_path
        ) VALUES (
            %(run_id)s,
            %(frame_index)s,
            %(timestamp_sec)s,
            %(brand)s,
            %(confidence)s,
            %(x1)s,
            %(y1)s,
            %(x2)s,
            %(y2)s,
            %(box_width)s,
            %(box_height)s,
            %(video_path)s,
            %(output_video_path)s,
            %(output_csv_path)s
        )
    """

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(create_table_sql)

            enriched_rows = []
            for row in detections:
                enriched = dict(row)
                enriched["run_id"] = run_id
                enriched["output_video_path"] = str(output_video_path)
                enriched["output_csv_path"] = str(output_csv_path)
                enriched_rows.append(enriched)

            cur.executemany(insert_sql, enriched_rows)


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
    writer = cv2.VideoWriter(
        str(output_video_path),
        fourcc,
        fps,
        (width, height),
    )

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

        results = model.predict(**predict_kwargs)
        result = results[0]

        if result.boxes is not None and len(result.boxes) > 0:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            for box, conf, class_id in zip(boxes_xyxy, confidences, class_ids):
                brand = get_brand_name(class_id, model.names)

                # Safety: keep only Adidas/Puma.
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

                csv_writer.writerow(
                    {
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
                )

                if postgres_dsn:
                    postgres_detections.append(
                        {
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
                    )

                detection_count += 1

        writer.write(frame)

        frame_index += 1

        if frame_index % 100 == 0:
            print(
                f"Processed {frame_index}/{total_frames} frames... detections: {detection_count}"
            )

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


def main():
    parser = argparse.ArgumentParser(
        description="Run Adidas/Puma logo detection on a video."
    )

    parser.add_argument(
        "--weights",
        # required=True,
        default="best.pt",
        help="Path to trained YOLO weights, e.g. runs/detect/train/weights/best.pt",
    )

    parser.add_argument(
        "--source",
        # required=True,
        default="https://youtu.be/obv2j5P2kys?si=8IWQr6FOtVncSzGG",
        help="Local video path or YouTube URL",
    )

    parser.add_argument(
        "--out-dir",
        default="outputs/logo_inference",
        help="Output directory",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU threshold",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Inference image size",
    )

    parser.add_argument(
        "--device",
        default=None,
        help="Device: cpu, mps, 0, etc. Leave empty for auto.",
    )

    args = parser.parse_args()

    try:
        run_video_inference(
            weights=Path(args.weights),
            source=args.source,
            out_dir=Path(args.out_dir),
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            imgsz=args.imgsz,
            device=args.device,
        )
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
