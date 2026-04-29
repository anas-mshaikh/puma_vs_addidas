# mobileFirst

Adidas / Puma logo detection for video with a Streamlit UI, SQLite-backed analytics, and YOLO training utilities.

## What It Does

- Runs YOLO inference on local videos or YouTube URLs
- Produces an annotated MP4 with bounding boxes and labels
- Exports frame-level detections to CSV
- Stores run metadata and detections in SQLite
- Provides an analytics tab with natural-language questions over the saved detections
- Includes dataset cleaning, balancing, visualization, and training entrypoints

## Repository Layout

- `backend/`
  - `inference/`
    - `video.py`
      - Core inference pipeline
      - Downloads YouTube videos when needed
      - Writes annotated video, CSV, and SQLite rows
    - `storage.py`
      - SQLite schema creation and persistence helpers
    - `cli.py`
      - CLI entrypoint for batch inference
  - `analytics/`
    - `db_views.py`
      - Creates analytics views over the SQLite database
    - `sql_agent.py`
      - Natural-language analytics assistant
    - `sqlite_tools.py`
      - Read-only SQLite query helpers and schema inspection
    - `sql_guard.py`
      - SQL safety validation for analytics queries
  - `data/`
    - `cleaning.py`
      - Builds a cleaned Adidas/Puma dataset from raw Roboflow exports
    - `balancing.py`
      - Balances the cleaned dataset
    - `visualization.py`
      - Generates annotated debug samples from the dataset
  - `training/`
    - `train.py`
      - YOLO training entrypoint
- `frontend/`
  - `app.py`
  - Streamlit app with `Inference` and `Analytics` tabs
- `streamlit_app.py`
  - Streamlit launcher
- `train_yolo.py`
  - Training launcher

## Requirements

- Python 3.10+ recommended
- OpenCV
- Streamlit
- Ultralytics
- `yt-dlp` for YouTube sources
- `HF_TOKEN` for the analytics tab

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

Run inference from the CLI:

```bash
python scripts/infer_logo_video.py --source path/to/video.mp4
```

Train a model:

```bash
python train_yolo.py
```

## Configuration

### Environment Variables

- `LOGO_DB_PATH`
  - Optional SQLite file path used by the app and analytics tab
  - Default: `outputs/streamlit_runs/video_detections.sqlite3`
- `SQLITE_DB_PATH`
  - Fallback SQLite file path if `LOGO_DB_PATH` is not set
- `HF_TOKEN`
  - Required for the analytics tab
- `HF_BASE_URL`
  - Optional OpenAI-compatible API base URL
  - Default: `https://router.huggingface.co/v1`
- `HF_MODEL`
  - Optional model name for analytics
  - Default: `openai/gpt-oss-120b:cerebras`

## Streamlit App

The app has two tabs:

- `Inference`
  - Select a YouTube URL or local video
  - Tune confidence, IoU, image size, and device
  - Run inference and preview the annotated video
  - Download the generated CSV
- `Analytics`
  - Ask natural-language questions about the saved SQLite data
  - Uses the current `active_run_id` when available
  - Falls back to querying across all runs if no active run exists

The active run is set automatically after inference finishes and the SQLite run row is created.

## SQLite Database

The app initializes these tables automatically:

```sql
CREATE TABLE IF NOT EXISTS inference_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type TEXT,
    source_value TEXT,
    input_video_path TEXT,
    output_video_path TEXT,
    csv_path TEXT,
    model_path TEXT,
    confidence_threshold REAL,
    iou_threshold REAL,
    total_frames INTEGER,
    fps REAL,
    width INTEGER,
    height INTEGER,
    total_detections INTEGER,
    started_at TEXT,
    finished_at TEXT,
    status TEXT
);

CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    frame_index INTEGER NOT NULL,
    timestamp_sec REAL NOT NULL,
    brand TEXT NOT NULL,
    confidence REAL NOT NULL,
    x1 INTEGER NOT NULL,
    y1 INTEGER NOT NULL,
    x2 INTEGER NOT NULL,
    y2 INTEGER NOT NULL,
    box_width INTEGER NOT NULL,
    box_height INTEGER NOT NULL,
    area INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES inference_runs(id)
);
```

Analytics views are also created automatically:

- `v_detections`
- `v_brand_summary`
- `v_first_appearance`
- `v_timeline_5s`

## Running Inference

### Streamlit

```bash
streamlit run streamlit_app.py
```

### CLI

```bash
python scripts/infer_logo_video.py \
  --source path/to/video.mp4 \
  --weights best.pt \
  --out-dir outputs/logo_inference \
  --sqlite-db outputs/logo_inference/video_detections.sqlite3
```

### Common Options

- `--weights`
  - Path to the YOLO weights file
  - Default: `best.pt`
- `--source`
  - Local video path or YouTube URL
- `--out-dir`
  - Output directory for annotated video and CSV
- `--conf`
  - Confidence threshold
- `--iou`
  - NMS IoU threshold
- `--imgsz`
  - Inference image size
- `--device`
  - `cpu`, `mps`, `0`, etc.
- `--sqlite-db`
  - SQLite database path for run and detection storage

## Analytics Workflow

1. Run inference and save a run to SQLite
2. Open the `Analytics` tab
3. Ask questions like:
   - When did Puma first appear?
   - Which brand appeared more?
   - Compare average confidence
   - Show the highest-confidence Puma detections
   - Which 5-second interval had the most detections?

The analytics layer uses read-only SQL and approved views only.

## Dataset Preparation

### Clean Raw Exports

```bash
python scripts/make_adidas_puma_subset.py \
  --raw-root /path/to/raw/roboflow_exports \
  --out datasets/adidas_puma_clean
```

This step:

- finds raw Roboflow datasets under the provided root
- keeps Adidas and Puma labels only
- normalizes class names
- writes a cleaned YOLO dataset with `train`, `valid`, and `test` splits

### Balance the Dataset

```bash
python scripts/make_balanced_subset.py
```

This copies from:

- `datasets/adidas_puma_clean`

and writes to:

- `datasets/adidas_puma_balanced`

### Visualize Samples

```bash
python scripts/visualize_yolo_samples.py
```

This generates annotated debug samples for manual inspection.

## Training

Train using the default balanced dataset:

```bash
python train_yolo.py
```

Defaults:

- Base model: `yolo26n.pt`
- Dataset: `datasets/adidas_puma_balanced/data.yaml`
- Epochs: `100`
- Image size: `640`
- Device: `mps`

Override them with CLI flags:

```bash
python train_yolo.py --model yolo26n.pt --data datasets/adidas_puma_balanced/data.yaml --epochs 100 --imgsz 640 --device cpu
```

## Output Locations

- Streamlit runs:
  - `outputs/streamlit_runs/<run_id>/`
- CLI inference:
  - `outputs/logo_inference/`
- SQLite database:
  - user-configured path, defaulting to `outputs/streamlit_runs/video_detections.sqlite3`

Each run produces:

- annotated MP4
- detections CSV
- SQLite run row
- SQLite detection rows

## Troubleshooting

- If the video player is disabled in Streamlit, check the codec used for the annotated MP4 and rerun inference.
- If YouTube downloads fail, confirm `yt-dlp` is installed and the URL is accessible.
- If analytics fails, make sure `HF_TOKEN` is set.
- If SQLite queries fail, delete the old database file and rerun inference to recreate the schema cleanly.
- If the model weights cannot be found, place `best.pt` in the project root or pass `--weights`.

## Notes

- The app is designed to run locally with no Postgres dependency.
- Analytics are intentionally read-only.
- The SQLite schema is created automatically on first run.
