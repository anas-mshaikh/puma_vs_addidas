# mobileFirst

Adidas / Puma logo detection with:
- balanced dataset generation
- YOLO video inference
- Streamlit UI
- optional SQLite storage for detections

## Layout

- `brand_detection/`
  - `video.py`: inference pipeline
  - `storage.py`: SQLite persistence
  - `cli.py`: command-line entry point
  - `web.py`: Streamlit app
  - `data_cleaning.py`: build a cleaned Adidas/Puma dataset from raw Roboflow exports
  - `data_balancing.py`: create a balanced subset
  - `visualization.py`: generate annotated debug samples
  - `training.py`: YOLO training entry point
- `scripts/`
  - `infer_logo_video.py`: CLI inference wrapper
  - `make_adidas_puma_subset.py`: dataset cleaning wrapper
  - `make_balanced_subset.py`: dataset balancing wrapper
  - `visualize_yolo_samples.py`: debug sample wrapper
- `streamlit_app.py`: Streamlit inference entry point
- `train_yolo.py`: training entry point

## Run

```bash
pip install -r requirements.txt
python scripts/infer_logo_video.py --source path/to/video.mp4
streamlit run streamlit_app.py
python train_yolo.py
```

## SQLite

Enable SQLite in the Streamlit sidebar and provide a database path like:

```text
outputs/streamlit_runs/video_detections.sqlite3
```
