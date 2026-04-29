from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import streamlit as st

from .video import run_video_inference


APP_TITLE = "Adidas / Puma Video Inference"
DEFAULT_WEIGHTS = Path("best.pt")
OUTPUT_ROOT = Path("outputs/streamlit_runs")
DEFAULT_SQLITE_DB = Path(
    os.environ.get("SQLITE_DB_PATH", "outputs/streamlit_runs/video_detections.sqlite3")
)
DEFAULT_SQLITE_TABLE = "video_detections"


def save_uploaded_video(uploaded_file, run_dir: Path) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    input_path = run_dir / f"uploaded_input{suffix}"
    input_path.write_bytes(uploaded_file.getbuffer())
    return input_path


def resolve_source(
    source_mode: str,
    youtube_url: str,
    local_path: str,
    uploaded_file,
    run_dir: Path,
) -> str:
    if source_mode == "YouTube URL":
        source = youtube_url.strip()
        if not source:
            raise ValueError("Provide a YouTube URL.")
        return source

    if uploaded_file is not None:
        return str(save_uploaded_video(uploaded_file, run_dir))

    source = local_path.strip()
    if not source:
        raise ValueError("Provide a local video path or upload a file.")

    video_path = Path(source).expanduser()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    return str(video_path)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption(
        "Run YOLO inference on a YouTube video or local file and preview the annotated result."
    )

    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    col_left, col_right = st.columns([2, 1], gap="large")

    with col_left:
        st.subheader("Source")
        source_mode = st.radio(
            "Source type", ["YouTube URL", "Local video"], index=0, horizontal=True
        )

        youtube_url = st.text_input(
            "YouTube URL",
            value="https://youtu.be/obv2j5P2kys?si=8IWQr6FOtVncSzGG",
            placeholder="https://youtu.be/...",
            disabled=source_mode != "YouTube URL",
        )

        local_path = st.text_input(
            "Local video path",
            placeholder="/absolute/path/to/video.mp4",
            disabled=source_mode != "Local video",
        )

        uploaded_file = None
        if source_mode == "Local video":
            uploaded_file = st.file_uploader(
                "Or upload a video",
                type=["mp4", "mov", "avi", "mkv", "m4v"],
            )

        run_clicked = st.button(
            "Run inference", type="primary", use_container_width=True
        )

    with col_right:
        with st.expander("Advanced settings", expanded=True):
            conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.01)
            iou = st.slider("IoU threshold", 0.05, 0.95, 0.45, 0.01)
            imgsz = st.select_slider(
                "Image size", options=[320, 416, 512, 640, 768, 960], value=640
            )
            device = st.text_input(
                "Device", value="", help="Leave empty for auto, or use cpu / mps / 0."
            )
            sqlite_db_path = st.text_input(
                "SQLite DB path",
                value=str(DEFAULT_SQLITE_DB),
                placeholder="outputs/streamlit_runs/video_detections.sqlite3",
                help="Local SQLite file used to store detection rows.",
            ).strip()
            st.caption("SQLite writes are local and do not require network access.")

    if run_clicked:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = OUTPUT_ROOT / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            source = resolve_source(
                source_mode=source_mode,
                youtube_url=youtube_url,
                local_path=local_path,
                uploaded_file=uploaded_file,
                run_dir=run_dir,
            )

            progress_placeholder = st.empty()
            progress_bar = st.progress(0)

            def on_progress(done: int, total: int, detections: int) -> None:
                if total > 0:
                    percent = int((done / total) * 100)
                    progress_bar.progress(min(100, percent))
                progress_placeholder.write(
                    f"Processed {done}/{total} frames. Detections: {detections}"
                )

            with st.spinner("Processing video..."):
                (
                    output_video_path,
                    output_csv_path,
                    detection_count,
                    input_video_path,
                ) = run_video_inference(
                    weights=DEFAULT_WEIGHTS,
                    source=source,
                    out_dir=run_dir,
                    conf_threshold=conf,
                    iou_threshold=iou,
                    imgsz=int(imgsz),
                    device=device.strip() or None,
                    progress_callback=on_progress,
                    sqlite_db_path=Path(sqlite_db_path) if sqlite_db_path else None,
                    sqlite_table=DEFAULT_SQLITE_TABLE,
                )

            st.session_state.last_result = {
                "run_dir": run_dir,
                "input_video_path": input_video_path,
                "output_video_path": output_video_path,
                "output_csv_path": output_csv_path,
                "detection_count": detection_count,
                "sqlite_enabled": bool(sqlite_db_path),
                "sqlite_db_path": sqlite_db_path,
                "sqlite_table": DEFAULT_SQLITE_TABLE,
            }

            progress_bar.progress(100)
            progress_placeholder.write("Processing complete.")
            st.success("Inference finished.")

        except Exception as exc:
            st.session_state.last_result = None
            st.error(str(exc))

    result = st.session_state.last_result

    if result:
        st.subheader("Annotated Video")

        output_video_path = Path(result["output_video_path"])
        if not output_video_path.exists():
            st.error(f"Annotated video was not created: {output_video_path}")
            return

        video_bytes = output_video_path.read_bytes()
        if not video_bytes:
            st.error(f"Annotated video is empty: {output_video_path}")
            return

        st.video(video_bytes, format="video/mp4")
        st.caption(f"Saved to {output_video_path}")

        with st.expander("Run details", expanded=False):
            cols = st.columns(3)
            cols[0].metric("Detections", result["detection_count"])
            cols[1].metric("Run folder", str(result["run_dir"]))
            cols[2].metric("Input", str(result["input_video_path"]))

            if result.get("sqlite_enabled"):
                st.info(
                    f"Detections were also written to SQLite table `{result['sqlite_table']}` in `{result['sqlite_db_path']}`."
                )


if __name__ == "__main__":
    main()
