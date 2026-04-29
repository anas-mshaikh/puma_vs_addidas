from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import streamlit as st

from infer_logo_video import run_video_inference


APP_TITLE = "Adidas / Puma Video Inference"
DEFAULT_WEIGHTS = Path("best.pt")
OUTPUT_ROOT = Path("outputs/streamlit_runs")


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
    st.set_page_config(
        page_title=APP_TITLE,
        layout="wide",
    )

    st.title(APP_TITLE)
    st.caption("Run YOLO inference on a YouTube video or local file and preview the annotated result.")

    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    with st.sidebar:
        st.header("Settings")

        source_mode = st.radio(
            "Source type",
            ["YouTube URL", "Local video"],
            index=0,
        )

        youtube_url = st.text_input(
            "YouTube URL",
            value="https://youtu.be/obv2j5P2kys?si=8IWQr6FOtVncSzGG",
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

        weights = st.text_input(
            "Weights path",
            value=str(DEFAULT_WEIGHTS),
        )

        out_root = st.text_input(
            "Output folder",
            value=str(OUTPUT_ROOT),
        )

        store_to_postgres = st.checkbox("Store detections in Postgres", value=False)
        postgres_dsn = st.text_input(
            "Postgres DSN",
            value=os.environ.get("DATABASE_URL", ""),
            placeholder="postgresql://user:password@host:5432/dbname",
            disabled=not store_to_postgres,
        )
        postgres_table = st.text_input(
            "Postgres table",
            value="video_detections",
            disabled=not store_to_postgres,
        )

        with st.expander("Advanced"):
            conf = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.01)
            iou = st.slider("IoU threshold", 0.05, 0.95, 0.45, 0.01)
            imgsz = st.select_slider("Image size", options=[320, 416, 512, 640, 768, 960], value=640)
            device = st.text_input("Device", value="mps", help="Leave empty for auto, or use cpu / mps / 0.")

        run_clicked = st.button("Run inference", type="primary", use_container_width=True)

    if run_clicked:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = Path(out_root).expanduser() / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            if store_to_postgres and not postgres_dsn.strip():
                raise ValueError("Enable Postgres only after providing a Postgres DSN.")

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
                output_video_path, output_csv_path, detection_count, input_video_path = run_video_inference(
                    weights=Path(weights),
                    source=source,
                    out_dir=run_dir,
                    conf_threshold=conf,
                    iou_threshold=iou,
                    imgsz=int(imgsz),
                    device=device.strip() or None,
                    progress_callback=on_progress,
                    postgres_dsn=postgres_dsn.strip() if store_to_postgres else None,
                    postgres_table=postgres_table.strip() or "video_detections",
                )

            st.session_state.last_result = {
                "run_dir": run_dir,
                "input_video_path": input_video_path,
                "output_video_path": output_video_path,
                "output_csv_path": output_csv_path,
                "detection_count": detection_count,
                "postgres_enabled": store_to_postgres,
                "postgres_table": postgres_table.strip() or "video_detections",
            }

            progress_bar.progress(100)
            progress_placeholder.write("Processing complete.")
            st.success("Inference finished.")

        except Exception as exc:
            st.session_state.last_result = None
            st.error(str(exc))

    result = st.session_state.last_result

    if result:
        st.subheader("Result")
        cols = st.columns(3)
        cols[0].metric("Detections", result["detection_count"])
        cols[1].metric("Run folder", str(result["run_dir"]))
        cols[2].metric("Input", str(result["input_video_path"]))

        if result.get("postgres_enabled"):
            st.info(f"Detections were also written to Postgres table `{result['postgres_table']}`.")

        st.video(str(result["output_video_path"]))

        video_bytes = Path(result["output_video_path"]).read_bytes()
        csv_bytes = Path(result["output_csv_path"]).read_bytes()

        download_cols = st.columns(2)
        download_cols[0].download_button(
            "Download annotated video",
            data=video_bytes,
            file_name=Path(result["output_video_path"]).name,
            mime="video/mp4",
            use_container_width=True,
        )
        download_cols[1].download_button(
            "Download detections CSV",
            data=csv_bytes,
            file_name=Path(result["output_csv_path"]).name,
            mime="text/csv",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
