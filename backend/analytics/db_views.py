from __future__ import annotations

import sqlite3


def create_analytics_views(db_path: str) -> None:
    conn = sqlite3.connect(db_path)

    try:
        conn.executescript(
            """
            CREATE VIEW IF NOT EXISTS v_detections AS
            SELECT
                d.id AS detection_id,
                d.run_id,
                r.source_type,
                r.source_value,
                r.output_video_path,
                d.frame_index,
                d.timestamp_sec,
                ROUND(d.timestamp_sec, 2) AS timestamp_rounded,
                LOWER(d.brand) AS brand,
                d.confidence,
                d.x1,
                d.y1,
                d.x2,
                d.y2,
                d.box_width,
                d.box_height,
                COALESCE(d.area, d.box_width * d.box_height) AS box_area
            FROM detections d
            JOIN inference_runs r ON r.id = d.run_id;

            CREATE VIEW IF NOT EXISTS v_brand_summary AS
            SELECT
                run_id,
                brand,
                COUNT(*) AS detection_count,
                ROUND(AVG(confidence), 4) AS avg_confidence,
                ROUND(MIN(timestamp_sec), 2) AS first_seen_sec,
                ROUND(MAX(timestamp_sec), 2) AS last_seen_sec,
                ROUND(MAX(timestamp_sec) - MIN(timestamp_sec), 2) AS visible_span_sec,
                ROUND(AVG(box_area), 2) AS avg_box_area,
                ROUND(SUM(confidence * box_area), 2) AS exposure_score
            FROM v_detections
            GROUP BY run_id, brand;

            CREATE VIEW IF NOT EXISTS v_first_appearance AS
            SELECT
                run_id,
                brand,
                ROUND(MIN(timestamp_sec), 2) AS first_seen_sec
            FROM v_detections
            GROUP BY run_id, brand;

            CREATE VIEW IF NOT EXISTS v_timeline_5s AS
            SELECT
                run_id,
                brand,
                CAST(timestamp_sec / 5 AS INTEGER) * 5 AS bucket_start_sec,
                CAST(timestamp_sec / 5 AS INTEGER) * 5 + 5 AS bucket_end_sec,
                COUNT(*) AS detection_count,
                ROUND(AVG(confidence), 4) AS avg_confidence
            FROM v_detections
            GROUP BY run_id, brand, CAST(timestamp_sec / 5 AS INTEGER);
            """
        )
        conn.commit()
    finally:
        conn.close()
