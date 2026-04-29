from __future__ import annotations

import random
from pathlib import Path

import cv2


DATASET_ROOT = Path("datasets/adidas_puma_clean")
OUT_DIR = Path("debug_samples")
CLASS_NAMES = {
    0: "adidas",
    1: "puma",
}


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    split = "train"
    num_samples = 50

    image_dir = DATASET_ROOT / "images" / split
    label_dir = DATASET_ROOT / "labels" / split

    images = list(image_dir.glob("*"))
    random.shuffle(images)

    saved = 0

    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            continue

        img = cv2.imread(str(image_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        lines = label_path.read_text().strip().splitlines()
        if not lines:
            continue

        for line in lines:
            parts = line.split()
            cls_id = int(parts[0])
            x, y, bw, bh = map(float, parts[1:5])

            x1 = int((x - bw / 2) * w)
            y1 = int((y - bh / 2) * h)
            x2 = int((x + bw / 2) * w)
            y2 = int((y + bh / 2) * h)

            label = CLASS_NAMES.get(cls_id, str(cls_id))

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                label,
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        out_path = OUT_DIR / image_path.name
        cv2.imwrite(str(out_path), img)

        saved += 1
        if saved >= num_samples:
            break

    print(f"Saved {saved} samples to {OUT_DIR}")


if __name__ == "__main__":
    main()
