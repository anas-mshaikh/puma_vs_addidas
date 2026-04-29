from pathlib import Path
import random
import shutil
from collections import Counter
import sys

SRC = Path("datasets/adidas_puma_clean")
DST = Path("datasets/adidas_puma_balanced")

SEED = 42
KEEP_ALL_PUMA_IMAGES = True

random.seed(SEED)


def count_classes(label_path: Path):
    counts = Counter()
    for line in label_path.read_text().splitlines():
        if line.strip():
            counts[int(line.split()[0])] += 1
    return counts


def reset_dst():
    if DST.exists():
        shutil.rmtree(DST)

    for split in ["train", "valid", "test"]:
        (DST / "images" / split).mkdir(parents=True, exist_ok=True)
        (DST / "labels" / split).mkdir(parents=True, exist_ok=True)


def copy_pair(image_path: Path, label_path: Path, split: str):
    shutil.copy2(image_path, DST / "images" / split / image_path.name)
    shutil.copy2(label_path, DST / "labels" / split / label_path.name)


def write_yaml():
    yaml_text = f"""path: {DST.resolve()}
train: images/train
val: images/valid
test: images/test

names:
  0: adidas
  1: puma
"""
    (DST / "data.yaml").write_text(yaml_text)


def print_progress(prefix: str, current: int, total: int) -> None:
    if total <= 0:
        return
    step = max(1, total // 100)
    if current != total and current % step != 0:
        return

    width = 28
    filled = int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    percent = 100 * current / total
    sys.stderr.write(f"\r{prefix} [{bar}] {current}/{total} ({percent:5.1f}%)".ljust(80))
    sys.stderr.flush()


reset_dst()

summary = Counter()

for split in ["train", "valid", "test"]:
    image_dir = SRC / "images" / split
    label_dir = SRC / "labels" / split
    image_lookup = {image_path.stem: image_path for image_path in image_dir.iterdir() if image_path.is_file()}

    puma_images = []
    mixed_images = []
    adidas_only_images = []

    label_paths = list(label_dir.glob("*.txt"))
    total_labels = len(label_paths)

    for idx, label_path in enumerate(label_paths, start=1):
        print_progress(f"Scanning {split}", idx, total_labels)
        image_path = image_lookup.get(label_path.stem)
        if not image_path:
            continue

        class_counts = count_classes(label_path)

        has_adidas = class_counts[0] > 0
        has_puma = class_counts[1] > 0

        if has_puma and has_adidas:
            mixed_images.append((image_path, label_path, class_counts[0], class_counts[1]))
        elif has_puma:
            puma_images.append((image_path, label_path, class_counts[0], class_counts[1]))
        elif has_adidas:
            adidas_only_images.append((image_path, label_path, class_counts[0], class_counts[1]))

    random.shuffle(adidas_only_images)
    adidas_only_images.sort(key=lambda item: item[2])

    fixed_adidas_boxes = sum(item[2] for item in puma_images + mixed_images)
    fixed_puma_boxes = sum(item[3] for item in puma_images + mixed_images)
    adidas_budget = max(0, fixed_puma_boxes - fixed_adidas_boxes)

    selected_adidas = []
    used_adidas_boxes = 0
    for item in adidas_only_images:
        adidas_boxes = item[2]
        if used_adidas_boxes + adidas_boxes <= adidas_budget or (
            not selected_adidas and adidas_budget > 0
        ):
            selected_adidas.append(item)
            used_adidas_boxes += adidas_boxes

    if not KEEP_ALL_PUMA_IMAGES:
        selected_puma = []
    else:
        selected_puma = puma_images

    selected = selected_puma + mixed_images + selected_adidas
    random.shuffle(selected)

    total_selected = len(selected)
    for idx, (image_path, label_path, adidas_boxes, puma_boxes) in enumerate(selected, start=1):
        print_progress(f"Copying  {split}", idx, total_selected)
        copy_pair(image_path, label_path, split)

        summary[f"{split}_adidas"] += adidas_boxes
        summary[f"{split}_puma"] += puma_boxes
        summary[f"{split}_images"] += 1

    sys.stderr.write("\n")

write_yaml()

print("Balanced dataset created:", DST)
for k, v in sorted(summary.items()):
    print(k, v)
