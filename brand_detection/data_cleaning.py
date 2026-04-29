from __future__ import annotations

import argparse
import csv
import hashlib
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CANONICAL_NAMES = {
    0: "adidas",
    1: "puma",
}

BRAND_TO_ID = {
    "adidas": 0,
    "puma": 1,
}


def normalize_name(name: str) -> str:
    s = str(name).lower().strip()
    for ch in ["_", "-", ".", "/", "\\"]:
        s = s.replace(ch, " ")
    return " ".join(s.split())


def map_to_brand(class_name: str) -> Optional[str]:
    s = normalize_name(class_name)

    adidas_aliases = [
        "adidas",
        "addidas",
        "adiddas",
        "adidas logo",
        "adidas text",
        "adidas_text",
    ]

    puma_aliases = [
        "puma",
        "puma logo",
        "puma text",
        "puma_text",
    ]

    if any(alias.replace("_", " ") in s for alias in adidas_aliases):
        return "adidas"

    if any(alias.replace("_", " ") in s for alias in puma_aliases):
        return "puma"

    return None


def load_class_names(data_yaml_path: Path) -> Dict[int, str]:
    with data_yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data.get("names")
    if names is None:
        raise ValueError(f"No 'names' found in {data_yaml_path}")

    if isinstance(names, list):
        return {i: str(name) for i, name in enumerate(names)}

    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}

    raise ValueError(f"Unsupported names format in {data_yaml_path}: {type(names)}")


def find_dataset_roots(raw_root: Path) -> List[Path]:
    yaml_files = list(raw_root.rglob("data.yaml"))
    roots = sorted({p.parent for p in yaml_files})
    return roots


def find_split_dirs(dataset_root: Path) -> List[Tuple[str, Path, Path]]:
    split_names = ["train", "valid", "val", "test"]
    found = []

    for split in split_names:
        image_dir = dataset_root / split / "images"
        label_dir = dataset_root / split / "labels"

        if image_dir.exists() and label_dir.exists():
            canonical_split = "valid" if split == "val" else split
            found.append((canonical_split, image_dir, label_dir))

    return found


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_and_filter_label_file(
    label_path: Path,
    old_to_new_class: Dict[int, int],
) -> Tuple[List[str], Counter]:
    cleaned_lines = []
    stats = Counter()

    if not label_path.exists():
        stats["missing_label_file"] += 1
        return cleaned_lines, stats

    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()

        if len(parts) < 5:
            stats["invalid_short_line"] += 1
            continue

        try:
            old_class_id = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:5])
        except ValueError:
            stats["invalid_number"] += 1
            continue

        if old_class_id not in old_to_new_class:
            stats["non_target_box"] += 1
            continue

        if not (
            0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0
        ):
            stats["invalid_yolo_box"] += 1
            continue

        new_class_id = old_to_new_class[old_class_id]
        cleaned_lines.append(f"{new_class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        stats[f"box_{CANONICAL_NAMES[new_class_id]}"] += 1

    return cleaned_lines, stats


def split_records(
    records: List[dict], train_ratio: float, valid_ratio: float, seed: int
) -> Dict[str, List[dict]]:
    random.seed(seed)

    groups = defaultdict(list)

    for rec in records:
        class_ids = set(int(line.split()[0]) for line in rec["label_lines"])
        if class_ids == {0}:
            group_key = "adidas"
        elif class_ids == {1}:
            group_key = "puma"
        else:
            group_key = "mixed"
        groups[group_key].append(rec)

    split_map = {"train": [], "valid": [], "test": []}

    for _, group_records in groups.items():
        random.shuffle(group_records)
        n = len(group_records)

        n_train = int(n * train_ratio)
        n_valid = int(n * valid_ratio)

        if n >= 10:
            n_train = max(1, n_train)
            n_valid = max(1, n_valid)
        else:
            n_train = max(1, n - 1)
            n_valid = 0

        train_records = group_records[:n_train]
        valid_records = group_records[n_train : n_train + n_valid]
        test_records = group_records[n_train + n_valid :]

        split_map["train"].extend(train_records)
        split_map["valid"].extend(valid_records)
        split_map["test"].extend(test_records)

    for split in split_map:
        random.shuffle(split_map[split])

    return split_map


def reset_output_dir(out_root: Path) -> None:
    if out_root.exists():
        shutil.rmtree(out_root)

    for split in ["train", "valid", "test"]:
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_data_yaml(out_root: Path) -> None:
    data = {
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/valid",
        "test": "images/test",
        "names": {
            0: "adidas",
            1: "puma",
        },
    }

    with (out_root / "data.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        required=True,
        help="Folder containing all downloaded Roboflow datasets",
    )
    parser.add_argument("--out", required=True, help="Output folder for cleaned Adidas/Puma dataset")
    parser.add_argument("--train", type=float, default=0.80)
    parser.add_argument("--valid", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-dedupe", action="store_true", help="Disable exact image duplicate removal"
    )
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    out_root = Path(args.out)

    if not raw_root.exists():
        raise FileNotFoundError(f"Raw root not found: {raw_root}")

    if args.train + args.valid >= 1.0:
        raise ValueError("--train + --valid must be less than 1.0 so test split can exist")

    dataset_roots = find_dataset_roots(raw_root)

    if not dataset_roots:
        raise RuntimeError(f"No data.yaml files found under {raw_root}")

    print(f"Found {len(dataset_roots)} dataset(s):")
    for root in dataset_roots:
        print(f"  - {root}")

    all_records = []
    global_stats = Counter()
    seen_hashes = set()

    for dataset_root in dataset_roots:
        data_yaml_path = dataset_root / "data.yaml"
        class_names = load_class_names(data_yaml_path)

        old_to_new_class = {}
        for old_id, old_name in class_names.items():
            brand = map_to_brand(old_name)
            if brand is not None:
                old_to_new_class[old_id] = BRAND_TO_ID[brand]

        if not old_to_new_class:
            print(f"\nSkipping {dataset_root}: no Adidas/Puma class found in data.yaml")
            continue

        print(f"\nUsing {dataset_root}")
        print("Class mapping:")
        for old_id, new_id in old_to_new_class.items():
            print(
                f"  {old_id}:{class_names[old_id]} -> {new_id}:{CANONICAL_NAMES[new_id]}"
            )

        split_dirs = find_split_dirs(dataset_root)
        if not split_dirs:
            print("  Warning: no train/valid/test image+label folders found")
            continue

        for source_split, image_dir, label_dir in split_dirs:
            for image_path in sorted(image_dir.iterdir()):
                if image_path.suffix.lower() not in IMAGE_EXTS:
                    continue

                label_path = label_dir / f"{image_path.stem}.txt"
                cleaned_lines, label_stats = parse_and_filter_label_file(
                    label_path, old_to_new_class
                )
                global_stats.update(label_stats)

                if not cleaned_lines:
                    global_stats["image_without_target_boxes"] += 1
                    continue

                image_hash = sha1_file(image_path)

                if not args.no_dedupe:
                    if image_hash in seen_hashes:
                        global_stats["duplicate_image_skipped"] += 1
                        continue
                    seen_hashes.add(image_hash)

                safe_dataset_name = dataset_root.name.replace(" ", "_")
                new_stem = f"{safe_dataset_name}_{source_split}_{image_path.stem}_{image_hash[:8]}"
                new_image_name = f"{new_stem}{image_path.suffix.lower()}"
                new_label_name = f"{new_stem}.txt"

                all_records.append(
                    {
                        "src_image": image_path,
                        "new_image_name": new_image_name,
                        "new_label_name": new_label_name,
                        "label_lines": cleaned_lines,
                        "source_dataset": dataset_root.name,
                        "source_split": source_split,
                    }
                )

    if not all_records:
        raise RuntimeError("No Adidas/Puma images found. Check class names in data.yaml files.")

    split_map = split_records(
        records=all_records,
        train_ratio=args.train,
        valid_ratio=args.valid,
        seed=args.seed,
    )

    reset_output_dir(out_root)

    rows = []
    final_stats = Counter()

    for split, records in split_map.items():
        for rec in records:
            dst_image = out_root / "images" / split / rec["new_image_name"]
            dst_label = out_root / "labels" / split / rec["new_label_name"]

            shutil.copy2(rec["src_image"], dst_image)
            dst_label.write_text("\n".join(rec["label_lines"]) + "\n", encoding="utf-8")

            box_counts = Counter(int(line.split()[0]) for line in rec["label_lines"])

            final_stats[f"images_{split}"] += 1
            final_stats[f"boxes_{split}_adidas"] += box_counts[0]
            final_stats[f"boxes_{split}_puma"] += box_counts[1]
            final_stats["total_images"] += 1
            final_stats["total_boxes_adidas"] += box_counts[0]
            final_stats["total_boxes_puma"] += box_counts[1]

            rows.append(
                {
                    "split": split,
                    "image": str(dst_image.relative_to(out_root)),
                    "label": str(dst_label.relative_to(out_root)),
                    "source_dataset": rec["source_dataset"],
                    "source_split": rec["source_split"],
                    "adidas_boxes": box_counts[0],
                    "puma_boxes": box_counts[1],
                }
            )

    write_data_yaml(out_root)

    stats_csv = out_root / "dataset_stats.csv"
    with stats_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "image",
                "label",
                "source_dataset",
                "source_split",
                "adidas_boxes",
                "puma_boxes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nDone.")
    print(f"Output dataset: {out_root}")
    print(f"data.yaml: {out_root / 'data.yaml'}")
    print(f"stats CSV: {stats_csv}")

    print("\nFinal clean dataset stats:")
    for key, value in sorted(final_stats.items()):
        print(f"  {key}: {value}")

    print("\nCleaning stats:")
    for key, value in sorted(global_stats.items()):
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
