"""Split chipped tiles + masks into train/val folders for training."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2

TILE_PREFIX = "tile_"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images",
        default="data/steyr_512/test/images",
        help="Folder with tile PNG images",
    )
    parser.add_argument(
        "--labels",
        default="data/steyr_512/test/labels",
        help="Folder with matching label PNG masks",
    )
    parser.add_argument(
        "--output",
        default="data/steyr_train",
        help="Dataset root with train/ and val/ splits (default: data/steyr_train)",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of tile columns held out for validation (default: 0.2)",
    )
    parser.add_argument(
        "--only-size",
        type=int,
        default=512,
        help="Only include tiles with this width and height (default: 512). Use 0 to keep all.",
    )
    return parser.parse_args()


def tile_x_offset(path: Path) -> int:
    parts = path.stem.split("_")
    if len(parts) < 3 or parts[0] != "tile":
        raise ValueError(f"Unexpected tile name: {path.name}")
    return int(parts[1])


def copy_pair(image_path: Path, label_path: Path, split: str, output_root: Path) -> None:
    image_dest = output_root / split / "images" / image_path.name
    label_dest = output_root / split / "labels" / label_path.name
    image_dest.parent.mkdir(parents=True, exist_ok=True)
    label_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, image_dest)
    shutil.copy2(label_path, label_dest)


def main() -> None:
    args = parse_args()
    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    output_root = Path(args.output)

    if not images_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels folder not found: {labels_dir}")

    image_paths = sorted(images_dir.glob(f"{TILE_PREFIX}*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No tiles found in {images_dir}")

    x_values = sorted({tile_x_offset(path) for path in image_paths})
    val_column_count = max(1, round(len(x_values) * args.val_fraction))
    val_x_values = set(x_values[-val_column_count:])

    train_count = 0
    val_count = 0
    skipped = 0
    size_skipped = 0

    for image_path in image_paths:
        if args.only_size:
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Skipping (unreadable): {image_path.name}")
                skipped += 1
                continue
            height, width = image.shape[:2]
            if width != args.only_size or height != args.only_size:
                size_skipped += 1
                continue

        label_path = labels_dir / image_path.name
        if not label_path.exists():
            print(f"Skipping (missing label): {image_path.name}")
            skipped += 1
            continue

        split = "val" if tile_x_offset(image_path) in val_x_values else "train"
        copy_pair(image_path, label_path, split, output_root)
        if split == "train":
            train_count += 1
        else:
            val_count += 1

    if train_count == 0 or val_count == 0:
        raise RuntimeError(
            f"Split failed: train={train_count}, val={val_count}. "
            "Try a smaller --val-fraction or check tile names."
        )

    print(f"Output: {output_root.resolve()}")
    print(f"Train tiles: {train_count}")
    print(f"Val tiles: {val_count}")
    print(f"Skipped (missing labels): {skipped}")
    if args.only_size:
        print(f"Skipped (not {args.only_size}x{args.only_size}): {size_skipped}")
    print(f"Val held-out x columns ({len(val_x_values)}): {sorted(val_x_values)}")


if __name__ == "__main__":
    main()
