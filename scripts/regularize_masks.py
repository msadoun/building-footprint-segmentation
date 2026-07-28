"""Regularize predicted building masks into sharp polygonal footprints."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from building_footprint_segmentation.utils.mask_regularize import regularize_binary_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="outputs/steyr_preds_512",
        help="Folder with raw *_mask.png predictions",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output folder (default: <input>_regularized)",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=64,
        help="Drop components smaller than this area in pixels",
    )
    parser.add_argument(
        "--morph-kernel",
        type=int,
        default=3,
        help="Morphology kernel size (odd integer, default: 3)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output or f"{input_dir}_regularized")
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_paths = sorted(input_dir.glob("*_mask.png"))
    if not mask_paths:
        mask_paths = sorted(input_dir.glob("*.png"))
    if not mask_paths:
        raise FileNotFoundError(f"No PNG masks found in {input_dir}")

    print(f"Input: {input_dir.resolve()} ({len(mask_paths)} masks)")
    print(f"Output: {output_dir.resolve()}")

    for path in mask_paths:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Skipping unreadable: {path.name}")
            continue
        sharp = regularize_binary_mask(
            mask, min_area=args.min_area, morph_kernel=args.morph_kernel
        )
        out_name = path.name.replace("_mask.png", "_mask_regularized.png")
        if out_name == path.name:
            out_name = f"{path.stem}_regularized.png"
        out_path = output_dir / out_name
        cv2.imwrite(str(out_path), sharp)
        print(f"Saved: {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
