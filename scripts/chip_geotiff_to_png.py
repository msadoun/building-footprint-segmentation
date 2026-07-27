"""Chip a large GeoTIFF into RGB PNG tiles for inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from osgeo import gdal

gdal.UseExceptions()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=r"D:\sadoun\Devs\TestAreaSteyr\Tif\TestAreaSteyr.tif",
        help="Path to source GeoTIFF",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Folder for PNG tiles. "
            "Default: data/steyr_<tile-size>/test/images"
        ),
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=1024,
        help="Tile width/height in pixels (default: 1024)",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=256,
        help="Skip edge tiles smaller than this (default: 256)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output or f"data/steyr_{args.tile_size}/test/images")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    ds = gdal.Open(str(input_path))
    if ds is None:
        raise RuntimeError(f"Could not open: {input_path}")

    width = ds.RasterXSize
    height = ds.RasterYSize
    bands = ds.RasterCount
    print(f"Opened: {input_path}")
    print(f"Size: {width} x {height}, bands: {bands}")

    if bands < 3:
        raise RuntimeError(f"Need at least 3 bands for RGB, found {bands}")

    tile = args.tile_size
    min_size = args.min_size
    written = 0

    for y in range(0, height, tile):
        for x in range(0, width, tile):
            win_w = min(tile, width - x)
            win_h = min(tile, height - y)
            if win_w < min_size or win_h < min_size:
                continue

            out_path = output_dir / f"tile_{x}_{y}.png"
            gdal.Translate(
                str(out_path),
                ds,
                format="PNG",
                srcWin=[x, y, win_w, win_h],
                bandList=[1, 2, 3],
            )
            written += 1
            print(f"Saved: {out_path} ({win_w}x{win_h})")

    print(f"Done. Wrote {written} tiles to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
