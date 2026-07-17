"""Run a quick GPU inference smoke test on dummy (or real) test images."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch
from albumentations import Compose

from building_footprint_segmentation._env import configure_windows_openmp
from building_footprint_segmentation.segmentation import init_segmentation
from building_footprint_segmentation.utils.py_network import convert_tensor_to_numpy, gpu_variable

configure_windows_openmp()


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="data/dummy",
        help="Dataset root with test/images (default: data/dummy)",
    )
    parser.add_argument(
        "--weights",
        default=str(project_root / "refine.pth"),
        help="Path to model weights (default: refine.pth in project root)",
    )
    parser.add_argument(
        "--output",
        default="outputs/inference_test",
        help="Folder for predicted masks (default: outputs/inference_test)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.20,
        help="Mask threshold after sigmoid (default: 0.20)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data)
    weights_path = Path(args.weights)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (data_root / "test" / "images").exists():
        raise FileNotFoundError(
            f"Missing {data_root / 'test' / 'images'}. "
            "Run: python scripts/create_dummy_data.py"
        )
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    segmentation = init_segmentation("binary")
    model = segmentation.load_model("ReFineNet", transfer_weights=str(weights_path))
    model.eval()

    loader = segmentation.load_loader(
        root_folder=str(data_root),
        image_normalization="divide_by_255",
        label_normalization="binary_label",
        augmenters=Compose([]),
        batch_size=1,
    )

    print(f"Running inference on {len(loader.test_loader.dataset)} test image(s)...")

    with torch.no_grad():
        for batch in loader.test_loader:
            images = gpu_variable(batch["images"])
            predictions = model(images).sigmoid()
            predictions = (predictions >= args.threshold).float()

            file_names = batch["file_name"]
            if isinstance(file_names, str):
                file_names = [file_names]

            batch_size, _, height, width = predictions.shape
            for index in range(batch_size):
                mask = convert_tensor_to_numpy(predictions[index]).reshape(height, width)
                mask_image = (mask * 255).astype("uint8")
                output_path = output_dir / f"{Path(file_names[index]).stem}_mask.png"
                cv2.imwrite(str(output_path), mask_image)
                print(f"Saved: {output_path}")

    print(f"Done. Predictions are in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
