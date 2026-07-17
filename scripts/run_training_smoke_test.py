"""Run a 1-epoch training smoke test on dummy data (optional)."""

from __future__ import annotations

import argparse
from pathlib import Path

import albumentations as A

from building_footprint_segmentation._env import configure_windows_openmp
from building_footprint_segmentation.helpers.callbacks import CallbackList, TimeCallback
from building_footprint_segmentation.segmentation import init_segmentation
from building_footprint_segmentation.trainer import Trainer

configure_windows_openmp()


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="data/dummy",
        help="Dataset root (default: data/dummy)",
    )
    parser.add_argument(
        "--weights",
        default=str(project_root / "refine.pth"),
        help="Optional pretrained weights (default: refine.pth)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs to run (default: 1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size (default: 2)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data)
    if not (data_root / "train" / "images").exists():
        raise FileNotFoundError(
            f"Missing training data at {data_root}. "
            "Run: python scripts/create_dummy_data.py"
        )

    segmentation = init_segmentation("binary")
    weights = str(args.weights) if Path(args.weights).exists() else None
    model = segmentation.load_model("ReFineNet", transfer_weights=weights)
    criterion = segmentation.load_criterion(name="BinaryCrossEntropy")
    loader = segmentation.load_loader(
        root_folder=str(data_root),
        image_normalization="divide_by_255",
        label_normalization="binary_label",
        augmenters=A.Compose(
            [
                A.HorizontalFlip(p=0.5),
            ]
        ),
        batch_size=args.batch_size,
    )
    metrics = segmentation.load_metrics(
        data_metrics=["accuracy", "precision", "f1", "recall", "iou"]
    )
    optimizer = segmentation.load_optimizer(model, name="Adam", lr=1e-4)
    callbacks = CallbackList([TimeCallback(log_dir="outputs/training_smoke_test")])

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loader=loader,
        metrics=metrics,
        callbacks=callbacks,
        scheduler=None,
    )

    print(f"Starting smoke training for {args.epochs} epoch(s)...")
    trainer.train(start_epoch=0, end_epoch=args.epochs)
    print("Training smoke test complete.")


if __name__ == "__main__":
    main()
