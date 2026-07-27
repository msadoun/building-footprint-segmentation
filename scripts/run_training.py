"""Fine-tune ReFineNet on a prepared train/val dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import albumentations as A

from building_footprint_segmentation._env import configure_windows_openmp
from building_footprint_segmentation.helpers.callbacks import (
    CallbackList,
    MetricsPlotCallback,
    TimeCallback,
    TrainChkCallback,
    TrainStateCallback,
)
from building_footprint_segmentation.segmentation import init_segmentation
from building_footprint_segmentation.trainer import Trainer

configure_windows_openmp()


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default="data/steyr_train",
        help="Dataset root with train/ and val/ splits",
    )
    parser.add_argument(
        "--weights",
        default=str(project_root / "refine.pth"),
        help="Pretrained weights for fine-tuning (default: refine.pth)",
    )
    parser.add_argument(
        "--output",
        default="outputs/steyr_training",
        help="Folder for checkpoints and logs",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of training epochs (default: 1000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (default: 8)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Adam learning rate (default: 1e-4)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data)
    output_dir = Path(args.output)
    weights_path = Path(args.weights)

    train_images = data_root / "train" / "images"
    val_images = data_root / "val" / "images"
    train_labels = data_root / "train" / "labels"
    val_labels = data_root / "val" / "labels"

    if not train_images.exists() or not val_images.exists():
        raise FileNotFoundError(
            f"Missing train/val folders under {data_root}. "
            "Run scripts/prepare_steyr_dataset.py first."
        )
    if not train_labels.exists() or not val_labels.exists():
        raise FileNotFoundError(
            f"Missing train/val labels under {data_root}. "
            "Run scripts/rasterize_building_masks.py first."
        )

    weights = str(weights_path) if weights_path.exists() else None
    if weights is None:
        print(f"Warning: weights not found at {weights_path}, training from scratch.")

    output_dir.mkdir(parents=True, exist_ok=True)

    segmentation = init_segmentation("binary")
    model = segmentation.load_model("ReFineNet", transfer_weights=weights)
    criterion = segmentation.load_criterion(name="BinaryCrossEntropy")
    loader = segmentation.load_loader(
        root_folder=str(data_root),
        image_normalization="divide_by_255",
        label_normalization="binary_label",
        augmenters=A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
            ]
        ),
        batch_size=args.batch_size,
    )
    metrics = segmentation.load_metrics(
        data_metrics=["accuracy", "precision", "f1", "recall", "iou"]
    )
    optimizer = segmentation.load_optimizer(model, name="Adam", lr=args.lr)

    callbacks = CallbackList(
        [
            TimeCallback(log_dir=str(output_dir)),
            TrainStateCallback(log_dir=str(output_dir)),
            TrainChkCallback(log_dir=str(output_dir)),
            MetricsPlotCallback(log_dir=str(output_dir)),
        ]
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        loader=loader,
        metrics=metrics,
        callbacks=callbacks,
        scheduler=None,
    )

    print(f"Training data: {data_root.resolve()}")
    print(f"Output: {output_dir.resolve()}")
    print(f"Epochs: {args.epochs}, batch size: {args.batch_size}, lr: {args.lr}")
    print(f"Train tiles: {len(loader.train_loader.dataset)}")
    print(f"Val tiles: {len(loader.val_loader.dataset)}")

    trainer.train(start_epoch=0, end_epoch=args.epochs)

    print("Training complete.")
    print(f"Results chart: {output_dir / 'results.png'}")
    print(f"Results CSV: {output_dir / 'results.csv'}")
    print(f"Best checkpoint: {output_dir / 'state' / 'best.pt'}")
    print(f"Latest weights: {output_dir / 'chk_pth' / 'chk_pth.pt'}")


if __name__ == "__main__":
    main()
