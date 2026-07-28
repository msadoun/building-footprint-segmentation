# Building Footprint Segmentation

Train and run inference for building footprint segmentation on satellite and aerial imagery.

This repository is a modernized fork of
[fuzailpalnak/building-footprint-segmentation](https://github.com/fuzailpalnak/building-footprint-segmentation)
with CUDA-ready PyTorch support, Windows fixes, and practical CLI scripts for tiling, training, and inference.

![Python](https://img.shields.io/badge/python-v3.9+-blue.svg)
![Licence](https://img.shields.io/github/license/fuzailpalnak/building-footprint-segmentation)

![merge1](https://user-images.githubusercontent.com/24665570/97859410-91fa6100-1d26-11eb-8a47-e41982c748d7.jpg)

---

## What's New (v0.2.5+)

| Area | Details |
|------|---------|
| Python / PyTorch | Python **3.9+**, PyTorch **2.4+** with CUDA |
| torchvision | Updated to `weights=` API |
| Windows | OpenMP workaround for PyTorch + OpenCV/MKL |
| Inference | Padding for non-multiple-of-32 tiles; image-only file filtering |
| Training | Full training CLI, checkpoints, Ultralytics-style `results.png` charts |
| Geospatial prep | Chip GeoTIFF → PNG, rasterize shapefile → masks, spatial train/val split |

### Helper scripts

| Script | Purpose |
|--------|---------|
| `scripts/check_gpu.py` | Verify CUDA / GPU detection |
| `scripts/create_dummy_data.py` | Tiny dummy train/val/test dataset |
| `scripts/chip_geotiff_to_png.py` | Chip a large GeoTIFF into RGB PNG tiles |
| `scripts/rasterize_building_masks.py` | Burn building shapefile into per-tile masks |
| `scripts/prepare_steyr_dataset.py` | Spatial train/val split from tiles + masks |
| `scripts/run_inference_test.py` | Run inference and save `*_mask.png` |
| `scripts/run_training.py` | Fine-tune ReFineNet with metrics charts |
| `scripts/regularize_masks.py` | Post-process raw masks into sharp polygons |
| `scripts/run_training_smoke_test.py` | Short training smoke test |

---

## Installation

### Recommended: conda env + CUDA PyTorch

```bash
conda create -n buildingfp python=3.11 -y
conda activate buildingfp

# Optional GIS tools (GeoTIFF chip / shapefile rasterize)
conda install -c conda-forge gdal -y

# CUDA PyTorch (do not install the default CPU wheel from PyPI)
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-gpu.txt
pip install -e .
python scripts/check_gpu.py
```

Expected GPU check:

```text
PyTorch version: 2.x.x+cu124
Selected device: cuda
GPU 0: NVIDIA ...
```

### CPU only

```bash
pip install -r requirements-cpu.txt
pip install -e .
```

### Requirements files

| File | Contents |
|------|----------|
| `requirements.txt` | Core deps (no PyTorch), includes matplotlib |
| `requirements-cpu.txt` | Core + CPU PyTorch |
| `requirements-gpu.txt` | Core only — install CUDA PyTorch separately |

---

## Data Format

The library expects **aerial / satellite RGB imagery**.
Pretrained `refine.pth` was trained on the
[INRIA Aerial Image Labeling Dataset](https://project.inria.fr/aerialimagelabeling/).

### Inference only

```text
dataset_root/
  test/
    images/
      tile_001.png
      tile_002.png
```

Labels are **not** required. Only image files are loaded (`.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.bmp`, `.webp`). Sidecars such as `*.aux.xml` are ignored.

### Training

```text
dataset_root/
  train/
    images/
      tile_001.png
    labels/
      tile_001.png
  val/
    images/
      tile_101.png
    labels/
      tile_101.png
  test/                 # optional
    images/
      tile_201.png
```

**Pairing rules**

- Image / label pairs must share the **exact same filename**
- Image and mask must have the **same height and width**
- Prefer a **fixed tile size** within a batch (e.g. all `512×512`) so `batch_size > 1` works

### Image / mask types

| Property | Images | Labels |
|----------|--------|--------|
| Content | RGB aerial / satellite | Binary building mask |
| Formats | PNG / JPG / OpenCV-readable TIF | PNG recommended |
| Building pixels | — | White (`255`) |
| Background | — | Black (`0`) |
| Dtype | `uint8` | `uint8` |
| Default norm | `divide_by_255` → `[0, 1]` | `binary_label` → `0.0` / `1.0` |

### Model output

| Stage | Value |
|-------|-------|
| Raw | Logits `[B, 1, H, W]` |
| Post-process | Sigmoid + threshold (default `0.20`) |
| Saved | Single-channel PNG (`0` background, `255` building) |

---

## End-to-end workflow (custom GeoTIFF + shapefile)

Example paths below match a Steyr-style dataset. Adjust paths as needed.

### 1. Chip the GeoTIFF into PNG tiles

```bash
python scripts/chip_geotiff_to_png.py --tile-size 512
```

Writes to `data/steyr_512/test/images/` (tile size is part of the folder name).

```bash
python scripts/chip_geotiff_to_png.py --tile-size 1024
# → data/steyr_1024/test/images/
```

### 2. Rasterize building footprints into masks

```bash
python scripts/rasterize_building_masks.py \
  --geotiff "D:/path/to/area.tif" \
  --shapefile "D:/path/to/buildings.shp" \
  --images data/steyr_512/test/images \
  --labels data/steyr_512/test/labels
```

Creates one label PNG per tile, georeferenced from the source GeoTIFF and tile offsets in names like `tile_1024_2048.png`.

### 3. Build train / validation split

Uses a **spatial** hold-out (eastern tile columns → val) to reduce leakage from neighboring tiles:

```bash
python scripts/prepare_steyr_dataset.py \
  --images data/steyr_512/test/images \
  --labels data/steyr_512/test/labels \
  --output data/steyr_train \
  --val-fraction 0.2 \
  --only-size 512
```

Produces:

```text
data/steyr_train/
  train/images + train/labels
  val/images   + val/labels
```

### 4. Train (fine-tune)

```bash
python scripts/run_training.py \
  --data data/steyr_train \
  --weights refine.pth \
  --output outputs/steyr_training_1000 \
  --epochs 1000 \
  --batch-size 8 \
  --lr 0.0001
```

Default is **1000 epochs**. Training writes:

| Output | Description |
|--------|-------------|
| `results.png` | Raw model metrics chart: loss, accuracy, precision, recall, F1, IoU, LR |
| `results.csv` | Raw model metrics, one row per epoch |
| `results_regularized.png` | Val metrics **after** mask regularization (sharp polygons) |
| `results_regularized.csv` | Regularized metrics, one row per epoch |
| `predictions.png` | Samples: image / GT / raw prediction / **regularized** prediction |
| `<timestamp>/state/best.pt` | Best validation-loss checkpoint |
| `<timestamp>/chk_pth/chk_pth.pt` | Latest model weights |

The chart is refreshed every epoch (also kept if you interrupt with `Ctrl+C`).

### 5. Inference with trained weights

```bash
python scripts/run_inference_test.py \
  --data data/steyr_512 \
  --weights outputs/steyr_training_1000/<timestamp>/chk_pth/chk_pth.pt \
  --output outputs/steyr_preds \
  --threshold 0.20
```

Non-multiple-of-32 edge tiles are padded automatically during inference.

---

## Inference (pretrained only)

```bash
python scripts/create_dummy_data.py
python scripts/run_inference_test.py \
  --data data/dummy \
  --weights refine.pth \
  --output outputs/inference_test \
  --threshold 0.20
```

### Python API

```python
import torch
from albumentations import Compose
from building_footprint_segmentation.segmentation import init_segmentation
from building_footprint_segmentation.utils.py_network import gpu_variable

segmentation = init_segmentation("binary")
model = segmentation.load_model("ReFineNet", transfer_weights="refine.pth")
model.eval()

loader = segmentation.load_loader(
    root_folder="data/my_inference",
    image_normalization="divide_by_255",
    label_normalization="binary_label",
    augmenters=Compose([]),
    batch_size=1,
)

with torch.no_grad():
    for batch in loader.test_loader:
        images = gpu_variable(batch["images"])
        predictions = model(images).sigmoid()
        mask = (predictions >= 0.20).float()
```

---

## Training (smoke test)

```bash
python scripts/create_dummy_data.py --output data/dummy
python scripts/run_training_smoke_test.py \
  --data data/dummy \
  --weights refine.pth \
  --epochs 1 \
  --batch-size 2
```

### Callbacks

| Callback | Role |
|----------|------|
| `MetricsPlotCallback` | `results.csv` + `results.png` |
| `RegularizedMetricsCallback` | `results_regularized.csv` + `results_regularized.png` |
| `PredictionSampleCallback` | `predictions.png` (image / GT / prediction / regularized) |
| `TrainStateCallback` | `best.pt` / `default.pt` training state |
| `TrainChkCallback` | Latest `chk_pth.pt` weights |
| `TensorBoardCallback` | TensorBoard event logs |
| `TimeCallback` | Total run time |

TensorBoard:

```bash
tensorboard --logdir="outputs/steyr_training_1000"
```

---

## Weight Files

| Weights | Source |
|---------|--------|
| `refine.pth` (local) | ReFineNet / INRIA — place in project root |
| [RefineNet (INRIA)](https://github.com/fuzailpalnak/building-footprint-segmentation/releases/download/alpha/refine.zip) | Upstream release zip |
| [DLinkNet (Massachusetts)](https://github.com/fuzailpalnak/building-footprint-segmentation/releases/download/alpha/DlinkNet.zip) | Upstream release zip |

Download example:

```bash
# unzip refine.zip into the project root as refine.pth
```

---

## Models

Binary segmentation models available via `load_model(...)`:

- `ReFineNet` (default / pretrained path)
- `ReFineNetLite`
- `DLinkNet34`
- `AlBuNet`
- `MFRN`

---

## Notes & limitations

- **Domain gap**: pretrained INRIA weights often look poor on new orthophotos until fine-tuned.
- **JPEG2000 (`.jp2`)**: GDAL may need `libgdal-jp2openjpeg`. Prefer chipping from GeoTIFF when JP2 support is missing.
- **GeoTIFF**: OpenCV inference does not preserve CRS. Keep the source georeference / world files if you need to reproject masks.
- **Batch size**: mixed tile sizes → use `batch_size=1` or filter to a fixed size (`--only-size` in dataset prep).
- **Large rasters**: do not feed a multi-GB mosaic to the model whole — chip first.

---

## Segmentation scope

- [x] Binary building footprint
- [ ] Building with boundary (multi-class)

---

## Notebooks / upstream extras

- [Train with config](https://github.com/fuzailpalnak/building-footprint-segmentation/blob/main/examples/Run%20with%20config.ipynb)
- [Train with arguments](https://github.com/fuzailpalnak/building-footprint-segmentation/blob/main/examples/Run%20with%20defined%20arguments.ipynb)
- [Test callback](https://github.com/fuzailpalnak/building-footprint-segmentation/blob/main/examples/TestCallBack.ipynb)
- [Prediction with augmentations](https://github.com/fuzailpalnak/building-footprint-segmentation/blob/main/examples/PredictionWithAugmentations.ipynb)

For additional GeoTIFF utilities, see [gtkit](https://github.com/fuzailpalnak/gtkit).

---

## License

Apache-2.0 (see `LICENSE`).
