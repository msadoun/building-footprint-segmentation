# Building Footprint Segmentation

Library to train and run inference for building footprint segmentation on satellite and aerial imagery.

![Python](https://img.shields.io/badge/python-v3.9+-blue.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)
![Licence](https://img.shields.io/github/license/fuzailpalnak/building-footprint-segmentation)
[![Downloads](https://static.pepy.tech/badge/building-footprint-segmentation)](https://pepy.tech/project/building-footprint-segmentation)

<a href='https://ko-fi.com/fuzailpalnak' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://az743702.vo.msecnd.net/cdn/kofi1.png?v=0' border='0' alt='Buy Me a Coffee at ko-fi.com' /></a>

![merge1](https://user-images.githubusercontent.com/24665570/97859410-91fa6100-1d26-11eb-8a47-e41982c748d7.jpg)

## What's New (v0.2.5)

This fork modernizes the original library for current Python/PyTorch stacks and adds local smoke-test scripts.

### Library upgrades

| Component | Before | Now |
|-----------|--------|-----|
| Python | 3.6+ (loose `~=3.3`) | **3.9+** |
| PyTorch | unpinned / often CPU-only | **2.4+** with CUDA support |
| torchvision | `pretrained=True` (deprecated) | **`weights=` API** via compatibility helper |
| NumPy | 1.19.1 | **1.26+** |
| OpenCV | 4.4.x | **4.9+** |
| albumentations | 1.3.0 | **1.4+** |
| PyYAML | 5.3.1 | **6.0+** |
| scikit-learn | pinned (unused) | **removed** |
| tensorboard | not listed | **added** (optional callbacks) |

### Code improvements

- Explicit GPU device handling (`get_device()`, `.to(device)` instead of hardcoded `.cuda()`)
- `DataParallel` only when multiple GPUs are available
- Safer checkpoint loading (`weights_only=False`, DataParallel key stripping)
- Lazy TensorBoard import (training works without tensorboard unless `TensorBoardCallback` is used)
- Windows OpenMP conflict workaround for PyTorch + MKL/OpenCV
- Bug fixes: metric sigmoid activation, training loss return type, tensor-to-numpy conversion

### New helper scripts

| Script | Purpose |
|--------|---------|
| `scripts/check_gpu.py` | Verify CUDA and GPU detection |
| `scripts/create_dummy_data.py` | Generate a small local test dataset |
| `scripts/run_inference_test.py` | Run inference with pretrained weights |
| `scripts/run_training_smoke_test.py` | Run a short training smoke test |

---

## Installation

### GPU (recommended)

Install the **CUDA build** of PyTorch. A `+cpu` build cannot use the GPU.

**Option A — pip (recommended)**

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-gpu.txt
pip install -e .
python scripts/check_gpu.py
```

**Option B — conda (use Python 3.11)**

On Windows with Python 3.13, `conda install pytorch ...` may resolve to a CPU build from `defaults`. If the install plan shows `pytorch-*-cpu_*`, cancel and use Option A or create a Python 3.11 environment:

```bash
conda create -n buildingfp python=3.11 -y
conda activate buildingfp
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements-gpu.txt
pip install -e .
python scripts/check_gpu.py
```

If conda reports a corrupted package (`InvalidArchiveError`):

```bash
conda clean --packages -y
```

Expected `check_gpu.py` output:

```
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
| `requirements.txt` | Core dependencies (no PyTorch) |
| `requirements-cpu.txt` | Core + CPU PyTorch |
| `requirements-gpu.txt` | Core only — install CUDA PyTorch separately (see above) |

---

## Data Format

The library expects **aerial or satellite RGB imagery**. The pretrained `refine.pth` weights were trained on the [INRIA Aerial Image Labeling Dataset](https://project.inria.fr/aerialimagelabeling/).

### Folder structure

#### Inference only

Labels are **not** required.

```
dataset_root/
  test/
    images/
      tile_001.png
      tile_002.jpg
```

#### Training

Train and validation require **paired** images and masks. Test images are optional.

```
dataset_root/
  train/
    images/
      sample_001.png
      sample_002.png
    labels/
      sample_001.png
      sample_002.png
  val/
    images/
      sample_101.png
    labels/
      sample_101.png
  test/                  # optional
    images/
      sample_201.png
```

**Pairing rules**

- Image/label pairs are matched by **sorted filename** — names must match exactly in `images/` and `labels/`
- Train and val splits must have the same number of files in `images/` and `labels/`
- Image and mask must have the **same height and width** (pixel-aligned)

### Data types

#### Input images (`images/`)

| Property | Inference | Training |
|----------|-----------|----------|
| Content | Aerial / satellite RGB | Aerial / satellite RGB |
| File formats | `.png`, `.jpg`, `.jpeg`, `.tif` (OpenCV-readable) | Same |
| Channels | 3 (RGB) | 3 (RGB) |
| Dtype | `uint8` (0–255) | `uint8` (0–255) |
| Size | Any H×W (fully convolutional model) | Same H×W within a batch |
| Labels required | No | Yes (train + val) |

**Normalization used by default:** `divide_by_255` → float values in `[0, 1]`

**Internal tensor shape:** `[batch, 3, height, width]` (`float32`)

#### Masks / labels (`labels/`) — training only

| Property | Value |
|----------|-------|
| Type | Binary building mask |
| Format | RGB image (PNG recommended); converted to grayscale internally |
| Building pixels | White (`255`) |
| Background pixels | Black (`0`) |
| Size | Must match the paired image (H×W) |

**Normalization used by default:** `binary_label` → single channel, values `0.0` or `1.0`

**Internal tensor shape:** `[batch, 1, height, width]` (`float32`)

#### Model output (inference)

| Property | Value |
|----------|-------|
| Raw output | Logits, shape `[batch, 1, H, W]` |
| After sigmoid + threshold | Binary mask (default threshold: `0.20`) |
| Saved format | Single-channel PNG (`0` = background, `255` = building) |

### Normalization options

Configured when creating the loader:

| Images | Labels | Use case |
|--------|--------|----------|
| `divide_by_255` | `binary_label` | **Default** — used in examples and scripts |
| `divide_255_image_net` | `binary_label` | ImageNet mean/std normalization |
| `min_max_image_net` | `binary_label` | Per-image min-max + ImageNet stats |

### Batch size note

All images in a batch must share the same dimensions. If your dataset has mixed sizes, use `batch_size=1` or tile images to a fixed patch size first (see [Split the images in smaller sample](#split-the-images-in-smaller-sample)).

---

## Inference Guide

### Quick start (CLI)

1. Place images in `test/images/`:

```
data/my_inference/
  test/
    images/
      aerial_01.png
```

2. Run inference with pretrained weights:

```bash
python scripts/run_inference_test.py \
  --data data/my_inference \
  --weights refine.pth \
  --output outputs/inference_test \
  --threshold 0.20
```

3. Masks are saved to `outputs/inference_test/` as `*_mask.png`.

### Quick start (Python)

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

### Dummy data smoke test

```bash
python scripts/create_dummy_data.py
python scripts/check_gpu.py
python scripts/run_inference_test.py
```

### Augmented inference

For test-time augmentation and result aggregation, see the upstream notebook:
[PredictionWithAugmentations.ipynb](https://github.com/fuzailpalnak/building-footprint-segmentation/blob/main/examples/PredictionWithAugmentations.ipynb)

---

## Training Guide

### Quick start (CLI smoke test)

1. Create or prepare a dataset (see [Data Format](#data-format)):

```bash
python scripts/create_dummy_data.py --output data/dummy
```

2. Run a short training test:

```bash
python scripts/run_training_smoke_test.py \
  --data data/dummy \
  --weights refine.pth \
  --epochs 1 \
  --batch-size 2
```

### Quick start (Python)

```python
import albumentations as A
from building_footprint_segmentation.segmentation import init_segmentation
from building_footprint_segmentation.helpers.callbacks import CallbackList, TimeCallback
from building_footprint_segmentation.trainer import Trainer

segmentation = init_segmentation("binary")

augmenters = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
])

model = segmentation.load_model("ReFineNet", transfer_weights="refine.pth")
criterion = segmentation.load_criterion(name="BinaryCrossEntropy")
loader = segmentation.load_loader(
    root_folder="data/my_dataset",
    image_normalization="divide_by_255",
    label_normalization="binary_label",
    augmenters=augmenters,
    batch_size=4,
)
metrics = segmentation.load_metrics(
    data_metrics=["accuracy", "precision", "f1", "recall", "iou"]
)
optimizer = segmentation.load_optimizer(model, name="Adam", lr=1e-4)
callbacks = CallbackList([TimeCallback(log_dir="outputs/training")])

trainer = Trainer(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    loader=loader,
    metrics=metrics,
    callbacks=callbacks,
    scheduler=None,
)

trainer.train(start_epoch=0, end_epoch=10)
```

### Training with notebooks

- [Train With Config](https://github.com/fuzailpalnak/building-footprint-segmentation/blob/main/examples/Run%20with%20config.ipynb) — YAML config; use [config template](https://codebeautify.org/yaml-validator/cbc60637)
- [Train With Arguments](https://github.com/fuzailpalnak/building-footprint-segmentation/blob/main/examples/Run%20with%20defined%20arguments.ipynb)
- [TestCallBack](https://github.com/fuzailpalnak/building-footprint-segmentation/blob/main/examples/TestCallBack.ipynb) — visualize predictions during training

### Supported datasets

- [Massachusetts Buildings Dataset](https://www.cs.toronto.edu/~vmnih/data/)
- [Inria Aerial Image Labeling Dataset](https://project.inria.fr/aerialimagelabeling/)

---

## Visualize Training

### Test images at end of every epoch

Follow [TestCallBack.ipynb](https://github.com/fuzailpalnak/building-footprint-segmentation/blob/main/examples/TestCallBack.ipynb).

### TensorBoard

```python
from building_footprint_segmentation.helpers.callbacks import CallbackList, TensorBoardCallback

where_to_log_the_callback = r"path_to_log_callback"
callbacks = CallbackList()
callbacks.append(TensorBoardCallback(where_to_log_the_callback))
```

```bash
tensorboard --logdir="path_to_log_callback"
```

Requires `pip install tensorboard`.

---

## Defining Custom Callback

```python
from building_footprint_segmentation.helpers.callbacks import CallbackList, Callback

class CustomCallback(Callback):
    def __init__(self, log_dir):
        super().__init__(log_dir)

where_to_log_the_callback = r"path_to_log_callback"
callbacks = CallbackList()
callbacks.append(CustomCallback(where_to_log_the_callback))
```

---

## Split the images in smaller sample

```python
import glob
import os

from image_fragment.fragment import ImageFragment

# FOR .jpg, .png, .jpeg
from imageio import imread, imsave

# FOR .tiff
from tifffile import imread, imsave

ORIGINAL_DIM_OF_IMAGE = (1500, 1500, 3)
CROP_TO_DIM = (384, 384, 3)

image_fragment = ImageFragment.image_fragment_3d(
    fragment_size=(384, 384, 3), org_size=ORIGINAL_DIM_OF_IMAGE
)

IMAGE_DIR = r"pth\to\input\dir"
SAVE_DIR = r"pth\to\save\dir"

for file in glob.glob(
    os.path.join(IMAGE_DIR, "*")
):
    image = imread(file)
    for i, fragment in enumerate(image_fragment):
        fragmented_image = fragment.get_fragment_data(image)

        imsave(
            os.path.join(
                SAVE_DIR,
                f"{i}_{os.path.basename(file)}",
            ),
            fragmented_image,
        )
```

---

## Segmentation for building footprint

- [x] binary
- [ ] building with boundary (multi class segmentation)

---

## Weight Files

- [RefineNet trained on INRIA](https://github.com/fuzailpalnak/building-footprint-segmentation/releases/download/alpha/refine.zip) — also available locally as `refine.pth`
- [DlinkNet trained on Massachusetts Buildings Dataset](https://github.com/fuzailpalnak/building-footprint-segmentation/releases/download/alpha/DlinkNet.zip)

---

## Geospatial utilities

Refer to [gtkit](https://github.com/fuzailpalnak/gtkit) for common GeoTIFF workflows:

- [Generate bitmap from shape file](https://github.com/fuzailpalnak/gtkit/blob/main/tutorials/shpToBitmap.ipynb)
- [Generate shape geometry from geo reference bitmap](https://github.com/fuzailpalnak/gtkit/blob/main/tutorials/bitmapToShp.ipynb)
- [Save Multi Band Imagery](https://github.com/fuzailpalnak/gtkit)
