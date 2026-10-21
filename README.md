# Stage-Selective SAD-Stem--Mamba

Official reproducibility code for **Stage-Selective SAD-Stem--Mamba Network for Lightweight Skin Lesion Segmentation**.

The released model is `UltraLightDynamicStemAll6SMBStageScaleFinal` with 506,990 trainable parameters:

- SAD-Stem at encoder stages E1--E3;
- GFC-Mamba at E4--E6 and decoder stages D1--D3 (historical source-code name: SMB/`PSMambaTransition2D`);
- five learnable identity-initialized Stage-Scale skip scalars.

The repository contains only the final model, data preparation/loading, formal training and evaluation code, locked configurations, and a lightweight verification script. It does not contain datasets, checkpoints, experiment logs, comparison models, or ablation code.

## 1. Environment

The formal experiments used Linux, Python 3.8.20, CUDA 11.7, PyTorch 1.13.0+cu117, torchvision 0.14.0+cu117, timm 0.4.12, and mamba-ssm 1.0.1.

```bash
conda create -n stage-selective-mamba python=3.8.20 -y
conda activate stage-selective-mamba
pip install -r requirements.txt
python verify_model.py
```

`verify_model.py` should report exactly 506,990 trainable parameters and output shape `(1, 1, 256, 256)` for a random `(1, 3, 256, 256)` tensor.

## 2. Download the datasets

- ISIC2017 and ISIC2018: [official ISIC Challenge data page](https://challenge.isic-archive.com/data/). Use the Task-1 dermoscopic images and segmentation masks.
- PH2: [official PH2 database page](https://www.fc.up.pt/addi/ph2%20database.html). PH2 permits research/educational use but prohibits redistribution and commercial use; download it directly from the provider.

Raw images are not included in this repository.

## 3. Prepare ISIC2017 and ISIC2018

The training code expects six arrays per dataset:

```text
datasets/isic2017/
  data_train.npy  mask_train.npy
  data_val.npy    mask_val.npy
  data_test.npy   mask_test.npy

datasets/isic2018/
  data_train.npy  mask_train.npy
  data_val.npy    mask_val.npy
  data_test.npy   mask_test.npy
```

The paper follows the original UltraLight VM-UNet preparation protocol: the official training images are read in the legacy unsorted filesystem `glob` order and sliced at fixed boundaries. The locked counts are 1250/150/600 for ISIC2017 and 1815/259/520 for ISIC2018. Images are RGB and masks are grayscale; both are resized to 256 x 256. Run:

```bash
python prepare_isic.py --dataset ISIC2017 --images /data/ISIC2017/images --masks /data/ISIC2017/masks --output datasets/isic2017
python prepare_isic.py --dataset ISIC2018 --images /data/ISIC2018/images --masks /data/ISIC2018/masks --output datasets/isic2018
```

Each invocation records the resulting IDs in `split_manifest.csv`; that manifest should be retained and reused for repeat runs. If an archived ordered manifest is available, pass it explicitly with `--split-manifest`. A portable filename-sorted alternative is also available:

```bash
python prepare_isic.py --dataset ISIC2017 --images /data/ISIC2017/images --masks /data/ISIC2017/masks --split-mode sorted --output datasets/isic2017
```

The sorted alternative has the correct counts but does not reproduce the legacy filesystem ordering. The exact historical ID manifest was not archived, so the repository reproduces the documented UltraLight VM-UNet split procedure but cannot independently prove byte-for-byte identity with the historical machine's filesystem order. The script writes `split_manifest.csv` and `SPLIT_SOURCE.txt` beside the arrays.

The runtime preprocessing matches the formal code: each split is standardized using its own array-wide mean and standard deviation, then each image is min-max scaled to `[0,255]`. Training augmentation independently applies random 90-degree rotation plus flip with probability 0.5 and an integer 20--79 degree nearest-neighbor rotation with probability 0.5. The original ISIC preparation used bilinear resizing for both images and masks; this behavior is retained.

## 4. Train

The locked protocol is seed 42, 250 epochs, batch size 8, `num_workers=0`, FP32, BCE+Dice loss, AdamW (`lr=1e-3`, `weight_decay=1e-2`), CosineAnnealingLR (`T_max=50`, `eta_min=1e-5`), EMA decay 0.999, and threshold 0.5.

```bash
python train.py --config configs/isic2017.yaml --data-root datasets/isic2017 --output runs/isic2017_r1
python train.py --config configs/isic2018.yaml --data-root datasets/isic2018 --output runs/isic2018_r1
```

For the paper's three repetitions, run the same command three times with new output directories (`r1`, `r2`, and `r3`) while keeping seed 42 and all settings unchanged. EMA is updated immediately after each optimizer step. The selected checkpoint is:

```text
checkpoints/ema_best_val_loss.pth
```

It is selected only by minimum validation loss; the test set is not used for checkpoint, run, or threshold selection.

## 5. Evaluate ISIC2017 or ISIC2018

```bash
python test.py --config configs/isic2017.yaml --data-root datasets/isic2017 --checkpoint runs/isic2017_r1/checkpoints/ema_best_val_loss.pth --output runs/isic2017_r1/test.json
python test.py --config configs/isic2018.yaml --data-root datasets/isic2018 --checkpoint runs/isic2018_r1/checkpoints/ema_best_val_loss.pth --output runs/isic2018_r1/test.json
```

Evaluation uses batch size 1, strict checkpoint loading, and threshold 0.5. DSC, mIoU, sensitivity, specificity, and accuracy are computed after accumulating TP/TN/FP/FN over all test pixels. The experiment implementation reports `TP/(TP+FP+FN)` as mIoU.

## 6. Frozen PH2 external test

`test_ph2.py` accepts either the original 200 PH2 case folders or a pair named `data_test_PH2.npy` and `mask_test_PH2.npy`. Raw images use bilinear resizing and masks use nearest-neighbor resizing. The complete PH2 evaluation array supplies the normalization statistics, matching the formal external-test implementation.

```bash
python test_ph2.py --ph2-root /data/PH2Dataset --isic2017-checkpoint runs/isic2017_r1/checkpoints/ema_best_val_loss.pth --output runs/isic2017_r1/ph2.json
```

PH2 uses the frozen ISIC2017 checkpoint. There is no PH2 training, fine-tuning, augmentation, checkpoint selection, or threshold search.

## 7. Repository layout

```text
configs/             locked ISIC training configurations
data/                ISIC and PH2 dataset loaders
models/              final network and required modules
utils/               EMA, loss, metrics, seeding, and runtime helpers
prepare_isic.py      ISIC array/split preparation
train.py             formal EMA training
test.py              ISIC test evaluation
test_ph2.py          frozen ISIC2017-to-PH2 evaluation
verify_model.py      parameter-count and random-forward check
CHECK_REPORT.md      lightweight validation report and known limitations
AVAILABILITY.md      manuscript availability-statement drafts
```

See `CHECK_REPORT.md` before publishing or resubmitting.
