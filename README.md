# Stage-Selective SAD-Stem--Mamba

Minimal code release for **Stage-Selective SAD-Stem--Mamba Network for Lightweight Skin Lesion Segmentation**. The released model is `UltraLightDynamicStemAll6SMBStageScaleFinal` (506,990 parameters). This package contains only the final model, data pipeline, formal training/evaluation code, and locked configurations; datasets and checkpoints are not included.

## 1. Environment

The paper environment used Python 3.8.20, CUDA 11.7, PyTorch 1.13.0, and `mamba-ssm` 1.0.1.

```bash
conda create -n stage-selective-mamba python=3.8.20 -y
conda activate stage-selective-mamba
pip install -r requirements.txt
```

## 2. Data

Download ISIC 2017 and ISIC 2018 from the [official ISIC challenge data page](https://challenge.isic-archive.com/data/). For each dataset, place the official training images and Task-1 segmentation masks in separate directories, then create the paper-format NPY arrays:

```bash
python prepare_isic.py --dataset ISIC2017 --images /data/ISIC2017/images --masks /data/ISIC2017/masks --output datasets/isic2017
python prepare_isic.py --dataset ISIC2018 --images /data/ISIC2018/images --masks /data/ISIC2018/masks --output datasets/isic2018
```

Each output directory contains `data_{train,val,test}.npy` and `mask_{train,val,test}.npy`. The locked split counts are 1250/150/600 for ISIC2017 and 1815/259/520 for ISIC2018. Images and masks are resized to 256 x 256; each split is normalized by its global mean and standard deviation and then each image is min-max scaled to [0, 255]. Training uses the mother-project rotation/flip augmentation.

The legacy preparation code divided filename-ordered challenge data at the locked boundaries. If you already have the six arrays used by the mother project, use them directly to retain their exact sample order.

Download PH2 from the [official PH2 page](https://www.fc.up.pt/addi/ph2%20database.html). `test_ph2.py` accepts either the original 200 case folders or one NPY pair named `data_test_PH2.npy`/`mask_test_PH2.npy` (also lowercase `_ph2` or plain `data_test.npy`/`mask_test.npy`). Raw PH2 images use bilinear resize and masks use nearest-neighbor resize. PH2 is inference-only.

## 3. Train

The locked protocol uses seed 42, 250 epochs, batch size 8, FP32, AdamW (`lr=1e-3`, `weight_decay=1e-2`), CosineAnnealingLR (`T_max=50`, `eta_min=1e-5`), BCE+Dice loss, and EMA decay 0.999. EMA is updated after every optimizer step. `checkpoints/best.pth` is selected strictly by minimum EMA validation loss; the test set is never used for selection.

```bash
python train.py --config configs/isic2017.yaml --data-root datasets/isic2017 --output runs/isic2017_r1
python train.py --config configs/isic2018.yaml --data-root datasets/isic2018 --output runs/isic2018_r1
```

For the paper's three repeated runs, execute the same command three times with new output directories (`r1`, `r2`, and `r3`) while keeping seed 42 and all configuration values unchanged.

## 4. Test ISIC2017 and ISIC2018

```bash
python test.py --config configs/isic2017.yaml --data-root datasets/isic2017 --checkpoint runs/isic2017_r1/checkpoints/best.pth --output runs/isic2017_r1/test.json
python test.py --config configs/isic2018.yaml --data-root datasets/isic2018 --checkpoint runs/isic2018_r1/checkpoints/best.pth --output runs/isic2018_r1/test.json
```

## 5. External PH2 test

Use a frozen ISIC2017 `best.pth`; PH2 performs no training, fine-tuning, checkpoint selection, or threshold search.

```bash
python test_ph2.py --ph2-root /data/PH2Dataset --isic2017-checkpoint runs/isic2017_r1/checkpoints/best.pth --output runs/isic2017_r1/ph2.json
```

All evaluations use batch size 1, threshold 0.5, and globally accumulated pixel counts for DSC, mIoU, Sensitivity, Specificity, and Accuracy. Checkpoints are loaded with `strict=True`. Use `--device cuda:1` if the desired GPU is not CUDA device 0.
