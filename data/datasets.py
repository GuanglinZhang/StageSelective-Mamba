import random
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from torch.utils.data import Dataset


def dataset_normalized(images):
    """Formal preprocessing: split-wide z-score, then per-image min-max to [0, 255]."""
    images = np.asarray(images)
    standard_deviation = np.std(images)
    if not np.isfinite(standard_deviation) or standard_deviation == 0:
        raise ValueError("Image array has an invalid global standard deviation")
    normalized = (images - np.mean(images)) / standard_deviation
    for index in range(len(normalized)):
        low = np.min(normalized[index])
        high = np.max(normalized[index])
        if not np.isfinite(low) or not np.isfinite(high) or high == low:
            raise ValueError(f"Image {index} is constant or non-finite")
        normalized[index] = (normalized[index] - low) / (high - low) * 255.0
    return normalized


def _mask_scale(mask):
    if not np.isfinite(mask).all():
        raise ValueError("Mask contains non-finite values")
    low, high = float(np.min(mask)), float(np.max(mask))
    if low < 0 or high > 255:
        raise ValueError(f"Mask range [{low}, {high}] is invalid")
    return 1.0 if high <= 1.0 else 255.0


class ISICNpyDataset(Dataset):
    def __init__(self, root, split, expected_count, augment=False):
        root = Path(root)
        image_path = root / f"data_{split}.npy"
        mask_path = root / f"mask_{split}.npy"
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(f"Expected {image_path.name} and {mask_path.name} in {root}")
        images = np.load(image_path, allow_pickle=False)
        masks = np.load(mask_path, allow_pickle=False)
        if images.shape != (expected_count, 256, 256, 3):
            raise ValueError(f"Unexpected {image_path.name} shape: {images.shape}")
        if masks.shape != (expected_count, 256, 256):
            raise ValueError(f"Unexpected {mask_path.name} shape: {masks.shape}")
        self.images = dataset_normalized(images)
        self.masks = np.expand_dims(masks / _mask_scale(masks), axis=3)
        self.augment = bool(augment)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        mask = self.masks[index]
        if self.augment and random.random() > 0.5:
            turns = np.random.randint(0, 4)
            axis = np.random.randint(0, 2)
            image = np.flip(np.rot90(image, turns), axis=axis).copy()
            mask = np.flip(np.rot90(mask, turns), axis=axis).copy()
        if self.augment and random.random() > 0.5:
            angle = np.random.randint(20, 80)
            image = ndimage.rotate(image, angle, order=0, reshape=False)
            mask = ndimage.rotate(mask, angle, order=0, reshape=False)
        image = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)
        mask = torch.from_numpy(np.ascontiguousarray(mask)).permute(2, 0, 1)
        return image, mask


def _case_id(path):
    match = re.search(r"(?i)IMD\s*0*([0-9]+)", str(path))
    return f"IMD{int(match.group(1)):03d}" if match else None


def _load_raw_ph2(root):
    image_paths = {}
    mask_paths = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".bmp", ".jpg", ".jpeg", ".png"}:
            continue
        case = _case_id(path)
        if case is None:
            continue
        is_mask = any(word in str(path).lower() for word in ("mask", "lesion", "groundtruth", "segmentation"))
        target = mask_paths if is_mask else image_paths
        if case in target:
            raise RuntimeError(f"Duplicate PH2 {'mask' if is_mask else 'image'} for {case}")
        target[case] = path
    if set(image_paths) != set(mask_paths) or len(image_paths) != 200:
        raise RuntimeError(
            f"PH2 must contain exactly 200 paired cases; found {len(image_paths)} images and {len(mask_paths)} masks"
        )
    bilinear = getattr(Image, "Resampling", Image).BILINEAR
    nearest = getattr(Image, "Resampling", Image).NEAREST
    images, masks = [], []
    for case in sorted(image_paths):
        with Image.open(image_paths[case]) as image:
            images.append(np.asarray(image.convert("RGB").resize((256, 256), bilinear), dtype=np.float64))
        with Image.open(mask_paths[case]) as mask:
            masks.append(np.asarray(mask.convert("L").resize((256, 256), nearest), dtype=np.float64))
    return np.stack(images), np.stack(masks)


def _load_ph2(root):
    candidates = (
        ("data_test_PH2.npy", "mask_test_PH2.npy"),
        ("data_test_ph2.npy", "mask_test_ph2.npy"),
        ("data_test.npy", "mask_test.npy"),
    )
    present = [(root / a, root / b) for a, b in candidates if (root / a).is_file() and (root / b).is_file()]
    if len(present) > 1:
        raise RuntimeError("Multiple PH2 NPY pairs found; keep exactly one pair")
    if present:
        images = np.load(present[0][0], allow_pickle=False)
        masks = np.load(present[0][1], allow_pickle=False)
    else:
        images, masks = _load_raw_ph2(root)
    if images.shape != (200, 256, 256, 3):
        raise ValueError(f"Unexpected PH2 image shape: {images.shape}")
    if masks.shape == (200, 256, 256, 1):
        masks = masks[..., 0]
    if masks.shape != (200, 256, 256):
        raise ValueError(f"Unexpected PH2 mask shape: {masks.shape}")
    return images, masks


class PH2Dataset(Dataset):
    def __init__(self, root):
        images, masks = _load_ph2(Path(root))
        self.images = dataset_normalized(images)
        self.masks = np.expand_dims(masks / _mask_scale(masks), axis=3)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = torch.from_numpy(np.ascontiguousarray(self.images[index])).permute(2, 0, 1)
        mask = torch.from_numpy(np.ascontiguousarray(self.masks[index])).permute(2, 0, 1)
        return image, mask
