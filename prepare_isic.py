#!/usr/bin/env python3
"""Create the six legacy NPY split files from official ISIC image/mask folders."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


COUNTS = {
    "ISIC2017": (1250, 150, 600),
    "ISIC2018": (1815, 259, 520),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(COUNTS))
    parser.add_argument("--images", required=True, help="Official challenge image directory")
    parser.add_argument("--masks", required=True, help="Official challenge segmentation-mask directory")
    parser.add_argument("--output", required=True, help="New directory for the six NPY files")
    args = parser.parse_args()

    image_dir, mask_dir, output = Path(args.images), Path(args.masks), Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing path: {output}")
    image_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    expected = sum(COUNTS[args.dataset])
    if len(image_paths) != expected:
        raise RuntimeError(f"Expected {expected} images, found {len(image_paths)}")

    bilinear = getattr(Image, "Resampling", Image).BILINEAR
    images = np.empty((expected, 256, 256, 3), dtype=np.float64)
    masks = np.empty((expected, 256, 256), dtype=np.float64)
    for index, image_path in enumerate(image_paths):
        mask_path = mask_dir / f"{image_path.stem}_segmentation.png"
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)
        with Image.open(image_path) as image:
            images[index] = np.asarray(image.convert("RGB").resize((256, 256), bilinear), dtype=np.float64)
        with Image.open(mask_path) as mask:
            # The mother preparation code used bilinear interpolation for ISIC masks.
            masks[index] = np.asarray(mask.convert("L").resize((256, 256), bilinear), dtype=np.float64)

    output.mkdir(parents=True)
    train, val, test = COUNTS[args.dataset]
    boundaries = {"train": (0, train), "val": (train, train + val), "test": (train + val, expected)}
    for split, (start, stop) in boundaries.items():
        np.save(output / f"data_{split}.npy", images[start:stop])
        np.save(output / f"mask_{split}.npy", masks[start:stop])
    print(f"Prepared {args.dataset}: train={train}, val={val}, test={test} in {output}")


if __name__ == "__main__":
    main()
