#!/usr/bin/env python3
"""Prepare the six ISIC NPY files used by the training and evaluation scripts."""
import argparse
import csv
import glob
from pathlib import Path

import numpy as np
from PIL import Image


COUNTS = {
    "ISIC2017": {"train": 1250, "val": 150, "test": 600},
    "ISIC2018": {"train": 1815, "val": 259, "test": 520},
}


def _available_images(directory):
    paths = [p for p in Path(directory).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    mapping = {p.stem: p for p in paths}
    if len(mapping) != len(paths):
        raise RuntimeError("Duplicate image stems were found")
    return mapping


def _read_manifest(path, images, dataset):
    """Read ordered rows with columns `split,image_id`."""
    rows = list(csv.DictReader(Path(path).open(encoding="utf-8-sig", newline="")))
    if not rows or not {"split", "image_id"}.issubset(rows[0]):
        raise ValueError("Split manifest must contain columns: split,image_id")
    ordered = {split: [] for split in ("train", "val", "test")}
    used = set()
    for row in rows:
        split, image_id = row["split"].strip().lower(), row["image_id"].strip()
        if split not in ordered:
            raise ValueError(f"Invalid split {split!r}")
        if image_id not in images:
            raise FileNotFoundError(f"Manifest image not found: {image_id}")
        if image_id in used:
            raise ValueError(f"Duplicate manifest image: {image_id}")
        ordered[split].append(images[image_id]); used.add(image_id)
    expected = COUNTS[dataset]
    actual = {key: len(value) for key, value in ordered.items()}
    if actual != expected:
        raise RuntimeError(f"Manifest counts {actual} do not match locked counts {expected}")
    if len(used) != sum(expected.values()):
        raise RuntimeError("Manifest does not contain the complete locked split")
    return ordered, "provided_manifest"


def _ultralight_legacy_split(image_directory, images, dataset):
    """Reproduce the original UltraLight VM-UNet glob-order split."""
    expected = COUNTS[dataset]
    paths = [Path(path) for path in glob.glob(str(Path(image_directory) / "*.jpg"))]
    if len(paths) != sum(expected.values()):
        raise RuntimeError(f"Expected {sum(expected.values())} JPG images, found {len(paths)}")
    if any(path.stem not in images for path in paths):
        raise RuntimeError("Legacy glob returned an image absent from the indexed image directory")
    train_end = expected["train"]
    val_end = train_end + expected["val"]
    return {
        "train": paths[:train_end],
        "val": paths[train_end:val_end],
        "test": paths[val_end:],
    }, "ultralight_vmunet_legacy_unsorted_glob_order"


def _sorted_reconstruction(images, dataset):
    expected = COUNTS[dataset]
    paths = [images[key] for key in sorted(images)]
    if len(paths) != sum(expected.values()):
        raise RuntimeError(f"Expected {sum(expected.values())} images, found {len(paths)}")
    train_end = expected["train"]
    val_end = train_end + expected["val"]
    return {
        "train": paths[:train_end],
        "val": paths[train_end:val_end],
        "test": paths[val_end:],
    }, "sorted_filename_reconstruction_NOT_verified_as_paper_split"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(COUNTS))
    parser.add_argument("--images", required=True, help="Official challenge image directory")
    parser.add_argument("--masks", required=True, help="Official challenge Task-1 mask directory")
    parser.add_argument("--output", required=True, help="New directory for six NPY files")
    parser.add_argument("--split-manifest", help="CSV with ordered columns split,image_id")
    parser.add_argument(
        "--split-mode", choices=("ultralight-legacy", "sorted"), default="ultralight-legacy",
        help="Without a manifest, reproduce UltraLight VM-UNet's unsorted glob split (default), or use a portable filename-sorted split",
    )
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing path: {output}")
    images = _available_images(args.images)
    if args.split_manifest:
        split_paths, split_source = _read_manifest(args.split_manifest, images, args.dataset)
    elif args.split_mode == "ultralight-legacy":
        split_paths, split_source = _ultralight_legacy_split(args.images, images, args.dataset)
        print("Using the original UltraLight VM-UNet unsorted glob-order split and recording the resulting IDs")
    else:
        split_paths, split_source = _sorted_reconstruction(images, args.dataset)
        print("Using a filename-sorted split; it is portable but is not the legacy UltraLight VM-UNet ordering")

    mask_dir = Path(args.masks)
    bilinear = getattr(Image, "Resampling", Image).BILINEAR
    output.mkdir(parents=True)
    manifest_rows = []
    for split in ("train", "val", "test"):
        paths = split_paths[split]
        data = np.empty((len(paths), 256, 256, 3), dtype=np.float64)
        masks = np.empty((len(paths), 256, 256), dtype=np.float64)
        for index, image_path in enumerate(paths):
            mask_path = mask_dir / f"{image_path.stem}_segmentation.png"
            if not mask_path.is_file():
                raise FileNotFoundError(mask_path)
            with Image.open(image_path) as image:
                data[index] = np.asarray(image.convert("RGB").resize((256, 256), bilinear), dtype=np.float64)
            with Image.open(mask_path) as mask:
                # The original mother-project preparation used bilinear interpolation for ISIC masks.
                masks[index] = np.asarray(mask.convert("L").resize((256, 256), bilinear), dtype=np.float64)
            manifest_rows.append({"split": split, "image_id": image_path.stem})
        np.save(output / f"data_{split}.npy", data)
        np.save(output / f"mask_{split}.npy", masks)
    with (output / "split_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "image_id"])
        writer.writeheader(); writer.writerows(manifest_rows)
    (output / "SPLIT_SOURCE.txt").write_text(split_source + "\n", encoding="utf-8")
    print(f"Prepared {args.dataset} in {output}; split_source={split_source}")


if __name__ == "__main__":
    main()
