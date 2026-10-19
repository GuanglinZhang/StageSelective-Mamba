#!/usr/bin/env python3
"""Inference-only PH2 external test using a frozen ISIC2017 checkpoint."""
import argparse
import json

import torch
from torch.utils.data import DataLoader

from data import PH2Dataset
from utils.runtime import build_model, evaluate, load_state, write_json_new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ph2-root", required=True)
    parser.add_argument("--isic2017-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = build_model()
    incompatible = model.load_state_dict(load_state(args.isic2017_checkpoint), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Strict checkpoint loading unexpectedly reported incompatible keys")
    model.to(device)
    dataset = PH2Dataset(args.ph2_root)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, pin_memory=device.type == "cuda",
        num_workers=0, drop_last=False,
    )
    result = {
        "dataset": "PH2",
        "source_dataset": "ISIC2017",
        "checkpoint_policy": "frozen ISIC2017 EMA minimum validation loss",
        "ph2_training": False,
        "ph2_fine_tuning": False,
        "ph2_checkpoint_selection": False,
        "ph2_threshold_search": False,
        **evaluate(model, loader, device, threshold=0.5),
    }
    write_json_new(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
