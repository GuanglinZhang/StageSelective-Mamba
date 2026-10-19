#!/usr/bin/env python3
"""Evaluate one frozen checkpoint on the ISIC test split."""
import argparse
import json

import torch
from torch.utils.data import DataLoader

from data import ISICNpyDataset
from utils.runtime import build_model, evaluate, load_state, read_config, write_json_new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config = read_config(args.config)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model = build_model()
    incompatible = model.load_state_dict(load_state(args.checkpoint), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Strict checkpoint loading unexpectedly reported incompatible keys")
    model.to(device)
    test_set = ISICNpyDataset(
        args.data_root, "test", config["split_counts"]["test"], augment=False,
    )
    loader = DataLoader(
        test_set, batch_size=1, shuffle=False, pin_memory=device.type == "cuda",
        num_workers=config["num_workers"], drop_last=True,
    )
    result = {
        "dataset": config["dataset"],
        "checkpoint_policy": "EMA minimum validation loss",
        **evaluate(model, loader, device, config["threshold"]),
    }
    write_json_new(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
