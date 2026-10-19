#!/usr/bin/env python3
"""Train the final model and select the EMA checkpoint with minimum validation loss."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import ISICNpyDataset
from utils.ema import EMAShadow
from utils.losses import BceDiceLoss
from utils.reproducibility import set_seed
from utils.runtime import build_model, evaluate, read_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="ISIC2017 or ISIC2018 YAML config")
    parser.add_argument("--data-root", required=True, help="Directory containing six split NPY files")
    parser.add_argument("--output", required=True, help="New run directory")
    args = parser.parse_args()

    config = read_config(args.config)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {output}")
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    if not torch.cuda.is_available():
        raise RuntimeError("Training the formal Mamba model requires a CUDA GPU")

    set_seed(config["seed"])
    counts = config["split_counts"]
    train_set = ISICNpyDataset(args.data_root, "train", counts["train"], augment=True)
    val_set = ISICNpyDataset(args.data_root, "val", counts["val"], augment=False)
    train_loader = DataLoader(
        train_set, batch_size=config["batch_size"], shuffle=True,
        pin_memory=True, num_workers=config["num_workers"], drop_last=False,
    )
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False,
        pin_memory=True, num_workers=config["num_workers"], drop_last=True,
    )

    device = torch.device("cuda:0")
    raw_model = build_model().to(device)
    model = torch.nn.DataParallel(raw_model, device_ids=[0], output_device=0)
    ema = EMAShadow(model.module, config["ema_decay"])
    criterion = BceDiceLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.001, betas=(0.9, 0.999),
        eps=1e-8, weight_decay=0.01, amsgrad=False,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=50, eta_min=1e-5,
    )

    best = None
    optimizer_steps = 0
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        training_losses = []
        for images, masks in tqdm(train_loader, desc=f"epoch {epoch}/{config['epochs']}"):
            images = images.to(device, non_blocking=True).float()
            masks = masks.to(device, non_blocking=True).float()
            optimizer.zero_grad()
            loss = criterion(model(images), masks)
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"Non-finite training loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            ema.update(model.module)
            training_losses.append(loss.item())
        scheduler.step()

        validation = evaluate(ema.module, val_loader, device, config["threshold"])
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(training_losses)),
            "ema_val_loss": validation["loss"],
            **{f"ema_val_{key}": value for key, value in validation.items() if key not in {"loss", "threshold"}},
        }
        print(json.dumps(record, sort_keys=True), flush=True)
        if best is None or validation["loss"] < best["val_loss"]:
            best = {"epoch": epoch, "val_loss": validation["loss"], **validation}
            torch.save(ema.module.state_dict(), checkpoint_dir / "best.pth")

    if ema.updates != optimizer_steps:
        raise RuntimeError("EMA update count differs from optimizer step count")
    selection = {
        "dataset": config["dataset"],
        "checkpoint": "checkpoints/best.pth",
        "checkpoint_policy": "EMA minimum validation loss",
        "ema_decay": config["ema_decay"],
        "ema_updates": ema.updates,
        "optimizer_steps": optimizer_steps,
        "test_based_selection": False,
        "selected_validation": best,
    }
    (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
