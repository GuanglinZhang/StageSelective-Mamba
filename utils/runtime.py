import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .losses import BceDiceLoss
from .metrics import ConfusionMeter


LOCKED = {
    "seed": 42,
    "epochs": 250,
    "batch_size": 8,
    "num_workers": 0,
    "amp": False,
    "ema_decay": 0.999,
    "checkpoint_policy": "ema_minimum_validation_loss",
    "threshold": 0.5,
}


def read_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key, expected in LOCKED.items():
        if config.get(key) != expected:
            raise RuntimeError(f"Locked setting {key} must be {expected!r}, got {config.get(key)!r}")
    if config.get("input_size") != 256:
        raise RuntimeError("The input size is locked to 256")
    if config.get("optimizer") != {"name": "AdamW", "lr": 0.001, "weight_decay": 0.01}:
        raise RuntimeError("Optimizer configuration differs from the paper protocol")
    if config.get("scheduler") != {"name": "CosineAnnealingLR", "T_max": 50, "eta_min": 0.00001}:
        raise RuntimeError("Scheduler configuration differs from the paper protocol")
    return config


def build_model():
    # Import lazily so CLI help and configuration checks remain available on
    # machines where the optional CUDA/Mamba stack has not yet been installed.
    from models import UltraLightDynamicStemAll6SMBStageScaleFinal

    model = UltraLightDynamicStemAll6SMBStageScaleFinal(
        num_classes=1,
        input_channels=3,
        c_list=[8, 16, 24, 32, 48, 64],
        split_att="fc",
        bridge=True,
    )
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != 506990:
        raise RuntimeError(f"Model parameter mismatch: expected 506990, got {parameters}")
    return model


def load_state(path):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict"):
            if key in state:
                state = state[key]
                break
    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a state dictionary")
    return {(name[7:] if name.startswith("module.") else name): value for name, value in state.items()}


def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    criterion = BceDiceLoss()
    meter = ConfusionMeter(threshold)
    losses = []
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=device.type == "cuda").float()
            masks = masks.to(device, non_blocking=device.type == "cuda").float()
            predictions = model(images)
            losses.append(criterion(predictions, masks).item())
            meter.update(predictions, masks)
    if not losses:
        raise RuntimeError("Evaluation loader is empty")
    result = {"loss": float(np.mean(losses)), "threshold": float(threshold)}
    result.update(meter.compute())
    return result


def write_json_new(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
