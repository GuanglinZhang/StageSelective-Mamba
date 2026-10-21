#!/usr/bin/env python3
"""Instantiate the final model and verify its deployment identity."""
import torch

from models import UltraLightDynamicStemAll6SMBStageScaleFinal


def main():
    model = UltraLightDynamicStemAll6SMBStageScaleFinal().eval()
    parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if parameters != 506990:
        raise RuntimeError(f"Expected 506990 trainable parameters, found {parameters}")
    with torch.no_grad():
        output = model(torch.randn(1, 3, 256, 256))
    if tuple(output.shape) != (1, 1, 256, 256):
        raise RuntimeError(f"Unexpected output shape: {tuple(output.shape)}")
    print(f"PASS parameters={parameters} output_shape={tuple(output.shape)}")


if __name__ == "__main__":
    main()
