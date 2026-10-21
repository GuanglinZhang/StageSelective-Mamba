"""Frozen final UltraLight VM-UNet: DynamicStem + six SMB + Stage-Scale skip.

This file intentionally contains no research candidates other than the retained
five-scalar auxiliary skip calibration.
"""
import torch
import torch.nn.functional as F
from torch import nn

from models.smb_dynamic_stem_all6_e456_d123 import SMBDynamicStemAll6E456D123UNet


class StageScaleSkip(nn.Module):
    def __init__(self):
        super().__init__()
        self.alphas = nn.ParameterList([nn.Parameter(torch.zeros(())) for _ in range(5)])

    def scales(self):
        return tuple(1.0 + torch.tanh(alpha) for alpha in self.alphas)

    def forward(self, *skips):
        return tuple(scale * skip for scale, skip in zip(self.scales(), skips))


class UltraLightDynamicStemAll6SMBStageScaleFinal(SMBDynamicStemAll6E456D123UNet):
    final_model_name = "ULTRALIGHT_DYNAMIC_STEM_ALL6_SMB_STAGE_SCALE_FINAL"
    parameter_count = 506990

    def __init__(self, num_classes=1, input_channels=3,
                 c_list=[8, 16, 24, 32, 48, 64], split_att="fc", bridge=True):
        super().__init__(num_classes, input_channels, c_list, split_att, bridge)
        self.stage_scale_skip = StageScaleSkip()

    def forward(self, x):
        out = F.gelu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2)); t1 = out
        out = F.gelu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2)); t2 = out
        out = F.gelu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2)); t3 = out
        out = F.gelu(F.max_pool2d(self.ebn4(self.encoder4(out)), 2, 2)); t4 = out
        out = F.gelu(F.max_pool2d(self.ebn5(self.encoder5(out)), 2, 2)); t5 = out
        if self.bridge:
            t1, t2, t3, t4, t5 = self.scab(t1, t2, t3, t4, t5)
        t1, t2, t3, t4, t5 = self.stage_scale_skip(t1, t2, t3, t4, t5)
        out = F.gelu(self.encoder6(out))

        out5 = F.gelu(self.dbn1(self.decoder1(out)))
        out5 = torch.add(out5, t5)
        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)), scale_factor=(2, 2), mode="bilinear", align_corners=True))
        out4 = torch.add(out4, t4)
        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)), scale_factor=(2, 2), mode="bilinear", align_corners=True))
        out3 = torch.add(out3, t3)
        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)), scale_factor=(2, 2), mode="bilinear", align_corners=True))
        out2 = torch.add(out2, t2)
        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)), scale_factor=(2, 2), mode="bilinear", align_corners=True))
        out1 = torch.add(out1, t1)
        out0 = F.interpolate(self.final(out1), scale_factor=(2, 2), mode="bilinear", align_corners=True)
        return torch.sigmoid(out0)
