"""Frozen formal candidate: PSMamba-SMB D1-D3 plus DynamicStem E1-E3."""
from torch import nn

from models.UltraLight_VM_UNet_official import UltraLight_VM_UNet as OfficialUNet
from models.module_mining_blocks import DynamicDepthwiseStem
from models.source_faithful_blocks import PSMambaTransition2D


def _append(stage, block):
    return nn.Sequential(*list(stage.children()), block)


class SMBDynamicStemE123UNet(OfficialUNet):
    module_mining_variant = "SMB_DYNAMIC_STEM_E123"
    source_fidelity = "FROZEN_FORMAL_VALIDATION"

    def __init__(self, num_classes=1, input_channels=3,
                 c_list=[8, 16, 24, 32, 48, 64], split_att="fc", bridge=True):
        super().__init__(num_classes, input_channels, c_list, split_att, bridge)
        self.encoder1 = _append(self.encoder1, DynamicDepthwiseStem(c_list[0]))
        self.encoder2 = _append(self.encoder2, DynamicDepthwiseStem(c_list[1]))
        self.encoder3 = _append(self.encoder3, DynamicDepthwiseStem(c_list[2]))
        self.decoder1 = nn.Sequential(PSMambaTransition2D(c_list[5], c_list[4]))
        self.decoder2 = nn.Sequential(PSMambaTransition2D(c_list[4], c_list[3]))
        self.decoder3 = nn.Sequential(PSMambaTransition2D(c_list[3], c_list[2]))
