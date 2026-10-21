"""Position ablation: frozen DynamicStem plus frozen SMB at all six former PVM sites.

PSMamba source:
Medical Image Registration via Spatial Feature Extraction Mamba and Substrate Iterative Refinement
Zilong Xue, Kangjian He, Dan Xu, Jian Gong; IET Image Processing.
DOI: 10.1049/ipr2.70117; https://github.com/VCMHE/PSMamba

The source SMB is adapted from 3D to 2D and mapped to the six original PVM
channel-transition positions of UltraLight VM-UNet. It is not claimed as a new SMB.
"""
from torch import nn

from models.smb_dynamic_stem_e123_formal import SMBDynamicStemE123UNet
from models.source_faithful_blocks import PSMambaTransition2D


class SMBDynamicStemAll6E456D123UNet(SMBDynamicStemE123UNet):
    module_mining_variant = "SMB_DYNAMIC_STEM_ALL6_E456_D123"
    source_fidelity = "FROZEN_WINNER_PLUS_ENCODER_POSITION_ABLATION"
    source_paper = ("Medical Image Registration via Spatial Feature Extraction Mamba "
                    "and Substrate Iterative Refinement")
    source_doi = "10.1049/ipr2.70117"
    source_code = "https://github.com/VCMHE/PSMamba"
    ultralight_mapping = "E4/E5/E6/D1/D2/D3; 3D SMB adapted to 2D"

    def __init__(self, num_classes=1, input_channels=3,
                 c_list=[8, 16, 24, 32, 48, 64], split_att="fc", bridge=True):
        super().__init__(num_classes, input_channels, c_list, split_att, bridge)
        # The sole variable versus the formal winner: replace Official E4/E5/E6 PVM.
        self.encoder4 = nn.Sequential(PSMambaTransition2D(c_list[2], c_list[3]))
        self.encoder5 = nn.Sequential(PSMambaTransition2D(c_list[3], c_list[4]))
        self.encoder6 = nn.Sequential(PSMambaTransition2D(c_list[4], c_list[5]))
