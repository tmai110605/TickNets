import re
import types

import torch.nn
import torch.nn.init

from .common import conv1x1_block, Classifier,conv3x3_dw_blockAll,conv3x3_block
from .SE_Attention import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class DualScaleDW(nn.Module):
    """
    Channel-split dual-scale depthwise convolution.
    Half channels: 3x3 DW (local), Half channels: 5x5 DW (contextual).
    """
    def __init__(self, channels, stride):
        super().__init__()
        assert channels % 2 == 0
        mid = channels // 2
        self.dw3 = nn.Conv2d(mid, mid, kernel_size=3, stride=stride,
                             padding=1, groups=mid, bias=False)
        self.dw5 = nn.Conv2d(mid, mid, kernel_size=5, stride=stride,
                             padding=2, groups=mid, bias=False)
        self.bn  = nn.BatchNorm2d(channels)

    def forward(self, x):
        mid = x.shape[1] // 2
        x_lo, x_hi = x[:, :mid, :, :], x[:, mid:, :, :]
        x_lo = self.dw3(x_lo)
        x_hi = self.dw5(x_hi)
        x = torch.cat([x_lo, x_hi], dim=1)
        return F.relu6(self.bn(x), inplace=True)


class ComplementaryGatedFusion(nn.Module):
    """
    Generates a per-channel gate g from global context.
    Fuses feature and shortcut as: g*feature + (1-g)*shortcut.
    Uses depthwise 1x1 (per-channel FC) on GAP output — no bottleneck.
    """
    def __init__(self, channels):
        super().__init__()
        # Depthwise 1x1 = per-channel learned weight, zero bias init → g≈0.5 at start
        self.gate_conv = nn.Conv2d(channels, channels, kernel_size=1,
                                   groups=channels, bias=True)
        nn.init.zeros_(self.gate_conv.weight)
        nn.init.zeros_(self.gate_conv.bias)

    def forward(self, feature, shortcut):
        # Global context from transformed feature
        gap = F.adaptive_avg_pool2d(feature, 1)           # (B, C, 1, 1)
        g   = torch.sigmoid(self.gate_conv(gap))          # (B, C, 1, 1)
        return g * feature + (1.0 - g) * shortcut

def conv1x1_bn(in_ch, out_ch, stride=1, activation=True):
    layers = [nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
              nn.BatchNorm2d(out_ch)]
    if activation:
        layers.append(nn.ReLU6(inplace=True))
    return nn.Sequential(*layers)


class MSAG_Block(nn.Module):
    """
    MSAG-Block: Multi-Scale Adaptive Gating Block.
    
    Flow:
        PW1 (mix) → Channel-Split Dual DW (3x3 + 5x5) → BN+ReLU6
                  → PW2 (project) → Complementary Gated Fusion with shortcut
    
    Args:
        in_channels  (int): Input channel count.
        out_channels (int): Output channel count.
        stride       (int): Spatial stride (1 or 2).
    """
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.stride      = stride
        self.in_channels  = in_channels
        self.out_channels = out_channels

        # Step 1: Pointwise mixing (no BN/act — mirrors FR_PDP Pw1)
        self.pw1 = conv1x1_bn(in_channels, in_channels, activation=False)
        # Overwrite BN/act to match original design intent (identity-like start)
        self.pw1 = nn.Conv2d(in_channels, in_channels, 1, bias=False)

        # Step 2+3: Dual-scale depthwise convolution
        self.dual_dw = DualScaleDW(in_channels, stride=stride)

        # Step 4: Pointwise projection
        self.pw2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
            # No activation — CGF acts as the nonlinear gating
        )

        # Step 5: Shortcut path
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        # Step 6: Complementary Gated Fusion
        self.cgf = ComplementaryGatedFusion(out_channels)

    def forward(self, x):
        residual = x

        # Feature transformation path
        out = self.pw1(x)          # pointwise mix
        out = self.dual_dw(out)    # dual-scale DW: 3x3 + 5x5
        out = self.pw2(out)        # project to out_channels

        # Shortcut path
        shortcut = self.shortcut(residual)

        # Complementary gated fusion
        out = self.cgf(out, shortcut)

        return out


# ── Drop-in replacement in TickNet ───────────────────────────────────────────

class TickNet(torch.nn.Module):
    """TickNet with MSAG_Block replacing FR_PDP_block. All other logic identical."""
    def __init__(self, num_classes, init_conv_channels, init_conv_stride,
                 channels, strides, in_channels=3,
                 in_size=(224, 224), use_data_batchnorm=True):
        super().__init__()
        self.use_data_batchnorm = use_data_batchnorm
        self.in_size = in_size
        self.backbone = torch.nn.Sequential()

        if use_data_batchnorm:
            self.backbone.add_module("data_bn",
                                     torch.nn.BatchNorm2d(in_channels))
        self.backbone.add_module("init_conv",
                                 conv3x3_block(in_channels, init_conv_channels,
                                               stride=init_conv_stride))
        in_channels = init_conv_channels

        for stage_id, stage_channels in enumerate(channels):
            stage = torch.nn.Sequential()
            for unit_id, unit_channels in enumerate(stage_channels):
                stride = strides[stage_id] if unit_id == 0 else 1
                stage.add_module(
                    f"unit{unit_id + 1}",
                    MSAG_Block(in_channels, unit_channels, stride)   # ← new block
                )
                in_channels = unit_channels
            self.backbone.add_module(f"stage{stage_id + 1}", stage)

        self.final_conv_channels = 1024
        self.backbone.add_module("final_conv",
            conv1x1_block(in_channels, self.final_conv_channels, activation="relu"))
        self.backbone.add_module("global_pool",
                                 torch.nn.AdaptiveAvgPool2d(1))
        self.classifier = Classifier(self.final_conv_channels, num_classes)
        self._init_params()

    def _init_params(self):
        for m in self.backbone.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
        self.classifier.init_params()

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)
###
#%% model definitions
###
def build_TickNet(num_classes, typesize='large', cifar=False):
    init_conv_channels = 32
    if typesize=='basic':
        channels = [[128],[64],[128],[256],[512]]
    if typesize=='small':
        channels = [[128],[64,128],[256,512,128],[64,128,256],[512]]
    if typesize=='large':
        channels = [[128],[64,128],[256,512,128,64,128,256],[512,128,64,128,256],[512]]
    if cifar:
        in_size = (32, 32)
        init_conv_stride = 1
        strides = [1, 1, 2, 2, 2]
    else:
        in_size = (224, 224)
        init_conv_stride = 2
        if typesize=='basic':
            strides = [1, 2, 2, 2, 2]
        else:
            strides = [2, 1, 2, 2, 2]
    return  TickNet(num_classes=num_classes,
                       init_conv_channels=init_conv_channels,
                       init_conv_stride=init_conv_stride,
                       channels=channels,
                       strides=strides,
                       in_size=in_size)
