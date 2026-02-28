import re
import types

import torch.nn
import torch.nn.init

from .common import conv1x1_block, Classifier,conv3x3_dw_blockAll,conv3x3_block
from .SE_Attention import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn

class MultiScaleFactorizedDWConvBlock(nn.Module):
    """
    Multi-Scale Factorized Depthwise Convolution:
    Branch A: 3x1 + 1x3
    Branch B: 5x1 + 1x5
    Output = sum(A, B)
    """

    def __init__(self, channels, stride=1, use_bn=True, activation="relu"):
        super().__init__()

        # ----- Branch 3x3 (factorized) -----
        self.dw3_vert = nn.Conv2d(
            channels, channels,
            kernel_size=(3, 1),
            stride=(stride, 1),
            padding=(1, 0),
            groups=channels,
            bias=False
        )

        self.dw3_hori = nn.Conv2d(
            channels, channels,
            kernel_size=(1, 3),
            stride=(1, stride),
            padding=(0, 1),
            groups=channels,
            bias=False
        )

        # ----- Branch 5x5 (factorized) -----
        self.dw5_vert = nn.Conv2d(
            channels, channels,
            kernel_size=(5, 1),
            stride=(stride, 1),
            padding=(2, 0),
            groups=channels,
            bias=False
        )

        self.dw5_hori = nn.Conv2d(
            channels, channels,
            kernel_size=(1, 5),
            stride=(1, stride),
            padding=(0, 2),
            groups=channels,
            bias=False
        )

        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm2d(channels)

        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        else:
            self.act = None

    def forward(self, x):

        # Branch 3x3
        b3 = self.dw3_vert(x)
        b3 = self.dw3_hori(b3)

        # Branch 5x5
        b5 = self.dw5_vert(x)
        b5 = self.dw5_hori(b5)

        # Sum fusion (không tăng channel)
        out = b3 + b5

        if self.use_bn:
            out = self.bn(out)

        if self.act is not None:
            out = self.act(out)

        return out


def conv_factorized_dw_blockAll(channels, stride):
    return MultiScaleFactorizedDWConvBlock(
        channels=channels,
        stride=stride,
        use_bn=True,
        activation="relu"
    )

class FR_PDP_block(torch.nn.Module):
    """
    FR_PDP_block for TickNet.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride):
        super().__init__()
        self.Pw1 = conv1x1_block(in_channels=in_channels,
                                out_channels=in_channels,                                
                                use_bn=False,
                                activation=None)
        self.Dw = conv_factorized_dw_blockAll(channels=in_channels, stride=stride)         
        self.Pw2 = conv1x1_block(in_channels=in_channels,
                                             out_channels=out_channels,                                             
                                             groups=1)
        self.PwR = conv1x1_block(in_channels=in_channels,
                                out_channels=out_channels,
                                stride=stride)
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.SE = SE(out_channels, 16)
    def forward(self, x):
        residual = x
        x = self.Pw1(x)        
        x = self.Dw(x)        
        x = self.Pw2(x)
        x = self.SE(x)
        if self.stride == 1 and self.in_channels == self.out_channels:
            x = x + residual
        else:            
            residual = self.PwR(residual)
            x = x + residual
        return x

class TickNet(torch.nn.Module):
    """
    Class for constructing TickNet.    
    """
    def __init__(self,
                 num_classes,
                 init_conv_channels,
                 init_conv_stride,
                 channels,
                 strides,
                 in_channels=3,
                 in_size=(224, 224),
                 use_data_batchnorm=True):
        super().__init__()
        self.use_data_batchnorm = use_data_batchnorm
        self.in_size = in_size

        self.backbone = torch.nn.Sequential()

        # data batchnorm
        if self.use_data_batchnorm:
            self.backbone.add_module("data_bn", torch.nn.BatchNorm2d(num_features=in_channels))

        # init conv
        self.backbone.add_module("init_conv", conv3x3_block(in_channels=in_channels, out_channels=init_conv_channels, stride=init_conv_stride))

        # stages
        in_channels = init_conv_channels
        for stage_id, stage_channels in enumerate(channels):
            stage = torch.nn.Sequential()
            for unit_id, unit_channels in enumerate(stage_channels):
                stride = strides[stage_id] if unit_id == 0 else 1                
                stage.add_module("unit{}".format(unit_id + 1), FR_PDP_block(in_channels=in_channels, out_channels=unit_channels, stride=stride))
                in_channels = unit_channels
            self.backbone.add_module("stage{}".format(stage_id + 1), stage)
        self.final_conv_channels = 1024        
        self.backbone.add_module("final_conv", conv1x1_block(in_channels=in_channels, out_channels=self.final_conv_channels, activation="relu"))
        self.backbone.add_module("global_pool", torch.nn.AdaptiveAvgPool2d(output_size=1))
        in_channels = self.final_conv_channels
        # classifier
        self.classifier = Classifier(in_channels=in_channels, num_classes=num_classes)

        self.init_params()

    def init_params(self):
        # backbone
        for name, module in self.backbone.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)

        # classifier
        self.classifier.init_params()

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

###
#%% model definitions
###
def build_TickNet(num_classes, typesize='small', cifar=False):
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
