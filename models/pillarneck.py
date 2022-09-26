import torch
import torch.nn as nn
import numpy as np
from det3d.models.utils.conv import BasicBlock, ConvBlock
import torch.utils.checkpoint as cp
from trainer.utils import force_fp32


class BiFPNLayer(nn.Module):
    def __init__(
        self,
        in_channels=256,
        is_last=False,
    ):
        super(BiFPNLayer, self).__init__()

        deconv_cfg = {'_target_': 'torch.nn.ConvTranspose2d', 'padding': 0, 'bias': False}
        conv_cfg = {'_target_': 'torch.nn.Conv2d', 'padding': 0, 'bias': False}

        self.is_last = is_last

        self.c7upsample = ConvBlock(in_channels, in_channels, kernel_size=2, stride=2, conv_cfg=deconv_cfg)

        self.c6upsample = ConvBlock(in_channels, in_channels, kernel_size=2, stride=2, conv_cfg=deconv_cfg)

        self.c5upsample = ConvBlock(in_channels, in_channels, kernel_size=2, stride=2, conv_cfg=deconv_cfg)
        
        self.c4upsample = ConvBlock(in_channels, in_channels, kernel_size=2, stride=2, conv_cfg=deconv_cfg)

        if not is_last:
            self.c3downsample = ConvBlock(in_channels, in_channels, kernel_size=2, stride=2, conv_cfg=conv_cfg)

            self.c4downsample = ConvBlock(in_channels, in_channels, kernel_size=2, stride=2, conv_cfg=conv_cfg)

            self.c5downsample = ConvBlock(in_channels, in_channels, kernel_size=2, stride=2, conv_cfg=conv_cfg)

            self.c6downsample = ConvBlock(in_channels, in_channels, kernel_size=2, stride=2, conv_cfg=conv_cfg)



    def forward(self, c3, c4, c5, c6, c7):
        c6mid = self.c7upsample(c7) + c6
        c5mid = self.c6upsample(c6mid) + c5
        c4mid = self.c5upsample(c5mid) + c4
        
        c3 = self.c4upsample(c4mid) + c3
        
        if not self.is_last:
            c4 = self.c3downsample(c3) + c4 + c4mid
            c5 = self.c4downsample(c4) + c5 + c5mid
            c6 = self.c5downsample(c5) + c6 + c6mid
            c7 = self.c6downsample(c6) + c7
            return c3, c4, c5, c6, c7
        return c3


class PillarNeck(nn.Module):
    def __init__(
        self,
        in_channels,
        conv_cfg=None,
        norm_cfg=None,
        deconv_cfg=None):

        super(PillarNeck, self).__init__()
        self._num_filters = in_channels

        if deconv_cfg is None:
            deconv_cfg = {'_target_': 'torch.nn.ConvTranspose2d', 'padding': 0, 'bias': False}
        if conv_cfg is None:
            conv_cfg = {'_target_': 'torch.nn.Conv2d', 'padding': 0, 'bias': False}

        self.c3downsample = ConvBlock(in_channels, in_channels, kernel_size=2, stride=2, conv_cfg=conv_cfg)
        self.c4downsample = ConvBlock(in_channels, in_channels, kernel_size=2, stride=2, conv_cfg=conv_cfg)
        self.c5downsample = ConvBlock(in_channels, in_channels, kernel_size=2, stride=2, conv_cfg=conv_cfg)
        self.c6downsample = ConvBlock(in_channels, in_channels, kernel_size=2, stride=2, conv_cfg=conv_cfg)        
        
        self.c3conv = BasicBlock(in_channels, in_channels)
        self.c4conv = BasicBlock(in_channels, in_channels)
        self.c5conv = BasicBlock(in_channels, in_channels)
        self.c6conv = BasicBlock(in_channels, in_channels)
        self.c7conv = BasicBlock(in_channels, in_channels)

        self.bifpn1 = BiFPNLayer()
        self.bifpn2 = BiFPNLayer()
        self.bifpn3 = BiFPNLayer(is_last=True)

        
    def _forward(self, x):
        c4 = self.c3downsample(x)
        c5 = self.c4downsample(c4)
        c6 = self.c5downsample(c5)
        c7 = self.c6downsample(c6)

        c3 = self.c3conv(x)
        c4 = self.c4conv(c4)
        c5 = self.c5conv(c5)
        c6 = self.c6conv(c6)
        c7 = self.c7conv(c7)

        c3, c4, c5, c6, c7 = self.bifpn1(c3, c4, c5, c6, c7)
        c3, c4, c5, c6, c7 = self.bifpn2(c3, c4, c5, c6, c7)
        c3 = self.bifpn3(c3, c4, c5, c6, c7)

        return c3

    def forward(self, x):
        if x.requires_grad:
            c3 = cp.checkpoint(self._forward, x)
        else:
            c3 = self._forward(x)

        return c3
