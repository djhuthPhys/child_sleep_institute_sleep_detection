import os
import math
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch

import torch.nn as nn

from torch import Tensor
from functools import partial
from collections import OrderedDict

# Define model
class Conv1dAutoPad(nn.Conv1d):
    """
    Auto-padding convolution depending on kernel size and dilation used
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = self.dilation[0] * (self.kernel_size[0] // 2)

autoconv = partial(Conv1dAutoPad, kernel_size=3, dilation=1, bias=True)


class WaveNetLayerInit(nn.Module):
    """
    Initializes the WaveNet layer with a dilated convolution and identity residuals and skips
    """
    def __init__(self, in_channels, out_channels, conv=autoconv, *args, **kwargs):
        """
        :param in_channels:
        :param out_channels:
        :param dilation:
        :param conv: determines what kind of convolutional layer is used
        :param args:
        :param kwargs:
        """
        super().__init__()
        self.in_channels, self.out_channels, self.conv = in_channels, out_channels, conv

        self.tanh_conv = nn.Sequential(OrderedDict(
            {
                'conv' : conv(self.in_channels, self.out_channels, *args, **kwargs),

                # 'drop': nn.Dropout(0.1)
            }))
        
        self.sig_conv = nn.Sequential(OrderedDict(
            {
                'conv' : conv(self.in_channels, self.out_channels, *args, **kwargs),
                
                # 'drop': nn.Dropout(0.1)
            }))

        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()

        self.skip = nn.Identity()

        self.shortcut = nn.Identity()

    def forward(self, x):
        """
        Forward propagation for a single WaveNet layer. Feature maps are split after dilated convolution as described in
        PixelCNN/RNN paper
        :param x: input data
        :return:
        """
        residual = x

        if self.should_apply_mapping:
            residual = self.shortcut(x)

        x_tanh = self.tanh_conv(x)
        x_sig = self.sig_conv(x)

        # Gating activation
        x = self.tanh(x_tanh) * self.sig(x_sig)

        x = self.skip(x)
        skip = x

        x += residual
        return x, skip

    @property
    def should_apply_mapping(self):
        return self.in_channels != self.out_channels


class addResidualConnection(WaveNetLayerInit):
    def __init__(self, in_channels, out_channels, expansion=1, conv=autoconv, *args, **kwargs):
        """
        Adds the parameterized residual connection if in_channels != out_channels. Standard WaveNet maintains channel
        number throughout the network
        :param in_channels:
        :param out_channels:
        :param expansion: expansion factor if in_channels != out_channels
        :param conv:
        :param args:
        :param kwargs:
        """
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.expansion = expansion

        self.shortcut = nn.Sequential(OrderedDict(
            {
                'conv' : conv(self.in_channels, self.map_channels, kernel_size=1),

                'bn' : nn.BatchNorm1d(self.map_channels),

                # 'drop': nn.Dropout(0.1)
            })) if self.should_apply_mapping else None


    @property
    def map_channels(self):
        return self.out_channels * self.expansion

    @property
    def should_apply_mapping(self):
        return self.in_channels != self.out_channels


class addSkipConnection(addResidualConnection):
    def __init__(self, in_channels, out_channels, conv=autoconv, *args, **kwargs):
        """
        Adds the 1X1 convolution after the gating of the dilated convolution and returns doubled number of feature maps.
        :param in_channels:
        :param out_channels:
        :param conv:
        :param args:
        :param kwargs:
        """
        super().__init__(in_channels, out_channels, *args, **kwargs)

        if in_channels != 1:
            self.skip = conv(self.in_channels, self.out_channels, kernel_size=1, dilation=1) # Expects in_channels to be half of that used by the dilated convolution
        else:
            self.skip = None


class WaveNetLayer(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, layer=addSkipConnection, *args, **kwargs):
        super().__init__()

        self.layer = layer(in_channels, out_channels, *args, **kwargs)

    def forward(self, x):
        x, skip = self.layer(x)
        return x, skip


class WaveNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, layer=WaveNetLayer, block_size=1, *args, **kwargs):
        """
        Constructs a block of WaveNet layers of size block_size where the dilation increases by a power of 2 between
        every layer
        :param in_channels:
        :param out_channels:
        :param layer:
        :param block_size: number of WaveNet layers toj include in the block
        """
        super().__init__()

        self.block = nn.Sequential(
            layer(in_channels, out_channels, dilation=1,*args, **kwargs),
            *[layer(out_channels * layer.expansion, out_channels, dilation=2**(i+1), *args, **kwargs)
              for i in range(block_size-1)]
        )

    def forward(self, x):
        skips_list = []
        for layer in self.block:
            x, layer_skip = layer(x)
            skips_list.append(layer_skip)
        layer_skips = torch.stack(skips_list, dim=0) # (depth, batch, channel, time)
        return x, layer_skips


class WaveNetConvs(nn.Module):
    def __init__(self, in_channels, feat_sizes=(16,32), depths=(2, 3), block=WaveNetBlock, *args, **kwargs):
        """
        Constructs the convolutional layers of the WaveNet
        :param in_channels:
        :param feat_sizes: tuple of the number of feature maps per layer in each block
        :param depths: number of layers in every block of the WaveNet
        :param block:
        :param args:
        :param kwargs:
        """
        super().__init__()

        self.feat_sizes, self.depths = feat_sizes, depths

        self.init_conv = nn.Sequential(
            nn.Conv1d(in_channels, feat_sizes[0], kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(feat_sizes[0]),
            # nn.Dropout(0.1),
            nn.SiLU()
        )

        self.in_out_pairs = list(zip(feat_sizes, feat_sizes[1:]))

        self.blocks = nn.ModuleList([
            block(feat_sizes[0], feat_sizes[0], block_size=depths[0], *args, **kwargs),
            *[block(in_channels, out_channels, block_size=depth, *args, **kwargs)
              for (in_channels, out_channels), depth in zip(self.in_out_pairs, depths[1:])]
        ])


    def forward(self, x):
        bskip_list = []
        x = self.init_conv(x)
        for block in self.blocks:
            x, block_skip = block(x)
            bskip_list.append(block_skip)
        block_skips = torch.stack(bskip_list, dim=0) # (block, depth, batch, channel, time)
        return x, block_skips
    

class Transpose12(nn.Module):
    def forward(self, x):
        return x.transpose(1, 2)

class WaveNetTail(nn.Module):
    def __init__(self, in_channels):
        """
        Last few layers of WaveNet where skip connections are integrated and final 1X1 convolutions occur
        :param in_channels:
        """
        super().__init__()

        # self.crit_tail = nn.Sequential(
        #     nn.Linear(in_channels, in_channels//2),
        #     Transpose12(),
        #     nn.BatchNorm1d(in_channels//2),
        #     Transpose12(),
        #     # nn.Dropout(0.1),
        #     nn.SiLU(),
        #     nn.Linear(in_channels//2, 2)
        # )

        self.gru = nn.GRU(in_channels, in_channels, num_layers=1, bidirectional=True)
        self.linear = nn.Linear(in_channels*2, 2)

    def forward(self, x):

        # Critical point regression
        x = x.transpose(1, 2) # (batch, time, features)
        x, h = self.gru(x)
        x = self.linear(nn.functional.silu(x))
        x = x.transpose(1, 2) # (batch, features, time)
        return x


class Regressor(nn.Module):
    def __init__(self, in_channels, *args, **kwargs):
        """
        The completed WaveNet model
        :param in_channels:
        :param args:
        :param kwargs:
        """
        super().__init__()

        self.feature_extraction = WaveNetConvs(in_channels, *args, **kwargs)
        self.tail = WaveNetTail(self.feature_extraction.feat_sizes[-1])
        self.gelu = nn.SiLU()

    def forward(self, x):
        x, skips = self.feature_extraction(x)
        skip_sum = torch.sum(skips, dim=(0, 1))
        x = self.tail(skip_sum)
        x = nn.functional.sigmoid(x)
        return x
    