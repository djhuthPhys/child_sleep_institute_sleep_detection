import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch

import torch.nn as nn

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

class SelfAttention(nn.Module):
    """
    Self attention Layer
    """
    def __init__(self,in_channels, activation):
        super(SelfAttention,self).__init__()
        self.chanel_in = in_channels
        self.activation = activation
        
        self.query_conv = nn.Conv1d(in_channels , out_channels=in_channels//8, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels , out_channels=in_channels//8, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels , out_channels=in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps(batch, channel, time)
            returns :
                out : self attention value + input feature 
                attention: (batch, time)
        """
        batch_size, channels, time= x.size()
        out = torch.zeros_like(x)
        chunk_size = 100
        num_chunks = time//chunk_size + 1
        for i in range(num_chunks):
            if i < num_chunks-1:
                energy =  torch.bmm(self.query_conv(x[:, :, chunk_size*i:chunk_size*(i+1)]).view(batch_size, -1, chunk_size).permute(0,2,1),
                                    self.key_conv(x[:, :, chunk_size*i:chunk_size*(i+1)]).view(batch_size, -1, chunk_size)) # (batch, time, time)
                self_attention = torch.bmm(self.value_conv(x[:, :, chunk_size*i:chunk_size*(i+1)]).view(batch_size, -1, chunk_size), self.softmax(energy))
                out[:, :, chunk_size*i:chunk_size*(i+1)] = self_attention.view(batch_size, channels, chunk_size)
            else:
                remain_size = x[:, :, chunk_size*i:].shape[2]
                energy =  torch.bmm(self.query_conv(x[:, :, chunk_size*i:]).view(batch_size, -1, remain_size ).permute(0,2,1),
                                    self.key_conv(x[:, :, chunk_size*i:]).view(batch_size, -1, remain_size )) # (batch, time, time)
                self_attention = torch.bmm(self.value_conv(x[:, :, chunk_size*i:]).view(batch_size, -1, remain_size ), self.softmax(energy))
                out[:, :, chunk_size*i:] = self_attention.view(batch_size, channels, remain_size )
        out = self.gamma*out + x
        return out


autoconv = partial(Conv1dAutoPad, kernel_size=3, dilation=1, bias=False)


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

        self.dilated_conv = nn.Sequential(OrderedDict(
            {
                'conv' : conv(self.in_channels, self.out_channels, *args, **kwargs),

                'bn' : nn.BatchNorm1d(self.out_channels), 

                'drop': nn.Dropout(0.1)
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

        x = self.dilated_conv(x)

        # Gating activation
        x = self.tanh(x) * self.sig(x)

        # x = self.skip(x)
        # skip = x

        x += residual
        return x

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

                'drop': nn.Dropout(0.1)
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
        x= self.layer(x)
        return x


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
        layer_skips = []
        for layer in self.block:
            x = layer(x)
        #     layer_skips.append(layer_skip)
        # skips = torch.cat(layer_skips, 0)
        return x


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
            nn.Dropout(0.1),
            nn.SiLU()
        )

        self.in_out_pairs = list(zip(feat_sizes, feat_sizes[1:]))

        self.blocks = nn.ModuleList([
            block(feat_sizes[0], feat_sizes[0], block_size=depths[0], *args, **kwargs),
            *[block(in_channels, out_channels, block_size=depth, *args, **kwargs)
              for (in_channels, out_channels), depth in zip(self.in_out_pairs, depths[1:])]
        ])


    def forward(self, x):
        block_skips = []
        x = self.init_conv(x)
        for block in self.blocks:
            x = block(x)
        #     block_skips.append(block_skip)
        # skips = torch.cat(block_skips, 0)
        return x

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

        self.tail = nn.Sequential(
            nn.Conv1d(in_channels, in_channels//2, kernel_size=1, padding=0),
            nn.BatchNorm1d(in_channels//2),
            nn.Dropout(0.1),
            nn.SiLU(),
            nn.Conv1d(in_channels//2, 3, kernel_size=1, padding=0)
        )

        self.bbox_reg = nn.Sequential(
            nn.Conv1d(in_channels, in_channels//2, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm1d(in_channels//2),
            nn.Dropout(0.1),
            nn.SiLU(),
            nn.Conv1d(in_channels//2, in_channels//4, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm1d(in_channels//4),
            nn.Dropout(0.1),
            nn.SiLU(),
            nn.Conv1d(in_channels//4, in_channels//8, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm1d(in_channels//8),
            nn.Dropout(0.1),
            nn.SiLU(),
            nn.Conv1d(in_channels//8, in_channels//16, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm1d(in_channels//16),
            nn.Dropout(0.1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(1080*(in_channels//16)-4, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            nn.SiLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        # Classification
        x_cls = self.tail(x)

        # Bbox regression
        x_bbox = self.bbox_reg(x)

        return x_cls, x_bbox


class Classifier(nn.Module):
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

    def forward(self, x):
        x = self.feature_extraction(x)
        x_cls, x_bbox = self.tail(x)
        return x_cls, x_bbox
    