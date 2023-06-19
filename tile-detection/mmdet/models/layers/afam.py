# coding:utf-8
import torch
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule, Sequential
from torch import nn, Tensor


class ChannelAttention(BaseModule):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = Sequential(
            ConvModule(in_channels=in_planes,
                       out_channels=int(in_planes // ratio),
                       kernel_size=1,
                       stride=1,
                       conv_cfg=None,
                       act_cfg=dict(type='ReLU')
                       ),
            ConvModule(in_channels=int(in_planes // ratio),
                       out_channels=in_planes,
                       kernel_size=1,
                       stride=1,
                       conv_cfg=None,
                       act_cfg=dict(type='Sigmoid')
                       ),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(BaseModule):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv = ConvModule(in_channels=2,
                               out_channels=1,
                               kernel_size=kernel_size,
                               padding=kernel_size // 2,
                               bias=False
                               )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class AFAM(BaseModule):

    def __init__(self, channels: int):
        super().__init__()
        self.ca = ChannelAttention(in_planes=channels)
        self.sa = SpatialAttention()

    def forward(self, x: Tensor) -> Tensor:
        '''
        X   ==> CA ==> Sigmoid ==> Residual Conn                        !
            ==> SA ==> Sigmoid ==> Local Cross-Dimension Interaction    => Out = WSA + WCA
        '''
        out1 = self.ca(x) * x + x
        out2 = self.sa(x) * out1 + x

        return out1 + out2

