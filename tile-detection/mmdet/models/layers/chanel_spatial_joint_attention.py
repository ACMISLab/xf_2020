# coding: utf-8

import logging

# Created by Xufang
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule

from mmdet.models.layers.axial_attention import AxialAttention


class PCALayer(BaseModule):

    def __init__(self, in_planes=1, out_planes=1, kernel_size=3):
        super(PCALayer, self).__init__(init_cfg=None)

        # print('Construct ECA Module!')

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)

        self.conv = ConvModule(in_channels=in_planes,
                               out_channels=out_planes,
                               kernel_size=kernel_size,
                               padding=(kernel_size - 1) // 2,
                               bias=False,
                               conv_cfg=dict(type='Conv1d'),
                               act_cfg=None)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        max_feature = self.max_pool(x)
        avg_feature = self.avg_pool(x)

        max_out = self.conv(max_feature.squeeze(-1).transpose(-1, -2).contiguous()).transpose(-1, -2).unsqueeze(
            -1).contiguous()
        avg_out = self.conv(avg_feature.squeeze(-1).transpose(-1, -2).contiguous()).transpose(-1, -2).unsqueeze(
            -1).contiguous()

        y = self.sigmoid(max_out + avg_out)

        return y.expand_as(x)


class EfficientAttention(BaseModule):

    def __init__(self, in_channels, key_channels, head_count, value_channels):
        super(EfficientAttention, self).__init__()

        # print('Construct Efficient Attention Module!')
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels

        self.keys = ConvModule(in_channels, self.key_channels, 1)
        self.queries = ConvModule(in_channels, self.key_channels, 1)
        self.values = ConvModule(in_channels, self.value_channels, 1)

        self.reprojection = ConvModule(self.value_channels, in_channels, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_):
        residual = input_

        n, _, h, w = input_.size()
        keys = self.keys(input_).reshape((n, self.key_channels, h * w))
        queries = self.queries(input_).reshape(n, self.key_channels, h * w)
        values = self.values(input_).reshape((n, self.value_channels, h * w))
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            key = torch.softmax(keys[
                                :,
                                i * head_key_channels: (i + 1) * head_key_channels,
                                :
                                ], dim=2)
            query = torch.softmax(queries[
                                  :,
                                  i * head_key_channels: (i + 1) * head_key_channels,
                                  :
                                  ], dim=1)
            value = values[
                    :,
                    i * head_value_channels: (i + 1) * head_value_channels,
                    :
                    ]
            context = key @ value.transpose(1, 2)
            attended_value = (
                    context.transpose(1, 2) @ query
            ).reshape(n, head_value_channels, h, w)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        reprojected_value = self.reprojection(aggregated_values)

        return self.sigmoid(reprojected_value)
        #
        # return residual - residual * attention.expand_as(input_)


class CSAM(BaseModule):

    def __init__(self,
                 in_channels=512,
                 ):
        super(CSAM, self).__init__()

        # self.sm = EfficientAttention(in_channels=in_channels,
        #                              key_channels=int(in_channels / 2),
        #                              head_count=16,
        #                              value_channels=int(in_channels / 2))
        self.cm = PCALayer(in_planes=1, out_planes=1)
        self.sm = AxialAttention(in_dim=in_channels)

    def forward(self, x):
        # b, c, _, _ = x.size()

        # if not self.disable_spatial:

        out = self.sm(x)
        out = self.cm(x) * out + x

        return out
