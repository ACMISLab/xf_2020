# coding:utf-8
import torch
from mmcv.cnn import ConvModule
from mmengine.model import Sequential, BaseModule
from torch import nn
from torch.nn import UpsamplingNearest2d, AdaptiveAvgPool2d


class AFF(BaseModule):
    '''
    多特征融合 AFF
    '''

    def __init__(self, spa_channels: int,
                 sem_channels: int,
                 r=4):
        super(AFF, self).__init__()

        fus_channels = int((spa_channels if spa_channels > sem_channels else sem_channels) / 2)

        self.conv_sem = ConvModule(
            in_channels=sem_channels,
            out_channels=fus_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN2d'),
        )
        self.up_sampling = UpsamplingNearest2d(scale_factor=4)

        self.conv_spa = ConvModule(
            in_channels=spa_channels,
            out_channels=fus_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN2d'),
        )

        inter_channels = int(fus_channels // r)

        self.local_att = Sequential(
            ConvModule(
                in_channels=fus_channels,
                out_channels=inter_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d'),
                act_cfg=dict(type='ReLU')
            ),
            ConvModule(
                in_channels=inter_channels,
                out_channels=fus_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d')
            )
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

        self.global_att_conv1 = ConvModule(
            in_channels=fus_channels,
            out_channels=inter_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN2d'),
            act_cfg=dict(type='ReLU')
        )
        self.global_att_conv2 = ConvModule(
            in_channels=inter_channels,
            out_channels=fus_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN2d')
        )

        self.conv = ConvModule(
            in_channels=fus_channels,
            out_channels=spa_channels,
            kernel_size=1,
            padding=1,
            bias=False,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN2d'),
            act_cfg=None)

    def forward(self, x_spa, x_sem):
        x_fus_spa = self.conv_spa(x_spa)
        x_fus_sem = self.conv_sem(self.up_sampling(x_sem))

        xa = x_fus_sem + x_fus_spa
        xl = self.local_att(xa)

        # xa_avg_result = torch.mean(xa, dim=1, keepdim=True)
        xa_avg_result = self.avg_pool(xa)
        xg = self.global_att_conv1(xa_avg_result)
        xg = self.global_att_conv2(xg)

        xlg = xl + xg
        wei = torch.sigmoid(xlg)

        xo = 2 * x_fus_spa * wei + 2 * x_fus_sem * (1 - wei)
        return self.conv(xo)
