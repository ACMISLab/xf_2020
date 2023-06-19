# coding:utf-8
import torch
from torch import nn

from mmcv.cnn import ConvModule
from mmengine.model import BaseModule, Sequential


class SelectiveFeatureFusionModule(BaseModule):

    def __init__(self,
                 spa_channels: int,
                 sem_channels: int,
                 init_cfg=None
                 ):
        super(SelectiveFeatureFusionModule, self).__init__(init_cfg=init_cfg)

        assert spa_channels < sem_channels

        self.up_sampling = nn.UpsamplingNearest2d(scale_factor=4)

        fus_channels = int(spa_channels if spa_channels > sem_channels else sem_channels) / 2
        self.fus_channels = int(fus_channels)

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv_sem = ConvModule(
            in_channels=sem_channels,
            out_channels=self.fus_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN2d'),
            # act_cfg=dict(type='ReLU')
        )

        self.conv_spa = ConvModule(
            in_channels=spa_channels,
            out_channels=self.fus_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN2d'),
            # act_cfg=dict(type='ReLU')
        )

        self.conv_attn = ConvModule(
            in_channels=self.fus_channels,
            out_channels=spa_channels,
            kernel_size=1,
            padding=1,
            bias=False,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN2d'),
            act_cfg=None)

        self.conv1d = Sequential(
            ConvModule(in_channels=2,
                       out_channels=1,
                       kernel_size=3,
                       stride=1,
                       padding=1,
                       bias=False,
                       norm_cfg=dict(type='BN', requires_grad=True),
                       act_cfg=dict(type='ReLU')),
            ConvModule(in_channels=1,
                       out_channels=1,
                       kernel_size=3,
                       stride=1,
                       padding=1,
                       bias=False,
                       norm_cfg=dict(type='BN', requires_grad=True),
                       act_cfg=dict(type='ReLU')),
            ConvModule(in_channels=1,
                       out_channels=1,
                       kernel_size=3,
                       stride=1,
                       padding=1,
                       bias=False,
                       norm_cfg=dict(type='BN', requires_grad=True),
                       act_cfg=dict(type='ReLU'))
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x_spatial, x_semantic):
        x_semantic_up = self.up_sampling(x_semantic)

        x_sem_fus = self.conv_sem(x_semantic_up)
        # sem_avg_feature = self.avg_pool(x_sem_fus)

        x_spa_fus = self.conv_spa(x_spatial)

        # max_result, _ = torch.max(x, dim=1, keepdim=True)
        spa_max_result, _ = torch.max(x_spa_fus, dim=1, keepdim=True)
        sem_max_result, _ = torch.max(x_sem_fus, dim=1, keepdim=True)

        spa_avg_result = torch.mean(x_spa_fus, dim=1, keepdim=True)
        sem_avg_result = torch.mean(x_sem_fus, dim=1, keepdim=True)

        # avg_out = torch.mean(x, dim=1, keepdim=True)
        # max_out, _ = torch.max(x, dim=1, keepdim=True)
        # out = torch.cat([avg_out, max_out], dim=1)

        fus1 = torch.cat([spa_avg_result, sem_avg_result], 1).contiguous()
        fus2 = torch.cat([spa_max_result, sem_max_result], 1).contiguous()

        weights = self.sigmoid(self.conv1d(fus1) + self.conv1d(fus2))

        # print('Weights shape: {}'.format(weights.shape))
        init_fus = (x_spa_fus + x_sem_fus)
        return self.conv_attn(init_fus * weights + init_fus)

        # print('fusion shape: {}, spatial shape: {}'.format(x_fusion.shape, x_spatial.shape))
        # x_fuse_attn = x_spatial + x_fusion

        # return x_fusion


class FeatureFusionModule(BaseModule):

    def __init__(self,
                 spa_channels,
                 sem_channels,
                 init_cfg=None
                 ):
        super(FeatureFusionModule, self).__init__(init_cfg=init_cfg)

        assert spa_channels < sem_channels

        self.up_sampling = nn.UpsamplingNearest2d(scale_factor=4)

        fus_channels = (spa_channels if spa_channels > sem_channels else sem_channels) / 2
        self.fus_channels = int(fus_channels)

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv_sem = ConvModule(
            in_channels=sem_channels,
            out_channels=self.fus_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN2d'),
            # act_cfg=dict(type='ReLU')
        )

        self.conv_spa = ConvModule(
            in_channels=spa_channels,
            out_channels=self.fus_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN2d'),
            # act_cfg=dict(type='ReLU')
        )

        self.conv_attn = ConvModule(
            in_channels=self.fus_channels,
            out_channels=spa_channels,
            kernel_size=1,
            padding=1,
            bias=False,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN2d'),
            act_cfg=None)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x_spatial, x_semantic):
        x_semantic_up = self.up_sampling(x_semantic)

        x_sem_fus = self.conv_sem(x_semantic_up)
        # sem_avg_feature = self.avg_pool(x_sem_fus)

        x_spa_fus = self.conv_spa(x_spatial)

        weights = self.ca(x_sem_fus)

        x_fusion = self.conv_attn(x_spa_fus + x_sem_fus * weights + x_spa_fus)

        # print('fusion shape: {}, spatial shape: {}'.format(x_fusion.shape, x_spatial.shape))
        # x_fuse_attn = x_spatial + x_fusion

        return x_fusion
