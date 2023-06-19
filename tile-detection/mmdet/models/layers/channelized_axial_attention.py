# coding:utf-8
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule


class ChannelizedAxialAttention(BaseModule):

    def __init__(self):
        super().__init__()

        self.entry_conv = ConvModule()