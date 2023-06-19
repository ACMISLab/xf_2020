_base_ = './retinanet_r101_fpn_2x_coco.py'
model = dict(
    backbone=dict(
        _delete_=True,
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')
    ),
    neck=dict(
        in_channels=[256, 512, 1024, 2048]
    )
)
