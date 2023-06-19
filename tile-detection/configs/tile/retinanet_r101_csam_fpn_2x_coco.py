_base_ = './retinanet_r101_cbam_fpn_2x_coco.py'

model = dict(
    backbone=dict(
        type='ResNetCSAM',
    ),
    neck=dict(
        in_channels=[256, 512, 1024, 2048]
    )
)

batch_size = 8

train_dataloader = dict(
    batch_size=8
)

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001))
