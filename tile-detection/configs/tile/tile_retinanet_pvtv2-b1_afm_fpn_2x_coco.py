_base_ = './tile_retinanet_pvtv2-b1_fpn_2x_coco.py'
model = dict(
    type='RetinaNet',
    backbone=dict(
        type='PVTv2AFM',
        num_layers=[3, 4, 6, 3],
        init_cfg=dict(checkpoint='https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b1.pth')
    ),
    # neck=dict(in_channels=[64, 256, 320, 512]),
)

batch_size = 2
train_dataloader = dict(
    batch_size=2
)

base_lr = 1e-4

optim_wrapper = dict(
    optimizer=dict(
        _delete_=True, type='AdamW', lr=base_lr * ((batch_size * 8) / (2 * 8)), weight_decay=0.0001))

find_unused_parameters = True

train_cfg = dict(max_epochs=28)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=24,
        by_epoch=True,
        milestones=[16, 25],
        gamma=0.1)
]
