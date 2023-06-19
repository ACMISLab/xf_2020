_base_ = './tile_retinanet_pvtv2-b1_fpn_2x_coco.py'
model = dict(
    type='RetinaNet',
    backbone=dict(
        type='PVTv2AFF',
        embed_dims=64,
        num_layers=[2, 2, 2, 2],
        init_cfg=dict(checkpoint='https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b1.pth')
    ),
    # neck=dict(in_channels=[64, 256, 320, 512]),

)

train_dataloader = dict(
    batch_size=6
)

_base_.default_hooks.logger.interval = 30


# find_unused_parameters = True