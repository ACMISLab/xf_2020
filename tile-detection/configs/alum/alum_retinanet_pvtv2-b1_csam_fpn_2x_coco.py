_base_ = './alum_retinanet_pvtv2-b1_fpn_2x_coco.py'
model = dict(
    type='RetinaNet',
    backbone=dict(
        type='PVTv2AFM',
        num_layers=[2, 2, 2, 2],
        init_cfg=dict(checkpoint='https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b1.pth')
    ),
    neck=dict(
        in_channels=[64, 128, 320, 512]
    ),
)

# train_dataloader = dict(
#     batch_size=4
# )

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True, type='AdamW', lr=1e-4, weight_decay=0.0001))

_base_.default_hooks.logger.interval = 20
# _base_.env_cfg.cudnn_benchmark = True

find_unused_parameters=True
