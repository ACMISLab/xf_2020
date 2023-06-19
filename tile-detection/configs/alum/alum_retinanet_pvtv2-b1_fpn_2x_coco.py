_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    './coco_detection_alum.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]
model = dict(
    type='RetinaNet',
    backbone=dict(
        _delete_=True,
        type='PyramidVisionTransformerV2',
        embed_dims=64,
        num_layers=[2, 2, 2, 2],
        init_cfg=dict(checkpoint='https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b1.pth')
    ),
    neck=dict(
        in_channels=[64, 128, 320, 512]
    ),
    bbox_head=dict(
        num_classes=len(_base_.class_name)
    )
)


# optimizer
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True, type='AdamW', lr=2e-4, weight_decay=0.0001))

_base_.default_hooks.logger.interval = 10
# _base_.env_cfg.cudnn_benchmark = True
