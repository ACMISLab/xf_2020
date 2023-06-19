_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    './coco_detection_tile.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]
model = dict(
    type='RetinaNet',
    backbone=dict(
        _delete_=True,
        type='PyramidVisionTransformerV2',
        embed_dims=64,
        num_layers=[3, 4, 6, 3],
        init_cfg=dict(checkpoint='https://github.com/whai362/PVT/'
                      'releases/download/v2/pvt_v2_b2.pth')),
    neck=dict(in_channels=[64, 128, 320, 512]),
    bbox_head=dict(
        num_classes=len(_base_.class_name),
        anchor_generator=dict(
            octave_base_scale=2,
            scales_per_octave=5,
            ratios=[0.3, 0.5, 1.0, 2.0, 3.33],
            strides=[8, 16, 32, 64, 128]),
    ),
    test_cfg=dict(
        nms_pre=100,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.1),
        max_per_img=100)
)

# optimizer
optim_wrapper = dict(
    # type='AmpOptimWrapper',
    optimizer=dict(
        _delete_=True, type='AdamW', lr=1e-4, weight_decay=0.0001))

_base_.default_hooks.logger.interval = 200
_base_.env_cfg.cudnn_benchmark = True
