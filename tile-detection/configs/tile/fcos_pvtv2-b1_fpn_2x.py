_base_ = [
    # '../fcos/fcos_r50_fpn_gn-head-center-normbbox-centeronreg-giou_8xb8-amp-lsj-200e_coco.py',
    './coco_detection_tile.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py'

    ]
image_size = (1024, 1024)
batch_augments = [dict(type='BatchFixedSizePad', size=image_size)]

model = dict(
    type='FCOS',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        batch_augments=batch_augments),
    backbone=dict(
        type='PyramidVisionTransformerV2',
        embed_dims=64,
        num_layers=[2, 2, 2, 2],
        init_cfg=dict(checkpoint='https://github.com/whai362/PVT/'
                      'releases/download/v2/pvt_v2_b1.pth')
    ),
    neck=dict(
        type='FPN',
        in_channels=[64, 128, 320, 512],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=len(_base_.class_name),
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        norm_on_bbox=True,
        centerness_on_reg=True,
        dcn_on_last_conv=False,
        center_sampling=True,
        conv_bias=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)
    ),
    test_cfg=dict(
            nms_pre=100,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.1),
            max_per_img=100)
)

# train_dataloader = dict(batch_size=8, num_workers=4)
optim_wrapper = dict(
    optimizer=dict(
        _delete_=True, type='AdamW', lr=1e-4, weight_decay=0.0001))

_base_.default_hooks.logger.interval = 100
_base_.env_cfg.cudnn_benchmark = True
auto_scale_lr = dict(base_batch_size=64)