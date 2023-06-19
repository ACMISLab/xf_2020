import os
import sys

_base_ = '../_base_/datasets/coco_detection.py'

dataset_type = 'BaseDetDataset'
# dataset_type = 'CocoDataset'

class_name = (
    'Background',
    'WritePot', 'LightBlock', 'DarkBlock', 'Aperture', 'Scratches')  # 根据 class_with_id.txt 类别信息，设置 class_name
num_classes = len(class_name)
metainfo = dict(
    classes=class_name,
    palette=[(20, 204, 143), (61, 153, 122), (1, 255, 28), (255, 64, 221), (172, 20, 204)]  # 画图时候的颜色，随便设置即可
)


data_root = '/remote-home/cs_acmis_yexf/datasets/tile' if sys.platform == 'linux' else r'D:\Datasets\tile'
backend_args = None
# dict(type='cudnn')

image_size = (1200, 1200)

file_client_args = dict(backend='disk')


train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=image_size, scale_factor=(1, 1.2), keep_ratio=True),
    dict(type='RandomCrop', crop_size=image_size, crop_type='absolute_range', recompute_bbox=True,
         allow_negative_crop=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='AutoAugment', policies=[
        [dict(type='Albu', transforms=[dict(type='ShiftScaleRotate', shift_limit=0.0625, scale_limit=0.0, rotate_limit=0, interpolation=1, p=1)],
            bbox_params=dict(type='BboxParams', format='pascal_voc',
                             label_fields=['gt_bboxes_labels', 'gt_ignore_flags'], min_visibility=0.0,
                             filter_lost_elements=True),
            keymap=dict(img='image', gt_bboxes='bboxes'),
            skip_img_without_anno=True)],
        [dict(type='Albu', transforms=[dict(type='RGBShift', r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=1.0)],
                # dict(type='Cutout', num_holes=10, max_h_size=20, max_w_size=20, fill_value=0, p=0.5)
            bbox_params=dict(type='BboxParams', format='pascal_voc',
                             label_fields=['gt_bboxes_labels', 'gt_ignore_flags'], min_visibility=0.0,
                             filter_lost_elements=True),
            keymap=dict(img='image', gt_bboxes='bboxes'),
            skip_img_without_anno=True)],
        [dict(type='Albu', transforms=[dict(type='HueSaturationValue', hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
            # dict(type='GaussNoise', p=1.0),
            # dict(type='Cutout', num_holes=10, max_h_size=20, max_w_size=20, fill_value=0, p=0.5)
            ],
              bbox_params=dict(type='BboxParams', format='pascal_voc',
                               label_fields=['gt_bboxes_labels', 'gt_ignore_flags'], min_visibility=0.0,
                               filter_lost_elements=True),
              keymap=dict(img='image', gt_bboxes='bboxes'),
              skip_img_without_anno=True)],
        [dict(type='Albu', transforms=[dict(type='Sharpen', p=1.0)],
              bbox_params=dict(type='BboxParams', format='pascal_voc',
                               label_fields=['gt_bboxes_labels', 'gt_ignore_flags'], min_visibility=0.0,
                               filter_lost_elements=True),
              keymap=dict(img='image', gt_bboxes='bboxes'),
              skip_img_without_anno=True)],
        [dict(type='Albu', transforms=[dict(type='Emboss', p=1.0), dict(type='ChannelShuffle', p=0.5)],
              bbox_params=dict(type='BboxParams', format='pascal_voc',
                               label_fields=['gt_bboxes_labels', 'gt_ignore_flags'], min_visibility=0.0,
                               filter_lost_elements=True),
              keymap=dict(img='image', gt_bboxes='bboxes'),
              skip_img_without_anno=True)],
        [dict(type='Albu', transforms=[dict(type='ChannelShuffle', p=1)],
              bbox_params=dict(type='BboxParams', format='pascal_voc',
                               label_fields=['gt_bboxes_labels', 'gt_ignore_flags'], min_visibility=0.0,
                               filter_lost_elements=True),
              keymap=dict(img='image', gt_bboxes='bboxes'),
              skip_img_without_anno=True
              )],
        [dict(type='Albu', transforms=[
            dict(type='RandomBrightnessContrast', brightness_limit=[0.1, 0.3], contrast_limit=[0.1, 0.3], p=1)],
              bbox_params=dict(type='BboxParams', format='pascal_voc',
                               label_fields=['gt_bboxes_labels', 'gt_ignore_flags'], min_visibility=0.0,
                               filter_lost_elements=True),
              keymap=dict(img='image', gt_bboxes='bboxes'), skip_img_without_anno=True)],
        [dict(type='Albu', transforms=[dict(type='CLAHE', clip_limit=2, p=1.0)],
              bbox_params=dict(type='BboxParams', format='pascal_voc',
                               label_fields=['gt_bboxes_labels', 'gt_ignore_flags'], min_visibility=0.0,
                               filter_lost_elements=True),
              keymap=dict(img='image', gt_bboxes='bboxes'),
              skip_img_without_anno=True
              )]
    ]
         ),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1300, 1300), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        _delete_=True,
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type='ClassBalancedDataset',
            oversample_thr=0.2,
            dataset=dict(
                type=dataset_type,
                data_root=data_root,
                metainfo=metainfo,
                ann_file='train.json',
                data_prefix=dict(img_path='train_imgs'),
                filter_cfg=dict(filter_empty_gt=False, min_size=32),
                pipeline=train_pipeline
            )
        )
    )
)

val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=metainfo,
        data_root=data_root,
        ann_file='val.json',
        data_prefix=dict(_delete_=True, img_path='train_imgs'),
        test_mode=True,
        pipeline=test_pipeline
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(
    _delete_=True,
    type='VOCMetric',
    metric='mAP',
    eval_mode='11points'
)
test_evaluator = val_evaluator
