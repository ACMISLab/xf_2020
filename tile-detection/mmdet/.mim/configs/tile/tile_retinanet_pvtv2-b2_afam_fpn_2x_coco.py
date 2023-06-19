_base_ = './tile_retinanet_pvtv2-b2_fpn_2x_coco.py'
model = dict(
    type='RetinaNet',
    backbone=dict(
        type='PVTv2AFAM',
    )
)