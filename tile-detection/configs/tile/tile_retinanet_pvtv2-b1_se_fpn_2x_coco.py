_base_ = './tile_retinanet_pvtv2-b1_fpn_2x_coco.py'
model = dict(
    type='RetinaNet',
    backbone=dict(
        type='PVTv2CSAM',
        embed_dims=64,
        num_layers=[3, 4, 6, 3],
        init_cfg=dict(checkpoint='https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b1.pth')
    )
)
# optimizer
optim_wrapper = dict(
    # type='AmpOptimWrapper',
    optimizer=dict(
        _delete_=True, type='AdamW', lr=3e-5, weight_decay=0.0001))
