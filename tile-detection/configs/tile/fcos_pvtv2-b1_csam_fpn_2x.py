_base_ = './fcos_pvtv2-b1_fpn_2x.py'
model = dict(

    backbone=dict(
        type='PVTv2CSAM',
        embed_dims=64,
        num_layers=[2, 2, 2, 2],
        init_cfg=dict(checkpoint='https://github.com/whai362/PVT/'
                      'releases/download/v2/pvt_v2_b1.pth')
    )
)
