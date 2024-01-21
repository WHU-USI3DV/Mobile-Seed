# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    type='EncoderDecoderRefine',
    down_ratio = 1,
    pretrained=None,
    backbone=dict(
        type='AFFormer_for_MS_base',
        strides=[4, 2, 2, 2],
        drop_path_rate = 0.1),
    decode_head=dict(),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

