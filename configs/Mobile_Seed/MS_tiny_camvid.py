_base_ = [
    '../_base_/models/Mobile_Seed.py', '../_base_/datasets/camvid_boundary.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
model = dict(
    pretrained='./ckpt/AFFormer_tiny_ImageNet1k.pth',
    backbone=dict(
        type='AFFormer_for_MS_tiny'),
    decode_head=[
        dict(
            type="BoundaryHead",
            in_channels = [16,64,216,216], # /2 /4 /8 /8
            bound_channels = [16,16,32,32],
            bound_ratio = 2,
            in_index = [1,3,5,6],
            channels= 16 + 16 + 32 + 32,
            num_classes=1,
            loss_decode= dict(type='ML_BCELoss', use_sigmoid=True, loss_weight=1.0,loss_name = "loss_be")),
        dict(
            type="RefineHead",
            fuse_channel = 96,
            in_channels=[216],
            in_index=[-1],
            channels=256,
            num_classes=11,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,loss_name = "loss_ce")),
                ],
    # test_cfg = dict(mode='slide',crop_size=(512, 512), stride=(384, 384))
    )

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(_delete_=True, type='AdamW', lr=0.0004, betas=(0.9, 0.999), weight_decay=0.01)

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0 , by_epoch=False)

# By default, models are trained on 2 GPUs with 4 images per GPU
data=dict(samples_per_gpu=16, workers_per_gpu=16)
find_unused_parameters=True


