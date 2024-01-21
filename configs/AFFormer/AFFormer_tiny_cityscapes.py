_base_ = [
    '../_base_/models/afformer.py', '../_base_/datasets/cityscapes_1024x1024.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
model = dict(
    pretrained='./ckpt/AFFormer_tiny_ImageNet1k.pth',
    backbone=dict(
        type='afformer_tiny',
        strides=[4, 2, 2, 2]),
    decode_head=dict(
        in_channels=[216],
        in_index=[3],
        channels=256,
        aff_channels=256,
        aff_kwargs=dict(MD_R=16),
        num_classes=19
    )
    )

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(_delete_=True, type='AdamW', lr=0.0004, betas=(0.9, 0.999), weight_decay=0.01)

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 2 GPUs with 4 images per GPU
data=dict(samples_per_gpu=8, workers_per_gpu=8)
find_unused_parameters=True
