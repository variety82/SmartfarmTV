# dataset settings
dataset_type = 'MultiImageMixDataset'
data_root = '/opt/ml/mmdetection/MMDetection/data/'
classes = ["Cabbage"]
albu_transforms = [
    dict(
        type='RandomBrightnessContrast',
        p=0.5),

]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Albu',
        transforms = albu_transforms),
    dict(type = 'Mosaic',img_scale=(1024, 1024)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
'''
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize',img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip',flip_ratio=0.5),
    dict(
        type='Albu',
        transforms = albu_transforms
        ),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Normalize', **img_norm_cfg),    
    dict(type='Collect', keys=['img']),
]
'''
#기존 
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize',img_scale=(1024, 1024), keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Albu',
                transforms = albu_transforms
                ),
            
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
'''
dict(type='LoadImageFromFile'),
dict(type='Resize',img_scale=(512, 512), keep_ratio=True),
dict(type='RandomFlip', flip_ratio=0.5),
dict(type='Normalize', **img_norm_cfg),
dict(type='Pad', size_divisor=32),
dict(type='ImageToTensor', keys=['img']),
dict(type='Collect', keys=['img']),
'''

        

data = dict(
    _delete_=True,
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        dataset=dict(
            classes=classes,
            type='CocoDataset',
            ann_file=data_root + '/coco/normal_annotations.json',
            img_prefix=data_root + 'train/images_normal',
            pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),]
            ),
        pipeline=train_pipeline
        ),
    val=dict(
        classes=classes,
        type='CocoDataset',
        ann_file=data_root + '/coco/val_normal_annotations.json',
        img_prefix=data_root + 'val/images_normal',
        pipeline=test_pipeline),
    test=dict(
        classes=classes,
        type='CocoDataset',
        ann_file=data_root + '/coco/val_normal_annotations.json',
        img_prefix=data_root + 'val/images_normal',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')


