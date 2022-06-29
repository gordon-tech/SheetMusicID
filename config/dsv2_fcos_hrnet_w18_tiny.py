model = dict(
    type='FCOS',
    pretrained='open-mmlab://msra/hrnetv2_w18',
    backbone=dict(
        type='HRNet',
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(18, 36)),
            stage3=dict(
                num_modules=4,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(18, 36)),
            stage4=dict(
                num_modules=3,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(18, 36)))),
    neck=dict(
        type='HRFPN',
        in_channels=[18, 36],
        out_channels=128,
        stride=1,
        num_outs=2),
    bbox_head=dict(
        type='FCOSHead',
        num_classes=34,
        in_channels=128,
        stacked_convs=4,
        feat_channels=128,
        strides=[2, 4],
        #regress_ranges=((-1, 9), (9, 20), (20, 40), (40, 1e8)),
        regress_ranges=((-1, 5), (5, 1e8)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='IoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)))
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.1,
    nms=dict(type='nms', iou_threshold=0.3),
    max_per_img=500)
dataset_type = 'DeepScoresV2Dataset'
data_root = '/home/data/cy/llj/dataset/mmdetection-test/data/ds2_dense/'
img_norm_cfg = dict(mean=[240, 240, 240], std=[57, 57, 57], to_rgb=False)

data = dict(
    imgs_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type='DeepScoresV2Dataset',
        ann_file='/home/data/cy/llj/dataset/mmdetection-test/data/ds2_dense/deepscores_train.json',
        img_prefix='/home/data/cy/llj/dataset/mmdetection-test/data/ds2_dense/images/',
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='PhotoMetricDistortion',
                brightness_delta=32,
                contrast_range=(0.5, 1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18),
            dict(type='Rotate', level=5, prob=0.5, max_rotate_angle=30),
            dict(type='Resize', img_scale=[(1400, 1920), (800, 1200)], keep_ratio=True),
            dict(type='RandomCrop', crop_size=(400, 400)),
            dict(type='RandomFlip', flip_ratio=0.0),
            dict(
                type='Normalize',
                mean=[240, 240, 240],
                std=[57, 57, 57],
                to_rgb=False),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type='DeepScoresV2Dataset',
        ann_file='/home/data/cy/llj/dataset/mmdetection-test/data/ds2_dense/deepscores_test.json',
        img_prefix='/home/data/cy/llj/dataset/mmdetection-test/data/ds2_dense/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=[(600, 800), (800, 600)],
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip', flip_ratio=0.0),
                    dict(
                        type='Normalize',
                        mean=[240, 240, 240],
                        std=[57, 57, 57],
                        to_rgb=False),
                    dict(type='Pad', size_divisor=32),
                    # dict(type='ImageToTensor', keys=['img']),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='DeepScoresV2Dataset',
        ann_file='/home/data/cy/llj/dataset/mmdetection-test/data/ds2_dense/deepscores_test.json',
        img_prefix='/home/data/cy/llj/dataset/mmdetection-test/data/ds2_dense/images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                # img_scale=[(800, 600), (600, 800), (800, 450), (450, 800)],
                img_scale=(800, 600),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip', flip_ratio=0.0),
                    dict(
                        type='Normalize',
                        mean=[240, 240, 240],
                        std=[57, 57, 57],
                        to_rgb=False),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    # dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]))

evaluation = dict(interval=100, metric='bbox')
optimizer = dict(
    type='SGD',
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(bias_lr_mult=2.0, bias_decay_mult=0.0))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='constant',
    warmup_iters=2000,
    warmup_ratio=0.3333333333333333,
    step=[500, 750])
total_epochs = 1000
checkpoint_config = dict(interval=10)
log_config = dict(
    interval=100,
    hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

work_dir = '/home/data/cy/llj/mmDetection/class34_hrnet_tiny_stride2/'
gpu_ids = range(0, 1)
seed = None
