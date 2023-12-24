auto_scale_lr = dict(base_batch_size=8)
custom_hooks = [
    dict(by_epoch=True, type='CustomTensorboardLoggerHook'),
]
default_hooks = dict(
    checkpoint=dict(interval=100, type='CheckpointHook'),
    logger=dict(interval=1, type='CustomLoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    sync_buffer=dict(type='SyncBuffersHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(
        draw_gt=True,
        draw_pred=True,
        enable=True,
        interval=1,
        show=False,
        type='VisualizationHook'))
default_scope = 'mmocr'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
file_client_args = dict(backend='disk')
launcher = 'none'
load_from = 'logs/yunus/epoch_200.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=10)
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=-1,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            1,
            2,
            3,
        ),
        style='pytorch',
        type='mmdet.ResNet'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            86.65888836888392,
            67.92744567921709,
            53.78325960605914,
        ],
        pad_size_divisor=32,
        std=[
            68.98970994105028,
            57.20489382979894,
            48.230552014910586,
        ],
        type='TextDetDataPreprocessor'),
    det_head=dict(
        fourier_degree=5,
        in_channels=256,
        module_loss=dict(
            level_proportion_range=(
                (
                    0,
                    0.25,
                ),
                (
                    0.2,
                    0.65,
                ),
                (
                    0.55,
                    1.0,
                ),
            ),
            num_sample=50,
            type='FCEModuleLoss'),
        postprocessor=dict(
            alpha=1.0,
            beta=2.0,
            num_reconstr_points=50,
            scales=(
                8,
                16,
                32,
            ),
            score_thr=0.3,
            text_repr_type='poly',
            type='FCEPostprocessor'),
        type='FCEHead'),
    neck=dict(
        act_cfg=None,
        add_extra_convs='on_output',
        in_channels=[
            512,
            1024,
            2048,
        ],
        num_outs=3,
        out_channels=256,
        relu_before_extra_convs=True,
        type='mmdet.FPN'),
    type='FCENet')
optim_wrapper = dict(
    optimizer=dict(lr=0.00098, momentum=0.9, type='SGD', weight_decay=0.0005),
    type='OptimWrapper')
optimizer_config = dict(
    cumulative_iters=2, type='GradientCumulativeOptimizerHook')
param_scheduler = [
    dict(end=1000, eta_min=1e-07, power=0.9, type='PolyLR'),
]
randomness = dict(seed=None)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file='textdet_test.json',
        data_root='data/icdar2015',
        pipeline=[
            dict(
                color_type='color_ignore_orientation',
                file_client_args=dict(backend='disk'),
                type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                2260,
                2260,
            ), type='Resize'),
            dict(
                type='LoadOCRAnnotations',
                with_bbox=True,
                with_label=True,
                with_polygon=True),
            dict(
                meta_keys=(
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackTextDetInputs'),
        ],
        test_mode=True,
        type='OCRDataset'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(type='HmeanIOUMetric')
train_cfg = dict(
    max_epochs=1000, type='EpochBasedTrainLoop', val_interval=1000)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file='textdet_train.json',
        data_root='data/icdar2015',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(
                color_type='color_ignore_orientation',
                file_client_args=dict(backend='disk'),
                type='LoadImageFromFile'),
            dict(
                type='LoadOCRAnnotations',
                with_bbox=True,
                with_label=True,
                with_polygon=True),
            dict(min_poly_points=4, type='FixInvalidPolygon'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.75,
                    2.5,
                ),
                scale=(
                    800,
                    800,
                ),
                type='RandomResize'),
            dict(
                crop_ratio=0.5,
                iter_num=1,
                min_area_ratio=0.2,
                type='TextDetRandomCropFlip'),
            dict(
                prob=0.8,
                transforms=[
                    dict(min_side_ratio=0.3, type='RandomCrop'),
                ],
                type='RandomApply'),
            dict(
                prob=0.6,
                transforms=[
                    dict(
                        max_angle=35,
                        pad_with_fixed_color=True,
                        type='RandomRotate',
                        use_canvas=True),
                ],
                type='RandomApply'),
            dict(
                prob=[
                    0.6,
                    0.4,
                ],
                transforms=[
                    [
                        dict(keep_ratio=True, scale=800, type='Resize'),
                        dict(size=(
                            800,
                            800,
                        ), type='Pad'),
                    ],
                    dict(keep_ratio=False, scale=800, type='Resize'),
                ],
                type='RandomChoice'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(direction='vertical', prob=0.5, type='RandomFlip'),
            dict(
                prob=0.3333333333333333,
                transforms=[
                    dict(
                        alpha=75.0,
                        op='ElasticTransform',
                        type='TorchVisionWrapper'),
                ],
                type='RandomApply'),
            dict(
                prob=0.75,
                transforms=[
                    dict(
                        prob=[
                            0.3333333333333333,
                            0.3333333333333333,
                            0.3333333333333333,
                        ],
                        transforms=[
                            dict(
                                op='RandomAdjustSharpness',
                                sharpness_factor=0,
                                type='TorchVisionWrapper'),
                            dict(
                                op='RandomAdjustSharpness',
                                sharpness_factor=60,
                                type='TorchVisionWrapper'),
                            dict(
                                op='RandomAdjustSharpness',
                                sharpness_factor=90,
                                type='TorchVisionWrapper'),
                        ],
                        type='RandomChoice'),
                ],
                type='RandomApply'),
            dict(
                brightness=0.15,
                contrast=0.3,
                op='ColorJitter',
                saturation=0.5,
                type='TorchVisionWrapper'),
            dict(
                prob=0.8,
                transforms=[
                    dict(
                        prob=[
                            0.5,
                            0.5,
                        ],
                        transforms=[
                            dict(
                                op='RandomEqualize',
                                type='TorchVisionWrapper'),
                            dict(
                                op='RandomAutocontrast',
                                type='TorchVisionWrapper'),
                        ],
                        type='RandomChoice'),
                ],
                type='RandomApply'),
            dict(min_poly_points=4, type='FixInvalidPolygon'),
            dict(
                meta_keys=(
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackTextDetInputs'),
        ],
        type='OCRDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(
        color_type='color_ignore_orientation',
        file_client_args=dict(backend='disk'),
        type='LoadImageFromFile'),
    dict(
        type='LoadOCRAnnotations',
        with_bbox=True,
        with_label=True,
        with_polygon=True),
    dict(min_poly_points=4, type='FixInvalidPolygon'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.75,
            2.5,
        ),
        scale=(
            800,
            800,
        ),
        type='RandomResize'),
    dict(
        crop_ratio=0.5,
        iter_num=1,
        min_area_ratio=0.2,
        type='TextDetRandomCropFlip'),
    dict(
        prob=0.8,
        transforms=[
            dict(min_side_ratio=0.3, type='RandomCrop'),
        ],
        type='RandomApply'),
    dict(
        prob=0.6,
        transforms=[
            dict(
                max_angle=35,
                pad_with_fixed_color=True,
                type='RandomRotate',
                use_canvas=True),
        ],
        type='RandomApply'),
    dict(
        prob=[
            0.6,
            0.4,
        ],
        transforms=[
            [
                dict(keep_ratio=True, scale=800, type='Resize'),
                dict(size=(
                    800,
                    800,
                ), type='Pad'),
            ],
            dict(keep_ratio=False, scale=800, type='Resize'),
        ],
        type='RandomChoice'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(direction='vertical', prob=0.5, type='RandomFlip'),
    dict(
        prob=0.3333333333333333,
        transforms=[
            dict(alpha=75.0, op='ElasticTransform', type='TorchVisionWrapper'),
        ],
        type='RandomApply'),
    dict(
        prob=0.75,
        transforms=[
            dict(
                prob=[
                    0.3333333333333333,
                    0.3333333333333333,
                    0.3333333333333333,
                ],
                transforms=[
                    dict(
                        op='RandomAdjustSharpness',
                        sharpness_factor=0,
                        type='TorchVisionWrapper'),
                    dict(
                        op='RandomAdjustSharpness',
                        sharpness_factor=60,
                        type='TorchVisionWrapper'),
                    dict(
                        op='RandomAdjustSharpness',
                        sharpness_factor=90,
                        type='TorchVisionWrapper'),
                ],
                type='RandomChoice'),
        ],
        type='RandomApply'),
    dict(
        brightness=0.15,
        contrast=0.3,
        op='ColorJitter',
        saturation=0.5,
        type='TorchVisionWrapper'),
    dict(
        prob=0.8,
        transforms=[
            dict(
                prob=[
                    0.5,
                    0.5,
                ],
                transforms=[
                    dict(op='RandomEqualize', type='TorchVisionWrapper'),
                    dict(op='RandomAutocontrast', type='TorchVisionWrapper'),
                ],
                type='RandomChoice'),
        ],
        type='RandomApply'),
    dict(min_poly_points=4, type='FixInvalidPolygon'),
    dict(
        meta_keys=(
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackTextDetInputs'),
]
train_pipeline_enhanced = [
    dict(
        color_type='color_ignore_orientation',
        file_client_args=dict(backend='disk'),
        type='LoadImageFromFile'),
    dict(
        type='LoadOCRAnnotations',
        with_bbox=True,
        with_label=True,
        with_polygon=True),
    dict(min_poly_points=4, type='FixInvalidPolygon'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.75,
            2.5,
        ),
        scale=(
            800,
            800,
        ),
        type='RandomResize'),
    dict(
        crop_ratio=0.5,
        iter_num=1,
        min_area_ratio=0.2,
        type='TextDetRandomCropFlip'),
    dict(
        prob=0.8,
        transforms=[
            dict(min_side_ratio=0.3, type='RandomCrop'),
        ],
        type='RandomApply'),
    dict(
        prob=0.6,
        transforms=[
            dict(
                max_angle=35,
                pad_with_fixed_color=True,
                type='RandomRotate',
                use_canvas=True),
        ],
        type='RandomApply'),
    dict(
        prob=[
            0.6,
            0.4,
        ],
        transforms=[
            [
                dict(keep_ratio=True, scale=800, type='Resize'),
                dict(size=(
                    800,
                    800,
                ), type='Pad'),
            ],
            dict(keep_ratio=False, scale=800, type='Resize'),
        ],
        type='RandomChoice'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(direction='vertical', prob=0.5, type='RandomFlip'),
    dict(
        prob=0.3333333333333333,
        transforms=[
            dict(alpha=75.0, op='ElasticTransform', type='TorchVisionWrapper'),
        ],
        type='RandomApply'),
    dict(
        prob=0.75,
        transforms=[
            dict(
                prob=[
                    0.3333333333333333,
                    0.3333333333333333,
                    0.3333333333333333,
                ],
                transforms=[
                    dict(
                        op='RandomAdjustSharpness',
                        sharpness_factor=0,
                        type='TorchVisionWrapper'),
                    dict(
                        op='RandomAdjustSharpness',
                        sharpness_factor=60,
                        type='TorchVisionWrapper'),
                    dict(
                        op='RandomAdjustSharpness',
                        sharpness_factor=90,
                        type='TorchVisionWrapper'),
                ],
                type='RandomChoice'),
        ],
        type='RandomApply'),
    dict(
        brightness=0.15,
        contrast=0.3,
        op='ColorJitter',
        saturation=0.5,
        type='TorchVisionWrapper'),
    dict(
        prob=0.8,
        transforms=[
            dict(
                prob=[
                    0.5,
                    0.5,
                ],
                transforms=[
                    dict(op='RandomEqualize', type='TorchVisionWrapper'),
                    dict(op='RandomAutocontrast', type='TorchVisionWrapper'),
                ],
                type='RandomChoice'),
        ],
        type='RandomApply'),
    dict(min_poly_points=4, type='FixInvalidPolygon'),
    dict(
        meta_keys=(
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackTextDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=8,
    dataset=dict(
        ann_file='textdet_test.json',
        data_root='data/icdar2015',
        pipeline=[
            dict(
                color_type='color_ignore_orientation',
                file_client_args=dict(backend='disk'),
                type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                2260,
                2260,
            ), type='Resize'),
            dict(
                type='LoadOCRAnnotations',
                with_bbox=True,
                with_label=True,
                with_polygon=True),
            dict(
                meta_keys=(
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackTextDetInputs'),
        ],
        test_mode=True,
        type='OCRDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(type='HmeanIOUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='TextDetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ])
work_dir = 'logs/yunus-resume-reduced-lr'
