custom_imports= dict(imports=['fcenet'], allow_failed_imports=False)


model = dict(
    type="FCENet",
    backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type="BN", requires_grad=True),
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
        norm_eval=True,
        style="pytorch",
        dcn=dict(type="DCNv2", deform_groups=2, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
    ),
    neck=dict(
        type="FPN",
        in_channels=[512, 1024, 2048],
        out_channels=256,
        add_extra_convs="on_output",
        num_outs=3,
        relu_before_extra_convs=True,
        act_cfg=None,
    ),
    det_head=dict(
        type="FCEHead",
        in_channels=256,
        fourier_degree=5,
        module_loss=dict(
            type="FCEModuleLoss",
            num_sample=50,
            level_proportion_range=((0, 0.25), (0.2, 0.65), (0.55, 1.0)),
        ),
        postprocessor=dict(
            type="FCEPostprocessor",
            scales=(8, 16, 32),
            text_repr_type="poly",
            num_reconstr_points=50,
            alpha=1.0,
            beta=2.0,
            score_thr=0.3,
        ),
    ),
    data_preprocessor=dict(
        type="TextDetDataPreprocessor",
        mean=[0, 0, 0],
        std=[1, 1, 1],
        bgr_to_rgb=True,
        pad_size_divisor=32,
    ),
)
"""
classes = ['ABZ579', 'ABZ13', 'ABZ342', 'ABZ70', 'ABZ461', 'ABZ142', 'ABZ318', 'ABZ231', 'ABZ1', 'ABZ480', 'ABZ533', 'ABZ449', 'ABZ75', 'ABZ354', 'ABZ61', 'ABZ597', 'ABZ536', 'ABZ139', 'ABZ381', 'ABZ308', 'ABZ86', 'ABZ328', 'ABZ330', 'ABZ69', 'ABZ214', 'ABZ73', 'ABZ545', 'ABZ15', 'ABZ295', 'ABZ296', 'ABZ151', 'ABZ55', 'ABZ335', 'ABZ537', 'ABZ371', 'ABZ68', 'ABZ457', 'ABZ84', 'ABZ366', 'ABZ5', 'ABZ353', 'ABZ396', 'ABZ411', 'ABZ206', 'ABZ58', 'ABZ324', 'ABZ99', 'ABZ376', 'ABZ532', 'ABZ384', 'ABZ334', 'ABZ383', 'ABZ74', 'ABZ59', 'ABZ343', 'ABZ145', 'ABZ589', 'ABZ586', 'ABZ211', 'ABZ212', 'ABZ399', 'ABZ7', 'ABZ367', 'ABZ78', 'ABZ115', 'ABZ322', 'ABZ207', 'ABZ38', 'ABZ319', 'ABZ144', 'ABZ85', 'ABZ97', 'ABZ112', 'ABZ60', 'ABZ79', 'ABZ427', 'ABZ232', 'ABZ80', 'ABZ167', 'ABZ312', 'ABZ535', 'ABZ52', 'ABZ172', 'ABZ331', 'ABZ554', 'ABZ314', 'ABZ128', 'ABZ142a', 'ABZ12', 'ABZ331e+152i', 'ABZ401', 'ABZ147', 'ABZ440', 'ABZ6', 'ABZ575', 'ABZ570', 'ABZ134', 'ABZ465', 'ABZ230', 'ABZ306', 'ABZ148', 'ABZ339', 'ABZ397', 'ABZ472', 'ABZ441', 'ABZ412', 'ABZ104', 'ABZ595', 'ABZ455', 'ABZ313', 'ABZ298', 'ABZ62', 'ABZ101', 'ABZ393', 'ABZ483', 'ABZ471', 'ABZ111', 'ABZ87', 'ABZ538', 'ABZ468', 'ABZ138', 'ABZ565', 'ABZ152', 'ABZ406', 'ABZ72', 'ABZ205', 'ABZ126', 'ABZ2', 'ABZ50', 'ABZ94', 'ABZ529', 'ABZ307', 'ABZ143', 'ABZ124', 'ABZ164', 'ABZ559', 'ABZ437', 'ABZ9', 'ABZ398', 'ABZ131']

metainfo = {
    'classes': classes,
    'palette': [
        (220, 20, 60),
    ] * len(classes),
}
"""
classes = ("null",)
metainfo = {
    'classes': ('null', ),
    'palette': [
        (220, 20, 60),
    ]
}


dataset_type = 'CocoDataset'
data_root = 'data/coco/'
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk'),         color_type="color_ignore_orientation"),
    dict(type="Resize", scale=(2260, 2260), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='CocoDataset',
        metainfo=metainfo,
        data_root='data/coco/',
        ann_file='annotations/instances_train2017.json',
        data_prefix=dict(img='train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(type='Resize', scale=(1333, 800), keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PackDetInputs')
        ]))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        metainfo= metainfo,
        data_root='data/coco/',
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk'), color_type="color_ignore_orientation"),
            dict(type='Resize', scale=(2260, 2260), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ]))



test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        metainfo=metainfo,
        data_root='data/coco/',
        ann_file='annotations/instances_val2017.json',
        data_prefix=dict(img='val2017/'),
        test_mode=True,
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk'),
                color_type="color_ignore_orientation"),

            dict(type='Resize', scale=(2260, 2260), keep_ratio=True),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
            dict(
                type='PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ]))
"""
val_evaluator = dict(
    type='CocoMetric',
    ann_file='data/coco/annotations/instances_val2017.json',
    metric=['bbox'],
    format_only=False)
"""
val_evaluator = dict(
    type='VOCMetric',
    iou_thrs=0.5,
)

test_evaluator = val_evaluator
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001))
auto_scale_lr = dict(enable=False, base_batch_size=16)
default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook', draw=True, show=True)
    )
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = 'checkpoints/fcenet_split1.pth'
resume = False
launcher = 'none'
work_dir = './work_dirs/mask-rcnn_r50_fpn_1x_coco'