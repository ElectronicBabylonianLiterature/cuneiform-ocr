custom_imports= dict(imports=['fcenet'], allow_failed_imports=False)


model = dict(
    backbone=dict(
        dcn=dict(deform_groups=2, fallback_on_stride=False, type='DCNv2'),
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
        stage_with_dcn=(
            False,
            True,
            True,
            True,
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

classes = [
    'ABZ13',
    'ABZ579',
    'ABZ480',
    'ABZ70',
    'ABZ597',
    'ABZ342',
    'ABZ461',
    'ABZ381',
    'ABZ61',
    'ABZ1',
    'ABZ142',
    'ABZ318',
    'ABZ231',
    'ABZ75',
    'ABZ449',
    'ABZ533',
    'ABZ354',
    'ABZ139',
    'ABZ545',
    'ABZ536',
    'ABZ330',
    'ABZ308',
    'ABZ86',
    'ABZ328',
    'ABZ214',
    'ABZ73',
    'ABZ15',
    'ABZ295',
    'ABZ296',
    'ABZ68',
    'ABZ55',
    'ABZ69',
    'ABZ537',
    'ABZ371',
    'ABZ5',
    'ABZ151',
    'ABZ411',
    'ABZ457',
    'ABZ335',
    'ABZ366',
    'ABZ324',
    'ABZ396',
    'ABZ206',
    'ABZ99',
    'ABZ84',
    'ABZ353',
    'ABZ532',
    'ABZ58',
    'ABZ384',
    'ABZ376',
    'ABZ59',
    'ABZ334',
    'ABZ74',
    'ABZ383',
    'ABZ144',
    'ABZ589',
    'ABZ586',
    'ABZ7',
    'ABZ97',
    'ABZ211',
    'ABZ399',
    'ABZ52',
    'ABZ145',
    'ABZ343',
    'ABZ367',
    'ABZ212',
    'ABZ78',
    'ABZ85',
    'ABZ319',
    'ABZ207',
    'ABZ115',
    'ABZ465',
    'ABZ322',
    'ABZ570',
    'ABZ331',
    'ABZ38',
    'ABZ427',
    'ABZ279',
    'ABZ112',
    'ABZ79',
    'ABZ80',
    'ABZ60',
    'ABZ535',
    'ABZ142a',
    'ABZ314',
    'ABZ232',
    'ABZ554',
    'ABZ312',
    'ABZ172',
    'ABZ128',
    'ABZ6',
    'ABZ595',
    'ABZ230',
    'ABZ167',
    'ABZ12',
    'ABZ331e+152i',
    'ABZ306',
    'ABZ339',
    'ABZ134',
    'ABZ575',
    'ABZ401',
    'ABZ313',
    'ABZ472',
    'ABZ441',
    'ABZ62',
    'ABZ111',
    'ABZ468',
    'ABZ148',
    'ABZ397',
    'ABZ104',
    'ABZ147',
    'ABZ455',
    'ABZ412',
    'ABZ471',
    'ABZ2',
    'ABZ440',
    'ABZ101',
    'ABZ538',
    'ABZ72',
    'ABZ298',
    'ABZ437',
    'ABZ143',
    'ABZ393',
    'ABZ483',
    'ABZ94',
    'ABZ565',
    'ABZ559',
    'ABZ138',
    'ABZ87',
    'ABZ50',
    'ABZ191',
    'ABZ152',
    'ABZ124',
    'ABZ398',
    'ABZ205',
    'ABZ9',
    'ABZ126',
    'ABZ164',
    'ABZ195',
    'ABZ307',
    'ABZ598a',
]

metainfo = {
    'classes': classes,
    'palette': [
        (220, 20, 60),
    ] * len(classes),
}



dataset_type = 'CocoDataset'
data_root = 'data/coco/'
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk') ),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
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
"""

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
    visualization=dict(type='DetVisualizationHook', draw=False, show=False)
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
work_dir = './work_dirs/recognition'