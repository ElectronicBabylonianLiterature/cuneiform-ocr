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
        mean=[86.65888836888392, 67.92744567921709, 53.78325960605914],
        std=[68.98970994105028, 57.20489382979894, 48.230552014910586],
        bgr_to_rgb=False,
        pad_size_divisor=32,
    ),
)
classes = [
    'ABZ579',
    'ABZ13',
    'ABZ480',
    'ABZ70',
    'ABZ597',
    'ABZ342',
    'ABZ461',
    'ABZ381',
    'ABZ1',
    'ABZ61',
    'ABZ142',
    'ABZ318',
    'ABZ231',
    'ABZ533',
    'ABZ449',
    'ABZ75',
    'ABZ354',
    'ABZ139',
    'ABZ545',
    'ABZ536',
    'ABZ330',
    'ABZ308',
    'ABZ15',
    'ABZ86',
    'ABZ73',
    'ABZ214',
    'ABZ328',
    'ABZ55',
    'ABZ296',
    'ABZ371',
    'ABZ68',
    'ABZ295',
    'ABZ537',
    'ABZ411',
    'ABZ457',
    'ABZ5',
    'ABZ335',
    'ABZ151',
    'ABZ69',
    'ABZ366',
    'ABZ396',
    'ABZ324',
    'ABZ99',
    'ABZ206',
    'ABZ353',
    'ABZ84',
    'ABZ532',
    'ABZ384',
    'ABZ58',
    'ABZ376',
    'ABZ59',
    'ABZ74',
    'ABZ334',
    'ABZ399',
    'ABZ97',
    'ABZ52',
    'ABZ586',
    'ABZ7',
    'ABZ211',
    'ABZ145',
    'ABZ383',
    'ABZ589',
    'ABZ367',
    'ABZ319',
    'ABZ343',
    'ABZ85',
    'ABZ144',
    'ABZ570',
    'ABZ78',
    'ABZ115',
    'ABZ212',
    'ABZ207',
    'ABZ465',
    'ABZ322',
    'ABZ112',
    'ABZ38',
    'ABZ331',
    'ABZ427',
    'ABZ60',
    'ABZ79',
    'ABZ80',
    'ABZ314',
    'ABZ142a',
    'ABZ595',
    'ABZ232',
    'ABZ535',
    'ABZ279',
    'ABZ172',
    'ABZ312',
    'ABZ6',
    'ABZ554',
    'ABZ230',
    'ABZ128',
    'ABZ468',
    'ABZ167',
    'ABZ401',
    'ABZ575',
    'ABZ12',
    'ABZ313',
    'ABZ148',
    'ABZ339',
    'ABZ104',
    'ABZ331e+152i',
    'ABZ472',
    'ABZ306',
    'ABZ134',
    'ABZ2',
    'ABZ441',
    'ABZ412',
    'ABZ147',
    'ABZ471',
    'ABZ397',
    'ABZ62',
    'ABZ111',
    'ABZ455',
    'ABZ72',
    'ABZ538',
    'ABZ143',
    'ABZ101',
    'ABZ440',
    'ABZ437',
    'ABZ393',
    'ABZ298',
    'ABZ50',
    'ABZ483',
    'ABZ559',
    'ABZ87',
    'ABZ94',
    'ABZ152',
    'ABZ138',
    'ABZ164',
    'ABZ565',
    'ABZ205',
    'ABZ598a',
    'ABZ307',
    'ABZ9',
    'ABZ398',
    'ABZ191',
    'ABZ126',
    'ABZ124',
    'ABZ195',
    'ABZ470',
    'ABZ131',
    'ABZ375',
    'ABZ56',
    'ABZ556',
    'ABZ170',
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