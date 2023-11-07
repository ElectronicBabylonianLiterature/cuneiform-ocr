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
#custom_classes = ['ABZ579', 'ABZ13', 'ABZ342', 'ABZ70', 'ABZ461', 'ABZ142', 'ABZ480', 'ABZ1', 'ABZ231', 'ABZ533', 'ABZ449', 'ABZ318', 'ABZ75', 'ABZ61', 'ABZ354', 'ABZ139', 'ABZ381', 'ABZ597', 'ABZ536', 'ABZ308', 'ABZ330', 'ABZ328', 'ABZ86', 'ABZ15', 'ABZ214', 'ABZ545', 'ABZ73', 'ABZ295', 'ABZ55', 'ABZ335', 'ABZ371', 'ABZ151', 'ABZ457', 'ABZ537', 'ABZ69', 'ABZ353', 'ABZ68', 'ABZ5', 'ABZ296', 'ABZ84', 'ABZ366', 'ABZ411', 'ABZ396', 'ABZ206', 'ABZ58', 'ABZ324', 'ABZ376', 'ABZ99', 'ABZ384', 'ABZ59', 'ABZ532', 'ABZ334', 'ABZ589', 'ABZ383', 'ABZ343', 'ABZ586', 'ABZ399', 'ABZ74', 'ABZ211', 'ABZ145', 'ABZ7', 'ABZ212', 'ABZ78', 'ABZ367', 'ABZ38', 'ABZ319', 'ABZ85', 'ABZ115', 'ABZ322', 'ABZ97', 'ABZ144', 'ABZ112', 'ABZ427', 'ABZ60', 'ABZ207', 'ABZ79', 'ABZ80', 'ABZ232', 'ABZ142a', 'ABZ312', 'ABZ52', 'ABZ331', 'ABZ128', 'ABZ314', 'ABZ535', 'ABZ575', 'ABZ134', 'ABZ465', 'ABZ167', 'ABZ172', 'ABZ339', 'ABZ6', 'ABZ331e+152i', 'ABZ306', 'ABZ12', 'ABZ2', 'ABZ148', 'ABZ397', 'ABZ554', 'ABZ570', 'ABZ441', 'ABZ147', 'ABZ472', 'ABZ104', 'ABZ440', 'ABZ230', 'ABZ595', 'ABZ455', 'ABZ313', 'ABZ298', 'ABZ412', 'ABZ62', 'ABZ468', 'ABZ101', 'ABZ111', 'ABZ483', 'ABZ538', 'ABZ471', 'ABZ87', 'ABZ143', 'ABZ565', 'ABZ205', 'ABZ152', 'ABZ72', 'ABZ138', 'ABZ401', 'ABZ50', 'ABZ406', 'ABZ307', 'ABZ126', 'ABZ124', 'ABZ164', 'ABZ529', 'ABZ559', 'ABZ94', 'ABZ437', 'ABZ56', 'ABZ393', 'ABZ398']
#not_found_class = ["SignClassNotInImageClassificationTrainData"]
#classes = [*custom_classes, *not_found_class]
classes = [
            'ABZ579',
            'ABZ13',
            'ABZ480',
            'ABZ70',
            'ABZ342',
            'ABZ597',
            'ABZ461',
            'ABZ142',
            'ABZ381',
            'ABZ1',
            'ABZ61',
            'ABZ318',
            'ABZ533',
            'ABZ231',
            'ABZ449',
            'ABZ75',
            'ABZ354',
            'ABZ545',
            'ABZ139',
            'ABZ330',
            'ABZ536',
            'ABZ308',
            'ABZ86',
            'ABZ15',
            'ABZ328',
            'ABZ214',
            'ABZ73',
            'ABZ295',
            'ABZ55',
            'ABZ537',
            'ABZ69',
            'ABZ371',
            'ABZ296',
            'ABZ457',
            'ABZ151',
            'ABZ411',
            'ABZ68',
            'ABZ335',
            'ABZ366',
            'ABZ5',
            'ABZ324',
            'ABZ396',
            'ABZ353',
            'ABZ99',
            'ABZ206',
            'ABZ84',
            'ABZ532',
            'ABZ376',
            'ABZ58',
            'ABZ384',
            'ABZ74',
            'ABZ334',
            'ABZ59',
            'ABZ383',
            'ABZ145',
            'ABZ399',
            'ABZ7',
            'ABZ589',
            'ABZ586',
            'ABZ97',
            'ABZ211',
            'ABZ343',
            'ABZ367',
            'ABZ52',
            'ABZ212',
            'ABZ85',
            'ABZ115',
            'ABZ319',
            'ABZ207',
            'ABZ78',
            'ABZ144',
            'ABZ465',
            'ABZ38',
            'ABZ570',
            'ABZ322',
            'ABZ331',
            'ABZ60',
            'ABZ427',
            'ABZ112',
            'ABZ80',
            'ABZ314',
            'ABZ79',
            'ABZ142a',
            'ABZ232',
            'ABZ312',
            'ABZ535',
            'ABZ554',
            'ABZ595',
            'ABZ128',
            'ABZ339',
            'ABZ12',
            'ABZ172',
            'ABZ331e+152i',
            'ABZ147',
            'ABZ575',
            'ABZ167',
            'ABZ230',
            'ABZ279',
            'ABZ401',
            'ABZ306',
            'ABZ468',
            'ABZ6',
            'ABZ472',
            'ABZ148',
            'ABZ2',
            'ABZ104',
            'ABZ313',
            'ABZ397',
            'ABZ134',
            'ABZ412',
            'ABZ441',
            'ABZ62',
            'ABZ455',
            'ABZ440',
            'ABZ471',
            'ABZ111',
            'ABZ538',
            'ABZ72',
            'ABZ101',
            'ABZ393',
            'ABZ50',
            'ABZ298',
            'ABZ437',
            'ABZ94',
            'ABZ143',
            'ABZ483',
            'ABZ205',
            'ABZ565',
            'ABZ191',
            'ABZ124',
            'ABZ152',
            'ABZ87',
            'ABZ138',
            'ABZ559',
            'ABZ164',
            'ABZ126',
            'ABZ598a',
            'ABZ195',
            'ABZ307',
            'ABZ9',
            'ABZ556',
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