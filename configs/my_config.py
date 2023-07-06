optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001))
param_scheduler = dict(
    type='MultiStepLR',
    by_epoch=True,
    milestones=[100, 150, 200, 250],
    gamma=0.1)
train_cfg = dict(by_epoch=True, max_epochs=700, val_interval=700)
val_cfg = dict()
test_cfg = dict()
auto_scale_lr = dict(base_batch_size=128)
default_scope = 'mmcls'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='CustomLoggerHook', interval=1),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=50),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='VisualizationHook', enable=False),
    sync_buffer=dict(type='SyncBuffersHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='ClsVisualizer', vis_backends=[dict(type='LocalVisBackend')])
log_level = 'INFO'
load_from = 'logs/heidel-140-12345677/epoch_500.pth'
resume = False
randomness = dict(seed=None, deterministic=False)
num_classes = 140
classes = [
    'ABZ579', 'ABZ13', 'ABZ342', 'ABZ70', 'ABZ461', 'ABZ142', 'ABZ318',
    'ABZ231', 'ABZ1', 'ABZ480', 'ABZ533', 'ABZ449', 'ABZ75', 'ABZ354', 'ABZ61',
    'ABZ597', 'ABZ536', 'ABZ139', 'ABZ381', 'ABZ308', 'ABZ86', 'ABZ328',
    'ABZ330', 'ABZ69', 'ABZ214', 'ABZ73', 'ABZ545', 'ABZ15', 'ABZ295',
    'ABZ296', 'ABZ151', 'ABZ55', 'ABZ335', 'ABZ537', 'ABZ371', 'ABZ68',
    'ABZ457', 'ABZ84', 'ABZ366', 'ABZ5', 'ABZ353', 'ABZ396', 'ABZ411',
    'ABZ206', 'ABZ58', 'ABZ324', 'ABZ99', 'ABZ376', 'ABZ532', 'ABZ384',
    'ABZ334', 'ABZ383', 'ABZ74', 'ABZ59', 'ABZ343', 'ABZ145', 'ABZ589',
    'ABZ586', 'ABZ211', 'ABZ212', 'ABZ399', 'ABZ7', 'ABZ367', 'ABZ78',
    'ABZ115', 'ABZ322', 'ABZ207', 'ABZ38', 'ABZ319', 'ABZ144', 'ABZ85',
    'ABZ97', 'ABZ112', 'ABZ60', 'ABZ79', 'ABZ427', 'ABZ232', 'ABZ80', 'ABZ167',
    'ABZ312', 'ABZ535', 'ABZ52', 'ABZ172', 'ABZ331', 'ABZ554', 'ABZ314',
    'ABZ128', 'ABZ142a', 'ABZ12', 'ABZ331e+152i', 'ABZ401', 'ABZ147', 'ABZ440',
    'ABZ6', 'ABZ575', 'ABZ570', 'ABZ134', 'ABZ465', 'ABZ230', 'ABZ306',
    'ABZ148', 'ABZ339', 'ABZ397', 'ABZ472', 'ABZ441', 'ABZ412', 'ABZ104',
    'ABZ595', 'ABZ455', 'ABZ313', 'ABZ298', 'ABZ62', 'ABZ101', 'ABZ393',
    'ABZ483', 'ABZ471', 'ABZ111', 'ABZ87', 'ABZ538', 'ABZ468', 'ABZ138',
    'ABZ565', 'ABZ152', 'ABZ406', 'ABZ72', 'ABZ205', 'ABZ126', 'ABZ2', 'ABZ50',
    'ABZ94', 'ABZ529', 'ABZ307', 'ABZ143', 'ABZ124', 'ABZ164', 'ABZ559',
    'ABZ437', 'ABZ9', 'ABZ398', 'ABZ131'
]
model = dict(
    type='ImageClassifier',
    backbone=dict(type='EfficientNet', arch='b0'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=140,
        in_channels=1280,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5)))
optimizer_config = dict(
    type='GradientCumulativeOptimizerHook', cumulative_iters=4)
dataset_type = 'CustomDataset'
data_preprocessor = dict(
    mean=[86.65888836888392, 67.92744567921709, 53.78325960605914],
    std=[68.98970994105028, 57.20489382979894, 48.230552014910586],
    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=380),
    dict(type='RandomFlip', prob=0.25, direction='horizontal'),
    dict(type='RandomFlip', prob=0.25, direction='vertical'),
    dict(type='PackClsInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=380),
    dict(type='PackClsInputs')
]
train_dataset = dict(
    type='CustomDataset',
    data_prefix='data/ebl/train_set/train_set',
    classes=[
        'ABZ579', 'ABZ13', 'ABZ342', 'ABZ70', 'ABZ461', 'ABZ142', 'ABZ318',
        'ABZ231', 'ABZ1', 'ABZ480', 'ABZ533', 'ABZ449', 'ABZ75', 'ABZ354',
        'ABZ61', 'ABZ597', 'ABZ536', 'ABZ139', 'ABZ381', 'ABZ308', 'ABZ86',
        'ABZ328', 'ABZ330', 'ABZ69', 'ABZ214', 'ABZ73', 'ABZ545', 'ABZ15',
        'ABZ295', 'ABZ296', 'ABZ151', 'ABZ55', 'ABZ335', 'ABZ537', 'ABZ371',
        'ABZ68', 'ABZ457', 'ABZ84', 'ABZ366', 'ABZ5', 'ABZ353', 'ABZ396',
        'ABZ411', 'ABZ206', 'ABZ58', 'ABZ324', 'ABZ99', 'ABZ376', 'ABZ532',
        'ABZ384', 'ABZ334', 'ABZ383', 'ABZ74', 'ABZ59', 'ABZ343', 'ABZ145',
        'ABZ589', 'ABZ586', 'ABZ211', 'ABZ212', 'ABZ399', 'ABZ7', 'ABZ367',
        'ABZ78', 'ABZ115', 'ABZ322', 'ABZ207', 'ABZ38', 'ABZ319', 'ABZ144',
        'ABZ85', 'ABZ97', 'ABZ112', 'ABZ60', 'ABZ79', 'ABZ427', 'ABZ232',
        'ABZ80', 'ABZ167', 'ABZ312', 'ABZ535', 'ABZ52', 'ABZ172', 'ABZ331',
        'ABZ554', 'ABZ314', 'ABZ128', 'ABZ142a', 'ABZ12', 'ABZ331e+152i',
        'ABZ401', 'ABZ147', 'ABZ440', 'ABZ6', 'ABZ575', 'ABZ570', 'ABZ134',
        'ABZ465', 'ABZ230', 'ABZ306', 'ABZ148', 'ABZ339', 'ABZ397', 'ABZ472',
        'ABZ441', 'ABZ412', 'ABZ104', 'ABZ595', 'ABZ455', 'ABZ313', 'ABZ298',
        'ABZ62', 'ABZ101', 'ABZ393', 'ABZ483', 'ABZ471', 'ABZ111', 'ABZ87',
        'ABZ538', 'ABZ468', 'ABZ138', 'ABZ565', 'ABZ152', 'ABZ406', 'ABZ72',
        'ABZ205', 'ABZ126', 'ABZ2', 'ABZ50', 'ABZ94', 'ABZ529', 'ABZ307',
        'ABZ143', 'ABZ124', 'ABZ164', 'ABZ559', 'ABZ437', 'ABZ9', 'ABZ398',
        'ABZ131'
    ],
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='Resize', scale=380),
        dict(type='RandomFlip', prob=0.25, direction='horizontal'),
        dict(type='RandomFlip', prob=0.25, direction='vertical'),
        dict(type='PackClsInputs')
    ])
train_dataloader = dict(
    pin_memory=True,
    persistent_workers=True,
    collate_fn=dict(type='default_collate'),
    batch_size=32,
    num_workers=12,
    dataset=dict(
        type='ClassBalancedDataset',
        dataset=dict(
            type='CustomDataset',
            data_prefix='data/ebl/train_set/train_set',
            classes=[
                'ABZ579', 'ABZ13', 'ABZ342', 'ABZ70', 'ABZ461', 'ABZ142',
                'ABZ318', 'ABZ231', 'ABZ1', 'ABZ480', 'ABZ533', 'ABZ449',
                'ABZ75', 'ABZ354', 'ABZ61', 'ABZ597', 'ABZ536', 'ABZ139',
                'ABZ381', 'ABZ308', 'ABZ86', 'ABZ328', 'ABZ330', 'ABZ69',
                'ABZ214', 'ABZ73', 'ABZ545', 'ABZ15', 'ABZ295', 'ABZ296',
                'ABZ151', 'ABZ55', 'ABZ335', 'ABZ537', 'ABZ371', 'ABZ68',
                'ABZ457', 'ABZ84', 'ABZ366', 'ABZ5', 'ABZ353', 'ABZ396',
                'ABZ411', 'ABZ206', 'ABZ58', 'ABZ324', 'ABZ99', 'ABZ376',
                'ABZ532', 'ABZ384', 'ABZ334', 'ABZ383', 'ABZ74', 'ABZ59',
                'ABZ343', 'ABZ145', 'ABZ589', 'ABZ586', 'ABZ211', 'ABZ212',
                'ABZ399', 'ABZ7', 'ABZ367', 'ABZ78', 'ABZ115', 'ABZ322',
                'ABZ207', 'ABZ38', 'ABZ319', 'ABZ144', 'ABZ85', 'ABZ97',
                'ABZ112', 'ABZ60', 'ABZ79', 'ABZ427', 'ABZ232', 'ABZ80',
                'ABZ167', 'ABZ312', 'ABZ535', 'ABZ52', 'ABZ172', 'ABZ331',
                'ABZ554', 'ABZ314', 'ABZ128', 'ABZ142a', 'ABZ12',
                'ABZ331e+152i', 'ABZ401', 'ABZ147', 'ABZ440', 'ABZ6', 'ABZ575',
                'ABZ570', 'ABZ134', 'ABZ465', 'ABZ230', 'ABZ306', 'ABZ148',
                'ABZ339', 'ABZ397', 'ABZ472', 'ABZ441', 'ABZ412', 'ABZ104',
                'ABZ595', 'ABZ455', 'ABZ313', 'ABZ298', 'ABZ62', 'ABZ101',
                'ABZ393', 'ABZ483', 'ABZ471', 'ABZ111', 'ABZ87', 'ABZ538',
                'ABZ468', 'ABZ138', 'ABZ565', 'ABZ152', 'ABZ406', 'ABZ72',
                'ABZ205', 'ABZ126', 'ABZ2', 'ABZ50', 'ABZ94', 'ABZ529',
                'ABZ307', 'ABZ143', 'ABZ124', 'ABZ164', 'ABZ559', 'ABZ437',
                'ABZ9', 'ABZ398', 'ABZ131'
            ],
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='Resize', scale=(380, 380)),
                dict(type='PackClsInputs')
            ]),
        oversample_thr=0.001),
    sampler=dict(type='DefaultSampler', shuffle=True))
val_dataset = ({
    'type':
    'CustomDataset',
    'data_prefix':
    'data/ebl/test_set/test_set',
    'classes': [
        'ABZ579', 'ABZ13', 'ABZ342', 'ABZ70', 'ABZ461', 'ABZ142', 'ABZ318',
        'ABZ231', 'ABZ1', 'ABZ480', 'ABZ533', 'ABZ449', 'ABZ75', 'ABZ354',
        'ABZ61', 'ABZ597', 'ABZ536', 'ABZ139', 'ABZ381', 'ABZ308', 'ABZ86',
        'ABZ328', 'ABZ330', 'ABZ69', 'ABZ214', 'ABZ73', 'ABZ545', 'ABZ15',
        'ABZ295', 'ABZ296', 'ABZ151', 'ABZ55', 'ABZ335', 'ABZ537', 'ABZ371',
        'ABZ68', 'ABZ457', 'ABZ84', 'ABZ366', 'ABZ5', 'ABZ353', 'ABZ396',
        'ABZ411', 'ABZ206', 'ABZ58', 'ABZ324', 'ABZ99', 'ABZ376', 'ABZ532',
        'ABZ384', 'ABZ334', 'ABZ383', 'ABZ74', 'ABZ59', 'ABZ343', 'ABZ145',
        'ABZ589', 'ABZ586', 'ABZ211', 'ABZ212', 'ABZ399', 'ABZ7', 'ABZ367',
        'ABZ78', 'ABZ115', 'ABZ322', 'ABZ207', 'ABZ38', 'ABZ319', 'ABZ144',
        'ABZ85', 'ABZ97', 'ABZ112', 'ABZ60', 'ABZ79', 'ABZ427', 'ABZ232',
        'ABZ80', 'ABZ167', 'ABZ312', 'ABZ535', 'ABZ52', 'ABZ172', 'ABZ331',
        'ABZ554', 'ABZ314', 'ABZ128', 'ABZ142a', 'ABZ12', 'ABZ331e+152i',
        'ABZ401', 'ABZ147', 'ABZ440', 'ABZ6', 'ABZ575', 'ABZ570', 'ABZ134',
        'ABZ465', 'ABZ230', 'ABZ306', 'ABZ148', 'ABZ339', 'ABZ397', 'ABZ472',
        'ABZ441', 'ABZ412', 'ABZ104', 'ABZ595', 'ABZ455', 'ABZ313', 'ABZ298',
        'ABZ62', 'ABZ101', 'ABZ393', 'ABZ483', 'ABZ471', 'ABZ111', 'ABZ87',
        'ABZ538', 'ABZ468', 'ABZ138', 'ABZ565', 'ABZ152', 'ABZ406', 'ABZ72',
        'ABZ205', 'ABZ126', 'ABZ2', 'ABZ50', 'ABZ94', 'ABZ529', 'ABZ307',
        'ABZ143', 'ABZ124', 'ABZ164', 'ABZ559', 'ABZ437', 'ABZ9', 'ABZ398',
        'ABZ131'
    ],
    'pipeline': [{
        'type': 'LoadImageFromFile'
    }, {
        'type': 'Resize',
        'scale': 380
    }, {
        'type': 'PackClsInputs'
    }]
}, )
val_dataloader = dict(
    pin_memory=True,
    persistent_workers=True,
    collate_fn=dict(type='default_collate'),
    batch_size=32,
    num_workers=12,
    dataset=dict(
        type='CustomDataset',
        data_prefix='data/ebl/test_set/test_set',
        classes=[
            'ABZ579', 'ABZ13', 'ABZ342', 'ABZ70', 'ABZ461', 'ABZ142', 'ABZ318',
            'ABZ231', 'ABZ1', 'ABZ480', 'ABZ533', 'ABZ449', 'ABZ75', 'ABZ354',
            'ABZ61', 'ABZ597', 'ABZ536', 'ABZ139', 'ABZ381', 'ABZ308', 'ABZ86',
            'ABZ328', 'ABZ330', 'ABZ69', 'ABZ214', 'ABZ73', 'ABZ545', 'ABZ15',
            'ABZ295', 'ABZ296', 'ABZ151', 'ABZ55', 'ABZ335', 'ABZ537',
            'ABZ371', 'ABZ68', 'ABZ457', 'ABZ84', 'ABZ366', 'ABZ5', 'ABZ353',
            'ABZ396', 'ABZ411', 'ABZ206', 'ABZ58', 'ABZ324', 'ABZ99', 'ABZ376',
            'ABZ532', 'ABZ384', 'ABZ334', 'ABZ383', 'ABZ74', 'ABZ59', 'ABZ343',
            'ABZ145', 'ABZ589', 'ABZ586', 'ABZ211', 'ABZ212', 'ABZ399', 'ABZ7',
            'ABZ367', 'ABZ78', 'ABZ115', 'ABZ322', 'ABZ207', 'ABZ38', 'ABZ319',
            'ABZ144', 'ABZ85', 'ABZ97', 'ABZ112', 'ABZ60', 'ABZ79', 'ABZ427',
            'ABZ232', 'ABZ80', 'ABZ167', 'ABZ312', 'ABZ535', 'ABZ52', 'ABZ172',
            'ABZ331', 'ABZ554', 'ABZ314', 'ABZ128', 'ABZ142a', 'ABZ12',
            'ABZ331e+152i', 'ABZ401', 'ABZ147', 'ABZ440', 'ABZ6', 'ABZ575',
            'ABZ570', 'ABZ134', 'ABZ465', 'ABZ230', 'ABZ306', 'ABZ148',
            'ABZ339', 'ABZ397', 'ABZ472', 'ABZ441', 'ABZ412', 'ABZ104',
            'ABZ595', 'ABZ455', 'ABZ313', 'ABZ298', 'ABZ62', 'ABZ101',
            'ABZ393', 'ABZ483', 'ABZ471', 'ABZ111', 'ABZ87', 'ABZ538',
            'ABZ468', 'ABZ138', 'ABZ565', 'ABZ152', 'ABZ406', 'ABZ72',
            'ABZ205', 'ABZ126', 'ABZ2', 'ABZ50', 'ABZ94', 'ABZ529', 'ABZ307',
            'ABZ143', 'ABZ124', 'ABZ164', 'ABZ559', 'ABZ437', 'ABZ9', 'ABZ398',
            'ABZ131'
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=380),
            dict(type='PackClsInputs')
        ]),
    sampler=dict(type='DefaultSampler', shuffle=False))
test_dataloader = dict(
    pin_memory=True,
    collate_fn=dict(type='default_collate'),
    persistent_workers=True,
    batch_size=1,
    num_workers=12,
    dataset=dict(
        type='CustomDataset',
        data_prefix='data/ebl/test_set/test_set',
        classes=[
            'ABZ579', 'ABZ13', 'ABZ342', 'ABZ70', 'ABZ461', 'ABZ142', 'ABZ318',
            'ABZ231', 'ABZ1', 'ABZ480', 'ABZ533', 'ABZ449', 'ABZ75', 'ABZ354',
            'ABZ61', 'ABZ597', 'ABZ536', 'ABZ139', 'ABZ381', 'ABZ308', 'ABZ86',
            'ABZ328', 'ABZ330', 'ABZ69', 'ABZ214', 'ABZ73', 'ABZ545', 'ABZ15',
            'ABZ295', 'ABZ296', 'ABZ151', 'ABZ55', 'ABZ335', 'ABZ537',
            'ABZ371', 'ABZ68', 'ABZ457', 'ABZ84', 'ABZ366', 'ABZ5', 'ABZ353',
            'ABZ396', 'ABZ411', 'ABZ206', 'ABZ58', 'ABZ324', 'ABZ99', 'ABZ376',
            'ABZ532', 'ABZ384', 'ABZ334', 'ABZ383', 'ABZ74', 'ABZ59', 'ABZ343',
            'ABZ145', 'ABZ589', 'ABZ586', 'ABZ211', 'ABZ212', 'ABZ399', 'ABZ7',
            'ABZ367', 'ABZ78', 'ABZ115', 'ABZ322', 'ABZ207', 'ABZ38', 'ABZ319',
            'ABZ144', 'ABZ85', 'ABZ97', 'ABZ112', 'ABZ60', 'ABZ79', 'ABZ427',
            'ABZ232', 'ABZ80', 'ABZ167', 'ABZ312', 'ABZ535', 'ABZ52', 'ABZ172',
            'ABZ331', 'ABZ554', 'ABZ314', 'ABZ128', 'ABZ142a', 'ABZ12',
            'ABZ331e+152i', 'ABZ401', 'ABZ147', 'ABZ440', 'ABZ6', 'ABZ575',
            'ABZ570', 'ABZ134', 'ABZ465', 'ABZ230', 'ABZ306', 'ABZ148',
            'ABZ339', 'ABZ397', 'ABZ472', 'ABZ441', 'ABZ412', 'ABZ104',
            'ABZ595', 'ABZ455', 'ABZ313', 'ABZ298', 'ABZ62', 'ABZ101',
            'ABZ393', 'ABZ483', 'ABZ471', 'ABZ111', 'ABZ87', 'ABZ538',
            'ABZ468', 'ABZ138', 'ABZ565', 'ABZ152', 'ABZ406', 'ABZ72',
            'ABZ205', 'ABZ126', 'ABZ2', 'ABZ50', 'ABZ94', 'ABZ529', 'ABZ307',
            'ABZ143', 'ABZ124', 'ABZ164', 'ABZ559', 'ABZ437', 'ABZ9', 'ABZ398',
            'ABZ131'
        ],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=380),
            dict(type='PackClsInputs')
        ]),
    sampler=dict(type='DefaultSampler', shuffle=False))
val_evaluator = [
    dict(type='Accuracy', topk=(1, 2, 3, 5)),
    dict(type='SingleLabelMetric', items=['precision', 'recall']),
    dict(type='AveragePrecision'),
    dict(type='MultiLabelMetric', average='macro'),
    dict(type='MultiLabelMetric', average='micro')
]
test_evaluator = [
    dict(type='Accuracy', topk=(1, 2, 3, 5)),
    dict(type='SingleLabelMetric', items=['precision', 'recall']),
    dict(type='AveragePrecision'),
    dict(type='MultiLabelMetric', average='macro'),
    dict(type='MultiLabelMetric', average='micro')
]
custom_hooks = [dict(type='CustomTensorboardLoggerHook', by_epoch=True)]
launcher = 'none'
work_dir = 'logs/heidel-140-12345677'
