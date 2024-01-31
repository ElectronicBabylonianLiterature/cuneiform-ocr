custom_imports = dict(imports=['efficient_net'], allow_failed_imports=False)


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

num_classes = len(classes)

model = dict(
    type='ImageClassifier',
    backbone=dict(type='EfficientNet', arch='b0'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=num_classes,
        in_channels=1280,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5)))


default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=50),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    #visualization=None,
    sync_buffer=dict(type='SyncBuffersHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]

log_level = 'INFO'
load_from = ''
resume = False
randomness = dict(seed=None, deterministic=False)
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0001))
param_scheduler = dict(
    type='MultiStepLR', by_epoch=True, milestones=[500, 300], gamma=0.1)
train_cfg = dict(by_epoch=True, max_epochs=2500, val_interval=1000)
val_cfg = dict()

auto_scale_lr = dict(base_batch_size=128)
optimizer_config = dict(
    type='GradientCumulativeOptimizerHook', cumulative_iters=4)
dataset_type = 'CustomDataset'
data_preprocessor = dict(
    mean=[86.65888836888392, 67.92744567921709, 53.78325960605914],
    std=[68.98970994105028, 57.20489382979894, 48.230552014910586],
    to_rgb=True)



test_dataloader = dict(
    pin_memory=True,
    collate_fn=dict(type='default_collate'),
    persistent_workers=True,
    batch_size=1,
    num_workers=12,
    dataset=dict(
        type='CustomDataset',
        data_prefix='data/ebl/test_set/test_set',
        classes=classes,
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=380),
            dict(type='PackClsInputs')
        ]),
    sampler=dict(type='DefaultSampler', shuffle=False))

test_evaluator = [
    dict(type='Accuracy', topk=(1, 2, 3, 5)),
    dict(type='SingleLabelMetric', items=['precision', 'recall']),
    dict(type='AveragePrecision'),
    dict(type='MultiLabelMetric', average='macro'),
    dict(type='MultiLabelMetric', average='micro')
]
train_dataloader = test_dataloader
val_dataloader = test_dataloader
val_evaluator = test_evaluator

launcher = 'none'

work_dir = "./logs"

test_cfg = dict()
