custom_imports = dict(imports=['efficient_net'], allow_failed_imports=False)

num_classes = 141
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
