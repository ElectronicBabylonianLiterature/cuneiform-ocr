custom_imports = dict(imports=['efficient_net'], allow_failed_imports=False)

num_classes = 147
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
