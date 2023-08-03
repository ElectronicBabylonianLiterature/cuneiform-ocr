custom_imports = dict(imports=['efficient_net'], allow_failed_imports=False)

num_classes = 142
classes = ['ABZ579', 'ABZ13', 'ABZ342', 'ABZ70', 'ABZ461', 'ABZ480', 'ABZ142', 'ABZ1', 'ABZ231', 'ABZ533', 'ABZ449', 'ABZ318', 'ABZ75', 'ABZ61', 'ABZ381', 'ABZ354', 'ABZ139', 'ABZ597', 'ABZ536', 'ABZ308', 'ABZ330', 'ABZ328', 'ABZ15', 'ABZ86', 'ABZ214', 'ABZ545', 'ABZ73', 'ABZ295', 'ABZ55', 'ABZ335', 'ABZ371', 'ABZ537', 'ABZ457', 'ABZ68', 'ABZ151', 'ABZ69', 'ABZ353', 'ABZ5', 'ABZ366', 'ABZ296', 'ABZ411', 'ABZ84', 'ABZ396', 'ABZ206', 'ABZ58', 'ABZ376', 'ABZ324', 'ABZ99', 'ABZ384', 'ABZ59', 'ABZ532', 'ABZ334', 'ABZ589', 'ABZ145', 'ABZ383', 'ABZ586', 'ABZ343', 'ABZ74', 'ABZ399', 'ABZ212', 'ABZ211', 'ABZ7', 'ABZ78', 'ABZ367', 'ABZ38', 'ABZ319', 'ABZ115', 'ABZ85', 'ABZ322', 'ABZ207', 'ABZ144', 'ABZ112', 'ABZ97', 'ABZ427', 'ABZ60', 'ABZ79', 'ABZ80', 'ABZ52', 'ABZ312', 'ABZ142a', 'ABZ232', 'ABZ535', 'ABZ314', 'ABZ331', 'ABZ167', 'ABZ128', 'ABZ172', 'ABZ6', 'ABZ575', 'ABZ331e+152i', 'ABZ554', 'ABZ134', 'ABZ339', 'ABZ465', 'ABZ12', 'ABZ306', 'ABZ397', 'ABZ570', 'ABZ2', 'ABZ147', 'ABZ148', 'ABZ440', 'ABZ401', 'ABZ230', 'ABZ441', 'ABZ104', 'ABZ472', 'ABZ313', 'ABZ595', 'ABZ298', 'ABZ412', 'ABZ455', 'ABZ62', 'ABZ101', 'ABZ468', 'ABZ471', 'ABZ111', 'ABZ483', 'ABZ538', 'ABZ87', 'ABZ143', 'ABZ565', 'ABZ205', 'ABZ126', 'ABZ50', 'ABZ72', 'ABZ152', 'ABZ138', 'ABZ393', 'ABZ406', 'ABZ307', 'ABZ124', 'ABZ94', 'ABZ164', 'ABZ398', 'ABZ529', 'ABZ559', 'ABZ131', 'ABZ437', 'ABZ56', 'ABZ9', 'ABZ191']


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
