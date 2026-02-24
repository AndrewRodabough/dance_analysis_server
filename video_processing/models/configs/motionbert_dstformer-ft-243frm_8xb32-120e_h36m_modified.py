# Path resolution removed - MMEngine eval() doesn't support __file__
# Checkpoint is passed separately via MMPoseInferencer weights parameter


#_base_ = ['../../../_base_/default_runtime.py']

# ---- Default Runtime Replacement ---- #

# defaults to use registries in mmpose
default_scope = 'mmpose'

# configure default hooks
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=10),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='PoseVisualizationHook', enable=False),
)

# configure environment
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# set log level
log_level = 'INFO'
load_from = None
resume = False

# ---- End Default Runtime Replacement ---- #


# Dataset metadata for Human3.6M - required for visualizer
dataset_name = 'h36m'
dataset_meta = dict(
    dataset_name='h36m',
    paper_info=dict(
        author='Ionescu, Catalin and Papava, Dragos and '
        'Olaru, Vlad and Sminchisescu,  Cristian',
        title='Human3.6M: Large Scale Datasets and Predictive '
        'Methods for 3D Human Sensing in Natural Environments',
        container='IEEE Transactions on Pattern Analysis and '
        'Machine Intelligence',
        year='2014',
        homepage='http://vision.imar.ro/human3.6m/',
    ),
    keypoint_info={
        0: dict(name='root', id=0, color=[255, 0, 0], type='lower', swap=''),
        1: dict(name='right_hip', id=1, color=[255, 128, 0], type='lower', swap='left_hip'),
        2: dict(name='right_knee', id=2, color=[255, 128, 0], type='lower', swap='left_knee'),
        3: dict(name='right_foot', id=3, color=[255, 128, 0], type='lower', swap='left_foot'),
        4: dict(name='left_hip', id=4, color=[0, 255, 0], type='lower', swap='right_hip'),
        5: dict(name='left_knee', id=5, color=[0, 255, 0], type='lower', swap='right_knee'),
        6: dict(name='left_foot', id=6, color=[0, 255, 0], type='lower', swap='right_foot'),
        7: dict(name='spine', id=7, color=[51, 153, 255], type='upper', swap=''),
        8: dict(name='thorax', id=8, color=[51, 153, 255], type='upper', swap=''),
        9: dict(name='neck_base', id=9, color=[51, 153, 255], type='upper', swap=''),
        10: dict(name='head', id=10, color=[51, 153, 255], type='upper', swap=''),
        11: dict(name='left_shoulder', id=11, color=[0, 255, 0], type='upper', swap='right_shoulder'),
        12: dict(name='left_elbow', id=12, color=[0, 255, 0], type='upper', swap='right_elbow'),
        13: dict(name='left_wrist', id=13, color=[0, 255, 0], type='upper', swap='right_wrist'),
        14: dict(name='right_shoulder', id=14, color=[255, 128, 0], type='upper', swap='left_shoulder'),
        15: dict(name='right_elbow', id=15, color=[255, 128, 0], type='upper', swap='left_elbow'),
        16: dict(name='right_wrist', id=16, color=[255, 128, 0], type='upper', swap='left_wrist'),
    },
    skeleton_info={
        0: dict(link=('root', 'left_hip'), id=0, color=[0, 255, 0]),
        1: dict(link=('left_hip', 'left_knee'), id=1, color=[0, 255, 0]),
        2: dict(link=('left_knee', 'left_foot'), id=2, color=[0, 255, 0]),
        3: dict(link=('root', 'right_hip'), id=3, color=[255, 128, 0]),
        4: dict(link=('right_hip', 'right_knee'), id=4, color=[255, 128, 0]),
        5: dict(link=('right_knee', 'right_foot'), id=5, color=[255, 128, 0]),
        6: dict(link=('root', 'spine'), id=6, color=[51, 153, 255]),
        7: dict(link=('spine', 'thorax'), id=7, color=[51, 153, 255]),
        8: dict(link=('thorax', 'neck_base'), id=8, color=[51, 153, 255]),
        9: dict(link=('neck_base', 'head'), id=9, color=[51, 153, 255]),
        10: dict(link=('thorax', 'left_shoulder'), id=10, color=[0, 255, 0]),
        11: dict(link=('left_shoulder', 'left_elbow'), id=11, color=[0, 255, 0]),
        12: dict(link=('left_elbow', 'left_wrist'), id=12, color=[0, 255, 0]),
        13: dict(link=('thorax', 'right_shoulder'), id=13, color=[255, 128, 0]),
        14: dict(link=('right_shoulder', 'right_elbow'), id=14, color=[255, 128, 0]),
        15: dict(link=('right_elbow', 'right_wrist'), id=15, color=[255, 128, 0]),
    },
    joint_weights=[1.] * 17,
    sigmas=[])

vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='Pose3dLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# runtime
train_cfg = dict(max_epochs=120, val_interval=10)

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.01))

# learning policy
param_scheduler = [
    dict(type='ExponentialLR', gamma=0.99, end=60, by_epoch=True)
]

auto_scale_lr = dict(base_batch_size=512)

# hooks
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        save_best='MPJPE',
        rule='less',
        max_keep_ckpts=1),
    logger=dict(type='LoggerHook', interval=20),
)

# codec settings
train_codec = dict(
    type='MotionBERTLabel', num_keypoints=17, concat_vis=True, mode='train')
val_codec = dict(
    type='MotionBERTLabel', num_keypoints=17, concat_vis=True, rootrel=True)

# model settings
model = dict(
    type='PoseLifter',
    backbone=dict(
        type='DSTFormer',
        in_channels=3,
        feat_size=512,
        depth=5,
        num_heads=8,
        mlp_ratio=2,
        seq_len=243,
        att_fuse=True,
    ),
    head=dict(
        type='MotionRegressionHead',
        in_channels=512,
        out_channels=3,
        embedding_size=512,
        loss=dict(type='MPJPEVelocityJointLoss'),
        decoder=val_codec,
    ),
    test_cfg=dict(flip_test=True),
    metainfo=dataset_meta,
    # Checkpoint passed separately via MMPoseInferencer
    # init_cfg=dict(
    #     type='Pretrained',
    #     checkpoint=None),
)

# base dataset settings
dataset_type = 'Human36mDataset'
data_root = 'data/h36m/'

# pipelines
train_pipeline = [
    dict(type='GenerateTarget', encoder=train_codec),
    dict(
        type='RandomFlipAroundRoot',
        keypoints_flip_cfg=dict(center_mode='static', center_x=0.),
        target_flip_cfg=dict(center_mode='static', center_x=0.),
        flip_label=True),
    dict(
        type='PackPoseInputs',
        meta_keys=('id', 'category_id', 'target_img_path', 'flip_indices',
                   'factor', 'camera_param'))
]
val_pipeline = [
    dict(type='GenerateTarget', encoder=val_codec),
    dict(
        type='PackPoseInputs',
        meta_keys=('id', 'category_id', 'target_img_path', 'flip_indices',
                   'factor', 'camera_param'))
]

# data loaders
train_dataloader = dict(
    batch_size=32,
    prefetch_factor=4,
    pin_memory=True,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file='annotation_body3d/fps50/h36m_train.npz',
        seq_len=1,
        multiple_target=243,
        multiple_target_step=81,
        camera_param_file='annotation_body3d/cameras.pkl',
        data_root=data_root,
        data_prefix=dict(img='images/'),
        pipeline=train_pipeline,
    ))

val_dataloader = dict(
    batch_size=32,
    prefetch_factor=4,
    pin_memory=True,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        ann_file='annotation_body3d/fps50/h36m_test.npz',
        seq_len=1,
        seq_step=1,
        multiple_target=243,
        camera_param_file='annotation_body3d/cameras.pkl',
        data_root=data_root,
        data_prefix=dict(img='images/'),
        pipeline=val_pipeline,
        test_mode=True,
    ))
test_dataloader = val_dataloader

# evaluators
skip_list = [
    'S9_Greet', 'S9_SittingDown', 'S9_Wait_1', 'S9_Greeting', 'S9_Waiting_1'
]
val_evaluator = [
    dict(type='MPJPE', mode='mpjpe', skip_list=skip_list),
    dict(type='MPJPE', mode='p-mpjpe', skip_list=skip_list)
]
test_evaluator = val_evaluator
