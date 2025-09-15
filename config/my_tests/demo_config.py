#import numpy as np #NON È POSSIBILE UTILIZZARE NUMPY QUI
from mmengine.config import Config
#import os.path as osp
from mmengine.config import read_base

with read_base():
    from .._base_.optimizer import *
    from .._base_.base_res101_bs16xep100_mmengine import *

mod = 'my_tests/demo_config'
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

dataset = 'once'
dataset_name = 'once'
data_dir = 'data/once/val/'
dataset_dir = 'data/once/'
eval_config_dir = 'config/_base_/once_eval_config.json'

save_path = './work_dirs/demo_eval'

max_lanes = 8
num_pt_per_line = 20

eta_min = 1e-6
clip_grad_norm = 20

batch_size = 8
nworkers = 10
num_category = 2
pos_threshold = 0.3

top_view_region = [[-10, 65], [10, 65], [-10, 0.5], [10, 0.5]]

enlarge_length = 10
position_range = [
    top_view_region[0][0] - enlarge_length,
    top_view_region[2][1] - enlarge_length,
    -5,
    top_view_region[1][0] + enlarge_length,
    top_view_region[0][1] + enlarge_length,
    5.]

anchor_y_steps = dict(type='linspace', start=0.5, stop=65, num=num_pt_per_line)
anchor_y_steps_dense = dict(type='linspace', start=0.5, stop=65, num=num_pt_per_line*10)
num_y_steps = num_pt_per_line

_dim_ = 256
num_query = 12
num_pt_per_line = 20
latr_cfg = dict(
    fpn_dim = _dim_,
    num_query = num_query,
    num_group = 1,
    sparse_num_group = 4,
    encoder = dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='caffe',
        # with_cp=True,
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    neck = dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=4,
        relu_before_extra_convs=True
    ),
    head=dict(
        xs_loss_weight=2.0,
        zs_loss_weight=10.0,
        vis_loss_weight=1.0,
        cls_loss_weight=10,
        project_loss_weight=1.0,
        pt_as_query=True,
        num_pt_per_line=num_pt_per_line,
    ),
    trans_params=dict(init_z=0, bev_h=150, bev_w=70),
)

ms2one=dict(
    type='DilateNaive',
    inc=_dim_, outc=_dim_, num_scales=4,
    dilations=(1, 2, 5, 9))

transformer=dict(
    type='LATRTransformer',
    decoder=dict(
        type='LATRTransformerDecoder',
        embed_dims=_dim_,
        num_layers=6,
        enlarge_length=enlarge_length,
        M_decay_ratio=1,
        num_query=num_query,
        num_anchor_per_query=num_pt_per_line,
        anchor_y_steps=anchor_y_steps,
        transformerlayers=dict(
            type='LATRDecoderLayer',
            attn_cfgs=[
                dict(
                    type='MultiheadAttention',
                    embed_dims=_dim_,
                    num_heads=4,
                    dropout=0.1),
                dict(
                    type='MSDeformableAttention3D',
                    embed_dims=_dim_,
                    num_heads=4,
                    num_levels=1,
                    num_points=8,
                    batch_first=False,
                    num_query=num_query,
                    num_anchor_per_query=num_pt_per_line,
                    anchor_y_steps=anchor_y_steps,
                    dropout=0.1),
                ],
            ffn_cfgs=dict(
                type='FFN',
                embed_dims=_dim_,
                feedforward_channels=_dim_*8,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True),
            ),
            feedforward_channels=_dim_ * 8,
            operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                            'ffn', 'norm')),
))

sparse_ins_decoder=Config(
    dict(
        encoder=dict(
            out_dims=_dim_),
        decoder=dict(
            num_query=latr_cfg['num_query'],
            num_group=latr_cfg['num_group'],
            sparse_num_group=latr_cfg['sparse_num_group'],
            hidden_dim=_dim_,
            kernel_dim=_dim_,
            num_classes=num_category,
            num_convs=4,
            output_iam=True,
            scale_factor=1.,
            ce_weight=2.0,
            mask_weight=5.0,
            dice_weight=2.0,
            objectness_weight=1.0,
        ),
        sparse_decoder_weight=5.0,
))

resize_h = 720
resize_w = 960
org_h = 1020
org_w = 1920
crop_y = 100
nepochs = 24
eval_freq = 8

optimizer_cfg = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'sampling_offsets': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

#==== FOR DEPLOYMENT ====#

model = dict(
    type='LATR',
    args=Config(
        dict(
            latr_cfg=latr_cfg,
            ms2one=ms2one,
            transformer=transformer,
            sparse_ins_decoder=sparse_ins_decoder,
            num_y_steps=num_y_steps,
            max_lanes=max_lanes,
            num_category=num_category,
            position_range=position_range,
            top_view_region=top_view_region,
            batch_size=batch_size,
            anchor_y_steps=anchor_y_steps,
            anchor_y_steps_dense=anchor_y_steps_dense,
            no_cuda=False
        )
    ),
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=mean,
        std=std,
        bgr_to_rgb=False
    ),
    test_cfg=dict()
)

test_pipeline = [dict(type='PackDetInputs')]
