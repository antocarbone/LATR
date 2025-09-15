# deploy_config.py
custom_imports = dict(
    imports=[
        'models.latr',
        'models.latr_head',
        'models.transformer_bricks'
    ],
    allow_failed_imports=False
)

onnx_config = dict(
    type='onnx',
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=13,
    save_file='latr.onnx',
    input_names=['image', 'lane_idx', 'seg', 'lidar2img', 'pad_shape'],
    output_names=['dets', 'labels'],
    dynamic_axes={
        'image': {0: 'batch', 2},
        'lane_idx': {0: 'batch'},
        'seg': {0: 'batch'},
        'lidar2img': {0: 'batch'},
        'pad_shape': {0: 'batch'},
        'dets': {0: 'batch', 1: 'num_dets'},
        'labels': {0: 'batch', 1: 'num_labels'},
    }
)

backend_config = dict(
    type='tensorrt',
    common_config=dict(
        fp16_mode=False,
        max_workspace_size=1 << 30
    ),
    model_inputs=[
        dict(
            input_shapes=dict(
                image=dict(min_shape=[1,3,320,320], opt_shape=[1,3,720,960], max_shape=[1,3,1344,1344]),
                lane_idx=dict(min_shape=[1,12,20], opt_shape=[1,12,20], max_shape=[1,12,20]),
                seg=dict(min_shape=[1,12,20], opt_shape=[1,12,20], max_shape=[1,12,20]),
                lidar2img=dict(min_shape=[1,4,4], opt_shape=[1,4,4], max_shape=[1,4,4]),
                pad_shape=dict(min_shape=[1,3], opt_shape=[1,3], max_shape=[1,3])
            )
        )
    ]
)

codebase_config = dict(
    type='mmdet',
    task='ObjectDetection'
)
