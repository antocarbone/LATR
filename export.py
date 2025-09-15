import os
import torch
import numpy as np
from mmengine.config import Config
from onnxconverter_common import float16
import onnx

import models.latr
import models.latr_head
import models.transformer_bricks

# ---------------- CONFIG ----------------
CFG_FILE = 'config/my_tests/demo_config.py'
cfg = Config.fromfile(CFG_FILE)
cfg.batch_size = 1
cfg.cam_height, cfg.pitch = 1.5, 0.0

DEPLOY_CFG = 'mmdeploy/configs/latr/latr_tensorrt_config.py'
dep_cfg = Config.fromfile(DEPLOY_CFG)
CHECKPOINT = 'pretrained_models/once.pth'

WORK_DIR = './latr_export'
ONNX_FILE = './latr_export/latr.onnx'
TRT_FILE = 'latr.engine'

os.makedirs(WORK_DIR, exist_ok=True)

# ---------------- DUMMY INPUTS ----------------
dummy_image = torch.zeros(1, 3, 720, 960, dtype=torch.float32).cuda()
dummy_lidar2img = torch.from_numpy(np.eye(4, dtype=np.float32)).unsqueeze(0).cuda()
dummy_pad_shape = torch.tensor([720, 960], dtype=torch.float32).unsqueeze(0).cuda()

dummy_inputs = (dummy_image, dummy_lidar2img, dummy_pad_shape)

# ---------------- LOAD MODEL ----------------
model = models.latr.LATR(cfg)
checkpoint = torch.load(CHECKPOINT)
model.load_state_dict(checkpoint['state_dict'], strict=False)
model.eval().cuda()

# ---------------- EXPORT ONNX ----------------
torch.onnx.export(
    model,
    args=dummy_inputs,
    f=ONNX_FILE,
    input_names=['image', 'lidar2img', 'pad_shape'],
    output_names=['all_cls_scores', 'all_line_preds'],
    opset_version=17,
    do_constant_folding=True
)
print(f"ONNX esportato in {ONNX_FILE} (float32, dimensioni fisse)")

# ---------------- CONVERTI A FLOAT16 ----------------
onnx_model = onnx.load(ONNX_FILE)
onnx_model_fp16 = float16.convert_float_to_float16(onnx_model)
onnx.save(onnx_model_fp16, ONNX_FILE)
print(f"ONNX convertito in float16: {ONNX_FILE}")

# ---------------- CARICA PLUGIN TRT ----------------
PLUGIN_PATH = 'mmdeploy/build/lib/libmmdeploy_tensorrt_ops.so'
if not os.path.exists(PLUGIN_PATH):
    raise FileNotFoundError(f"Plugin TensorRT non trovato in {PLUGIN_PATH}")
import ctypes
ctypes.CDLL(PLUGIN_PATH)
print(f"Plugin TensorRT caricato da {PLUGIN_PATH}")

# ---------------- CREAZIONE ENGINE TensorRT ----------------
from mmdeploy.backend.tensorrt.onnx2tensorrt import onnx2tensorrt

onnx2tensorrt(
    work_dir=WORK_DIR,
    save_file=TRT_FILE,
    model_id=0,
    deploy_cfg=DEPLOY_CFG,
    onnx_model=ONNX_FILE
)
print(f"TensorRT engine creato in {os.path.join(WORK_DIR, TRT_FILE)}")
