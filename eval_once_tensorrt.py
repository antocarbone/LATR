import os
import time
import json
import numpy as np
from tqdm import tqdm
import argparse
import ctypes
import torch
from mmengine.config import Config
from data.Load_Data import LaneDataset
from torch.utils.data import DataLoader
from utils.eval_3D_once import LaneEval
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("Errore: TensorRT e PyCUDA non disponibili.")
    exit()

class TensorRTInference:
    def __init__(self, engine_path, plugin_path=None):
        if not TRT_AVAILABLE:
            raise ImportError("TensorRT non disponibile")
        self.engine_path = engine_path
        self.plugin_path = plugin_path or "mmdeploy/build/lib/libmmdeploy_tensorrt_ops.so"
        cuda.init()
        self.device = cuda.Device(0)
        self.context = self.device.make_context()
        if os.path.exists(self.plugin_path):
            self.plugin_lib = ctypes.CDLL(self.plugin_path)
        self.engine = self._load_engine()
        self.context_exec = self.engine.create_execution_context()
        self.buffers = self._allocate_buffers()

    def _load_engine(self):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            if engine is None:
                raise RuntimeError(f"Impossibile deserializzare: {self.engine_path}")
            return engine

    def _allocate_buffers(self):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            tensor_mode = self.engine.get_tensor_mode(name)
            size = trt.volume(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if tensor_mode == trt.TensorIOMode.INPUT:
                inputs.append({'host': host_mem, 'device': device_mem, 'name': name})
            else:
                outputs.append({'host': host_mem, 'device': device_mem, 'name': name})
        return {'inputs': inputs, 'outputs': outputs, 'bindings': bindings, 'stream': stream}

    def infer(self, images, lidar2img, pad_shape):
        stream = self.buffers['stream']
        inputs_list = [images, lidar2img, pad_shape]
        for i, input_buffer in enumerate(self.buffers['inputs']):
            np.copyto(input_buffer['host'], inputs_list[i].cpu().numpy().ravel())
            cuda.memcpy_htod_async(input_buffer['device'], input_buffer['host'], stream)
        self.context_exec.execute_async_v2(bindings=self.buffers['bindings'], stream_handle=stream.handle)
        for out_buffer in self.buffers['outputs']:
            cuda.memcpy_dtoh_async(out_buffer['host'], out_buffer['device'], stream)
        stream.synchronize()
        out_dict = {}
        for out_buffer in self.buffers['outputs']:
            out_dict[out_buffer['name']] = out_buffer['host'].reshape(self.engine.get_tensor_shape(out_buffer['name']))
        return out_dict

    def __del__(self):
        try:
            if hasattr(self, 'context'):
                self.context.pop()
            if hasattr(self, 'context_exec'):
                del self.context_exec
            if hasattr(self, 'engine'):
                del self.engine
            for buf in self.buffers['inputs']+self.buffers['outputs']:
                buf['device'].free()
        except:
            pass

def preprocess_for_tensorrt(batch_data, precision="fp32"):
    if precision=="fp16":
        images = batch_data["image"].contiguous().half().cuda(non_blocking=True)
        lidar2img = batch_data["lidar2img"].contiguous().half().cuda(non_blocking=True)
        pad_shape = batch_data["pad_shape"].contiguous().half().cuda(non_blocking=True)
    else:
        images = batch_data["image"].contiguous().float().cuda(non_blocking=True)
        lidar2img = batch_data["lidar2img"].contiguous().float().cuda(non_blocking=True)
        pad_shape = batch_data["pad_shape"].contiguous().float().cuda(non_blocking=True)
    return images, lidar2img, pad_shape

def parse_model_output(output_dict, cfg, score_threshold=0.3):
    all_cls_scores = torch.from_numpy(output_dict["all_cls_scores"])
    all_line_preds = torch.from_numpy(output_dict["all_line_preds"])
    line_preds = all_line_preds[-1].cpu().numpy()
    cls_scores = all_cls_scores[-1].softmax(-1).cpu().numpy()

    cam_pitch = 0.3/180*np.pi
    cam_height = 1.5
    cam_extrinsics = np.array([[np.cos(cam_pitch),0,-np.sin(cam_pitch),0],
                               [0,1,0,0],
                               [np.sin(cam_pitch),0,np.cos(cam_pitch),cam_height],
                               [0,0,0,1]], dtype=float)
    R_vg = np.array([[0,1,0],[-1,0,0],[0,0,1]], dtype=float)
    R_gc = np.array([[1,0,0],[0,0,1],[0,-1,0]], dtype=float)
    cam_extrinsics[:3,:3] = np.matmul(np.matmul(np.linalg.inv(R_vg), cam_extrinsics[:3,:3]), R_vg)
    cam_extrinsics[:3,:3] = np.matmul(cam_extrinsics[:3,:3], R_gc)
    cam_extrinsics[0:2,3]=0.0

    batch_results=[]
    for b_idx in range(line_preds.shape[0]):
        pred_lanes=[]
        for lane_idx in range(line_preds.shape[1]):
            lane_cls_scores=cls_scores[b_idx,lane_idx]
            pred_class=np.argmax(lane_cls_scores)
            max_score=np.max(lane_cls_scores)
            if pred_class==0 or max_score<score_threshold:
                continue
            lane_pred=line_preds[b_idx,lane_idx]
            xs=lane_pred[0:cfg.num_y_steps]
            zs=lane_pred[cfg.num_y_steps:2*cfg.num_y_steps]
            vis=lane_pred[2*cfg.num_y_steps:3*cfg.num_y_steps]

            lane=[]
            for x,y,z,v in zip(xs,cfg.anchor_y_steps,zs,vis):
                if v>0.5:
                    lane.append([float(x), float(y), float(z)])
            if len(lane)>=2:
                lane=np.array(lane)
                lane=np.flip(lane, axis=0)
                lane=np.vstack((lane.T, np.ones((1,lane.shape[0]))))
                lane=np.matmul(np.linalg.inv(cam_extrinsics), lane)
                lane=lane[:3,:].T
                pred_lanes.append({"points":lane.tolist(),"score":float(max_score)})
        batch_results.append({"lanes":pred_lanes})
    return batch_results

def save_predictions(predictions, sample_info, pred_dir):
    for i, (pred, sample) in enumerate(zip(predictions, sample_info)):
        json_path_parts=sample['idx_json_file'].split(os.sep)
        scene_id=json_path_parts[-3] if len(json_path_parts)>=3 else f"scene_{i:06d}"
        cam_id=json_path_parts[-2] if len(json_path_parts)>=3 else "cam01"
        frame_name=os.path.basename(sample['idx_json_file'])
        save_path=os.path.join(pred_dir, scene_id, cam_id)
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, frame_name),"w") as f:
            json.dump(pred,f,indent=2)

def plot_and_overlay(predictions, dataset, sample_info, pred_dir, cfg):
    if len(sample_info) == 0:
        return

    random_idx = random.randint(0, len(sample_info)-1)
    sample_json = sample_info[random_idx]["idx_json_file"]

    json_path_parts = sample_json.split(os.sep)
    scene_id = json_path_parts[-3]
    cam_id = json_path_parts[-2]
    frame_name = os.path.basename(sample_json)

    pred_path = os.path.join(pred_dir, scene_id, cam_id, frame_name)
    with open(pred_path, "r") as f:
        pred_data = json.load(f)

    overlay_dir = os.path.join(pred_dir, "overlays")
    os.makedirs(overlay_dir, exist_ok=True)

    img_path = sample_json.replace(".json", ".jpg").replace("val", "data")
    img_pil = Image.open(img_path).convert('RGB')
    h_crop = getattr(cfg, 'crop_y', 0)
    h_org = getattr(cfg, 'org_h', img_pil.height)
    w_org = getattr(cfg, 'org_w', img_pil.width)
    img_cropped = F.crop(img_pil, h_crop, 0, h_org-h_crop, w_org)
    img_resized = F.resize(img_cropped, size=(cfg.resize_h, cfg.resize_w), interpolation=InterpolationMode.BILINEAR)
    img = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]

    cam_pitch = 0.3 / 180 * np.pi
    cam_height = 1.5
    cam_extrinsics = np.array([[np.cos(cam_pitch), 0, -np.sin(cam_pitch), 0],
                               [0, 1, 0, 0],
                               [np.sin(cam_pitch), 0, np.cos(cam_pitch), cam_height],
                               [0, 0, 0, 1]], dtype=float)
    R_vg = np.array([[0,1,0],[-1,0,0],[0,0,1]], dtype=float)
    R_gc = np.array([[1,0,0],[0,0,1],[0,-1,0]], dtype=float)
    cam_extrinsics[:3,:3] = np.matmul(np.matmul(np.linalg.inv(R_vg), cam_extrinsics[:3,:3]), R_vg)
    cam_extrinsics[:3,:3] = np.matmul(cam_extrinsics[:3,:3], R_gc)
    cam_extrinsics[0:2,3] = 0.0

    lidar2img = dataset[random_idx]["lidar2img"]

    for lane in pred_data["lanes"]:
        points_after_inv = np.array(lane["points"])
        points_h = np.concatenate([points_after_inv, np.ones((points_after_inv.shape[0],1))], axis=1).T
        points_original = cam_extrinsics @ points_h
        proj = lidar2img @ points_original
        
        valid_mask = proj[2, :] > 0.1
        if not np.any(valid_mask):
            continue
            
        proj_valid = proj[:, valid_mask].copy()
        proj_valid[:2] /= proj_valid[2:3]
        
        valid_indices = np.where(valid_mask)[0]
        for i in range(len(valid_indices)-1):
            if valid_indices[i+1] - valid_indices[i] == 1:
                x1, y1 = int(proj_valid[0, i]), int(proj_valid[1, i])
                x2, y2 = int(proj_valid[0, i+1]), int(proj_valid[1, i+1])
                if 0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h:
                    cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    out_img = os.path.join(overlay_dir, f"prediction_{random_idx:06d}_overlay.jpg")
    cv2.imwrite(out_img, img)
    print(f"Overlay salvato: {out_img}")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=str,default="./config/my_tests/demo_config.py")
    parser.add_argument("--engine",type=str,required=True)
    parser.add_argument("--plugin_path",type=str,default="mmdeploy/build/lib/libmmdeploy_tensorrt_ops.so")
    parser.add_argument("--score_threshold",type=float,default=0.3)
    parser.add_argument("--batch_size",type=int,default=1)
    parser.add_argument("--precision",type=str,default="fp32",choices=["fp32","fp16"])
    args=parser.parse_args()

    cfg=Config.fromfile(args.config)
    if isinstance(cfg.top_view_region, list):
        cfg.top_view_region = np.array(cfg.top_view_region)
    cfg.anchor_y_steps = np.linspace(cfg.anchor_y_steps['start'], cfg.anchor_y_steps['stop'], cfg.anchor_y_steps['num'])
    cfg.anchor_y_steps_dense = np.linspace(cfg.anchor_y_steps_dense['start'], cfg.anchor_y_steps_dense['stop'], cfg.anchor_y_steps_dense['num'])

    trt_inference=TensorRTInference(args.engine,args.plugin_path)
    dataset=LaneDataset(cfg.dataset_dir,cfg.data_dir,cfg,data_aug=False)
    dataloader=DataLoader(dataset,batch_size=args.batch_size,shuffle=False,num_workers=4,pin_memory=True)
    pred_dir="./work_dirs/tensorrt_test_predictions/"
    os.makedirs(pred_dir,exist_ok=True)

    all_sample_info=[]
    start_time=time.time()
    with torch.no_grad():
        for batch_idx,batch_data in enumerate(tqdm(dataloader)):
            images,lidar2img,pad_shape=preprocess_for_tensorrt(batch_data,args.precision)
            output_dict=trt_inference.infer(images,lidar2img,pad_shape)
            batch_predictions=parse_model_output(output_dict,cfg,args.score_threshold)
            batch_sample_info=[{"idx_json_file": dataset._label_list[batch_idx*args.batch_size+i]} for i in range(len(batch_predictions))]
            save_predictions(batch_predictions,batch_sample_info,pred_dir)
            all_sample_info.extend(batch_sample_info)
    total_time=time.time()-start_time
    print(f"FPS medio: {len(all_sample_info)/total_time:.2f}")

    plot_and_overlay(batch_predictions,dataset,all_sample_info,pred_dir,cfg)

    evaluator=LaneEval()
    eval_config_path="./config/_base_/once_eval_config.json"
    class DummyArgs:
        def __init__(self):
            self.proc_id=0
    dummy_args=DummyArgs()
    metrics=evaluator.lane_evaluation(cfg.data_dir,pred_dir,eval_config_path,args=dummy_args)
    print("Validazione completata!")

if __name__=="__main__":
    main()
