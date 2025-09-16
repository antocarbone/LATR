import os
import time
import torch
import json
import numpy as np
from tqdm import tqdm
from mmengine.config import Config
from models.latr import LATR
from data.Load_Data import LaneDataset
from utils.eval_3D_once import LaneEval
import argparse
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

import random
import matplotlib.pyplot as plt
import cv2

def load_model_checkpoint(model, checkpoint_path, device):
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
    return model

def preprocess_for_inference(batch_data, device, precision="fp32"):
    images = batch_data["image"].contiguous().float().to(device)
    lidar2img = batch_data["lidar2img"].contiguous().float().to(device)
    pad_shape = batch_data["pad_shape"].contiguous().float().to(device)
    return images, lidar2img, pad_shape

def parse_model_output(output, cfg, score_threshold=0.3):
    all_cls_scores, all_line_preds = output
    line_preds = all_line_preds[-1].cpu().numpy()
    cls_scores = all_cls_scores[-1]

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

    batch_results = []
    for b_idx in range(line_preds.shape[0]):
        pred_lanes = []
        cls_pred = torch.argmax(cls_scores[b_idx], dim=-1).cpu().numpy()
        pos_lanes = line_preds[b_idx][cls_pred > 0]
        
        if pos_lanes.shape[0]:
            if cfg.num_category > 1:
                scores_pred = torch.softmax(cls_scores[b_idx][cls_pred > 0], dim=-1).cpu().numpy()
            else:
                scores_pred = torch.sigmoid(cls_scores[b_idx][cls_pred > 0]).cpu().numpy()

            xs = pos_lanes[:, 0:cfg.num_y_steps]
            ys = np.tile(cfg.anchor_y_steps.copy()[None, :], (xs.shape[0], 1))
            zs = pos_lanes[:, cfg.num_y_steps:2*cfg.num_y_steps]
            vis = pos_lanes[:, 2*cfg.num_y_steps:]
            
            for tmp_idx in range(pos_lanes.shape[0]):
                cur_vis = vis[tmp_idx] > 0
                cur_xs = xs[tmp_idx][cur_vis]
                cur_ys = ys[tmp_idx][cur_vis]
                cur_zs = zs[tmp_idx][cur_vis]

                if cur_vis.sum() < 2:
                    continue

                if cfg.num_category > 1:
                    max_score = np.max(scores_pred[tmp_idx])
                else:
                    max_score = scores_pred[tmp_idx][0]
                    
                if max_score < score_threshold:
                    continue

                lane = []
                for tmp_inner_idx in range(cur_xs.shape[0]):
                    lane.append([cur_xs[tmp_inner_idx], 
                                cur_ys[tmp_inner_idx], 
                                cur_zs[tmp_inner_idx]])
                
                if len(lane) >= 2:
                    lane = np.array(lane)
                    lane = np.flip(lane, axis=0)
                    lane = np.vstack((lane.T, np.ones((1, lane.shape[0]))))
                    lane = np.matmul(np.linalg.inv(cam_extrinsics), lane)
                    lane = lane[:3, :].T
                    
                    pred_lanes.append({"points": lane.tolist(), "score": float(max_score)})
        
        batch_results.append({"lanes": pred_lanes})
    
    return batch_results

def save_predictions(predictions, sample_info, pred_dir):
    for pred, sample in zip(predictions, sample_info):
        json_path_parts = sample['idx_json_file'].split(os.sep)
        scene_id = json_path_parts[-3] if len(json_path_parts)>=3 else "scene_000000"
        cam_id = json_path_parts[-2] if len(json_path_parts)>=3 else "cam01"
        frame_name = os.path.basename(sample['idx_json_file'])

        save_path = os.path.join(pred_dir, scene_id, cam_id)
        os.makedirs(save_path, exist_ok=True)

        full_json_path = os.path.join(save_path, frame_name)
        with open(full_json_path, "w") as f:
            json.dump(pred, f, indent=2)

def overlay_predictions(pred_data, dataset, sample_idx, cfg, pred_dir):
    overlay_dir = os.path.join(pred_dir, "overlays")
    os.makedirs(overlay_dir, exist_ok=True)

    sample_json = dataset._label_list[sample_idx]
    img_path = sample_json.replace(".json", ".jpg").replace("val", "data")
    img_pil = Image.open(img_path).convert('RGB')
    h_crop = getattr(cfg, 'crop_y', 0)
    h_org = getattr(cfg, 'org_h', img_pil.height)
    w_org = getattr(cfg, 'org_w', img_pil.width)
    img_cropped = F.crop(img_pil, h_crop, 0, h_org-h_crop, w_org)
    img_resized = F.resize(img_cropped, size=(cfg.resize_h, cfg.resize_w),
                           interpolation=InterpolationMode.BILINEAR)
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

    lidar2img = dataset[sample_idx]["lidar2img"]
    
    points_drawn = 0
    points_outside = 0
    
    for lane_idx, lane in enumerate(pred_data["lanes"]):
        points_after_inv = np.array(lane["points"])

        points_h = np.concatenate([points_after_inv, np.ones((points_after_inv.shape[0],1))], axis=1).T
        points_original = cam_extrinsics @ points_h

        proj = lidar2img @ points_original

        valid_mask = proj[2, :] > 1e-5
        if not np.any(valid_mask):
            continue
            
        proj_valid = proj[:, valid_mask].copy()
        proj_valid[:2] /= proj_valid[2:3]

        valid_indices = np.where(valid_mask)[0]
        for i in range(len(valid_indices)-1):
            if valid_indices[i+1] - valid_indices[i] == 1:
                x1, y1 = int(proj_valid[0, i]), int(proj_valid[1, i])
                x2, y2 = int(proj_valid[0, i+1]), int(proj_valid[1, i+1])
                
                if (0 <= x1 < w and 0 <= y1 < h and 0 <= x2 < w and 0 <= y2 < h):
                    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
                    color = colors[lane_idx % len(colors)]
                    cv2.line(img, (x1, y1), (x2, y2), color, 3)
    
    out_img = os.path.join(overlay_dir, f"prediction_{sample_idx:06d}.jpg")
    cv2.imwrite(out_img, img)
    print(f"Overlay salvato: {out_img}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/my_tests/demo_config.py")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--score_threshold", type=float, default=0.3)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    if isinstance(cfg.top_view_region, list):
        cfg.top_view_region = np.array(cfg.top_view_region)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LATR(cfg)
    model = load_model_checkpoint(model, args.checkpoint, device)
    model = model.to(device).eval()

    dataset = LaneDataset(cfg.dataset_dir, cfg.data_dir, cfg, data_aug=False)
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    pred_dir = "./work_dirs/test_predictions"
    os.makedirs(pred_dir, exist_ok=True)

    evaluator = LaneEval()

    start_time = time.time()
    all_sample_info = []

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(dataloader)):
            images, lidar2img, pad_shape = preprocess_for_inference(batch_data, device)
            output = model(images, lidar2img, pad_shape)
            batch_predictions = parse_model_output(output, cfg, args.score_threshold)

            batch_sample_info = [{"idx_json_file": dataset._label_list[batch_idx*args.batch_size + i]}
                                 for i in range(len(batch_predictions))]
            save_predictions(batch_predictions, batch_sample_info, pred_dir)
            all_sample_info.extend(batch_sample_info)

    total_time = time.time() - start_time
    fps = len(all_sample_info)/total_time
    print(f"FPS medio: {fps:.2f}")

    if len(all_sample_info) > 0:
        sample_idx = random.randint(0, len(all_sample_info)-1)
        sample_json = all_sample_info[sample_idx]["idx_json_file"]
        json_path_parts = sample_json.split(os.sep)
        scene_id = json_path_parts[-3] if len(json_path_parts)>=3 else "scene_000000"
        cam_id = json_path_parts[-2] if len(json_path_parts)>=3 else "cam01"
        frame_name = os.path.basename(sample_json)
        pred_path = os.path.join(pred_dir, scene_id, cam_id, frame_name)

        with open(pred_path, "r") as f:
            pred_data = json.load(f)

        overlay_predictions(pred_data, dataset, sample_idx, cfg, pred_dir)

    eval_config_path = "./config/_base_/once_eval_config.json"
    dummy_args = type("DummyArgs", (), {"proc_id":0})()
    metrics = evaluator.lane_evaluation(cfg.data_dir, pred_dir, eval_config_path, dummy_args)
    print("Valutazione completata!")

if __name__ == "__main__":
    main()