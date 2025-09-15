import os, json, cv2
import numpy as np
import torch
import psutil
from torch import nn
import time

#from mmcv.utils import Config #SPOSTATO IN MMENGINE
from mmengine.config import Config

from torchvision import transforms
from models.latr import LATR
from utils.utils import projection_g2im
from utils.utils import homography_crop_resize, projective_transformation
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

CFG_FILE = "config/my_tests/demo_config.py"
#CKPT = "pretrained_models/once.pth"
CKPT = "pretrained_models/apollo_standard.pth"
ORG_H, ORG_W = 768, 1200
CROP_Y = 100
RESIZE_H, RESIZE_W = 720, 960
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

fx = fy = 1200; cx, cy = ORG_W/2, ORG_H/2
K  = np.array([[fx,  0, cx],
               [ 0, fy, cy],
               [ 0,  0,  1]], dtype=np.float32)
cam_height, cam_pitch = 1.5, 0.0
P_g2im = projection_g2im(cam_pitch, cam_height, K)
H_crop = homography_crop_resize([ORG_H, ORG_W], CROP_Y, [RESIZE_H, RESIZE_W])
lidar2img = np.eye(4, dtype=np.float32)
lidar2img[:3] = H_crop @ P_g2im

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

def plot_waypoints(txt_file, pdf_file="./data/waypoints_plot.pdf"):
    with open(txt_file, "r") as f:
        lines = f.readlines()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for line in lines:
        coords = [tuple(map(float, pt.split(","))) for pt in line.strip().split("; ")]
        xs, ys, zs = zip(*coords)
        ax.plot(xs, ys, zs)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=30, azim=-60)

    with PdfPages(pdf_file) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    print(f"PDF salvato in {pdf_file}")

def preprocess_rgb(path):
    img_bgr = cv2.imread(path)
    img_bgr = img_bgr[CROP_Y:, :, :]
    img_bgr = cv2.resize(img_bgr, (RESIZE_W, RESIZE_H), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor  = to_tensor(img_rgb)
    return tensor

def build_model(cfg_file, ckpt_path, device):
    cfg = Config.fromfile(cfg_file)
    cfg.batch_size = 1
    cfg.org_h, cfg.org_w = ORG_H, ORG_W
    cfg.crop_y  = CROP_Y
    cfg.resize_h, cfg.resize_w = RESIZE_H, RESIZE_W
    cfg.cam_height, cfg.pitch, cfg.K = cam_height, cam_pitch, K
    model = LATR(cfg).to(device).eval()
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["state_dict"])
    return model, cfg

@torch.no_grad()
def latr_inference(img_path, out_txt, model, cfg, device, num_runs=5):
    img = preprocess_rgb(img_path).unsqueeze(0).to(device)
    pad_shape = torch.tensor([RESIZE_H, RESIZE_W]).float().to(device)
    extra = {
        "lidar2img": torch.from_numpy(lidar2img).unsqueeze(0).to(device),
        "pad_shape": pad_shape.unsqueeze(0)
    }

    times = []
    gpu_mems = []
    outputs = None

    for _ in range(num_runs):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        start_time = time.time()
        all_cls_scores, all_line_preds = model(img, extra["lidar2img"], extra["pad_shape"])
        end_time = time.time()
        times.append((end_time - start_time) * 1000.0)

        if device.type == "cuda":
            used_mem = torch.cuda.memory_allocated(device) / (1024 ** 2)
            gpu_mems.append(used_mem)

        outputs = (all_cls_scores, all_line_preds)

    avg_time = sum(times) / len(times)
    print(f"Tempo inferenza medio su {num_runs} run: {avg_time:.2f} ms")

    if device.type == "cuda" and gpu_mems:
        avg_gpu = sum(gpu_mems) / len(gpu_mems)
        print(f"Memoria GPU media durante inferenza: {avg_gpu:.2f} MB")

    all_cls_scores, all_line_preds = outputs
    line_pred   = all_line_preds[-1][0].cpu().numpy()
    cls_scores  = all_cls_scores[-1][0].softmax(-1).cpu().numpy()

    anchor_y = cfg.anchor_y_steps
    n_pts    = anchor_y.shape[0]
    waypoints = []
    for lane_idx in range(line_pred.shape[0]):
        if cls_scores[lane_idx,1] < 0.25:
            continue
        xs = line_pred[lane_idx,:n_pts]
        zs = line_pred[lane_idx,n_pts:2*n_pts]
        vis= line_pred[lane_idx,2*n_pts:] > 0
        ys = anchor_y
        pts = np.stack([xs, ys, zs], axis=1)[vis]
        if len(pts) > 1:
            waypoints.append(pts.tolist())

    with open(out_txt, "w") as f:
        for lane in waypoints:
            line = "; ".join([f"{x:.2f},{y:.2f},{z:.2f}" for x,y,z in lane])
            f.write(line + "\n")
    print(f"Salvati {len(waypoints)} lane in {out_txt}")

if __name__ == "__main__":
    import argparse, pathlib
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="path immagine RGB")
    parser.add_argument("-o","--out", default="latr_pred.txt")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, cfg = build_model(CFG_FILE, CKPT, device)
    latr_inference(args.image, args.out, model, cfg, device)
    print(f"Inferenza completata")
    plot_waypoints(args.out)
