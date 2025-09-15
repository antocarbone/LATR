import os, cv2
import time
import numpy as np
import torch
import onnxruntime as ort
import psutil
from mmengine.config import Config
from torchvision import transforms
from utils.utils import projection_g2im, homography_crop_resize
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

CFG_FILE = "config/my_tests/demo_config.py"
ONNX_PATH = "latr_export_fp32/latr.onnx"
ORG_H, ORG_W = 768, 1200
CROP_Y = 100
RESIZE_H, RESIZE_W = 720, 960
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

fx = fy = 1200; cx, cy = ORG_W / 2, ORG_H / 2
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
cam_height, cam_pitch = 1.5, 0.0
P_g2im = projection_g2im(cam_pitch, cam_height, K)
H_crop = homography_crop_resize([ORG_H, ORG_W], CROP_Y, [RESIZE_H, RESIZE_W])
lidar2img = np.eye(4, dtype=np.float32)
lidar2img[:3] = H_crop @ P_g2im

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

def preprocess_rgb(path):
    img_bgr = cv2.imread(path)
    img_bgr = img_bgr[CROP_Y:, :, :]
    img_bgr = cv2.resize(img_bgr, (RESIZE_W, RESIZE_H), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = to_tensor(img_rgb)
    return tensor.unsqueeze(0).numpy() 

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

@torch.no_grad()
def latr_infer_onnx(img_path, out_txt, cfg):
    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss

    session = ort.InferenceSession(ONNX_PATH, providers=["ONNXExecutionProvider"])

    img_np = preprocess_rgb(img_path)
    B, _, H, W = img_np.shape

    seg_idx_label = np.zeros((B, cfg.max_lanes, H, W), dtype=np.uint8)
    seg_label     = np.zeros((B, 1, H, W), dtype=np.float32)
    pad_shape     = np.array([[RESIZE_H, RESIZE_W]], dtype=np.float32)
    lidar2img_np  = lidar2img[np.newaxis]

    inputs = {
        "image": img_np,
        "seg_idx_label": seg_idx_label,
        "seg_label": seg_label,
        "lidar2img": lidar2img_np,
        "pad_shape": pad_shape
    }

    start_time = time.time()
    outputs = session.run(None, inputs)
    inference_time = time.time() - start_time
    print(f"Tempo di inferenza ONNX: {inference_time * 1000:.2f} ms")

    pred = outputs[0]

    end_mem = process.memory_info().rss
    print(f"Memoria RAM usata: {(end_mem - start_mem) / 1024**2:.2f} MB")

    line_pred = pred[0, :, :-1]
    cls_scores = pred[0, :, -1]

    anchor_y = cfg.anchor_y_steps
    n_pts = len(anchor_y)
    waypoints = []

    for lane_idx in range(line_pred.shape[0]):
        if cls_scores[lane_idx] < 0.25:
            continue
        xs = line_pred[lane_idx, :n_pts]
        zs = line_pred[lane_idx, n_pts:2*n_pts]
        vis = line_pred[lane_idx, 2*n_pts:] > 0
        ys = anchor_y
        pts = np.stack([xs, ys, zs], axis=1)[vis]
        if len(pts) > 1:
            waypoints.append(pts.tolist())

    with open(out_txt, "w") as f:
        for lane in waypoints:
            line = "; ".join([f"{x:.2f},{y:.2f},{z:.2f}" for x, y, z in lane])
            f.write(line + "\n")
    print(f"Salvati {len(waypoints)} lane in {out_txt}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path immagine RGB")
    parser.add_argument("-o", "--out", default="latr_pred.txt")
    args = parser.parse_args()

    cfg = Config.fromfile(CFG_FILE)
    latr_infer_onnx(args.image, args.out, cfg)
    print("Inferenza completata")
    plot_waypoints(args.out)
