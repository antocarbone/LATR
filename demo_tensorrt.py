import os
import ctypes
import cv2
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from torchvision import transforms
from utils.utils import projection_g2im, homography_crop_resize

ORG_H, ORG_W = 768, 1200
CROP_Y = 100
RESIZE_H, RESIZE_W = 720, 960
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
fx = fy = 1200
cx, cy = ORG_W / 2, ORG_H / 2
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]], dtype=np.float16)
cam_height, cam_pitch = 1.5, 0.0
P_g2im = projection_g2im(cam_pitch, cam_height, K)
H_crop = homography_crop_resize([ORG_H, ORG_W], CROP_Y, [RESIZE_H, RESIZE_W])
lidar2img = np.eye(4, dtype=np.float16)
lidar2img[:3] = H_crop @ P_g2im

to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

def preprocess_rgb(path):
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise FileNotFoundError(f"Immagine non trovata: {path}")
    img_bgr = img_bgr[CROP_Y:, :, :]
    img_bgr = cv2.resize(img_bgr, (RESIZE_W, RESIZE_H), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = to_tensor(img_rgb)
    return tensor, img_rgb

def load_engine(engine_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError(f"Impossibile deserializzare l'engine: {engine_path}")
        return engine

import time
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt

def infer_trt_with_peak(engine, img_np, lidar2img_np, pad_shape_np, num_runs=5):
    context = engine.create_execution_context()
    stream = cuda.Stream()

    num_tensors = engine.num_io_tensors
    io_names = [engine.get_tensor_name(i) for i in range(num_tensors)]
    input_names = [n for n in io_names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
    output_names = [n for n in io_names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]

    host_buffers, device_buffers = {}, {}

    def alloc_input(name, arr_fp16):
        host = np.ascontiguousarray(arr_fp16.astype(np.float16, copy=False))
        dev = cuda.mem_alloc(host.nbytes)
        host_buffers[name] = host
        device_buffers[name] = dev
        context.set_tensor_address(name, int(dev))

    def alloc_output(name):
        out_shape = tuple(context.get_tensor_shape(name))
        host = np.ascontiguousarray(np.zeros(out_shape, dtype=np.float16))
        dev = cuda.mem_alloc(host.nbytes)
        host_buffers[name] = host
        device_buffers[name] = dev
        context.set_tensor_address(name, int(dev))

    free0, _ = cuda.mem_get_info()

    alloc_input("image", img_np)
    alloc_input("lidar2img", lidar2img_np)
    alloc_input("pad_shape", pad_shape_np)
    for name in output_names:
        alloc_output(name)

    free1, _ = cuda.mem_get_info()
    mem_alloc_init_mb = (free0 - free1) / (1024**2)

    for n in ["image", "lidar2img", "pad_shape"]:
        cuda.memcpy_htod_async(device_buffers[n], host_buffers[n], stream)
    stream.synchronize()

    times, outputs = [], None
    min_free_mem = free1

    for _ in range(num_runs):
        start = time.time()
        ok = context.execute_async_v3(stream_handle=stream.handle)
        if not ok:
            raise RuntimeError("Esecuzione TensorRT fallita")

        free_now, _ = cuda.mem_get_info()
        if free_now < min_free_mem:
            min_free_mem = free_now

        for n in output_names:
            cuda.memcpy_dtoh_async(host_buffers[n], device_buffers[n], stream)
        stream.synchronize()
        end = time.time()
        times.append(end - start)
        outputs = {n: np.copy(host_buffers[n]) for n in output_names}

    avg_time = sum(times) / max(1, num_runs)

    peak_mem_used_mb = (free0 - min_free_mem) / (1024**2)

    stream.synchronize()
    for dev in device_buffers.values():
        try:
            dev.free()
        except:
            pass
    del stream
    del context

    return outputs, avg_time, mem_alloc_init_mb, peak_mem_used_mb

def save_lanes_txt(lanes, path):
    with open(path, "w") as f:
        for lane in lanes:
            for (x, y, z) in lane:
                f.write(f"{x:.2f} {y:.2f} {z:.2f} ")
            f.write("\n")

def plot_lanes_3d(lanes, out_pdf):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for lane in lanes:
        if not lane:
            continue
        xs, ys, zs = zip(*lane)
        ax.plot(xs, ys, zs)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=30, azim=-60)
    with PdfPages(out_pdf) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    print(f"Plot 3D salvato in {out_pdf}")

def decode_lanes(all_cls_scores_t, all_line_preds_t, anchor_y,
                 vis_thresh=0.5, score_thresh=0.25,
                 lane_cls_idx=1, layer=-1, batch=0):
    cls = all_cls_scores_t[layer, batch].softmax(-1)
    preds = all_line_preds_t[layer, batch]
    num_points = preds.shape[1] // 3
    lanes = []
    for q in range(preds.shape[0]):
        if cls[q, lane_cls_idx] < score_thresh:
            continue
        xs = preds[q, 0:num_points].cpu().numpy()
        zs = preds[q, num_points:2 * num_points].cpu().numpy()
        vis = preds[q, 2 * num_points:].cpu().numpy()
        ys = anchor_y
        lane = [(float(x), float(y), float(z)) for x, y, z, v in zip(xs, ys, zs, vis) if v > vis_thresh]
        if len(lane) >= 2:
            lanes.append(lane)
    return lanes

if __name__ == "__main__":
    import argparse, gc, os, time, ctypes
    import numpy as np
    import torch
    import pycuda.driver as cuda

    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path immagine RGB")
    parser.add_argument("--engine", default="latr_export/latr.engine")
    parser.add_argument("--out_txt", default="lanes_pred.txt")
    parser.add_argument("--out_pdf", default="lanes_plot3d.pdf")
    args = parser.parse_args()

    cuda.init()
    dev = cuda.Device(0)
    ctx = dev.make_context()

    plugin_path = "mmdeploy/build/lib/libmmdeploy_tensorrt_ops.so"
    if not os.path.exists(plugin_path):
        raise FileNotFoundError(f"Plugin TensorRT non trovato: {plugin_path}")
    PLUGIN_LIB = ctypes.CDLL(plugin_path)
    print(f"[INFO] Plugin TensorRT caricato: {plugin_path}")

    try:
        img_tensor, _ = preprocess_rgb(args.image)
        img_np = img_tensor.unsqueeze(0).numpy().astype(np.float16)
        lidar2img_np = np.expand_dims(lidar2img, axis=0).astype(np.float16)
        pad_shape_np = np.array([[RESIZE_H, RESIZE_W]], dtype=np.float16)

        engine = load_engine(args.engine)

        out_dict, avg_time, mem_alloc_init_mb, peak_mem_used_mb = infer_trt_with_peak(
            engine, img_np, lidar2img_np, pad_shape_np, num_runs=10
        )

        print("\n===== RISULTATI INFERENZA =====")
        print(f"Tempo medio inferenza : {avg_time*1000:.2f} ms")
        print(f"Frame rate (FPS) : {1.0/avg_time:.2f} FPS")
        print(f"VRAM allocata iniziale : {mem_alloc_init_mb:.2f} MB")
        print(f"Picco VRAM utilizzata : {peak_mem_used_mb:.2f} MB")
        print("==============================\n")

        all_cls_scores = torch.from_numpy(out_dict["all_cls_scores"])
        all_line_preds = torch.from_numpy(out_dict["all_line_preds"])

        num_points = all_line_preds.shape[-1] // 3
        anchor_y = np.linspace(0, 50, num_points, dtype=np.float16)
        last_layer = all_line_preds.shape[0] - 1

        lanes = decode_lanes(
            all_cls_scores, all_line_preds, anchor_y,
            vis_thresh=0.5, score_thresh=0.25,
            lane_cls_idx=1, layer=last_layer, batch=0
        )

        save_lanes_txt(lanes, args.out_txt)
        plot_lanes_3d(lanes, args.out_pdf)

        print(f"File punti salvato in: {args.out_txt}")
        print(f"Plot 3D salvato in  : {args.out_pdf}")

    finally:
        try:
            del out_dict
        except Exception:
            pass
        
        try:
            gc.collect()
        except Exception:
            pass

        try:
            if 'engine' in globals():
                del engine
        except Exception:
            pass

        try:
            if 'PLUGIN_LIB' in globals():
                del PLUGIN_LIB
        except Exception:
            pass
        
        gc.collect()
        time.sleep(0.05)

        try:
            ctx.pop()
        except Exception as e:
            print(f"[WARN] ctx.pop() ha fallito: {e}")

        try:
            del ctx
        except Exception:
            pass
        
        gc.collect()

