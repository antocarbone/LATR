import os
import numpy as np
import ctypes

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