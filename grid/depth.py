import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from grid.geometry import DEPTH_SIZE, MAX_DEPTH_M

DEPTH_INPUT_SIZE = 518
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def build_depth_model(max_depth=MAX_DEPTH_M):
    from depth_anything_v2.dpt import DepthAnythingV2

    return DepthAnythingV2(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024], max_depth=max_depth)


def load_depth_state(path):
    state = torch.load(str(path), map_location="cpu", weights_only=True)
    if "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    return {k[len("depth_model."):] if k.startswith("depth_model.") else k: v for k, v in state.items()}


def preprocess_view(image):
    image = image.convert("RGB").resize((DEPTH_INPUT_SIZE, DEPTH_INPUT_SIZE), Image.Resampling.BICUBIC)
    array = (np.asarray(image, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))


class AerialDepth:
    def __init__(self, checkpoint, device="cuda", dtype=torch.bfloat16):
        self.device = torch.device(device)
        self.dtype = dtype
        self.model = build_depth_model()
        self.model.load_state_dict(load_depth_state(checkpoint), strict=True)
        self.model.requires_grad_(False).eval().to(self.device)

    @torch.inference_mode()
    def predict(self, views, size=DEPTH_SIZE):
        batch = torch.stack([preprocess_view(view) for view in views]).to(self.device)
        with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"):
            depth = self.model(batch)
        if depth.ndim == 3:
            depth = depth[:, None]
        depth = F.interpolate(depth.float(), size=(size, size), mode="bilinear", align_corners=False)
        return depth[:, 0].clamp(0, MAX_DEPTH_M).cpu().numpy().astype(np.float16)
