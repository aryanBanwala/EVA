import warnings

# ignore only the CLIP/video‐deprecation warning from torchvision
warnings.filterwarnings(
    "ignore",
    r".*video decoding and encoding capabilities of torchvision are deprecated.*",
    category=UserWarning,
    module=r"torchvision\.io\._video_deprecation_warning"
)

import torch
import numpy as np
import torch.nn.functional as F
from fractions import Fraction
from contextlib import suppress
from decord import VideoReader, cpu , gpu , bridge
bridge.set_bridge('torch') 

from models.clip_loader import load_openclip

_model_cache = {"model": None, "preprocess": None, "device": None}

def get_cached_model(device: str):
    if _model_cache["model"] is None or _model_cache["device"] != device:
        print("📦 Loading OpenCLIP on", device)
        model, preprocess = load_openclip(
            model_name="ViT-B-32",
            ckpt="laion2b_s34b_b79k",
            device=device
        )
        _model_cache.update(model=model, preprocess=preprocess, device=device)
        print("✅ Model cached")
    return _model_cache["model"], _model_cache["preprocess"]

def _as_float_fps(fps_meta):
    if isinstance(fps_meta, (int, float)):
        return float(fps_meta)
    if isinstance(fps_meta, Fraction):
        return fps_meta.numerator / fps_meta.denominator
    if isinstance(fps_meta, (list, tuple)):
        num_den = fps_meta if len(fps_meta) == 2 else (fps_meta[0], 1)
        return float(num_den[0]) / float(num_den[1] or 1)
    return float(fps_meta)

def _safe_close_vr(vr):
    with suppress(Exception):
        vr.close()
    for attr in ("_c", "_reader", "_container", "container"):
        with suppress(Exception):
            getattr(vr, attr).close()

def extract_and_preprocess_frames(
    video_path: str,
    frame_per_second: int,
    max_frames: int,
    device: str,
) -> torch.Tensor:
    """
    Fast frame sampler using Decord.
    1) Reads with zero-copy into numpy.
    2) Converts to torch → GPU, resize, normalize.
    """

    # 1. Open video (Decord chooses best backend automatically)
    if ( device == "cpu" ):
        vr = VideoReader(video_path, ctx=cpu(0))
    elif ( device == "cuda" ):
        vr = VideoReader(video_path, ctx=gpu(0))  
    fps = vr.get_avg_fps() or 30  # fallback if header missing
    fps = _as_float_fps(fps)

    # 2. Decide sampling step
    step = max(1, int(round(fps / frame_per_second)))

    # 3. Sample frames
    indices = list(range(0, len(vr), step))[:max_frames]
    if not indices:
        raise ValueError(f"No frames sampled from {video_path}")
    
    batch = vr.get_batch(indices)           # shape (T, H, W, C) uint8
    
    if isinstance(batch, torch.Tensor):
        tensor_batch = batch                       # already torch
    else:
        # Decord NDArray  ➜  numpy ➜  torch  (zero-copy if possible)
        tensor_batch = torch.from_numpy(batch.asnumpy())

    # 4. Move to GPU  &  [T,C,H,W]  &  float
    frames = (tensor_batch.to(device)
                          .contiguous()
                          .permute(0, 3, 1, 2)      # CHW
                          .float() / 255.0)
    # 5. Resize on GPU
    frames = F.interpolate(frames, (224, 224), mode="bilinear", align_corners=False)

    # 6. CLIP normalization
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                        device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                        device=device).view(1, 3, 1, 1)
    frames = (frames - mean) / std
    return frames  # shape [T, 3, 224, 224]

def embed_batch(frames_list: list[torch.Tensor], device: str) -> list[torch.Tensor]:
    model, _ = get_cached_model(device)
    model = model.to(device).eval().half()

    all_frames = torch.cat(frames_list, dim=0).half().to(device)
    with torch.cuda.amp.autocast():
        feats = model.encode_image(all_frames)  # [sum(N), 512]

    embeddings = []
    idx = 0
    for frames in frames_list:
        cnt = frames.shape[0]
        chunk = feats[idx : idx + cnt]
        embeddings.append(chunk.mean(0).cpu())
        idx += cnt

    return embeddings