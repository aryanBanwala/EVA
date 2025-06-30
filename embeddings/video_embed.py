import warnings

# ignore only the CLIP/video‐deprecation warning from torchvision
warnings.filterwarnings(
    "ignore",
    r".*video decoding and encoding capabilities of torchvision are deprecated.*",
    category=UserWarning,
    module=r"torchvision\.io\._video_deprecation_warning"
)

import torch
import torch.nn.functional as F
from torchvision.io import read_video

from models.clip_loader import load_openclip
from utils.video_utils     import extract_frames

_model_cache = {
    "model": None,
    "preprocess": None,
    "device": None
}

def get_cached_model(device='cpu'):
    if _model_cache["model"] is None or _model_cache["device"] != device:
        print("📦 Loading OpenCLIP once for device:", device)
        model, preprocess = load_openclip(
            model_name="ViT-B-32",
            ckpt="laion2b_s34b_b79k",
            device=device
        )
        _model_cache["model"] = model
        _model_cache["preprocess"] = preprocess
        _model_cache["device"] = device
        print("✅ Model cached.")
    return _model_cache["model"], _model_cache["preprocess"]

def get_video_embedding(
    video_path: str,
    frame_per_second: int = 1,
    max_frames: int      = None,
    device: str          = 'cpu'
) -> torch.Tensor:
    # 1) Load video frames into a [T,H,W,C] uint8 tensor
    video, _, info = read_video(video_path, pts_unit="sec")
    fps = info["video_fps"]
    step = max(1, int(fps / frame_per_second))
    frames = video[::step]
    if max_frames:
        frames = frames[:max_frames]

    # 2) Move into GPU, reorder dims to [T,C,H,W], normalize to [0,1]
    frames = frames.to(device=device).permute(0,3,1,2).float() / 255.0

    # 3) Resize all frames at once to 224×224
    frames = F.interpolate(frames, size=(224,224), mode="bilinear", align_corners=False)

    # 4) Normalize with CLIP’s mean/std (broadcasted)
    mean = torch.tensor([0.48145466, 0.4578275,  0.40821073], device=device).view(1,3,1,1)
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1,3,1,1)
    frames = (frames - mean) / std

    # 5) One batched forward pass
    model, _ = get_cached_model(device)
    with torch.no_grad():
        feats = model.encode_image(frames)  # shape: [T, 512]

    if feats.numel() == 0:
        raise ValueError(f"No frames to embed in {video_path}")

    # 6) Mean-pool over time → [512]
    return feats.mean(dim=0).cpu()