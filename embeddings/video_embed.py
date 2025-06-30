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
from fractions import Fraction
from contextlib import suppress
from torchvision.io import VideoReader

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

def _as_float_fps(fps_meta):
    if isinstance(fps_meta, (int, float)):
        return float(fps_meta)
    if isinstance(fps_meta, Fraction):
        return fps_meta.numerator / fps_meta.denominator
    if isinstance(fps_meta, (list, tuple)):
        if len(fps_meta) == 2:
            num, den = fps_meta
            return float(num) / float(den or 1)
        if len(fps_meta) == 1:
            return float(fps_meta[0])
    return float(fps_meta)

def _safe_close_vr(vr):
    with suppress(Exception):
        vr.close()
    for attr in ("_c", "_reader", "_container", "container"):
        with suppress(Exception):
            getattr(vr, attr).close()

def get_video_embedding(
    video_path: str,
    frame_per_second: int = 1,
    max_frames: int      = 100,
    device: str          = "cpu",
) -> torch.Tensor:
    vr = None
    try:
        vr = VideoReader(video_path, "video")
        meta = vr.get_metadata().get("video", {})
        fps  = _as_float_fps(meta.get("fps"))

        step = max(1, int(round(fps / frame_per_second)))

        sampled = []
        for idx, pkt in enumerate(vr):
            if idx % step == 0:
                sampled.append(pkt["data"])
                if len(sampled) >= max_frames:
                    break

        if not sampled:
            raise ValueError(f"No frames sampled from {video_path}")

        # stack into a tensor
        frames = torch.stack(sampled)  # shape could be [T,C,H,W] or [T,H,W,C]

        # detect and reorder dims if needed:
        if frames.ndim == 4 and frames.shape[-1] == 3:
            # frames are [T, H, W, C] → permute to [T, C, H, W]
            frames = frames.permute(0, 3, 1, 2)

        # now frames is [T, C, H, W]
        frames = frames.to(device).float() / 255.0

        # resize on GPU
        frames = F.interpolate(frames, (224, 224),
                               mode="bilinear", align_corners=False)

        # CLIP normalization
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                            device=device).view(1, 3, 1, 1)
        std  = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                            device=device).view(1, 3, 1, 1)
        frames = (frames - mean) / std

        model, _ = get_cached_model(device)
        with torch.no_grad():
            feats = model.encode_image(frames)  # [T, 512]

        return feats.mean(0).cpu()

    finally:
        if vr is not None:
            _safe_close_vr(vr)
            del vr