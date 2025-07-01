import torch
import torch.nn.functional as F
from fractions import Fraction
from contextlib import suppress
from torchvision.io import VideoReader

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
    device: str
) -> torch.Tensor:
    vr   = VideoReader(video_path, "video")
    meta = vr.get_metadata().get("video", {})
    fps  = _as_float_fps(meta.get("fps", frame_per_second))
    step = max(1, int(round(fps / frame_per_second)))

    sampled = []
    for idx, pkt in enumerate(vr):
        if idx % step == 0:
            sampled.append(pkt["data"])
            if len(sampled) >= max_frames:
                break

    _safe_close_vr(vr)
    if not sampled:
        raise ValueError(f"No frames from {video_path}")

    frames = torch.stack(sampled)
    if frames.ndim == 4 and frames.shape[-1] == 3:
        frames = frames.permute(0, 3, 1, 2)

    frames = frames.to(device).float() / 255.0
    frames = F.interpolate(frames, (224, 224), mode="bilinear", align_corners=False)

    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
    return (frames - mean) / std

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