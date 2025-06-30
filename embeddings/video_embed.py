import os
import shutil
import torch
from PIL import Image

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
    """
    1) Samples `frame_per_second` frames per second (up to max_frames),
    2) Encodes each via OpenCLIP,
    3) Mean‐pools to a single 512‐d vector,
    4) Cleans up temporary frames dir.
    """
    # print(f"🚀 Starting video embedding for: {video_path}")
    model, preprocess = get_cached_model(device)

    # 1) fps‐based frame extraction
    frame_paths = extract_frames(
        video_path,
        fps=frame_per_second,
        max_frames=max_frames
    )
    # print(f"🖼️ Total sampled frames: {len(frame_paths)}")

    # 2) encode each frame
    embeddings = []
    for idx, frame_path in enumerate(frame_paths, start=1):
        # print(f"➡️ Frame {idx}/{len(frame_paths)}: {frame_path}")
        # Load image from path
        image = Image.open(frame_path).convert("RGB")
        img_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(img_tensor)      # returns (1,512) tensor
        embeddings.append(feat.squeeze(0))             # now (512,)

    if not embeddings:
        raise ValueError("❌ No frames extracted.")

    # 3) cleanup temporary frames
    frames_dir = os.path.join("assets", "frames")
    if os.path.isdir(frames_dir):
        shutil.rmtree(frames_dir)
        # print(f"🗑️ Removed temporary frames directory: {frames_dir}")

    # 4) mean‐pool to get final vector
    # print("🧮 Averaging embeddings …")
    video_embedding = torch.stack(embeddings).mean(0)  # shape: (512,)
    return video_embedding