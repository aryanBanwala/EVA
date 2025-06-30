# models/clip_loader.py
import open_clip
import torch

def load_openclip(model_name="ViT-B-32", ckpt="laion2b_s34b_b79k", device="cpu"):
    """
    Returns (model, preprocess) for OpenCLIP.
    model_name examples:
        ViT-B-32  | ViT-B-16 | ViT-L-14
    ckpt examples:
        laion2b_s34b_b79k  |  laion400m_e32  | openai
    """
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=ckpt, device=device
    )
    model.to(device).eval()
    return model, preprocess