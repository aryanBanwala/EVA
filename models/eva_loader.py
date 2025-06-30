import torch
from transformers import ViTModel, ViTConfig

def load_eva_model(ckpt_path: str, device='cpu'):
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = ViTModel(ViTConfig())
    model.load_state_dict(checkpoint, strict=False)
    model.to(device).eval()
    return model