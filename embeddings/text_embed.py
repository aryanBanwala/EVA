# text_embed.py
import torch
import open_clip

# Lazy‐loaded model, transforms and tokenizer
_model = None
_preprocess = None
_tokenizer = None

def load_and_prepare_text_model(device: str = 'cpu'):
    global _model, _preprocess, _tokenizer
    if _model is None:
        print("📦 Loading CLIP text encoder (ViT-B/32, laion2b)…")
        # open_clip.create_model_and_transforms returns (model, transform, preprocess)
        _model, _transform, _preprocess = open_clip.create_model_and_transforms(
            model_name="ViT-B-32",
            pretrained="laion2b_s34b_b79k",
            device=device
        )
        _tokenizer = open_clip.get_tokenizer("ViT-B-32")
        _model.eval()
        print("✅ CLIP text encoder ready.")
    return _model, _preprocess, _tokenizer

def get_text_embedding(text: str, device: str = 'cpu'):
    model, preprocess, tokenizer = load_and_prepare_text_model(device)
    tokens = tokenizer([text])
    tokens = tokens.to(device)
    with torch.no_grad():
        features = model.encode_text(tokens)
    # remove the batch dimension
    return features.squeeze(0)