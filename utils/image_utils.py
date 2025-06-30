from PIL import Image
from torchvision import transforms

def get_preprocessor():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)