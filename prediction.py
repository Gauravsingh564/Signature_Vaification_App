import torch
from PIL import Image
from torchvision import transforms

def predict_signature(model, image_path, device, img_size=(224,224), class_names=None):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    x = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        pred = out.argmax(dim=1).item()
    if class_names:
        return class_names[pred]
    return pred
