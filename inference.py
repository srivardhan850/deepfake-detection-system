import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm

# Validation transforms used during training for consistency
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def get_model(num_classes=2, model_path='best_model.pth', device=None):
    """
    Load the trained Xception model for inference.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = timm.create_model('xception', pretrained=False)
    n_features = model.fc.in_features
    model.fc = nn.Linear(n_features, num_classes)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

def predict_image(model, device, pil_image):
    """
    Predict if an aligned face image is real or fake (AI generated).

    Args:
        model: Loaded PyTorch model for inference.
        device: torch.device.
        pil_image: PIL Image of size 224x224 (aligned face).

    Returns:
        prediction (str): 'real' or 'fake'
        confidence (float): confidence score for the predicted class
    """
    input_tensor = val_transforms(pil_image).unsqueeze(0).to(device)  # add batch dim
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)
        pred_label = 'fake' if pred_idx.item() == 1 else 'real'
    return pred_label, confidence.item()
