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

DEFAULT_MODEL_NAME = 'mobilenetv3_small_100'


def build_model(model_name=DEFAULT_MODEL_NAME, num_classes=2, pretrained=False):
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model


def get_model(num_classes=2, model_path='best_model.pth', device=None, model_name=DEFAULT_MODEL_NAME):
    """
    Load the trained Xception model for inference.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not model_path:
        model_path = 'best_model.pth'

    if not isinstance(model_path, str):
        raise ValueError("model_path must be a string path to a .pth checkpoint.")

    import os
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}. "
            "Train the model first or place best_model.pth in the project root."
        )

    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=device)
    if isinstance(state, dict) and 'model_state_dict' in state:
        model_name = state.get('model_name', model_name)
        num_classes = state.get('num_classes', num_classes)
        state_dict = state['model_state_dict']
    else:
        state_dict = state

    model = build_model(model_name=model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(state_dict)
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
        fake_probability (float): probability assigned to the fake class
    """
    input_tensor = val_transforms(pil_image).unsqueeze(0).to(device)  # add batch dim
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)
        pred_label = 'fake' if pred_idx.item() == 1 else 'real'
        fake_probability = probs[0, 1].item()
    return pred_label, confidence.item(), fake_probability
