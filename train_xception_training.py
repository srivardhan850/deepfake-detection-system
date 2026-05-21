# Deepfake Detection Training Notebook Script (Python)

# Cell 1: Imports and Setup
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score, accuracy_score
from PIL import Image
import numpy as np
import timm
import glob
import random

DEFAULT_MODEL_NAME = "mobilenetv3_small_100"
IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.webp")


def find_images(root_dir):
    files = []
    for pattern in IMAGE_EXTENSIONS:
        files.extend(glob.glob(os.path.join(root_dir, "**", pattern), recursive=True))
    return sorted(files)


def select_balanced_files(real_files, fake_files, max_images_per_class=0, seed=42):
    rng = random.Random(seed)
    real_files = list(real_files)
    fake_files = list(fake_files)
    rng.shuffle(real_files)
    rng.shuffle(fake_files)

    if max_images_per_class and max_images_per_class > 0:
        real_files = real_files[:max_images_per_class]
        fake_files = fake_files[:max_images_per_class]

    return sorted(real_files), sorted(fake_files)


# Cell 2: Dataset Class
class FaceDataset(Dataset):
    def __init__(self, root_dir, transforms=None, real_files=None, fake_files=None):
        """
        Args:
            root_dir (str): Root dataset directory with subfolders 'real' and 'fake'.
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        self.real_dir = os.path.join(root_dir, "real")
        self.fake_dir = os.path.join(root_dir, "fake")
        self.real_files = list(real_files) if real_files is not None else find_images(self.real_dir)
        self.fake_files = list(fake_files) if fake_files is not None else find_images(self.fake_dir)
        self.all_files = self.real_files + self.fake_files
        self.labels = [0]*len(self.real_files) + [1]*len(self.fake_files)
        self.transforms = transforms

        if len(self.real_files) == 0 or len(self.fake_files) == 0:
            print(
                "Warning: dataset should contain both real and fake images. "
                f"Found real={len(self.real_files)}, fake={len(self.fake_files)}"
            )

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        img_path = self.all_files[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transforms:
            image = self.transforms(image)
        return image, label

# Cell 3: Transforms
import random
from PIL import ImageFilter, ImageEnhance
import io

class RandomGaussianBlur(object):
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.0):
        self.p = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if random.random() < self.p:
            radius = random.uniform(self.radius_min, self.radius_max)
            return img.filter(ImageFilter.GaussianBlur(radius))
        return img

class RandomJPEGCompression(object):
    def __init__(self, p=0.5, quality_min=30, quality_max=90):
        self.p = p
        self.quality_min = quality_min
        self.quality_max = quality_max

    def __call__(self, img):
        if random.random() < self.p:
            quality = random.randint(self.quality_min, self.quality_max)
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            return Image.open(buffer)
        return img

class RandomNoise(object):
    def __init__(self, p=0.5, noise_level=0.05):
        self.p = p
        self.noise_level = noise_level

    def __call__(self, img):
        if random.random() < self.p:
            np_img = np.array(img).astype(np.float32) / 255.0
            noise = np.random.normal(0, self.noise_level, np_img.shape)
            np_img = np.clip(np_img + noise, 0, 1)
            np_img = (np_img * 255).astype(np.uint8)
            return Image.fromarray(np_img)
        return img

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Cell 4: Create Dataloaders
def split_class_files(files, val_split=0.2, seed=42):
    files = list(files)
    rng = random.Random(seed)
    rng.shuffle(files)
    split = max(1, int(np.floor(val_split * len(files))))
    return files[split:], files[:split]


def create_dataloaders(dataset_path, batch_size=32, val_split=0.2, seed=42, max_images_per_class=0):
    real_dir = os.path.join(dataset_path, "real")
    fake_dir = os.path.join(dataset_path, "fake")
    real_files, fake_files = select_balanced_files(
        find_images(real_dir),
        find_images(fake_dir),
        max_images_per_class=max_images_per_class,
        seed=seed,
    )

    full_dataset = FaceDataset(dataset_path, transforms=None, real_files=real_files, fake_files=fake_files)
    dataset_size = len(full_dataset)
    if dataset_size == 0:
        raise ValueError(
            f"No training images found in {dataset_path}. "
            "Expected dataset/real/... and dataset/fake/... images."
        )
    if len(full_dataset.real_files) == 0 or len(full_dataset.fake_files) == 0:
        raise ValueError(
            "Training needs both classes. Add images under dataset/real and dataset/fake."
        )
    if len(full_dataset.real_files) < 2 or len(full_dataset.fake_files) < 2:
        raise ValueError(
            "Dataset is too small for training. "
            f"Found real={len(full_dataset.real_files)}, fake={len(full_dataset.fake_files)}. "
            "Add at least 2 real and 2 fake images for a test run. "
            "For useful results, use 100+ real and 100+ fake face images."
        )

    train_real, val_real = split_class_files(real_files, val_split=val_split, seed=seed)
    train_fake, val_fake = split_class_files(fake_files, val_split=val_split, seed=seed + 1)

    if not train_real or not train_fake or not val_real or not val_fake:
        raise ValueError("Train/validation split is empty. Add more images or reduce val_split.")

    train_dataset = FaceDataset(
        dataset_path,
        transforms=train_transforms,
        real_files=train_real,
        fake_files=train_fake,
    )
    val_dataset = FaceDataset(
        dataset_path,
        transforms=val_transforms,
        real_files=val_real,
        fake_files=val_fake,
    )

    print(
        "Dataset split: "
        f"train real={len(train_real)}, train fake={len(train_fake)}, "
        f"val real={len(val_real)}, val fake={len(val_fake)}"
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader

# Cell 5: Model Setup
def get_model(model_name=DEFAULT_MODEL_NAME, num_classes=2, pretrained=False):
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model

# Cell 6: Training and Validation Functions
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            probs = torch.softmax(outputs, dim=1)[:,1]
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    if len(set(all_labels)) < 2:
        epoch_auc = 0.0
        print("Warning: validation split has only one class, so AUC is set to 0.0")
    else:
        epoch_auc = roc_auc_score(all_labels, all_preds)
    return epoch_loss, epoch_auc

# Cell 7: Main Training Loop
def main():
    import argparse
    import time
    parser = argparse.ArgumentParser(description="Train the deepfake detection model")
    parser.add_argument("--dataset", default="./dataset", help="Dataset root with real and fake folders")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--output", default="best_model.pth", help="Path to save best checkpoint")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="timm model name")
    parser.add_argument("--max-images-per-class", type=int, default=0, help="Limit images per class for quick CPU training")
    parser.add_argument("--pretrained", action="store_true", help="Download and use ImageNet pretrained weights")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = args.dataset
    batch_size = args.batch_size
    num_epochs = args.epochs

    train_loader, val_loader = create_dataloaders(
        dataset_path,
        batch_size=batch_size,
        max_images_per_class=args.max_images_per_class,
    )

    model = get_model(model_name=args.model, pretrained=args.pretrained)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_auc = -1.0
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(
                {
                    "model_name": args.model,
                    "num_classes": 2,
                    "model_state_dict": model.state_dict(),
                    "best_auc": float(best_auc),
                },
                args.output,
            )

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train loss: {train_loss:.4f} Acc: {train_acc:.4f} "
              f"Val loss: {val_loss:.4f} AUC: {val_auc:.4f} "
              f"Time: {(time.time()-start_time):.2f}s")

if __name__ == "__main__":
    main()
