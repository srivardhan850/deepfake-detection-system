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
from torch.cuda.amp import autocast, GradScaler
import glob
import random

# Cell 2: Dataset Class
class FaceDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        """
        Args:
            root_dir (str): Root dataset directory with subfolders 'real' and 'fake'.
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        self.real_dir = os.path.join(root_dir, "real")
        self.fake_dir = os.path.join(root_dir, "fake")
        self.real_files = glob.glob(os.path.join(self.real_dir, "**", "*.jpg"), recursive=True)
        self.fake_files = glob.glob(os.path.join(self.fake_dir, "**", "*.jpg"), recursive=True)
        self.all_files = self.real_files + self.fake_files
        self.labels = [0]*len(self.real_files) + [1]*len(self.fake_files)
        self.transforms = transforms

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
    RandomGaussianBlur(p=0.5),
    RandomJPEGCompression(p=0.5),
    RandomNoise(p=0.5),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
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
def create_dataloaders(dataset_path, batch_size=32, val_split=0.2, seed=42):
    dataset = FaceDataset(dataset_path, transforms=train_transforms)
    # Shuffle and split indices
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    random.seed(seed)
    random.shuffle(indices)
    split = int(np.floor(val_split * dataset_size))
    train_idx, val_idx = indices[split:], indices[:split]

    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, num_workers=4)

    return train_loader, val_loader

# Cell 5: Model Setup
def get_model(num_classes=2):
    model = timm.create_model('xception', pretrained=True)
    n_features = model.fc.in_features
    model.fc = nn.Linear(n_features, num_classes)
    return model

# Cell 6: Training and Validation Functions
def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

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
    epoch_auc = roc_auc_score(all_labels, all_preds)
    return epoch_loss, epoch_auc

# Cell 7: Main Training Loop
def main():
    import time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = "./dataset"  # Update this path to your dataset root
    batch_size = 32
    num_epochs = 10

    train_loader, val_loader = create_dataloaders(dataset_path, batch_size=batch_size)

    model = get_model()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler()

    best_auc = 0.0
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), "best_model.pth")

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train loss: {train_loss:.4f} Acc: {train_acc:.4f} "
              f"Val loss: {val_loss:.4f} AUC: {val_auc:.4f} "
              f"Time: {(time.time()-start_time):.2f}s")

if __name__ == "__main__":
    main()
