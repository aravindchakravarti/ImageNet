from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import torch
import os
from config import Config


class HFDataset(torch.utils.data.Dataset):
    """Wraps a HuggingFace dataset (ImageNet) to work with PyTorch transforms."""
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        label = item["label"]

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        # 2. Force RGB (critical for grayscale images)
        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)
        return image, label

def get_train_test_loaders(path, batch_size=64, image_size=224, norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225]):
    """
    Loads ImageNet train and validation datasets from local Hugging Face cache
    and returns PyTorch DataLoaders.
    """
    hf_cache_dir = path

    print(f"Loading ImageNet dataset from local cache: {hf_cache_dir}")

    # Load datasets directly from local cache
    train_ds = load_dataset("imagenet-1k", split="train", cache_dir=hf_cache_dir)
    val_ds = load_dataset("imagenet-1k", split="validation", cache_dir=hf_cache_dir)

    # Define transforms (standard ImageNet preprocessing)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
        transforms.RandomErasing(
            p=0.5,                 # probability of applying
            scale=(0.02, 0.33),    # area range of erasing
            ratio=(0.3, 3.3),      # aspect ratio range
            value=0,               # fill with black value
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(image_size+32),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std),
    ])

    # Wrap HuggingFace datasets
    train_dataset = HFDataset(train_ds, transform=train_transform)
    val_dataset = HFDataset(val_ds, transform=val_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True)

    print("✅ ImageNet train and validation DataLoaders are ready!")
    return train_loader, val_loader
