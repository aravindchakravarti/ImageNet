from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from config import Config


def get_train_test_loaders(path, batch_size=64, image_size=32, norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225]):
    # Define transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_mean, std=norm_std)
    ])

    train_path = os.path.join(path, 'train')
    val_path = os.path.join(path, 'val')

    # Load datasets
    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_path, transform=val_transform)

    # Create dataloaders

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=Config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=Config.NUM_WORKERS)

    return train_loader, val_loader