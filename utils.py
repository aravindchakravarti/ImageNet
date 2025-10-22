import matplotlib.pyplot as plt
import torch
import logging
from torch_lr_finder import LRFinder
from model_v1 import ResNet
from torch import nn

logger = logging.getLogger(__name__)

def  show_dataset_stats(image_loader):
    imgs = []
    for index, item in enumerate(image_loader):
        # Sampling 1024 images only
        if index > 1024:
            break
        else:
            imgs.append(item[0])

    imgs = torch.stack(imgs, dim=0).numpy()
    imgs_r = imgs[:,0,:,:].flatten()
    imgs_g = imgs[:,1,:,:].flatten()
    imgs_b = imgs[:,2,:,:].flatten()
    logger.info(f"Flatten images size {imgs_r.shape}, {imgs_g.shape}, {imgs_b.shape}")

    plt.hist(imgs_r, bins=50, alpha=0.33, color='r', label='Red')
    plt.hist(imgs_g, bins=50, alpha=0.33, color='g', label='Green')
    plt.hist(imgs_b, bins=50, alpha=0.33, color='b', label='Blue')
    plt.legend()
    plt.show()

def dataset_visualizer(dataset_loader, n_images=12):
    """Visualize a few samples from the dataset loader with labels."""
    batch_data, batch_label = next(iter(dataset_loader))
    n_images = min(n_images, len(batch_data))

    # Get class names if available (e.g., from ImageFolder)
    try:
        classes = dataset_loader.dataset.classes
    except AttributeError:
        classes = None  # fallback if classes not available

    fig = plt.figure(figsize=(12, 9))
    for i in range(n_images):
        plt.subplot(3, 4, i + 1)
        plt.tight_layout()

        # Denormalize for better visualization (optional but recommended)
        img = batch_data[i].cpu().permute(1, 2, 0)
        # If you used ImageNet normalization, undo it:
        if img.min() < 0:  # implies normalized
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
            img = img * std + mean
            img = torch.clamp(img, 0, 1)

        plt.imshow(img)
        label = batch_label[i].item()
        label_text = classes[label] if classes else f"Class {label}"
        plt.title(label_text, fontsize=10)
        plt.xticks([])
        plt.yticks([])
    plt.show()


def indentify_optim_lr(device, train_loader):
    amp_config = {
        'device_type': 'cuda',
        'dtype': torch.float16,
    }
    grad_scaler = torch.cuda.amp.GradScaler()

    model = ResNet(layers=[2,2,3,2], num_classes=10, use_depthwise=(False, False, True, True)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, weight_decay=1e-2)

    lr_finder = LRFinder(
        model, optimizer, criterion, device=device,
        amp_backend='torch', amp_config=amp_config, grad_scaler=grad_scaler
    )
    lr_finder.range_test(train_loader, end_lr=0.01, num_iter=60, step_mode='exp')
    lr_finder.plot()
    lr_finder.reset()