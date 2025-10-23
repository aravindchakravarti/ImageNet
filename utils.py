import matplotlib.pyplot as plt
import torch
import logging
from torch_lr_finder import LRFinder
import numpy as np

from torch import nn
from config import Config

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
            mean = torch.tensor(Config.NORM_MEAN).view(1, 1, 3)
            std = torch.tensor(Config.NORM_STD).view(1, 1, 3)
            img = img * std + mean
            img = torch.clamp(img, 0, 1)

        plt.imshow(img)
        label = batch_label[i].item()
        label_text = classes[label] if classes else f"Class {label}"
        plt.title(label_text, fontsize=10)
        plt.xticks([])
        plt.yticks([])
    plt.show()


def identify_optim_lr(model, device, train_loader, lr_finder_end_lr, lr_finder_num_iter):
    amp_config = {
        'device_type': 'cuda',
        'dtype': torch.float16,
    }
    grad_scaler = torch.cuda.amp.GradScaler()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, weight_decay=1e-2)

    lr_finder = LRFinder(
        model, optimizer, criterion, device=device,
        amp_backend='torch', amp_config=amp_config, grad_scaler=grad_scaler
    )
    lr_finder.range_test(train_loader, end_lr=lr_finder_end_lr, num_iter=lr_finder_num_iter, step_mode='exp')

    # Convert to numpy arrays
    losses = np.array(lr_finder.history['loss'])
    lrs = np.array(lr_finder.history['lr'])

    # Compute rate of change (gradient) of loss
    loss_grad = np.gradient(losses)
    # Find index where slope is steepest (largest negative gradient)
    min_grad_idx = np.argmin(loss_grad)
    suggested_lr = lrs[min_grad_idx]

    # Alternatively, choose slightly lower (10x smaller)
    safe_lr = suggested_lr / 10

    # Plot and save
    fig = lr_finder.plot()
    fig.savefig('lr_finder_plot.png')
    plt.close(fig)

    lr_finder.reset()

    return safe_lr
