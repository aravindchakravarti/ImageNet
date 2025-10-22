import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import os
import sys
import time
from torch.optim.lr_scheduler import OneCycleLR


import logging
logger = logging.getLogger(__name__)


def save_checkpoint(model, optimizer, scheduler, epoch, best_loss, path):
    state = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'best_loss': best_loss
    }
    torch.save(state, path)
    logger.info(f"✅ Checkpoint saved at epoch {epoch+1} to {path}")


def load_checkpoint(model, optimizer, scheduler, path, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    scheduler.load_state_dict(checkpoint['scheduler_state'])
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['best_loss']
    logger.info(f"✅ Resumed from checkpoint: epoch {start_epoch}, best loss {best_loss:.4f}")
    return start_epoch, best_loss

criterion = nn.CrossEntropyLoss()
train_loss_data = []
train_accuracy = []
test_loss_data = []
test_accuracy = []
learning_rate_over_steps = []

best_loss = float('inf')   # initialize with infinity

scaler = GradScaler()

def train(model, device, train_loader, optimizer, scheduler, epoch):
    model.train()
    pbar = tqdm(train_loader)

    correct = 0
    total = 0
    running_loss = 0.0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # ✅ Enable autocast for forward + loss
        with autocast():
            output = model(data)
            loss = criterion(output, target)

        # ✅ Backward using gradient scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()  # OneCycleLR updates per batch
        learning_rate_over_steps.append(optimizer.param_groups[0]['lr'])

        running_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)

        pbar.set_description(desc=f'loss={loss.item():.4f} batch_id={batch_idx}')

    # Epoch-level stats
    avg_loss = running_loss / len(train_loader)
    acc = 100. * correct / total

    train_loss_data.append(avg_loss)
    train_accuracy.append(acc)

    print(f'\nTrain set (epoch {epoch}): Average loss: {avg_loss:.4f}, Accuracy: {correct}/{total} ({acc:.2f}%)')

# ============================================================
# 3️⃣ Testing Loop
# ============================================================

def test(model, device, test_loader, epoch):
    global best_loss  # to update across epochs
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_loss_data.append(test_loss)
    acc = 100. * correct / len(test_loader.dataset)
    test_accuracy.append(acc)

    print(f'Test set (epoch {epoch}): Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({acc:.2f}%)')

    # ✅ Save best model by lowest test loss
    # if test_loss < best_loss:
    #     best_loss = test_loss
    #     save_dir = "/content/drive/MyDrive/Imagenet" if 'google.colab' in sys.modules else "."
    #     os.makedirs(save_dir, exist_ok=True)
    #     torch.save(model.state_dict(), os.path.join(save_dir, "best_model_imagenette.pt"))
    #     print(f"✅ Saved new best model at epoch {epoch} with loss {best_loss:.4f}")

    # print("\n\n")

    return test_loss


def prepare_and_train(model, lr_from_lr_finder, device, train_loader, val_loader, total_epochs):

    # Initialize optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=1e-4)

    max_lr = float(lr_from_lr_finder)   # Use from LR Finder output
    total_epochs = 15 # Full plan (even if train 20+20)
    scaler = GradScaler()
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=len(train_loader),
        epochs=total_epochs,
        pct_start=0.3,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1e4,
    )

    start_epoch = 0
    save_dir = "/content/drive/MyDrive/Imagenet" if 'google.colab' in sys.modules else "."
    os.makedirs(save_dir, exist_ok=True)
    save_dir_file_name = os.path.join(save_dir, "checkpoint_imagenette.pt")
    if os.path.exists(save_dir_file_name):
        logger.info("Found checkpoint, loading checkpoint ...")
        start_epoch, best_loss = load_checkpoint(model, optimizer, scheduler, save_dir_file_name, device)
    else:
        best_loss = float("inf")

    # ============================================================
    # 5️⃣ Main training loop
    # ============================================================
    start_time = time.time()
    for epoch in range(start_epoch, total_epochs):
        train(model, device, train_loader, optimizer, scheduler, epoch)
        test_loss = test(model, device, val_loader, epoch)
        save_checkpoint(model, optimizer, scheduler, epoch, best_loss, save_dir_file_name)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time:", elapsed_time, "seconds")