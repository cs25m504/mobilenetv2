import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across Python, NumPy, and PyTorch for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate_top1(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device) -> tuple[float, float]:
    """Evaluates average cross-entropy loss and Top-1 accuracy (%) on a dataset."""
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, targets in dataloader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * targets.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / total
    top1_acc = 100.0 * correct / total
    return avg_loss, top1_acc


def save_training_curves(history: dict, save_path: str = "baseline_curves.png") -> None:
    """Plots and saves training/validation loss and Top-1 accuracy curves for Question 1(c)."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss subplot
    ax1.plot(epochs, history["train_loss"], label="Train Loss", linewidth=1.8)
    ax1.plot(epochs, history["test_loss"], label="Test Loss", linewidth=1.8)
    ax1.set_title("MobileNet-v2 CIFAR-10 Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    # Accuracy subplot
    ax2.plot(epochs, history["train_acc"], label="Train Top-1 Acc", linewidth=1.8)
    ax2.plot(epochs, history["test_acc"], label="Test Top-1 Acc", linewidth=1.8)
    ax2.set_title("MobileNet-v2 CIFAR-10 Top-1 Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)