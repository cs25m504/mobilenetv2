import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_cifar10_dataloaders
from model import get_cifar10_mobilenetv2
from utils import set_seed, evaluate_top1, save_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Train MobileNet-v2 Baseline on CIFAR-10")
    parser.add_argument("--data_dir", type=str, default="./data", help="Dataset storage path")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save checkpoints and plots")
    parser.add_argument("--epochs", type=int, default=180, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=0.1, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="L2 weight decay")
    parser.add_argument("--width_mult", type=float, default=1.0, help="MobileNet-v2 width multiplier")
    parser.add_argument("--dropout", type=float, default=0.2, help="Classifier dropout rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader worker processes")
    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * targets.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return running_loss / total, 100.0 * correct / total


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    train_loader, test_loader = get_cifar10_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = get_cifar10_mobilenetv2(
        num_classes=10,
        width_mult=args.width_mult,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_acc = 0.0
    best_ckpt_path = os.path.join(args.save_dir, "baseline_mobilenetv2.pth")

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate_top1(model, test_loader, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_acc": best_acc,
                "config": vars(args),
            }, best_ckpt_path)

        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            elapsed = (time.time() - start_time) / 60.0
            print(
                f"Epoch [{epoch:3d}/{args.epochs}] | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}% | "
                f"Best: {best_acc:.2f}% ({elapsed:.1f}m)"
            )

    curves_path = os.path.join(args.save_dir, "baseline_curves.png")
    save_training_curves(history, save_path=curves_path)
    print(f"\nTraining complete. Best Test Top-1 Accuracy: {best_acc:.2f}%")
    print(f"Checkpoint saved to: {best_ckpt_path}")
    print(f"Training curves saved to: {curves_path}")


if __name__ == "__main__":
    main()