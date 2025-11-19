import os
import json
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.model_v1 import SimpleCNN


def get_dataloaders(data_root: str, batch_size: int = 32):
    """
    Create train, val, test DataLoaders for User 1 Animals dataset.
    Expects:
      data_root/train/<class>/
      data_root/val/<class>/
      data_root/test/<class>/
    """

    
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    eval_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")
    test_dir = os.path.join(data_root, "test")

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=eval_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=eval_transforms)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    return train_loader, val_loader, test_loader, train_dataset.classes


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser(description="Train model_v1 on Animals dataset (User 1)")
    parser.add_argument("--data-root", type=str, default="data/user1_animals",
                        help="Root folder with train/val/test")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--model-out", type=str, default="models/model_v1.pth",
                        help="Path to save trained weights")
    parser.add_argument("--metrics-out", type=str, default="results/metrics_v1.json",
                        help="Path to save training/val/test metrics as JSON")
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size
    )
    num_classes = len(class_names)
    print(f"Detected {num_classes} classes: {class_names}")

    
    model = SimpleCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    best_val_acc = 0.0
    history = {
        "model_name": "model_v1",
        "dataset": "animals_user1",
        "class_names": class_names,
        "epochs": args.epochs,
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"  Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}"
        )
        print(
            f"  Val   loss: {val_loss:.4f} | Val   acc: {val_acc:.4f}"
        )

        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.model_out)
            print(f"  >> Saved best model to {args.model_out}")

    
    model.load_state_dict(torch.load(args.model_out, map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    elapsed = time.time() - start_time

    history["best_val_acc"] = best_val_acc
    history["test_loss"] = test_loss
    history["test_acc"] = test_acc
    history["training_time_sec"] = elapsed

    with open(args.metrics_out, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTest loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")
    print(f"Metrics saved to {args.metrics_out}")


if __name__ == "__main__":
    main()
