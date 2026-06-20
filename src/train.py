import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.dataset import PatchCamelyonDataset, get_transforms
from src.model import get_resnet18
from src.utils import plot_training_history

def train_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU Model: {torch.cuda.get_device_name(0)}")

    train_transform, val_test_transform = get_transforms()

    # Configure datasets
    if args.mock:
        print("Configuring with mock datasets...")
        train_dataset = PatchCamelyonDataset(split='train', num_mock_samples=args.num_mock_samples, transform=train_transform, stain_norm=args.stain_norm)
        val_dataset = PatchCamelyonDataset(split='val', num_mock_samples=args.num_mock_samples // 2, transform=val_test_transform, stain_norm=args.stain_norm)
    else:
        print(f"Loading real train datasets from: {args.train_x}")
        if not os.path.exists(args.train_x) or not os.path.exists(args.train_y):
            raise FileNotFoundError(f"Train path not found: {args.train_x} or {args.train_y}. Please run with --mock if datasets are not downloaded.")
        train_dataset = PatchCamelyonDataset(args.train_x, args.train_y, transform=train_transform, stain_norm=args.stain_norm)
        val_dataset = PatchCamelyonDataset(args.val_x, args.val_y, transform=val_test_transform, stain_norm=args.stain_norm)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Initialize model, loss, optimizer
    model = get_resnet18(pretrained=True, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print("Starting Training Loop...")
    for epoch in range(args.epochs):
        start_time = time.time()

        # ==================== TRAINING PHASE ====================
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).squeeze().long()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / len(train_dataset)
        epoch_train_acc = correct / total
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        # ==================== VALIDATION PHASE ====================
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).squeeze().long()

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_loss = val_running_loss / len(val_dataset)
        epoch_val_acc = val_correct / val_total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)

        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.epochs} ({elapsed:.1f}s) | "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")

    print("Training Complete!")

    # Save model checkpoint
    os.makedirs(os.path.dirname(args.output_model_path), exist_ok=True)
    torch.save(model.state_dict(), args.output_model_path)
    print(f"Model checkpoint saved to {args.output_model_path}")

    # Plot and save history
    if args.output_plot_path:
        os.makedirs(os.path.dirname(args.output_plot_path), exist_ok=True)
        plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path=args.output_plot_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet-18 Histopathology Tumor Classifier")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="DataLoader batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Adam optimizer learning rate")
    parser.add_argument("--mock", action="store_true", help="Use mock datasets for testing")
    parser.add_argument("--num_mock_samples", type=int, default=500, help="Number of mock samples to generate")
    parser.add_argument("--stain_norm", action="store_true", help="Apply Macenko stain normalization")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of loader workers")
    parser.add_argument("--train_x", type=str, default="/content/patchcamelyon/camelyonpatch_level_2_split_valid_x.h5", help="Path to train x H5 file")
    parser.add_argument("--train_y", type=str, default="/content/patchcamelyon/camelyonpatch_level_2_split_valid_y.h5", help="Path to train y H5 file")
    parser.add_argument("--val_x", type=str, default="/content/patchcamelyon/camelyonpatch_level_2_split_test_x.h5", help="Path to val x H5 file")
    parser.add_argument("--val_y", type=str, default="/content/patchcamelyon/camelyonpatch_level_2_split_test_y.h5", help="Path to val y H5 file")
    parser.add_argument("--output_model_path", type=str, default="checkpoints/resnet18_tumor_model.pth", help="Path to output model checkpoint")
    parser.add_argument("--output_plot_path", type=str, default="plots/training_history.png", help="Path to output loss/accuracy history plot")
    
    args = parser.parse_args()
    train_model(args)
