import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from src.dataset import PatchCamelyonDataset, get_transforms
from src.model import get_resnet18
from src.utils import plot_predictions

def evaluate_model(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, val_test_transform = get_transforms()

    # Configure datasets
    if args.mock:
        print("Configuring with mock dataset...")
        test_dataset = PatchCamelyonDataset(split='test', num_mock_samples=args.num_mock_samples, transform=val_test_transform, stain_norm=args.stain_norm)
    else:
        print(f"Loading real test dataset from: {args.test_x}")
        if not os.path.exists(args.test_x) or not os.path.exists(args.test_y):
            raise FileNotFoundError(f"Test path not found: {args.test_x} or {args.test_y}. Please run with --mock if datasets are not downloaded.")
        test_dataset = PatchCamelyonDataset(args.test_x, args.test_y, transform=val_test_transform, stain_norm=args.stain_norm)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Initialize model and load checkpoint
    model = get_resnet18(pretrained=False, num_classes=2)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Successfully loaded model checkpoint from {args.model_path}")
    else:
        print(f"Warning: Checkpoint not found at {args.model_path}. Running with random weights.")
    
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    print("Running evaluation on test set...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).squeeze().long()

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Generate Metrics
    print("\n" + "="*40)
    print("Classification Report:")
    print("="*40)
    print(classification_report(all_labels, all_preds, target_names=["Healthy (0)", "Tumour (1)"]))

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    print("\nConfusion Matrix / Caught-vs-Missed Table:")
    print("-" * 42)
    print(f"| True Negatives (Healthy correctly identified): {tn:<5} |")
    print(f"| False Positives (Healthy called Tumour):       {fp:<5} |")
    print(f"| False Negatives (Tumour MISSED!):              {fn:<5} |")
    print(f"| True Positives (Tumour correctly caught):      {tp:<5} |")
    print("-" * 42)

    # Visualize predictions
    if args.output_plot_dir:
        os.makedirs(args.output_plot_dir, exist_ok=True)
        
        # If in mock mode, we have images loaded in self.images
        # Let's get mock images for visualization
        viz_dataset = PatchCamelyonDataset(split='test', num_mock_samples=len(test_dataset))
        
        correct_idx = np.where(all_preds == all_labels)[0]
        wrong_idx = np.where(all_preds != all_labels)[0]

        if len(correct_idx) > 0:
            correct_imgs = [viz_dataset.images[idx] for idx in correct_idx[:3]]
            correct_trues = [all_labels[idx] for idx in correct_idx[:3]]
            correct_preds = [all_preds[idx] for idx in correct_idx[:3]]
            plot_predictions(correct_imgs, correct_trues, correct_preds, 
                             title="Correct Predictions Examples", 
                             save_path=os.path.join(args.output_plot_dir, "correct_examples.png"))

        if len(wrong_idx) > 0:
            wrong_imgs = [viz_dataset.images[idx] for idx in wrong_idx[:3]]
            wrong_trues = [all_labels[idx] for idx in wrong_idx[:3]]
            wrong_preds = [all_preds[idx] for idx in wrong_idx[:3]]
            plot_predictions(wrong_imgs, wrong_trues, wrong_preds, 
                             title="Model Errors (Incorrect Predictions)", 
                             save_path=os.path.join(args.output_plot_dir, "error_examples.png"))
        else:
            print("Zero errors on the test split!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ResNet-18 Tumor Classifier")
    parser.add_argument("--model_path", type=str, default="checkpoints/resnet18_tumor_model.pth", help="Path to saved model checkpoint")
    parser.add_argument("--mock", action="store_true", help="Use mock dataset for evaluation")
    parser.add_argument("--num_mock_samples", type=int, default=100, help="Number of mock samples to generate")
    parser.add_argument("--batch_size", type=int, default=32, help="DataLoader batch size")
    parser.add_argument("--stain_norm", action="store_true", help="Apply Macenko stain normalization")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of loader workers")
    parser.add_argument("--test_x", type=str, default="/content/patchcamelyon/camelyonpatch_level_2_split_test_x.h5", help="Path to test x H5 file")
    parser.add_argument("--test_y", type=str, default="/content/patchcamelyon/camelyonpatch_level_2_split_test_y.h5", help="Path to test y H5 file")
    parser.add_argument("--output_plot_dir", type=str, default="plots", help="Directory to save sample plots")

    args = parser.parse_args()
    evaluate_model(args)
