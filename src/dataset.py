import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import h5py

class PatchCamelyonDataset(Dataset):
    def __init__(self, x_path=None, y_path=None, split='train', num_mock_samples=500, transform=None, stain_norm=False):
        self.transform = transform
        self.stain_norm = stain_norm
        self.is_mock = x_path is None or y_path is None
        self.split = split

        if self.is_mock:
            print(f"Creating mock PatchCamelyon dataset ({split} split, {num_mock_samples} samples)...")
            self.num_samples = num_mock_samples
            self.images = []
            self.labels = []

            # Generate simulated H&E patches (circles for nuclei)
            np.random.seed(42 if split == 'train' else 43)
            for i in range(self.num_samples):
                label = np.random.choice([0, 1])
                bg_color = np.array([235, 215, 225]) + np.random.randint(-10, 10, 3)
                img = np.ones((96, 96, 3), dtype=np.uint8) * np.clip(bg_color, 0, 255).astype(np.uint8)

                # Add nuclei (tumours have more nuclei clustered in center)
                num_nuclei = np.random.randint(25, 45) if label == 1 else np.random.randint(5, 15)
                for _ in range(num_nuclei):
                    cx = np.random.randint(30, 66) if label == 1 else np.random.randint(5, 91)
                    cy = np.random.randint(30, 66) if label == 1 else np.random.randint(5, 91)
                    radius = np.random.randint(2, 6)
                    n_color = [np.random.randint(40, 90), np.random.randint(30, 70), np.random.randint(110, 160)]
                    y, x = np.ogrid[-cy:96-cy, -cx:96-cx]
                    img[x*x + y*y <= radius*radius] = n_color
                self.images.append(img)
                self.labels.append(label)
        else:
            self.x_file = h5py.File(x_path, 'r')
            self.y_file = h5py.File(y_path, 'r')
            self.x_data = self.x_file['x']
            self.y_data = self.y_file['y']
            self.num_samples = len(self.x_data)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.is_mock:
            img = self.images[idx]
            label = self.labels[idx]
        else:
            img = self.x_data[idx]
            label = int(self.y_data[idx].squeeze())

        if self.stain_norm:
            try:
                from src.utils import normalize_stain_macenko
                img = normalize_stain_macenko(img)
            except Exception:
                try:
                    from utils import normalize_stain_macenko
                    img = normalize_stain_macenko(img)
                except Exception:
                    pass

        if self.transform:
            img = self.transform(Image.fromarray(img))
        else:
            img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0

        return img, label

def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_test_transform
