"""
Dataset and DataLoader utilities for Monk Skin Tone classification.
Handles class imbalance through weighted sampling and augmentation.
"""

import os
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import yaml


class MonkSkinToneDataset(Dataset):
    """
    Dataset for Monk Skin Tone Scale classification.

    Expects directory structure:
        root/
        ├── scale_1/
        │   ├── image1.jpg
        │   └── ...
        ├── scale_2/
        └── ...
    """

    # Class grouping presets
    CLASS_GROUPINGS = {
        "10class": {i: i for i in range(10)},  # Original 10 classes
        "3class": {
            0: 0, 1: 0, 2: 0,  # Light (scales 1-3)
            3: 1, 4: 1, 5: 1, 6: 1,  # Medium (scales 4-7)
            7: 2, 8: 2, 9: 2,  # Dark (scales 8-10)
        },
        "5class": {
            0: 0, 1: 0,  # Very Light (scales 1-2)
            2: 1, 3: 1,  # Light (scales 3-4)
            4: 2, 5: 2,  # Medium (scales 5-6)
            6: 3, 7: 3,  # Dark (scales 7-8)
            8: 4, 9: 4,  # Very Dark (scales 9-10)
        },
    }

    def __init__(self, root_dir: str, transform=None, class_grouping: str = "10class"):
        """
        Args:
            root_dir: Path to dataset split (train/val/test)
            transform: Optional torchvision transforms
            class_grouping: One of "10class", "3class", "5class"
        """
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.class_grouping = class_grouping

        # Get class mapping
        if class_grouping not in self.CLASS_GROUPINGS:
            raise ValueError(f"Unknown class_grouping: {class_grouping}. Choose from {list(self.CLASS_GROUPINGS.keys())}")
        self.class_map = self.CLASS_GROUPINGS[class_grouping]

        self.samples = []  # List of (image_path, label)
        self.class_counts = Counter()

        # Load all images from scale_1 through scale_10
        for scale in range(1, 11):
            scale_dir = self.root_dir / f"scale_{scale}"
            if not scale_dir.exists():
                print(f"Warning: {scale_dir} does not exist")
                continue

            original_label = scale - 1  # Convert to 0-indexed
            mapped_label = self.class_map[original_label]
            images = list(scale_dir.glob("*.jpg"))

            for img_path in images:
                self.samples.append((img_path, mapped_label))
                self.class_counts[mapped_label] += 1

        print(f"Loaded {len(self.samples)} images from {self.root_dir}")
        print(f"Class grouping: {class_grouping}")
        print(f"Class distribution: {dict(sorted(self.class_counts.items()))}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        return image, label

    @property
    def num_classes(self):
        """Get the number of classes based on the grouping."""
        return len(set(self.class_map.values()))

    def get_class_weights(self, method="inverse_frequency"):
        """
        Calculate class weights for handling imbalanced data.

        Args:
            method: "inverse_frequency" or "effective_samples"

        Returns:
            Tensor of class weights (shape: num_classes)
        """
        num_classes = self.num_classes
        counts = torch.zeros(num_classes)

        for label, count in self.class_counts.items():
            counts[label] = count

        if method == "inverse_frequency":
            # Inverse frequency weighting
            weights = 1.0 / counts
            weights = weights / weights.sum() * num_classes

        elif method == "effective_samples":
            # Effective number of samples (from "Class-Balanced Loss" paper)
            beta = 0.9999
            effective_num = 1.0 - torch.pow(beta, counts)
            weights = (1.0 - beta) / effective_num
            weights = weights / weights.sum() * num_classes
        else:
            weights = torch.ones(num_classes)

        return weights

    def get_sample_weights(self):
        """
        Get per-sample weights for WeightedRandomSampler.

        Returns:
            List of weights, one per sample
        """
        class_weights = self.get_class_weights()
        sample_weights = [class_weights[label].item() for _, label in self.samples]
        return sample_weights


def get_transforms(config: dict, split: str):
    """
    Build transforms based on configuration.

    Args:
        config: Full configuration dictionary
        split: "train" or "val"

    Returns:
        torchvision.transforms.Compose
    """
    aug_config = config["augmentation"][split]

    transform_list = []

    # Resize
    if "resize" in aug_config:
        transform_list.append(transforms.Resize(aug_config["resize"]))

    # Crop
    if split == "train":
        transform_list.append(transforms.RandomCrop(aug_config["crop_size"]))
    else:
        transform_list.append(transforms.CenterCrop(aug_config["crop_size"]))

    # Training augmentations
    if split == "train":
        if aug_config.get("horizontal_flip", False):
            transform_list.append(transforms.RandomHorizontalFlip())

        if aug_config.get("rotation", 0) > 0:
            transform_list.append(transforms.RandomRotation(aug_config["rotation"]))

        if "color_jitter" in aug_config:
            cj = aug_config["color_jitter"]
            transform_list.append(transforms.ColorJitter(
                brightness=cj.get("brightness", 0),
                contrast=cj.get("contrast", 0),
                saturation=cj.get("saturation", 0),
                hue=cj.get("hue", 0)
            ))

    # Convert to tensor
    transform_list.append(transforms.ToTensor())

    # Normalize
    if "normalize" in aug_config:
        transform_list.append(transforms.Normalize(
            mean=aug_config["normalize"]["mean"],
            std=aug_config["normalize"]["std"]
        ))

    return transforms.Compose(transform_list)


def create_dataloaders(config: dict):
    """
    Create train, validation, and test dataloaders.

    Args:
        config: Full configuration dictionary

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Get transforms
    train_transform = get_transforms(config, "train")
    val_transform = get_transforms(config, "val")

    # Get class grouping from config (default: 10class for backward compatibility)
    class_grouping = config["data"].get("class_grouping", "10class")

    # Create datasets
    train_dataset = MonkSkinToneDataset(
        config["data"]["train_dir"],
        transform=train_transform,
        class_grouping=class_grouping
    )

    val_dataset = MonkSkinToneDataset(
        config["data"]["val_dir"],
        transform=val_transform,
        class_grouping=class_grouping
    )

    test_dataset = MonkSkinToneDataset(
        config["data"]["test_dir"],
        transform=val_transform,
        class_grouping=class_grouping
    )

    # Training sampler for class imbalance
    train_sampler = None
    shuffle_train = True

    if config["training"].get("use_oversampling", False):
        sample_weights = train_dataset.get_sample_weights()
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        shuffle_train = False  # Sampler handles shuffling

    # Hardware settings
    num_workers = config["hardware"].get("num_workers", 4)
    pin_memory = config["hardware"].get("pin_memory", True)
    batch_size = config["training"]["batch_size"]

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # Store class weights for loss function
    class_weights = train_dataset.get_class_weights(
        method=config["training"].get("class_weights", "inverse_frequency")
    )

    return train_loader, val_loader, test_loader, class_weights


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    # Test the dataset
    config = load_config("training/configs/config.yaml")

    print("Creating dataloaders...")
    train_loader, val_loader, test_loader, class_weights = create_dataloaders(config)

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Class weights: {class_weights}")

    # Test a batch
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape: {images.shape}")
    print(f"Labels: {labels}")
