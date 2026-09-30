import os

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from .config import Config


def get_cifar100_loaders(augment: bool = True):
    """Create reproducible CIFAR-100 train/validation/test loaders."""
    stats = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(Config.IMG_SIZE, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*stats),
        ]
        if augment
        else [transforms.ToTensor(), transforms.Normalize(*stats)]
    )
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(*stats)]
    )

    train_data = datasets.CIFAR100(
        root=Config.DATA_ROOT, train=True, download=True, transform=train_transform
    )
    test_data = datasets.CIFAR100(
        root=Config.DATA_ROOT, train=False, download=True, transform=test_transform
    )

    val_size = int(0.1 * len(train_data))
    train_size = len(train_data) - val_size
    train_subset, val_subset = random_split(
        train_data,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(Config.SEED),
    )

    num_workers = 0 if os.name == "nt" else Config.NUM_WORKERS
    train_loader = DataLoader(train_subset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_subset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
