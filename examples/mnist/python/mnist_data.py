from __future__ import annotations

import torch
from mnist_config import DataConfig
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def dataset_class(name: str) -> type[datasets.MNIST]:
    if name == "mnist":
        return datasets.MNIST
    if name == "fashion_mnist":
        return datasets.FashionMNIST
    raise ValueError(f"unsupported dataset: {name}")


def create_data_loaders(
    config: DataConfig,
    *,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    transform = transforms.ToTensor()
    dataset = dataset_class(config.dataset)
    train_dataset = dataset(config.root, train=True, download=True, transform=transform)
    test_dataset = dataset(config.root, train=False, download=True, transform=transform)
    loader_options = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": config.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)
    return train_loader, test_loader
