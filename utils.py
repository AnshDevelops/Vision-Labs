import os
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from torch.utils.data import DataLoader


def get_num_files(path) -> None:
    """
    Walks through a directory and prints its contents
    :param path: Path to directory.
    """

    for path, _, filenames in os.walk(path):
        print(f"There are {len(filenames)} images in '{path}'.")


def get_normalization_params(loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the mean and standard deviation across all images in a DataLoader.
    :param loader: DataLoader containing the dataset.
    :return: Tuple of mean and standard deviation tensors for each channel.
    """

    mean = 0.0
    std = 0.0
    total_images = 0

    for images, _ in loader:
        images_per_batch = images.size(0)
        images = images.view(images_per_batch, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += images_per_batch
    
    mean /= total_images
    std /= total_images

    return mean, std
