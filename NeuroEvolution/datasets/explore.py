#conda_env: NeuroEvolution

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random
import torch

from NeuroEvolution.datasets.load import load_cifar, load_doodle

def view_doodle(dataset, dataset_type, image_index):
    """
    View an image from dataset
    """
    raw_data = dataset[dataset_type][image_index]["img"]
    torch_tensor = torch.tensor(raw_data) * 255
    image = np.array(torch_tensor, dtype=np.uint8)
    name = dataset[dataset_type][image_index]["label"]
    # Show image
    display = Image.fromarray(image)
    plt.imshow(display, cmap="gray")
    plt.title(f"{name}")
    plt.axis("off")  
    plt.show()
    return image


def view_cifar(dataset_type, image_index):
    dataset = load_cifar()
    image = dataset[dataset_type][image_index]["img"]
    name = dataset[dataset_type][image_index]["label"]

    plt.imshow(image)
    plt.title(f"{name}")
    plt.axis("off")
    plt.show()

    return 0

if __name__ == "__main__":
    #view_cifar("train", 0)
    view_doodle(load_doodle(), "train", 0)
