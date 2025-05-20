#conda_env: NeuroEvolution

from datasets import load_dataset
import json
import time

from NeuroEvolution.utils.device import DEVICE
from NeuroEvolution.utils.dir import BASE


def batch(dataset, start_index, batch_size, labels = {"crocodile": 0, "dragon": 1, "duck": 3, "lion": 4, "sheep": 5}):
    """
    Batch image into tensors

    Args: 
        dataset (dataset): Dataset we want to train model on, split should already be specified 
        start_index (int): Where to start in the dataset
        batch_size (int): Number of examples to pass through model at once 

    Returns:
        tensor.float() (tensor): Batched tensor that's now a float 
        truth (list): Ground truth labels
    """
    ground_truth = []
    images = []
    for i in range(start_index, start_index + batch_size):
        ground_truth.append(labels[dataset[i]["label"]])
        images.append(dataset[i]["img"])
    return ground_truth, images

if __name__ == "__main__":
    t0 = time.time()
    dataset = load_dataset("json", data_files=f"{BASE}/NeuroEvolution/data/datasets/doodle.jsonl", split="train")
    t1 = time.time()
    print(batch(dataset, 1000, 8))
    print(t1 - t0)
    
    


