#conda_env: NeuroEvolution 
import torch 
import time

from NeuroEvolution.datasets.load import load_doodle
from NeuroEvolution.models.model_loader import *
from NeuroEvolution.utils.device import DEVICE

def batch(batch_size, start_index, dataset, animals={"crocodile": 0, "dragon": 1, "duck": 3, "lion": 4, "sheep": 5}, img_size=28, num_channels=1):
    """
    Batch image into tensors

    Args: 
        batch_size (int): Number of examples to pass through model at once 
        start_index (int): Where to start in the dataset
        dataset (dataset): Dataset we want to train model on
        animals (dict): Dictionary containing all classes

    Returns:
        tensor.float() (tensor): Batched tensor that's now a float 
        truth (list): Ground truth labels
    """
    truth = []
    images = [sample for sample in dataset["train"][start_index:start_index + batch_size]["img"]]
    tensor = torch.tensor(images)
    tensor = tensor.view(batch_size, num_channels, img_size, img_size)
    for animal in dataset["train"][start_index:start_index + batch_size]["label"]:
        truth.append(animals[animal])
    return tensor.float(), torch.tensor(truth)

def accuracy(model, batch, ground_truth): 
    correct = 0 
    output = model(batch)
    predictions = torch.argmax(output, dim = 1)
    print(ground_truth) 
    correct_predictions = (predictions == ground_truth)
    accuracy = correct_predictions.float().mean() 
    return accuracy.item()

if __name__ == "__main__":
    batch, truth = batch(10, 0, load_doodle().shuffle())
    vit_config = {
        "image_size": 28,
        "patch_size": 7,
        "num_classes": 5,
        "dim": 1024,
        "depth": 6,
        "heads": 16,
        "mlp_dim": 2048,
        "dropout": 0.1,
        "emb_dropout": 0.1,
        "channels": 1 
    }
    model = torch_vit_init(vit_config)
    t0 = time.time()
    for i in range(1000):
        print(accuracy(model.to(DEVICE), batch.to(DEVICE), truth.to(DEVICE)))
    t1 = time.time()
    print(t1 - t0)
    """
    model = torch_vit_init(vit_config)
    t0 = time() 
    output = model(batch)
    t1 = time() 
    print(output, output.size())
    print(truth)
    print(t1 - t0)
    


    batch, truth = batch(10, 0, load_doodle())
    model = torch_cnn_init(5, 1) 
    output = model(batch)
    print(output)
    print(output.size())
    """




