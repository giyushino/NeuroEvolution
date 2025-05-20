#conda_env: NeuroEvolution
from datasets import load_dataset
import torch

from NeuroEvolution.models.model_loader import *

def generation(size, model, config):
    """
    Creates population with size # of models 
    
    Args: 
        size (int): Number of members in generation 
        model (torch.nn.module): Model

    Return: 
        population (list): List containing the models
    """
    models = []
    failed = True  
    try: 
        model(*config)
        failed = False 
    except: 
        pass
    if failed is True: 
        for i in range(size):
            models.append(model(config))
    else:
        for i in range(size):
            models.append(model(*config))
    return models 


if __name__ == "__main__":
    model = torch_vit_init
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
    population = generation(10, model, vit_config)
    print(population)


