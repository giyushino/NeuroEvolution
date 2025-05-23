#conda_env: NeuroEvolution
from datasets import load_dataset
import torch

from NeuroEvolution.models.model_loader import *
from NeuroEvolution.utils.timed import timed
from NeuroEvolution.utils.device import DEVICE

# look into torch.cuda.Stream()

def generation(size, model, config, quantize = False):
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
    # the ViT loader unpacks the config in the function, 
    # not sure if I want to change that 
    try: 
        model(*config)
        failed = False 
    except: 
        pass
    if failed is True: 
        for i in range(size):
            if quantize is True: 
                model_int8 = torch.ao.quantization.quantize_dynamic(
                    model(config),  
                    {torch.nn.Linear},  
                    dtype=torch.qint8)  
                models.append(model_int8)
            else:
                models.append(model(config).to(DEVICE))
    else:
        for i in range(size):
            if quantize is True: 
                model_int8 = torch.ao.quantization.quantize_dynamic(
                    model(*config),  
                    {torch.nn.Linear},  
                    dtype=torch.qint8)  
                models.append(model_int8)
            else:
                models.append(model(*config).to(DEVICE))
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
    timed_generation = timed(generation)
    population = timed_generation(100, model, vit_config, True)
    #print(population)


