#conda_env: NeuroEvolution

import time
import torch 
import numpy as np
from flax.core import unfreeze
from flax.traverse_util import flatten_dict

from NeuroEvolution.models.torch import vit_torch
from NeuroEvolution.utils.device import DEVICE
from NeuroEvolution.models.model_loader import linear_classifier_model, torch_cnn_init

def randomize(tensor, strength: float):
    """
    Adds noise to tensor by some given strength 

    Args: 
        tensor (torch.tensor): Tensor we want to modify 
        random (float): Strength by which we want to introduce randomness into the tensor 

    Returns: 
        tensor * randTensor: Randomized tensor
    """
    min_val = 1 - strength 
    max_val = 1 + strength 
    rand_tensor =min_val + (max_val -min_val) * torch.rand(tensor.shape)
    return tensor.to(DEVICE) * rand_tensor.to(DEVICE)

def modify(model, specific_layers, random_strength: float): 
    """
    Modify the parameters in a model by given strength by multiplying each layer by matrix of equivalent size, each value ranges from 1-random to 1+random 
    
    Args: 
        model (torch.nn.Module): Model we want to manipulate 
        specific_layers (list): Which layers we want to manipulate 
        random (float): Strength by which we want the model to be randomized

    Returns: 
        model (torch.nn.Module): Modified model
    """
    for name, param in model.named_parameters():
        if specific_layers is None or name in specific_layers:
            original_tensor = param.clone()
            swap_tensor = randomize(original_tensor, random_strength)
            param.data = swap_tensor
    return model

def merge(model_0, model_1, model_2, specific_layers, should_randomize, randomize_strength: float):    
    """
    Merges model_0 and model_1 weights to create model_2,
    essentially mimicking sexual reproduction
    
    Args: 
        model_0 (torch.nn.Module): Parent 0
        model_1 (torch.nn.Module): Parent 1
        model_2 (torch.nn.Module): Child
        specific_layers (list): Which layers we want to manipulate. If left empty, just merge all layers 
        should_randomize (bool): Whether or not to randomize layers after merging
        randomize_strength (float): The amount of noise to add to each layer  

    Returns: 
        model_2 (torch.nn.Module): Child
    """
    
    for name, param in model_2.named_parameters():
        if specific_layers is None or name in specific_layers: 
            tensor_0 = model_0.state_dict()[name].clone() 
            tensor_1 = model_1.state_dict()[name].clone() 
            new_tensor = (tensor_0 + tensor_1) / 2 
            if should_randomize is True: 
                new_tensor = randomize(new_tensor, randomize_strength)
            param.data = new_tensor 

    return model_2

def jax_model_summary(params):
    """
    Prints all layers with weights in the model + their size 
    
    Args:
        model (flax): PyTorch model to observe layers of
    
    Returns: 
        None
    """ 
    flat_params =  flatten_dict(unfreeze(params))
    printed = set()
    print("Model Weights:")
    for path, value in flat_params.items():
        layer = path[0]
        if layer not in printed:
            printed.add(layer)
            print(f"{layer}:")
        print(f"  {'.'.join(path[1:])} → shape: {value.shape}")

    n_params_flax = sum(
        jax.tree.leaves(jax.tree.map(lambda x: np.prod(x.shape), params))
    )
    print(f"Number of parameters in Flax model: {n_params_flax}")

def torch_model_summary(model, print_weights = False):
    """
    Prints all layers with weights in the model + their size 
    
    Args:
        model (torch.nn.Module): PyTorch model to observe layers of
        print_weights (bool): Whether or not to print the params  
    
    Returns: 
        None
    """ 
    #total_params = sum(p.numel() for p in model.parameters())
    print("Model Weights")
    if print_weights is True: 
        for name, param in model.named_parameters():
            #print(f"  {name.replace(".", "/")} -> shape: {(param.shape)}")
            print(f"  {name} -> shape: {(param.shape)}")
            print(f"  Params: {param}")
    else:
        for name, param in model.named_parameters():
            #print(f"  {name.replace(".", "/")} -> shape: {(param.shape)}")
            print(f"  {name} -> shape: {(param.shape)}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in PyTorch model: {total_params}")


def similarity(model_1, model_2):
    count = 0
    layers = 0
    for name, param in model_1.named_parameters():
        layers += 1
        count += torch.cosine_similarity(model_1.state_dict()[name].detach().flatten(), model_2.state_dict()[name].detach().flatten(), 0, 1e-6)
    print(count / layers)
    return count / layers


if __name__ == "__main__":
    t0 = time.time()
    #def __init__(self, image_size, num_classes):
    config = {
        "image_size": 28, 
        "num_classes": 5
             }
    model_1 = linear_classifier_model(config)
    model_2 = linear_classifier_model(config)
    model_3 = linear_classifier_model(config)
    model_2.load_state_dict(model_1.state_dict())  
    model_1 = modify(model_1, None, 0.9)
    #child = merge(model_1, model_2, model_3, None, True, 0.3) 
    #torch_model_summary(child, True)
    model_1.to("cpu")
    model_2.to("cpu")
    similarity(model_1, model_2)
    t1 = time.time()
    print(t1 - t0)
     

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
    
    #torch_vit = torch_vit_init(vit_config)
    #torch_model_summary(torch_vit)
    #t0 = time.time()
    #torch_vit = torch_vit_init(vit_config)
    #torch_model_summary(torch_vit)
    #t1 = time.time()
    #print(f"Time taken: {t1 - t0}")


#    model, params = jax_vit_init(16, vit_config) 
#    jax_model_summary(params)
#    jax_cnn, params = jax_cnn_init(16, [28, 28], 6, 1)
#    jax_model_summary(params)
#    torch_cnn = torch_cnn_init(6, 1) 
#    torch_model_summary(torch_cnn)

    



