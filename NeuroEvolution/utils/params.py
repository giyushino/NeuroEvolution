#conda_env: NeuroEvolution

import numpy as np
from flax.core import unfreeze
from flax.traverse_util import flatten_dict

from NeuroEvolution.models.jax.cnn_jax import * 
from NeuroEvolution.models.torch import vit_torch
from NeuroEvolution.models.torch.cnn_torch import * 
from NeuroEvolution.models.model_loader import *

# rewrite 
def jax_model_summary(params):
    flat_params =  flatten_dict(unfreeze(params))
    printed = set()
    print("Model Weights:")
    for path, value in flat_params.items():
        layer = path[0]
        if layer not in printed:
            printed.add(layer)
            print(f"{layer}:")
        print(f"  {'/'.join(path[1:])} → shape: {value.shape}")

    n_params_flax = sum(
        jax.tree.leaves(jax.tree.map(lambda x: np.prod(x.shape), params))
    )
    print(f"Number of parameters in Flax model: {n_params_flax}")
def torch_model_summary(model):
    """
    Prints all layers in the model
    
    Args:
        model (torch.nn.Module): PyTorch model to observe layers of
        specificLayers (list or None): List of specific layer names to observe. If None, observe all layers.
        seeWeights (bool): Whether or not to print the weight tensors
    
    Returns: 
        None
    """ 
    #total_params = sum(p.numel() for p in model.parameters())
    print("Model Weights")
    for name, param in model.named_parameters():
        print(f"  {name.replace(".", "/")} -> shape: {(param.shape)}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters in PyTorch model: {total_params}")


if __name__ == "__main__":
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
    model, params = jax_vit_init(16, vit_config) 
    jax_model_summary(params)


#    jax_cnn, params = jax_cnn_init(16, [28, 28], 6, 1)
#    jax_model_summary(params)
#    torch_cnn = torch_cnn_init(6, 1) 
#    torch_model_summary(torch_cnn)

    



