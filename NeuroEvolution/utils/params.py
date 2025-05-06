#conda_env: NeuroEvolution

from flax.core import unfreeze
from flax.traverse_util import flatten_dict

from NeuroEvolution.models.jax.cnn_jax import * 
from NeuroEvolution.models.torch.cnn_torch import * 

def jax_model_summary(params):
    flat_params =  flatten_dict(unfreeze(params))
    printed = set()
    print("Model Architecture:")
    for path, value in flat_params.items():
        layer = path[0]
        if layer not in printed:
            printed.add(layer)
            print(f"{layer}:")
        print(f"  {'/'.join(path[1:])} → shape: {value.shape}")

def torch_model_summary(model):
    return 0



