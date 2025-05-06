#conda_env: NeuroEvolution

import jax
import torch

from NeuroEvolution.models.jax.cnn_jax import * 
from NeuroEvolution.models.torch.cnn_torch import *
from NeuroEvolution.utils.params import print_jax_model_summary 

def jax_cnn_init(batch_size, image_size, num_classes, num_channels):
    example_input = jnp.ones((batch_size, 
                              image_size[0], 
                              image_size[1], 
                              num_channels))
    key = jax.random.PRNGKey(0)    
    model = JaxCNN(num_classes=num_classes)
    rng = {'params': key, 'droput': key} 
    params = model.init(rng, example_input)
    
    return model, params

def torch_cnn_init(num_classes, num_channels):
    return TorchCNN(num_classes = num_classes, num_channels = num_channels)


if __name__ == "__main__":
    torch_cnn = torch_cnn_init(5, 2)
    print(torch_cnn)
    #model, params = jax_cnn_init(16, [28, 28], 1, 5)
    #print_jax_model_summary(params)

