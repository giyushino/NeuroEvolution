#conda_env: NeuroEvolution

import jax
from numpy import imag
import torch
from transformers import CLIPModel, CLIPProcessor

from NeuroEvolution.models.jax.cnn_jax import * 
from NeuroEvolution.models.jax.vit_jax import * 
from NeuroEvolution.models.torch.cnn_torch import *
from NeuroEvolution.models.torch.vit_torch import *

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

def jax_vit_init(batch_size, config):
    num_channels = config["channels"] 
    image_size = config["image_size"]

    key = jax.random.PRNGKey(0)
    img = jax.random.normal(key, (batch_size, image_size, image_size, num_channels))

    init_rngs = {'params': jax.random.PRNGKey(1), 
                'dropout': jax.random.PRNGKey(2), 
                'emb_dropout': jax.random.PRNGKey(3)}
    config.pop("channels")
    model = JaxViT(**config) 
    params = model.init(init_rngs, img)
    return model, params

def torch_cnn_init(num_classes, num_channels):
    return TorchCNN(num_classes = num_classes, num_channels = num_channels)

def torch_vit_init(config):
    return TorchViT(**config) 

def openai_clip():
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    return model, processor 



if __name__ == "__main__":
    temp = 0
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

    
        
    # example for Jax ViT 
    """
    model, params = jax_vit_init(16, vit_config) 
    init_rngs = {'params': jax.random.PRNGKey(1), 
                'dropout': jax.random.PRNGKey(2), 
                'emb_dropout': jax.random.PRNGKey(3)}
    img = jax.random.normal(jax.random.PRNGKey(0), (16, 28, 28, 1))
    output = model.apply(params, img, rngs=init_rngs)
    print(output)
    print(output.shape)
    """

