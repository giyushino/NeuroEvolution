#conda_env: NeuroEvolution

import jax
from numpy import imag
import torch
from transformers import CLIPModel, CLIPProcessor

from NeuroEvolution.models.jax.cnn_jax import * 
from NeuroEvolution.models.jax.vit_jax import * 
from NeuroEvolution.models.torch import linear_classifier_torch
from NeuroEvolution.models.torch.cnn_torch import *
from NeuroEvolution.models.torch.vit_torch import *
from NeuroEvolution.models.torch.linear_classifier_torch import *

def jax_cnn(batch_size, image_size, num_classes, num_channels):
    example_input = jnp.ones((batch_size, 
                              image_size[0], 
                              image_size[1], 
                              num_channels))
    key = jax.random.PRNGKey(0)    
    model = JaxCNN(num_classes=num_classes)
    rng = {'params': key, 'droput': key} 
    params = model.init(rng, example_input)
    
    return model, params

def jax_vit(batch_size, config):
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

def torch_cnn(num_classes, num_channels, name = None, parent_1 = None, parent_2 = None):
    return TorchCNN(num_classes = num_classes, num_channels = num_channels, name = name, parent_1 = parent_1, parent_2 = parent_2)

def torch_vit(config):
    return TorchViT(**config) 

def torch_linear_classifier(config):
    return TorchLinear(**config)

def openai_clip():
    # reinit model 
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    return model, processor 




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

    model = torch_linear_classifier(28, 2)
    example = torch.rand(10, 784)
    output = model(example)
    print(output)
    """ 
    model_fp32 = torch_cnn_init(5, 2)
    print(model_fp32)
    # quantize
    model_int8 = torch.ao.quantization.quantize_dynamic(
        model_fp32,  # the original model
        {torch.nn.Linear, torch.nn.Conv2d, torch.nn.MaxPool2d, 
         torch.nn.functional.relu},  # a set of layers to dynamically quantize
        dtype=torch.qint8)  # the target dtype for quantized weights
    print(model_int8)
    model, params = jax_vit_init(16, vit_config) 
    init_rngs = {'params': jax.random.PRNGKey(1), 
                'dropout': jax.random.PRNGKey(2), 
                'emb_dropout': jax.random.PRNGKey(3)}
    img = jax.random.normal(jax.random.PRNGKey(0), (16, 28, 28, 1))
    output = model.apply(params, img, rngs=init_rngs)
    print(output)
    print(output.shape)
    """

