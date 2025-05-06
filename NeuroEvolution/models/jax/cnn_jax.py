#conda_env: NeuroEvolution

import jax
import jax.numpy as jnp
from jax import random, jit, grad
from flax.core.frozen_dict import freeze, unfreeze
from pprint import pprint
import flax.linen as nn
import optax
from flax.traverse_util import flatten_dict
from flax.core.frozen_dict import unfreeze

class JaxCNN(nn.Module):
    """
    Basic CNN written in JAX
    """
    num_classes: int  

    def setup(self):
        self.conv1 = nn.Conv(features=32, kernel_size=(3, 3))
        self.conv2 = nn.Conv(features=64, kernel_size=(3, 3))
        self.fc1 = nn.Dense(features=128)
        self.fc2 = nn.Dense(features=self.num_classes)  
        self.dropout = nn.Dropout(rate=0.5)

    def __call__(self, x, training=True, dropout_key=None):
        x = self.conv1(x)
        x = nn.relu(x)
        x = self.conv2(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = self.fc1(x)
        x = nn.relu(x)
        x = self.dropout(x, deterministic=not training, rng=dropout_key)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    batch_size = 16
    train_image = jnp.ones((batch_size, 28, 28, 1))
    key = random.PRNGKey(0)
    
    model = JaxCNN(num_classes=5)  
    
    rngs = {'params': key, 'dropout': key}
    params = model.init(rngs, train_image[0:1])
    tx = optax.adam(0.001)
    opt_state = tx.init(params)

    output = model.apply(params, train_image, training=False, dropout_key=key)
