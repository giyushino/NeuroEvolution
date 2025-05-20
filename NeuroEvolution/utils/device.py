#conda_env: NeuroEvolution
import jax 
import torch 

# By default, jax should be using GPU if installed correctly, then DEVICE only needs to be set for torch... I want to test CPU only computations later  
DEVICE = torch.device("cuda")

if __name__ == "__main__":
    print(jax.devices())
    device = torch.device("cpu")
    print(device)
