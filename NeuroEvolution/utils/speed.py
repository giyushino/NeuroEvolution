#conda_env: NeuroEvolution
import sys
import time
import random as pyrandom 

from NeuroEvolution.models.model_loader import *
from NeuroEvolution.utils import device
from NeuroEvolution.utils.device import DEVICE
from NeuroEvolution.eval.inference import *

def generation_speed(size, model, config):
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
    t0 = time.time()
    if failed is True: 
        for i in range(size):
            models.append(model(config))
    else:
        for i in range(size):
            models.append(model(*config))
    t1 = time.time()
    model = str(model).split(" ")[1]
    length = len(f"|| {model}: {size} || Time Taken: {(t1 - t0):.4f} || Time Per Model: {(t1 - t0)/size:.4f} ||")
    print("+" + ("-" * (length - 2)) + "+")
    print(f"|| {model}: {size} || Time Taken: {(t1 - t0):.4f} || Time Per Model: {(t1 - t0)/size:.4f} ||")
    print("+" + ("-" * (length - 2)) + "+")
    return models 

def run_inference_benchmarks(model, dataset, batch_sizes):
    """
    Runs inference benchmarks for a given model across multiple batch sizes.

    Args:
        model (torch.nn.Module): The neural network model to benchmark.
        dataset (torch.utils.data.Dataset): The dataset to draw batches from.
        batch_sizes (list): A list of integer batch sizes to test.
    """
    # First entry in batch is always slowest for some reason
    model.to(DEVICE) 
    model.eval() 

    # Define column widths for consistent formatting
    batch_col_width = 12
    time_col_width = 16
    per_image_col_width = 16

    # Header and Footer
    border_line = "+" + "-" * (batch_col_width + 2) + \
                  "+" + "-" * (time_col_width + 2) + \
                  "+" + "-" * (per_image_col_width + 2) + "+"

    print(f"{DEVICE} \n" + border_line)
    print(f"| {'Batch Size':^{batch_col_width}} | {'Time (s)':^{time_col_width}} | {'Per Image (s)':^{per_image_col_width}} |")
    print(border_line)

    with torch.no_grad(): # Disable gradient calculations for faster inference
        for batch_size in batch_sizes:
            max_start_index = len(dataset["train"]) - batch_size
            if max_start_index < 0: # Handle cases where dataset is smaller than batch_size
                start_index = 0
            else:
                start_index = pyrandom.randint(0, max_start_index)

            batch_torch, _ = batch(batch_size, start_index, dataset)

            # Move batch to device *before* passing to model for timing
            batch_torch = batch_torch.to(DEVICE)

            t0 = time.time()
            output = model(batch_torch)
            t1 = time.time()

            time_taken = t1 - t0
            per_image_time = time_taken / batch_size

            print(f"| {str(batch_size):^{batch_col_width}} | {time_taken:^{time_col_width}.6f} | {per_image_time:^{per_image_col_width}.6f} |")
    print(border_line)


if __name__ == "__main__":
    #generation_speed(100, linear_classifier_model, (28, 5))
    dataset = load_doodle().shuffle() 
    model = torch_cnn_init(5, 1)
    batches = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    run_inference_benchmarks(model, dataset, batches) 

    """
    batch_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512]
    for batch_size in batch_sizes:
        inference_speed(model, batch_size, dataset)
    """
