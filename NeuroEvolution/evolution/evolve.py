#conda_env: NeuroEvolution
import gc
import torch
import numpy as np
import random as pyrandom # there's an overlab with the jax.random for some reason so have to rename
from datasets import load_dataset

from NeuroEvolution.utils.timed import timed
from NeuroEvolution.utils.params import merge
from NeuroEvolution.utils.device import DEVICE
from NeuroEvolution.models.model_loader import *
from NeuroEvolution.eval.inference import accuracy, batch
from NeuroEvolution.datasets.load import load_doodle, load_doodle_two_classes

# look into torch.cuda.Stream()
def generation(size, model, config, models = None):
    """
    Creates population with size # of models 
    
    Args: 
        size (int): Number of members in generation 
        model (torch.nn.module): Model

    Return: 
        population (list): List containing the models
    """
    models = {}
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
            models[model(config).to(DEVICE)] = None
    else:
        for i in range(size):
            models[model(*config).to(DEVICE)] = None
    return models 

# parallelize using torch.stream()?
# eventually start counting which images we have seen
# try hvaing the threshold being gen size 
def culling(models, dataset, batch_size, threshold = 0.5):
    start_index = pyrandom.randint(0, len(dataset["train"]) - batch_size - 1)
    #img_batch, truth_batch = batch(batch_size, start_index, dataset, animals={"lion": 0, "sheep": 1}) 
    img_batch, truth_batch = batch(batch_size, start_index, dataset) 
    count = 0
    
    for model in models:
        count +=1 
        models[model] = accuracy(model.to(DEVICE), img_batch.to(DEVICE), truth_batch.to(DEVICE))

    sorted_dict = dict(
        sorted(
            ((k, v) for k, v in models.items() if v >= threshold),
            key=lambda item: item[1],
            reverse=True
        )
    ) 

    for model in models.items():
        if model not in sorted_dict: 
            del model 
    gc.collect()
    torch.cuda.empty_cache()
    #print(len(sorted_dict))
    return models

def new_generation(generation_size, models, config):
    failed = True
    try: 
        model(*config)
        failed = False 
    except: 
        pass
  
     
    if failed is True: 
        for i in range(generation_size - len(models)):
            s = np.random.exponential(scale=15, size=100)
            s = s[s <= 100]   
            parent1_num = pyrandom.choice(s); parent2_num = 0 
            while parent2_num == parent1_num:
                parent2_num = pyrandom.choice(s)
            child = 
            models[model(config).to(DEVICE)] = None
    else:
        for i in range(generation_size - len(models)):
            temp_model = model(*config).to(DEVICE)
            models[model(*config).to(DEVICE)] = None
            

def logging(save_path):
    with open(save_path, "a"):
        pass

def evolve(generation_size, dataset, num_generations):
    for i in range(num_generations):
        
        pass
    
if __name__ == "__main__":
    # might be fun to do lineage modeling
    #timed_generation = timed(generation)
    #population = timed_generation(100, torch_cnn_init, (5, 1))
    #print(population)
    #dataset = load_doodle_two_classes().shuffle()

    dataset = load_doodle().shuffle()
    print("loaded dataset")
    population = generation(10, torch_cnn_init, (5, 1))
    print("created population")
    culling(population, dataset, 10, 0.15)
    

    """
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
    """
