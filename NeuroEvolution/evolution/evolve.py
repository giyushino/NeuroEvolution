#conda_env: NeuroEvolution
import gc
import torch
import numpy as np
import random as pyrandom # there's an overlap with the jax.random for some reason so have to rename
from datasets import load_dataset
import matplotlib.pyplot as plt

from NeuroEvolution.utils.timed import timed
from NeuroEvolution.utils.params import merge
from NeuroEvolution.utils.device import DEVICE
from NeuroEvolution.models.model_loader import torch_cnn, torch_linear_classifier, torch_vit
from NeuroEvolution.eval.inference import accuracy, batch
from NeuroEvolution.datasets.load import load_doodle, load_doodle_two_classes

class Evolve:
    def __init__(self, model, dataset, config, population_size, 
                 generations, threshold, batch_size, random_strength):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.size = population_size
        self.models = {}
        self.generations = generations
        self.threshold = threshold
        self.batch_size = batch_size
        self.random_strength = random_strength
        self.current_generation = 0
        self.count = 0


    def create_model(self):
        self.config["name"] = f"Generation{self.current_generation}_Model{self.count}"
        return self.model(**self.config).to(DEVICE)


    def generation(self):
        """
        Creates population with size # of models 
        
        Args: 
            size (int): Number of members in generation 
            model (torch.nn.module): Model

        Return: 
            population (list): List containing the models
        """
        self.count = 0

        for i in range(self.size - len(self.models)):
            self.count += 1
            self.config["name"] = f"Generation{self.current_generation}_Model{self.count}"
            self.models[self.create_model()] = None

        return self.models


    def culling(self, threshold=None):
        print(f"{len(self.models)} present")
        if threshold is not None:
            # overide threshold to allow for dynamic selection
            self.threshold = threshold
        
        # testing 
        self.threshold += 0.005

        start_index = pyrandom.randint(0, len(self.dataset["train"]) - self.batch_size - 1)
        img_batch, truth_batch = batch(self.batch_size, start_index, self.dataset, animals={"lion": 0, "sheep": 1}) 
        #img_batch, truth_batch = batch(self.batch_size, start_index, self.dataset) 
    
        count = 0
        for model in self.models:
            count +=1 
            # set show = True for debugging
            percent_right = accuracy(model.to(DEVICE), img_batch.to(DEVICE), truth_batch.to(DEVICE))
            self.models[model] = accuracy(model.to(DEVICE), img_batch.to(DEVICE), truth_batch.to(DEVICE))

        self.models = {k: v for k, v in self.models.items() if v >= self.threshold}
        self.models = dict(sorted(self.models.items(), key=lambda item: item[1], reverse=True))

        gc.collect()
        torch.cuda.empty_cache()
        print(f"{len(self.models)} survived")
        return self.models


    def see_accuracy(self, max_display=10):
        """
        Prints model accuracies in a formatted box
        """
        lines = []
        count = 0
        accuracy = 0

        if len(self.models) == 0: 
            return 
        for model in self.models:
            acc = self.models[model]
            accuracy += acc
            if count <= max_display: 
                lines.append(f"|| {model.name:<20} || Accuracy: {acc:.4f} ||")
                count += 1

        width = max(len(line) for line in lines)
        border = "+" + "-" * (width - 2) + "+"
        
        print(f"GENERATION {self.current_generation} || AVERAGE ACC: {accuracy / len(self.models)} || CURRENT THRESHOLD: {self.threshold}")
        print(border)
        for line in lines:
            print(line.ljust(width))
        print(border)


    def skewed_left_choice(self, n, power=1.5):
        x = np.arange(1, n+1)
        probs = 1 / x**power
        probs /= probs.sum()
        return np.random.choice(x, p=probs)


    def procreate(self, scale=20, size=200):
        possible_parents = list(range(1, len(self.models)))
        parent_1_num = self.skewed_left_choice(len(possible_parents)); parent_2_num = 1

        while parent_1_num == parent_2_num:
            parent_2_num = self.skewed_left_choice(len(possible_parents))

        child = self.create_model() 
        parent_1, parent_2 = list(self.models.keys())[parent_1_num], list(self.models)[parent_2_num]
        self.config["name"] = f"Generation{self.current_generation}_Model{self.count}"
        self.config["parent_1"] = parent_1.name; self.config["parent_2"] = parent_2.name
        return merge(parent_1, parent_2, child, None, True, self.random_strength)


    def new_generation(self):
        self.current_generation += 1
        if len(self.models) == 0 or self.current_generation == self.generations: 
            return 
        
        for i in range(self.size - len(self.models)):
            self.count += 1 
            child = self.procreate()
            self.models[child] = None
                
        return self.models


"""
args = {
    "num_classes" : 2, 
    "num_channels" : 1, 
}
"""


#brah = Evolve(torch_cnn_init, load_doodle_two_classes().shuffle(), args, population_size=100, generations=10000, threshold = 0.5, batch_size=64, random_strength = 0.8)
args = {
    "image_size": 28,
    "num_classes": 5
}
brah = Evolve(linear_classifier_model, load_doodle_two_classes().shuffle(), args, population_size=100, generations=10000, threshold = 0.5, batch_size=64, random_strength = 0.8)
brah.generation()
for i in range(100):
    brah.culling()
    brah.see_accuracy()
    brah.new_generation()


