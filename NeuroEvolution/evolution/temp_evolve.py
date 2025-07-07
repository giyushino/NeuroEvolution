#conda_env: NeuroEvolution
import gc
import time
import torch
import numpy as np
import random as pyrandom # there's an overlap with the jax.random for some reason so have to rename
from datasets import load_dataset
import matplotlib.pyplot as plt

from NeuroEvolution.utils.timed import timed
from NeuroEvolution.utils.params import merge
from NeuroEvolution.utils.device import DEVICE
from NeuroEvolution.models.model_loader import linear_classifier_model
from NeuroEvolution.eval.inference import accuracy, batch
from NeuroEvolution.datasets.load import load_doodle, load_doodle_two_classes


if __name__ == "__main__":
    print("temp")


