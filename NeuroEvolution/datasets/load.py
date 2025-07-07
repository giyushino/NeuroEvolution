#conda_env: NeuroEvolution
import torch

from json import load
from datasets import load_dataset
from NeuroEvolution.utils.dir import BASE

ANIMALS = {"sheep": 0, "lion": 1, "duck": 3, "dragon": 4, "crocodile": 5}

def normalize_img(example, mean, std):
        return {"img": (torch.tensor(example["img"], dtype=torch.float32) - mean) / std }
    
# mean: 0.17667199671268463 || std: 0.3353184163570404
def load_doodle(dataset_path = "NeuroEvolution/data/datasets/doodle.jsonl", normalize = True):
    if normalize:
        return load_dataset("json", data_files = BASE + dataset_path).map(lambda example: normalize_img(example, 0.176, 0.335))
    else:
        return load_dataset("json", data_files = BASE + dataset_path)

# mean: 0.20931898057460785 || std: 0.3547174036502838
def load_doodle_two_classes(dataset_path = "NeuroEvolution/data/datasets/doodle_2_classes.jsonl", normalize = True):
    if normalize:
        return load_dataset("json", data_files = BASE + dataset_path).map(lambda example: normalize_img(example, 0.209, 0.354))
    else:
        return load_dataset("json", data_files = BASE + dataset_path)

def load_cifar():
    return load_dataset("uoft-cs/cifar10")



if __name__ == "__main__":
    doodle = load_doodle_two_classes()
    print(doodle)
    print(doodle["train"][10000]["label"])
#    cifar = load_cifar()
#    print(cifar["train"][0])
#
