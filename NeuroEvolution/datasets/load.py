#conda_env: NeuroEvolution

from json import load
from datasets import load_dataset
from NeuroEvolution.utils.dir import BASE

def load_doodle(dataset_path = "NeuroEvolution/data/datasets/doodle.jsonl"):
    return load_dataset("json", data_files = BASE + dataset_path)

def load_cifar():
    return load_dataset("uoft-cs/cifar10")


if __name__ == "__main__":
    #doodle = load_doodle()
    cifar = load_cifar()
    print(cifar["train"][0])

