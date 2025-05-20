#conda_env: NeuroEvolution

from io import RawIOBase
from datasets import load_dataset
import numpy as np
import json 
import os

from NeuroEvolution.utils.dir import BASE


def create_doodle(raw_data_path = "NeuroEvolution/data/raw_data/", save_path = "NeuroEvolution/data/datasets/doodle.jsonl", num_img = 70_000):
    """
    Replicating MNIST -> 60k training images, 10k test set
    5 classes: Crocodile, Dragon, Duck, Lion, Sheep 
    
    Args:
        raw_data_path (str): Where the raw data was downloaded to 
        save_path (str): Where to write the new jsonl dataset to 
        num_imgs (int): Number of images to add to the dataset

    Returns: 
        None
    """

    raw_data_path = BASE + raw_data_path 
    classes = 0
    for file in os.listdir(raw_data_path):
        classes += 1

    img_per_class = num_img // classes
    with open(BASE + save_path, "w") as json_file:
        for file in os.listdir(raw_data_path):
            class_name = file.split("_")[::-1][0].split(".")[0]
            print(f"On class {class_name}")
            class_data = np.load(os.path.join(raw_data_path, file))
            for i in range(img_per_class):
                image_data = class_data[i].reshape(28, 28)/ 255 
                data = {
                        "label": class_name,
                        "img": image_data.tolist() 
                        }
                json.dump(data, json_file)
                json_file.write("\n")

if __name__ == "__main__":
    create_doodle()
