#conda_env: NeuroEvolution 
import torch 
import time

from NeuroEvolution.datasets.load import load_doodle, load_doodle_two_classes
from NeuroEvolution.models.model_loader import torch_cnn, torch_linear_classifier, torch_vit
from NeuroEvolution.utils.device import DEVICE

def batch(batch_size, start_index, dataset, set_type="train", img_size=28, num_channels=1, debug=False):
    """
    Batch image into tensors

    Args: 
        batch_size (int): Number of examples to pass through model at once 
        start_index (int): Where to start in the dataset
        dataset (dataset): Dataset we want to train model on
        animals (dict): Dictionary containing all classes
        img_size (int): size of one side of square input image
        num_channels (int): number of channels in image 

    Returns:
        tensor.float() (tensor): Batched tensor that's now a float 
        truth (list): Ground truth labels
    """
    truth = []
    images, truth = [], []

    # get the images and truths and place in appropriate arrays
    for i in range(start_index, min(start_index + batch_size, len(dataset[set_type]))):
        images.append(dataset[set_type][i]["img"])
        truth.append(dataset[set_type][i]["label"])

    # reshape the tensor to be the correct shape to go through the model
    tensor = torch.tensor(images).view(batch_size, num_channels, img_size, img_size)
    if debug: 
        print(tensor.float(), torch.tensor(truth, dtype=torch.long))
    return tensor.float(), torch.tensor(truth, dtype=torch.long)

def accuracy(model, batch, ground_truth, show = False): 
    output = model(batch)
    predictions = torch.argmax(output, dim = 1)
    if show is True: 
        print(f"Prediction: {predictions}")
        print(f"Ground Truth: {ground_truth}")
    correct_predictions = (predictions == ground_truth)
    accuracy = correct_predictions.float().mean() 
    #print(f"Accuracy: {accuracy}")
    return accuracy.item()


if __name__ == "__main__":
    datset = load_doodle_two_classes(normalize = False).shuffle()
    for i in range(10):
        import random as pyrandom
        brah, truth = batch(10, pyrandom.randint(0, 1000), datset)
        print(truth)
    """
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
    config = {
        "image_size": 28,
        "num_classes": 5,
    } 
    for i in range(10):
        model = torch_linear_classifier(config)
        print(accuracy(model, batch, truth))
    """

    """
    #model = torch_cnn(5, 1)
    #print(model(batch))
    model = torch_vit(vit_config)
    t0 = time.time() 
    output = model(batch)
    t1 = time.time() 
    print(output, output.size())
    print(truth)
    print(t1 - t0)
    

    #batch, truth = batch(10, 0, load_doodle())
    model = torch_cnn(5, 1) 
    output = model(batch)
    print(output)
    print(truth)
    print(output.size())
    """


