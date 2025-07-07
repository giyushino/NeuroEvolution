#conda_env: NeuroEvolution 
import torch 
import random as pyrandom

from NeuroEvolution.utils.device import DEVICE
from NeuroEvolution.eval.inference import batch, accuracy
from NeuroEvolution.models.model_loader import torch_linear_classifier, torch_cnn
from NeuroEvolution.datasets.load import load_cifar, load_doodle, load_doodle_two_classes


def train(model, dataset, epochs, batch_size):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    total_loss = 0
    model.to(DEVICE)
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, (len(dataset["train"]) // batch_size * batch_size), batch_size):
            img_batch, truth = batch(batch_size, i, dataset, debug=False)
            #print(truth)
            output = model(img_batch.to(DEVICE))
            loss = criterion(output, truth.to(DEVICE))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() / batch_size
            if i % 100 == 0:
                print(f"Epoch {epoch} || {(i / len(dataset["train"]) * 100):.4f}  || loss: {epoch_loss}")
        with torch.no_grad():
            img_batch, truth = batch(batch_size, pyrandom.randint(0, 10000), dataset, debug=False)
            print(accuracy(model, img_batch.to(DEVICE), truth.to(DEVICE), show = True))
        print(f"completed epoch {epoch} || loss: {epoch_loss}")



if __name__ == "__main__":
    print(DEVICE)
    DEVICE = "cpu"
    dataset = load_doodle_two_classes().shuffle()
    #model = torch_linear_classifier({"image_size": 28, "num_classes": 2}).to(DEVICE) 
    config = {"num_classes": 2, "num_channels": 1}
    model = torch_cnn(2, 1)
    train(model, dataset, 100, 128)
