#conda_env: NeuroEvolution

import torch 
import torch.nn as nn 

class TorchLinear(nn.Module):
    def __init__(self, image_size, num_classes):
        super(TorchLinear, self).__init__()
        self.linear = nn.Linear(in_features = image_size**2, out_features=num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(self.linear(x.reshape(x.size(0), -1)))

"""
def linear_classifier_model(image_size = 28, num_classes = 5):
    linear_classifier = torch.nn.Sequential( 
        torch.nn.Linear(in_features = image_size**2, out_features =num_classes), 
        torch.nn.Softmax(dim=1) 
    )   
    return linear_classifier
"""
if __name__ == "__main__":
    linear = TorchLinear(28, 5)
    test_img = torch.ones(1, 28**2)
    print(linear(test_img))
