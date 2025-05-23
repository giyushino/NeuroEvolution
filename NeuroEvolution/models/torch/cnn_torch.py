#conda_env: NeuroEvolution 

import torch
import torch.nn as nn
import torch.nn.functional as F

class TorchCNN(nn.Module):
  def __init__(self, num_classes, num_channels):
    super(TorchCNN, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=32, kernel_size=3)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    self.fc1 = nn.Linear(64 * 12 * 12, 128) # 14 for size 32 x 32 images
    self.dropout = nn.Dropout(p=0.5)
    self.fc2 = nn.Linear(128, num_classes)

  def forward(self, x):

    x = F.relu(self.conv1(x)) 
    x = F.relu(self.conv2(x)) 
    x = self.pool(x)     
    x = x.view(x.size(0), -1) 
    x = F.relu(self.fc1(x))
    x = self.dropout(x)
    x = self.fc2(x)
    return x

if __name__ == "__main__":
    cnn = TorchCNN(2, 1)
    img = torch.rand(16, 1,  28, 28)
    output = cnn(img)
    print(output)

    """
    if x.shape[1] == 32 and x.shape[-1] == 3:
      x = x.permute(0, 3, 1, 2)  
    """
