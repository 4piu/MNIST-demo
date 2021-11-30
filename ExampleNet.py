import torch.nn as nn
import torch
import torch.nn.functional as F

class ExampleNet(nn.Module):
    def __init__(self):
        super(ExampleNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)  # [1, 28, 28] -> [32, 26, 26]
        x = F.relu(x)
        x = self.conv2(x)  # [32, 26, 26] -> [64, 24, 24]
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # [64, 24, 24] -> [64, 12, 12]
        x = self.dropout1(x)
        x = torch.flatten(x, 1)  # [64, 12, 12] -> [9216]
        x = self.fc1(x)  # [9216] -> [128]
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)  # [128] -> [10]
        output = F.log_softmax(x, dim=1)
        return output

