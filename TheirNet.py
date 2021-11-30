import torch.nn as nn
import torch
import torch.nn.functional as F

class TheirNet(nn.Module):
    def __init__(self):
        super(TheirNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 32, 5, 2)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout(0.4)
        self.conv4 = nn.Conv2d(32, 64, 3, 1)
        self.conv5 = nn.Conv2d(64, 64, 3, 1)
        self.conv6 = nn.Conv2d(64, 64, 5, 2)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(64, 128)
        self.batch_norm3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.batch_norm3(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output