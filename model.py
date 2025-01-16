import torch.nn as nn
from torchvision import models


class BinaryResNet18(nn.Module):
    def __init__(self):
        super(BinaryResNet18, self).__init__()
        self.base_model = models.resnet18(weights="ResNet18_Weights.DEFAULT")
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 1) # single output node

    def forward(self, x):
        return self.base_model(x)