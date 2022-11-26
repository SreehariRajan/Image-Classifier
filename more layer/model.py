import torch
import torch.nn as nn
import torch.nn.functional as F

# Class to define the model which we will use for training
# Stuff to fill in: The Architecture of your model, the forward function to define the forward pass
# NOTE!: You are NOT allowed to use pretrained models for this task


class Classifier(nn.Module):
    def __init__(self, n_classes):
        super(Classifier, self).__init__()
        # Useful Link: https://pytorch.org/docs/stable/nn.html

        self.layer1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2), nn.BatchNorm2d(
            64), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2), nn.BatchNorm2d(
            128), nn.ReLU())
        self.pool_1 = nn.MaxPool2d(kernel_size=3, stride=1)

        self.layer3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=2), nn.BatchNorm2d(
            256), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=2), nn.BatchNorm2d(
            512), nn.ReLU())
        self.pool_2 = nn.MaxPool2d(kernel_size=3, stride=1)

        self.layer5 = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=2), nn.BatchNorm2d(
            1024), nn.ReLU())
        # self.layer6 = nn.Sequential(nn.Conv2d(1024, 2048, kernel_size=3, stride=1, padding=2), nn.BatchNorm2d(
        #     2048), nn.ReLU())

        self.pool_3 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, n_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool_1(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool_2(x)
        x = self.layer5(x)
        # x = self.layer6(x)
        x = self.pool_3(x)
        x = x.reshape(x.size(0), -1)
        return F.softmax(x, dim=1)
