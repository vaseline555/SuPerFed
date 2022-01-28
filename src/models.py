import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

#################################
# Modules for subspace learning #
#################################
# Credits to: https://github.com/apple/learning-subspaces (Wortsman et al., 2021)
class SubspaceConv(nn.Conv2d):
    def forward(self, x):
        # call get_weight, which samples from the subspace, then use the corresponding weight.
        w = self.get_weight()
        x = F.conv2d(
            x,
            w,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return x

class TwoParamConv(SubspaceConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight1 = nn.Parameter(torch.zeros_like(self.weight))

class LinesConv(TwoParamConv):
    def get_weight(self):
        w = (1 - self.alpha) * self.weight + self.alpha * self.weight1
        return w

#################################
# Models for federated learning #
#################################
# for MNIST, EMNIST
class MNISTConvNet(nn.Module):
    def __init__(self, name, in_channels, hidden_channels, num_hiddens, num_classes):
        super(MNISTConvNet, self).__init__()
        self.name = name
        self.activation = nn.ReLU(True)

        self.conv1 = LinesConv(in_channels=in_channels, out_channels=hidden_channels, kernel_size=(5, 5), padding=0, stride=1, bias=True)
        self.conv2 = LinesConv(in_channels=hidden_channels, out_channels=hidden_channels * 2, kernel_size=(5, 5), padding=0, stride=1, bias=True)
        self.conv3 = LinesConv(in_channels=hidden_channels * 2, out_channels=hidden_channels, kernel_size=(2, 2), padding=0, stride=1, bias=True)
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), padding=0)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2), padding=0)

        self.fc1 = LinesConv(in_channels=hidden_channels, out_channels=num_hiddens, kernel_size=(1, 1), bias=True)
        self.fc2 = LinesConv(in_channels=num_hiddens, out_channels=num_classes, kernel_size=(1, 1), bias=True)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.maxpool1(x)

        x = self.activation(self.conv2(x))
        x = self.maxpool2(x)

        x = self.activation(self.conv3(x))
        x = self.maxpool3(x)
        
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze()
    
# for CIFAR10, CIFAR100
class CIFARConvNet(nn.Module):
    def __init__(self, name, in_channels, hidden_channels, num_hiddens, num_classes):
        super(CIFARConvNet, self).__init__()
        self.name = name
        self.activation = nn.ReLU(True)

        self.conv1 = LinesConv(in_channels=in_channels, out_channels=hidden_channels, kernel_size=(5, 5), padding=0, stride=1, bias=True)
        self.conv2 = LinesConv(in_channels=hidden_channels, out_channels=hidden_channels * 2, kernel_size=(5, 5), padding=0, stride=1, bias=True)
        self.conv3 = LinesConv(in_channels=hidden_channels * 2, out_channels=hidden_channels, kernel_size=(3, 3), padding=0, stride=1, bias=True)
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), padding=0)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2), padding=0)

        self.fc1 = LinesConv(in_channels=hidden_channels, out_channels=num_hiddens, kernel_size=(1, 1), bias=True)
        self.fc2 = LinesConv(in_channels=num_hiddens, out_channels=num_classes, kernel_size=(1, 1), bias=True)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.maxpool1(x)

        x = self.activation(self.conv2(x))
        x = self.maxpool2(x)

        x = self.activation(self.conv3(x))
        x = self.maxpool3(x)
        
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze()

# for TinyImageNet
class TINConvNet(nn.Module):
    def __init__(self, name, in_channels, hidden_channels, num_hiddens, num_classes):
        super(TINConvNet, self).__init__()
        self.name = name
        self.activation = nn.ReLU(True)

        self.conv1 = LinesConv(in_channels=in_channels, out_channels=hidden_channels, kernel_size=(5, 5), padding=0, stride=1, bias=True)
        self.conv2 = LinesConv(in_channels=hidden_channels, out_channels=hidden_channels * 2, kernel_size=(5, 5), padding=0, stride=1, bias=True)
        self.conv3 = LinesConv(in_channels=hidden_channels * 2, out_channels=hidden_channels, kernel_size=(2, 2), padding=0, stride=1, bias=True)
        
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3, 3), padding=0)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(3, 3), padding=0)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(3, 3), padding=0)

        self.fc1 = LinesConv(in_channels=hidden_channels, out_channels=num_hiddens, kernel_size=(1, 1), bias=True)
        self.fc2 = LinesConv(in_channels=num_hiddens, out_channels=num_classes, kernel_size=(1, 1), bias=True)

    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.maxpool1(x)

        x = self.activation(self.conv2(x))
        x = self.maxpool2(x)

        x = self.activation(self.conv3(x))
        x = self.maxpool3(x)
        
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze()