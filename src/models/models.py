import string
import torch.nn as nn
import torch.nn.functional as F


###################################################
# Models from FedAvg paper (McMahan et al., 2016) #
###################################################
class TwoNN(nn.Module):
    def __init__(self, builder, args, seed=None, block=None):
        super(TwoNN, self).__init__()
        self._seed = seed
        self.layers = nn.Sequential(
            builder.linear(in_features=784, out_features=200, bias=False, seed=seed),
            nn.ReLU(True),
            builder.linear(in_features=200, out_features=200, bias=False, seed=seed),
            nn.ReLU(True)
        )
        self.classifier = nn.Sequential(
            builder.linear(in_features=200, out_features=200, bias=False, seed=seed),
            nn.ReLU(),
            builder.linear(in_features=200, out_features=args.num_classes, bias=False, seed=seed)
        )

    def forward(self, x):
        if x.ndim == 4:
            x = x.view(x.size(0), -1)
        x = self.layers(x)
        x = self.classifier(x)
        return x

class TwoCNN(nn.Module):
    def __init__(self, builder, args, seed, block=None):
        super(TwoCNN, self).__init__()
        self._seed = seed
        self.activation = nn.ReLU(True)
        self.layers = nn.Sequential(
            builder.conv(in_channels=args.in_channels, out_channels=32, kernel_size=5, padding=1, stride=1, bias=False, seed=seed),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, padding=1),
            builder.conv(in_channels=32, out_channels=64, kernel_size=5, padding=1, stride=1, bias=False, seed=seed),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, padding=1),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            builder.linear(in_features=(32 * 2) * (7 * 7), out_features=200, bias=False, seed=seed),
            nn.ReLU(True),
            builder.linear(in_features=200, out_features=200, bias=False, seed=seed),
            nn.ReLU(True),
            builder.linear(in_features=200, out_features=args.num_classes, bias=False, seed=seed)
        )
        
    def forward(self, x):
        x = self.layers(x)
        x = self.classifier(x)
        return x

class NextCharLM(nn.Module):
    def __init__(self, builder, args, seed, block=None):
        super(NextCharLM, self).__init__()
        self._seed = seed
        self.num_layers = 2
        self.encoder = builder.embedding(len(string.printable), 8, seed=seed)
        self.rnn = builder.lstm(
            input_size=8,
            hidden_size=256,
            num_layers=self.num_layers,
            batch_first=True,
            bias=False,
            seed=seed
        )
        self.classifier = builder.linear(256, len(string.printable), bias=False, seed=seed)

    def forward(self, x):
        encoded = self.encoder(x)
        output, _ = self.rnn(encoded)
        output = self.classifier(output[:, -1, :])
        return output

    
###########
# ResNet9 #
###########
# https://github.com/apple/learning-subspaces/blob/9e4cdcf4cb92835f8e66d5ed13dc01efae548f67/models/resnet.py
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, builder, in_channels, out_channels, stride=1, downsample=None, seed=None):
        super(BasicBlock, self).__init__()        
        self.module = nn.Sequential(
            builder.conv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False, seed=seed),
            builder.bn(out_channels, seed=seed),
            nn.ReLU(inplace=True),
            builder.conv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False, seed=seed),
            builder.bn(out_channels, seed=seed)
        )
        self.downsample = downsample

    def forward(self, x):
        out = self.module(x)        
        if self.downsample is not None:
            out += self.downsample(x)
        out = F.relu(out)
        return out

class ResNet9(nn.Module):
    def __init__(self, builder, args, seed, block):
        super(ResNet9, self).__init__()
        self._seed = seed
        self.base_width = 64
        
        # input layer
        self.in_conv = nn.Sequential(
            builder.conv(in_channels=args.in_channels, out_channels=self.base_width, kernel_size=3, stride=2, padding=1, bias=False, seed=seed),
            builder.bn(self.base_width, seed=seed),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # intermediate layers
        self.layer1 = self._make_layer(builder, block, 64, 1)
        self.layer2 = self._make_layer(builder, block, 128, 1)
        self.layer3 = self._make_layer(builder, block, 256, 1, stride=2)
        self.layer4 = self._make_layer(builder, block, 512, 1, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        
        # classifier
        self.classifier = builder.linear(in_features=512 * block.expansion, out_features=args.num_classes, bias=False, seed=seed)

    def _make_layer(self, builder, block, planes, num_layers, stride=1):
        # define downsample operation
        downsample = None
        if stride != 1 or self.base_width != planes * block.expansion:
            dconv = builder.conv(in_channels=self.base_width, out_channels=planes * block.expansion, kernel_size=1, stride=stride, bias=False, seed=self._seed)
            dbn = builder.bn(planes * block.expansion, seed=self._seed)
            if dbn is not None:
                downsample = nn.Sequential(dconv, dbn)
            else:
                downsample = dconv

        # construct layers
        layers = []
        layers.append(
            block(builder, self.base_width, planes, stride, downsample, seed=self._seed)
        )
        self.base_width = planes * block.expansion
        for i in range(1, num_layers):
            layers.append(
                block(builder, self.base_width, planes, seed=self._seed)
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        # input layers
        x = self.in_conv(x)
        x = self.maxpool(x)
        
        # intermediate layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # classifier
        feats = self.avgpool(x)
        x = self.classifier(self.flatten(feats))
        return x
    

#############
# MobileNet #
#############
# https://github.com/jmjeon94/MobileNet-Pytorch
def efficient_conv(builder, seed, in_channels, out_channels, stride):
    return nn.Sequential(
        builder.conv(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=stride, padding=0, groups=in_channels, bias=False, seed=seed),
        builder.bn(in_channels, seed=seed),
        nn.ReLU(inplace=True),
        
        builder.conv(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False, seed=seed),
        builder.bn(out_channels, seed=seed),
        nn.ReLU(inplace=True)
    )

class MobileNet(nn.Module):
    def __init__(self, builder, args, seed, block):
        super(MobileNet, self).__init__()
        self._seed = seed
        self.builder = builder
        
        # input layer
        self.in_conv = nn.Sequential(
            builder.conv(in_channels=args.in_channels, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False, seed=seed),
            builder.bn(32, seed=seed),
            nn.ReLU(inplace=True)
        )
        
        # intermediate layers
        self.layers = nn.Sequential(
            efficient_conv(builder, seed, 32, 64, 1),
            efficient_conv(builder, seed, 64, 128, 2),
            efficient_conv(builder, seed, 128, 256, 2),
            efficient_conv(builder, seed, 256, 512, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

        # classifier
        self.classifier = builder.linear(in_features=512, out_features=args.num_classes, bias=False, seed=seed)

    def forward(self, x):
        x = self.in_conv(x)
        x = self.layers(x)
        x = self.classifier(x)
        return x
    
    
    
########
# VGG9 #
########
class VGG9(nn.Module):
    def __init__(self, builder, args, seed, block):
        super(VGG9, self).__init__()
        self._seed = seed
        self.layers = nn.Sequential(
            builder.conv(in_channels=args.in_channels, out_channels=16, kernel_size=3, padding=1, bias=False, seed=seed),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            builder.conv(in_channels=16, out_channels=32, kernel_size=3, padding=1, bias=False, seed=seed),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            builder.conv(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False, seed=seed),
            nn.ReLU(True),
            builder.conv(in_channels=64, out_channels=64, kernel_size=3, padding=1, bias=False, seed=seed),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            builder.conv(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False, seed=seed),
            nn.ReLU(True),
            builder.conv(in_channels=128, out_channels=64, kernel_size=3, padding=1, bias=False, seed=seed),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            builder.linear(in_features=64, out_features=64, bias=False, seed=seed),
            nn.ReLU(True),
            builder.linear(in_features=64, out_features=args.num_classes, bias=False, seed=seed)
        )

    def forward(self, x):
        x = self.layers(x)
        x = self.classifier(x)
        return x