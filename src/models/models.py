import string
import torch.nn as nn
import torch.nn.functional as F


###################################################
# Models from FedAvg paper (McMahan et al., 2016) #
###################################################
class TwoNN(nn.Module):
    def __init__(self, builder, args, block=None):
        super(TwoNN, self).__init__()
        self.layers = nn.Sequential(
            builder.linear(in_features=784, out_features=200, bias=True),
            nn.ReLU(True),
            builder.linear(in_features=200, out_features=200, bias=True),
            nn.ReLU(True)
        )
        self.classifier = builder.linear(in_features=200, out_features=args.num_classes, bias=True)

    def forward(self, x):
        if x.ndim == 4:
            x = x.view(x.size(0), -1)
        x = self.layers(x)
        x = self.classifier(x)
        return x

class TwoCNN(nn.Module):
    def __init__(self, builder, args, block=None):
        super(TwoCNN, self).__init__()
        self.activation = nn.ReLU(True)
        self.layers = nn.Sequential(
            builder.conv(in_channels=args.in_channels, out_channels=32, kernel_size=5, padding=1, stride=1, bias=False),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, padding=1),
            builder.conv(in_channels=32, out_channels=64, kernel_size=5, padding=1, stride=1, bias=False),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, padding=1),
            nn.Flatten(),
            builder.linear(in_features=(32 * 2) * (7 * 7), out_features=200, bias=True),
            nn.ReLU(True)
        )
        self.classifier = builder.linear(in_features=200, out_features=args.num_classes, bias=True)
        
    def forward(self, x):
        x = self.layers(x)
        x = self.classifier(x)
        return x

class NextCharLM(nn.Module):
    def __init__(self, builder, args, block=None):
        super(NextCharLM, self).__init__()
        self.num_layers = 2
        self.encoder = builder.embedding(len(string.printable), 8)
        self.rnn = builder.lstm(
            input_size=8,
            hidden_size=256,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.classifier = builder.linear(256, len(string.printable))

    def forward(self, x):
        encoded = self.encoder(x)
        self.rnn.flatten_parameters()
        output, _ = self.rnn(encoded)
        output = self.classifier(output[:, -1, :])
        return output

    
############
# ResNet18 #
############
# https://github.com/apple/learning-subspaces/blob/9e4cdcf4cb92835f8e66d5ed13dc01efae548f67/models/resnet.py
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, builder, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()        
        self.module = nn.Sequential(
            builder.conv(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1),
            builder.bn(out_channels),
            nn.ReLU(inplace=True),
            builder.conv(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            builder.bn(out_channels)
        )
        self.downsample = downsample

    def forward(self, x):
        out = self.module(x)        
        if self.downsample is not None:
            out += self.downsample(x)
        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, builder, args, block):
        super(ResNet18, self).__init__()
        self.base_width = 64
        
        # input layer
        self.in_conv = nn.Sequential(
            builder.conv(in_channels=args.in_channels, out_channels=self.base_width, kernel_size=3, stride=2, padding=1),
            builder.bn(self.base_width),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # intermediate layers
        self.layer1 = self._make_layer(builder, block, 64, 2)
        self.layer2 = self._make_layer(builder, block, 128, 2)
        self.layer3 = self._make_layer(builder, block, 256, 2, stride=2)
        self.layer4 = self._make_layer(builder, block, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        
        # classifier
        self.classifier = builder.linear(in_features=512 * block.expansion, out_features=args.num_classes)

    def _make_layer(self, builder, block, planes, num_layers, stride=1):
        # define downsample operation
        downsample = None
        if stride != 1 or self.base_width != planes * block.expansion:
            dconv = builder.conv(in_channels=self.base_width, out_channels=planes * block.expansion, kernel_size=1, stride=stride)
            dbn = builder.bn(planes * block.expansion)
            if dbn is not None:
                downsample = nn.Sequential(dconv, dbn)
            else:
                downsample = dconv

        # construct layers
        layers = []
        layers.append(
            block(builder, self.base_width, planes, stride, downsample)
        )
        self.base_width = planes * block.expansion
        for i in range(1, num_layers):
            layers.append(
                block(builder, self.base_width, planes)
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
    

###############
# MobileNetv2 #
###############
# https://github.com/jmjeon94/MobileNet-Pytorch
class InvertedBlock(nn.Module):
    def __init__(self, builder, in_channels, out_channels, expansion, stride):
        super(InvertedBlock, self).__init__()
        assert stride in [1, 2]
        self.use_skip_connection = stride == 1 and in_channels == out_channels
        
        # layer construction
        layers = []
        if expansion != 1:
            layers.append(
                nn.Sequential(
                    builder.conv(in_channels=in_channels, out_channels=in_channels * expansion, kernel_size=1, stride=1),
                    builder.bn(in_channels * expansion),
                    nn.ReLU6(inplace=True)
                )
            )
        layers.extend(
            [
                nn.Sequential( # depthwise convolution
                    builder.conv(in_channels=in_channels * expansion, out_channels=in_channels * expansion, kernel_size=3, stride=stride, groups=in_channels * expansion, padding=1),
                    builder.bn(in_channels * expansion),
                    nn.ReLU6(inplace=True)
                ),
                nn.Sequential( # pointwise convolution
                    builder.conv(in_channels=in_channels * expansion, out_channels=out_channels, kernel_size=1, stride=1),
                    builder.bn(out_channels)
                )
            ]
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_skip_connection:
            return x + self.layers(x)
        return self.layers(x)

class MobileNetv2(nn.Module):
    def __init__(self, builder, args, block):
        super(MobileNetv2, self).__init__()
        self.builder = builder
        self.configs = [ # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]
        
        # input layer
        self.in_conv = nn.Sequential(
            builder.conv(in_channels=args.in_channels, out_channels=args.in_channels, kernel_size=3, padding=1),
            builder.conv(in_channels=args.in_channels, out_channels=32, kernel_size=3, stride=1 if args.is_small else 2, padding=1),
            builder.bn(32),
            nn.ReLU6(inplace=True)
        )
        
        # intermediate layers
        layers = []
        in_channels = 32
        for t, c, n, s in self.configs:
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(
                    InvertedBlock(
                        builder=self.builder,
                        in_channels=in_channels,
                        out_channels=c,
                        expansion=t,
                        stride=stride
                    )
                )
                in_channels = c
        self.layers = nn.Sequential(*layers)
        
        self.out_conv = nn.Sequential(
                builder.conv(in_channels=in_channels, out_channels=1280, kernel_size=1, stride=1),
                builder.bn(1280),
                nn.ReLU6(inplace=True)
            )
        self.flatten = nn.Flatten()
        
        # classifier
        self.classifier = nn.Sequential(
            nn.Dropout2d(0.2),
            builder.linear(in_features=5120, out_features=args.num_classes)
        )

    def forward(self, x):
        x = self.in_conv(x)
        x = self.layers(x)
        x = self.out_conv(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x