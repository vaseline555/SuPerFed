import torch
import torch.nn as nn
import torch.nn.functional as F

from . import layers

class Builder(object):
    """Module for building a layer.
    """
    def __init__(self, args):
        self.fc_layer = getattr(layers, args.fc_type)
        self.conv_layer = getattr(layers, args.conv_type)
        self.bn_layer = getattr(layers, args.bn_type)
        self.embedding_lyaer = getattr(layers, args.embedding_type)
        self.lstm_layer = getattr(layers, args.lstm_type)

    def conv(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        return self.conv_layer(in_channels, out_channels, kernel_size=kernel_size, padding=padding, groups=groups, stride=stride, bias=bias)

    def bn(self, num_features):
        return self.bn_layer(num_features)
    
    def linear(self, in_features, out_features, bias=False):
        return self.fc_layer(in_features, out_features, bias=bias)
    
    def embedding(self, num_embeddings, embedding_dim):
        return self.embedding_layer(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
    
    def lstm(self, input_size, hidden_size, num_layers, batch_first):
        return self.lstm_laeyr(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first)