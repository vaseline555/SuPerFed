import math
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
        self.embedding_layer = getattr(layers, args.embedding_type)
        self.lstm_layer = getattr(layers, args.lstm_type)

    def conv(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False, seed=None):
        conv = self.conv_layer(in_channels, out_channels, kernel_size=kernel_size, padding=padding, groups=groups, stride=stride, bias=bias)
        torch.manual_seed(seed[0]); torch.nn.init.xavier_normal_(conv.weight)
        if len(seed) == 2: conv.initialize(seed[-1])
        return conv

    def bn(self, num_features, seed=None):
        bn = self.bn_layer(num_features)
        torch.manual_seed(seed[0]); torch.nn.init.ones_(bn.weight); torch.nn.init.zeros_(bn.bias)
        if len(seed) == 2: bn.initialize(seed[-1])
        return bn
    
    def linear(self, in_features, out_features, bias=False, seed=None):
        fc = self.fc_layer(in_features, out_features, bias=bias)
        torch.manual_seed(seed[0]); torch.nn.init.xavier_normal_(fc.weight)
        if len(seed) == 2: fc.initialize(seed[-1])
        return fc
    
    def embedding(self, num_embeddings, embedding_dim, seed=None):
        emb = self.embedding_layer(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        torch.manual_seed(seed[0]); torch.nn.init.normal_(emb.weight)
        if len(seed) == 2: emb.initialize(seed[-1])
        return emb
    
    def lstm(self, input_size, hidden_size, num_layers, batch_first, bias=False, seed=None):
        lstm = self.lstm_layer(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first, bias=bias)
        for l in range(lstm.num_layers):
            torch.manual_seed(seed[0])
            torch.nn.init.uniform_(getattr(lstm, f'weight_hh_l{l}'), a=math.sqrt(1 / hidden_size) * -1, b=math.sqrt(1 / hidden_size))
            torch.nn.init.uniform_(getattr(lstm, f'weight_ih_l{l}'), a=math.sqrt(1 / hidden_size) * -1, b=math.sqrt(1 / hidden_size))
        if len(seed) == 2: lstm.initialize(seed[-1])
        return lstm