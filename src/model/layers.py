import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


# for compatibility
StandardLinear = nn.Linear
StandardConv = nn.Conv2d
StandardBN = nn.BatchNorm2d
StandardLSTM = nn.LSTM
StandardEmbedding = nn.Embedding

#################################
# Modules for connected subspace learning #
#################################
# Credits to: https://github.com/apple/learning-subspaces (Wortsman et al., 2021)
# convolution layer
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

# dense layer
class SubspaceLinear(nn.Linear):
    def forward(self, x):
        # call get_weight, which samples from the subspace, then use the corresponding weight.
        w = self.get_weight()
        x = F.linear(
            x,
            w,
            self.bias
        )
        return x

class TwoParamLinear(SubspaceLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight1 = nn.Parameter(torch.zeros_like(self.weight))

class LinesLinear(TwoParamLinear):
    def get_weight(self):
        w = (1 - self.alpha) * self.weight + self.alpha * self.weight1
        return w

# LSTM layer
class SubspaceLSTM(nn.LSTM):
    def forward(self, x):
        # call get_weight, which samples from the subspace, then use the corresponding weight.
        weight_dict = self.get_weight()
        mixed_lstm = nn.LSTM(
            input_size=self.input_size, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers, 
            batch_first=self.batch_first
        )
        for l in range(self.num_layers):
            setattr(mixed_lstm, f'weight_hh_l{l}', nn.Parameter(weight_dict[f'weight_hh_l{l}_mixed']))
            setattr(mixed_lstm, f'weight_ih_l{l}', nn.Parameter(weight_dict[f'weight_ih_l{l}_mixed']))
            if self.bias:
                setattr(mixed_lstm, f'bias_hh_l{l}', nn.Parameter(weight_dict[f'bias_hh_l{l}_mixed']))
                setattr(mixed_lstm, f'bias_ih_l{l}', nn.Parameter(weight_dict[f'bias_ih_l{l}_mixed']))
        return mixed_lstm(x)

class TwoParamLSTM(SubspaceLSTM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for l in range(self.num_layers):
            setattr(self, f'weight_hh_l{l}_1', nn.Parameter(torch.zeros_like(getattr(self, f'weight_hh_l{l}'))))
            setattr(self, f'weight_ih_l{l}_1', nn.Parameter(torch.zeros_like(getattr(self, f'weight_ih_l{l}'))))
            if self.bias:
                setattr(self, f'bias_hh_l{l}_1', nn.Parameter(torch.zeros_like(getattr(self, f'bias_hh_l{l}'))))
                setattr(self, f'bias_ih_l{l}_1', nn.Parameter(torch.zeros_like(getattr(self, f'bias_ih_l{l}'))))
                
class LinesLSTM(TwoParamLSTM):
    def get_weight(self):
        weight_dict = dict()
        for l in range(self.num_layers):
            weight_dict[f'weight_hh_l{l}_mixed'] = (1 - self.alpha) * getattr(self, f'weight_hh_l{l}') + self.alpha * getattr(self, f'weight_hh_l{l}_1') 
            weight_dict[f'weight_ih_l{l}_mixed'] = (1 - self.alpha) * getattr(self, f'weight_ih_l{l}') + self.alpha * getattr(self, f'weight_ih_l{l}_1') 
            if self.bias:
                weight_dict[f'bias_hh_l{l}_mixed'] = (1 - self.alpha) * getattr(self, f'bias_hh_l{l}') + self.alpha * getattr(self, f'bias_hh_l{l}_1') 
                weight_dict[f'bias_ih_l{l}_mixed'] = (1 - self.alpha) * getattr(self, f'bias_ih_l{l}') + self.alpha * getattr(self, f'bias_ih_l{l}_1')
        return weight_dict

# Embedding layer
class SubspaceEmbedding(nn.Embedding):
    def forward(self, x):
        w = self.get_weight()
        x = F.embedding(
            x,
            w,
        )
        return x

class TwoParamEmbedding(SubspaceEmbedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight1 = nn.Parameter(torch.zeros_like(self.weight))
                
class LinesEmbedding(TwoParamEmbedding):
    def get_weight(self):
        w = (1 - self.alpha) * self.weight + self.alpha * self.weight1
        return w

# BatchNorm layer
class SubspaceBN(nn.BatchNorm2d):
    def forward(self, input):
        # call get_weight, which samples from the subspace, then use the corresponding weight.
        w, b = self.get_weight()

        # The rest is code in the PyTorch source forward pass for batchnorm.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(
                        self.num_batches_tracked
                    )
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (
                self.running_var is None
            )
        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var
            if not self.training or self.track_running_stats
            else None,
            w,
            b,
            bn_training,
            exponential_average_factor,
            self.eps,
        )
    
class TwoParamBN(SubspaceBN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight1 = nn.Parameter(torch.Tensor(self.num_features))
        self.bias1 = nn.Parameter(torch.Tensor(self.num_features))
        torch.nn.init.ones_(self.weight1)
        torch.nn.init.zeros_(self.bias1)
        
class LinesBN(TwoParamBN):
    def get_weight(self):
        w = (1 - self.alpha) * self.weight + self.alpha * self.weight1
        b = (1 - self.alpha) * self.bias + self.alpha * self.bias1
        return w, b
