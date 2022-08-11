import numpy as np
import math

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
        x = F.conv2d(input=x, weight=w, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        return x

class TwoParamConv(SubspaceConv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_local = nn.Parameter(torch.zeros_like(self.weight))
    
    def initialize(self, seed):
        if seed == -1: # SCAFFOLD
            torch.nn.init.zeros_(self.weight_local)
        else:
            torch.manual_seed(seed)
            torch.nn.init.xavier_normal_(self.weight_local)
        
class LinesConv(TwoParamConv):
    def get_weight(self):
        w = (1 - self.lam) * self.weight + self.lam * self.weight_local
        return w

    

# dense layer
class SubspaceLinear(nn.Linear):
    def forward(self, x):
        # call get_weight, which samples from the subspace, then use the corresponding weight.
        w = self.get_weight()
        x = F.linear(input=x, weight=w, bias=self.bias)
        return x

class TwoParamLinear(SubspaceLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_local = nn.Parameter(torch.zeros_like(self.weight))

    def initialize(self, seed):
        if seed == -1: # SCAFFOLD
            torch.nn.init.zeros_(self.weight_local)
        else:
            torch.manual_seed(seed)
            torch.nn.init.xavier_normal_(self.weight_local)
                                     
class LinesLinear(TwoParamLinear):
    def get_weight(self):
        w = (1 - self.lam) * self.weight + self.lam * self.weight_local
        return w

    
    
# LSTM layer
# https://discuss.pytorch.org/t/defining-weight-manually-for-lstm/102360/2
class SubspaceLSTM(nn.LSTM):
    def forward(self, x):
        w = self.get_weight()
        h = (
            torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(x.device), 
            torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(x.device)
        )
        with torch.no_grad():
            torch._cudnn_rnn_flatten_weight(
                weight_arr=w, 
                weight_stride0=(4 if self.bias else 2),
                input_size=self.input_size,
                mode=torch.backends.cudnn.rnn.get_cudnn_mode('LSTM'),
                hidden_size=self.hidden_size,
                proj_size=0,
                num_layers=self.num_layers,
                batch_first=True,
                bidirectional=False
            )
        result = torch._VF.lstm(x, h, w, self.bias, self.num_layers, 0.0, self.training, self.bidirectional, self.batch_first) 
        return result[0], result[1:]
    
class TwoParamLSTM(SubspaceLSTM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for l in range(self.num_layers):
            setattr(self, f'weight_hh_l{l}_local', nn.Parameter(torch.zeros_like(getattr(self, f'weight_hh_l{l}'))))
            setattr(self, f'weight_ih_l{l}_local', nn.Parameter(torch.zeros_like(getattr(self, f'weight_ih_l{l}'))))
            if self.bias:
                setattr(self, f'bias_hh_l{l}_local', nn.Parameter(torch.zeros_like(getattr(self, f'bias_hh_l{l}'))))
                setattr(self, f'bias_ih_l{l}_local', nn.Parameter(torch.zeros_like(getattr(self, f'bias_ih_l{l}'))))
        
    def initialize(self, seed):
        if seed == -1: # SCAFFOLD
            for l in range(self.num_layers):
                torch.nn.init.zeros_(getattr(self, f'weight_hh_l{l}_local'))
                torch.nn.init.zeros_(getattr(self, f'weight_ih_l{l}_local'))
        else:
            for l in range(self.num_layers):
                torch.manual_seed(seed)
                torch.nn.init.uniform_(getattr(self, f'weight_hh_l{l}_local'), a=math.sqrt(1 / self.hidden_size) * -1, b=math.sqrt(1 / self.hidden_size))
                torch.nn.init.uniform_(getattr(self, f'weight_ih_l{l}_local'), a=math.sqrt(1 / self.hidden_size) * -1, b=math.sqrt(1 / self.hidden_size))
            
class LinesLSTM(TwoParamLSTM):
    def get_weight(self):
        weight_list = []
        for l in range(self.num_layers):
            weight_list.append((1 - self.lam) * getattr(self, f'weight_ih_l{l}') + self.lam * getattr(self, f'weight_ih_l{l}_local'))
            weight_list.append((1 - self.lam) * getattr(self, f'weight_hh_l{l}') + self.lam * getattr(self, f'weight_hh_l{l}_local'))
        return weight_list

    
    
# Embedding layer
class SubspaceEmbedding(nn.Embedding):
    def forward(self, x):
        w = self.get_weight()
        x = F.embedding(input=x, weight=w)
        return x

class TwoParamEmbedding(SubspaceEmbedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_local = nn.Parameter(torch.zeros_like(self.weight))
    
    def initialize(self, seed):
        if seed == -1: # SCAFFOLD
            torch.nn.init.zeros_(self.weight_local)
        else:
            torch.manual_seed(seed)
            torch.nn.init.normal_(self.weight_local)
        
class LinesEmbedding(TwoParamEmbedding):
    def get_weight(self):
        w = (1 - self.lam) * self.weight + self.lam * self.weight_local
        return w

    
    
# BatchNorm layer
class SubspaceBN(nn.BatchNorm2d):
    def forward(self, x):
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
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
                    
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)
        return F.batch_norm(
            x,
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
        self.weight_local = nn.Parameter(torch.Tensor(self.num_features))
        self.bias_local = nn.Parameter(torch.Tensor(self.num_features))
        
    def initialize(self, seed):
        torch.nn.init.ones_(self.weight_local)
        torch.nn.init.zeros_(self.bias_local)
        
class LinesBN(TwoParamBN):
    def get_weight(self):
        w = (1 - self.lam) * self.weight + self.lam * self.weight_local
        b = (1 - self.lam) * self.bias + self.lam * self.bias_local
        return w, b
