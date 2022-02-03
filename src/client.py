import gc
import logging
import copy
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from .utils import init_weights
from .algorithm import *

logger = logging.getLogger(__name__)



class Client(object):
    """Class for client object having its own (private) data and resources to train a model.
    """
    def __init__(self, args, client_id, training_set, test_set, device):
        # default attributes
        self.args = args
        self.client_id = client_id
        self.device = device
        
        # dataset
        self.training_set = training_set
        self.test_set = test_set
        
        # model related attributes
        self._model = None
        self._optimizer = None
        self._criterion = None
        
        # training related attributes
        self.local_epoch = args.E
        self.batch_size = args.B
        self.lr = args.lr
        self.lr_decay = args.lr_decay
        
    def __len__(self):
        return len(self.training_set)
    
    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model
        
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
    
    @property
    def criterion(self):
        return self._criterion

    @criterion.setter
    def criterion(self, criterion):
        self._criterion = criterion
    
    def initialize_model(self):
        # initialize model
        if self.args.algorithm in ['superfed-mm', 'superfed-lm']:
            self._model = init_weights(self.model, self.args.init_type, self.args.init_gain, [self.args.global_seed, self.client_id])
        else:
             self._model = init_weights(self.model, self.args.init_type, self.args.init_gain, [self.args.global_seed])

    def client_update(self, current_round):
        
 
    def client_evaluate(self, is_finetune):
        if is_finetune:
            for epoch in range(1):
                self.client_update()
        self.test_dataloader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, drop_last=True)
        losses, accs, eces = [], [], []
        
        self.model.eval()
        self.model.to(self.device)
        
        for alpha in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]:
            for m in self.model.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d):
                    setattr(m, f"alpha", alpha)

            test_loss, correct, ece_loss = 0, 0, 0
            with torch.no_grad():
                for data, labels in self.test_dataloader:
                    data, labels = data.float().to(self.device), labels.long().to(self.device)
                    outputs = self.model(data)
                    test_loss += self.criterion()(outputs, labels).item()
                    ece_loss += ECELoss()(outputs, labels).item()
                    
                    predicted = outputs.argmax(dim=1, keepdim=True)
                    correct += predicted.eq(labels.view_as(predicted)).sum().item()

                    if self.device == "cuda": torch.cuda.empty_cache()
                    gc.collect()
            test_loss = test_loss / len(self.test_dataloader)
            test_accuracy = correct / len(self.test_data)
            ece_loss = ece_loss / len(self.test_dataloader)
            
            message = f"\t[Client {str(self.id).zfill(4)}] ...finished evaluation!\
            \n\t=> Test loss: {test_loss:.4f}\
            \n\t=> Test accuracy: {100. * test_accuracy:.2f}%\
            \n\t=> Test ECE loss: {ece_loss:.4f}\n"
            print(message, flush=True);
            logging.info(message)
            del message; gc.collect()
        
            losses.append(test_loss)
            accs.append(test_accuracy)
            eces.append(ece_loss)
            
        self.model.to("cpu")
        return losses, accs, eces
