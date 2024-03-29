import gc
import copy
import numpy as np

import torch
import torch.nn as nn

from .algorithm import *



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
        self.lr = args.lr
        self.lr_decay = args.lr_decay
        self.algorithm = args.algorithm
        
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

    def client_update(self, current_round):
        # adjust learning rate
        current_lr = self.lr * self.lr_decay**current_round
        
        # call update function corresponded to each algorithm
        if self.algorithm in ['fedavg', 'lg-fedavg', 'fedper']:
            basic_update(identifier=self.client_id, args=self.args, model=self._model, criterion=self.criterion, dataset=self.training_set, optimizer=self.optimizer, lr=current_lr)
        elif self.algorithm in ['scaffold']:
            self.control_variates = copy.deepcopy(self._model)
            scaffold_update(identifier=self.client_id, args=self.args, model=self._model, criterion=self.criterion, dataset=self.training_set, optimizer=self.optimizer, lr=current_lr)
        elif self.algorithm in ['fedrep']:
            fedrep_update(identifier=self.client_id, args=self.args, model=self._model, criterion=self.criterion, dataset=self.training_set, optimizer=self.optimizer, lr=current_lr)
        elif self.algorithm in ['fedprox']:
            fedprox_update(identifier=self.client_id, args=self.args, model=self._model, criterion=self.criterion, dataset=self.training_set, optimizer=self.optimizer, lr=current_lr)
        elif self.algorithm in ['apfl']:
            apfl_update(identifier=self.client_id, args=self.args, model=self._model, criterion=self.criterion, dataset=self.training_set, optimizer=self.optimizer, lr=current_lr)
        elif self.algorithm in ['ditto']:
            ditto_update(identifier=self.client_id, args=self.args, model=self._model, criterion=self.criterion, dataset=self.training_set, optimizer=self.optimizer, lr=current_lr)
        elif self.algorithm in ['pfedme']:
            pfedme_update(identifier=self.client_id, args=self.args, model=self._model, criterion=self.criterion, dataset=self.training_set, optimizer=self.optimizer, lr=current_lr)
        elif self.algorithm in ['superfed-mm', 'superfed-lm']:
            superfed_update(identifier=self.client_id, args=self.args, model=self._model, criterion=self.criterion, dataset=self.training_set, optimizer=self.optimizer, lr=current_lr, start_mix=True if current_round > int(self.args.L * self.args.R) else False)
        else:
            raise NotImplementedError(f'[ERROR] {self.algorithm} is NOT supported!')
 
    def client_evaluate(self, current_round):
        if self.algorithm in ['fedavg', 'fedprox', 'scaffold', 'lg-fedavg', 'fedper', 'fedrep', 'apfl', 'ditto', 'pfedme']:
            return basic_evaluate(identifier=self.client_id, args=self.args, model=self._model, criterion=self.criterion, dataset=self.test_set)
        elif self.algorithm in ['superfed-mm', 'superfed-lm']:
            if current_round > int(self.args.L * self.args.R):
                return superfed_evaluate(identifier=self.client_id, args=self.args, model=self._model, criterion=self.criterion, dataset=self.test_set, current_round=current_round)
            else:
                return basic_evaluate(identifier=self.client_id, args=self.args, model=self._model, criterion=self.criterion, dataset=self.test_set)