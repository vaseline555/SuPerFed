import gc
import logging
import copy
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from .utils import init_weights
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
        self.mu = args.mu # proximity regularization
        self.nu = args.nu # connectivity regularization
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
        if self.args.algorithm == 'SuPerFed':
            self._model = init_weights(self.model, self.args.init_type, self.args.init_gain, [self.args.global_seed, self.client_id])
        else:
             self._model = init_weights(self.model, self.args.init_type, self.args.init_gain, [self.args.global_seed])

    def client_update(self, current_round):
        training_dataloader = torch.utils.data.DataLoader(self.training_set, batch_size=self.batch_size, shuffle=True)
        
        if self.mu > 0:
            # fix global model for calculating a proximity term
            self.global_model = copy.deepcopy(self.model)
            self.global_model.to(self.device)
        
        for param in self.global_model.parameters():
            param.requires_grad = False

        # update local model
        self.model.train()
        self.model.to(self.device)
        
        parameters = list(self.model.named_parameters())
        parameters_to_opimizer = [v for n, v in parameters if v.requires_grad]            
        optimizer = self.optimizer(parameters_to_opimizer, lr=self.lr * self.lr_decay**current_round, momentum=0.9)

        flag = False
        if epoch is None:
            epoch = self.local_epoch
        else:
            flag = True
        
        for e in range(epoch):
            for data, labels in self.training_dataloader:
                data, labels = data.float().to(self.device), labels.long().to(self.device)

                # mixing models
                if not start_local_training:
                    alpha = 0.0
                else:
                    if flag:
                        alpha = 0.5
                    else:
                        alpha = np.random.uniform(0.0, 1.0)
                for m in self.model.modules():
                    if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d):
                        setattr(m, f"alpha", alpha)

                # inference
                outputs = self.model(data)
                loss = self.criterion()(outputs, labels)
                
                # subspace construction
                if start_local_training:
                    num, norm1, norm2, cossim = 0., 0., 0., 0.
                    for m in self.model.modules():
                        if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) and hasattr(m, 'weight1'):
                            num += (m.weight * m.weight1).sum()
                            norm1 += m.weight.pow(2).sum()
                            norm2 += m.weight1.pow(2).sum()
                    cossim = self.beta * (num.pow(2) / (norm1 * norm2))
                    loss += cossim

                # proximity regularization
                prox = 0.
                for (n, w), w_g in zip(self.model.named_parameters(), self.global_model.parameters()):
                    if "weight1" not in n:
                        prox += (w - w_g).norm(2)
                loss += self.mu * (1 - alpha) * prox

                # update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step() 
                
                if self.device == "cuda": torch.cuda.empty_cache() 
        self.model.to("cpu")
        self.model.eval()
 
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
