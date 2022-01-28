import gc
import pickle
import logging
import copy
import numpy as np
import pdb

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from .utils import *
logger = logging.getLogger(__name__)


class Client(object):
    """Class for client object having its own (private) data and resources to train a model.

    Participating client has its own dataset which are usually non-IID compared to other clients.
    Each client only communicates with the center server with its trained parameters or globally aggregated parameters.

    Attributes:
        id: Integer indicating client's id.
        data: torch.utils.data.Dataset instance containing local data.
        device: Training machine indicator (e.g. "cpu", "cuda").
        __model: torch.nn instance as a local model.
    """
    def __init__(self, client_id, local_data, device):
        """Client object is initiated by the center server."""
        self.id = client_id
        self.data = local_data
        self.device = device
        self.__model = None
        self.global_model = None
        
    @property
    def model(self):
        """Local model getter for parameter aggregation."""
        return self.__model

    @model.setter
    def model(self, model):
        """Local model setter for passing globally aggregated model parameters."""
        self.__model = model

    def __len__(self):
        """Return a total size of the client's local data."""
        return len(self.data)

    def setup(self, **client_config):
        """Set up common configuration of each client; called by center server."""
        self.local_epoch = client_config["num_local_epochs"]
        self.optim_config = client_config["optim_config"]
        self.batch_size = client_config["batch_size"]
        self.criterion = client_config["criterion"]
        self.optimizer = client_config["optimizer"]
        self.mu = client_config["mu"]
        self.beta = client_config["beta"]
        
        training_length = int(len(self.data) * 0.8)
        test_length = len(self.data) - training_length
        
        self.training_data, self.test_data = torch.utils.data.random_split(self.data, [training_length, test_length])
        self.training_dataloader = DataLoader(self.training_data, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.test_dataloader = DataLoader(self.test_data, batch_size=self.batch_size, shuffle=False, drop_last=True)
    
    def client_update(self, lr, epoch, start_local_training=False):
        """Update local model using local dataset."""
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
        optimizer = self.optimizer(parameters_to_opimizer, lr=lr, **self.optim_config)

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
 
    def client_evaluate(self):
        """Evaluate local model using local dataset (same as training set for convenience)."""
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
