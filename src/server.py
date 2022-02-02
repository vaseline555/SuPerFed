import json
import copy
import gc
import os
import logging

import numpy as np
import torch
import torch.nn as nn

from multiprocessing import pool, cpu_count
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from collections import OrderedDict

from utils import *
from client import Client

logger = logging.getLogger(__name__)


class Server(object):
    """Central server orchestrating the whole process of a federated learning.
    """
    def __init__(self, args, writer, model, server_testset, client_datasets):
        # default attributes
        self.args = args
        self.writer = writer
        self.model = model
        
        # datasets
        self.server_testset = server_testset
        self.client_datasets = client_datasets
        
        # federated learning related attributes
        self.clients = None
        self._round = 0
        self.fraction = args.C
        self.num_clients = args.K
        self.num_rounds = args.R
        self.algorithm = args.algorithm
        
        # personalization realted attributes
        self.start_personalization = int(args.L * args.R)
        self.optimizer = torch.optim.SGD
        self.criterion = torch.nn.CrossEntropyLoss
        
        # result container
        self.results = {'global_loss': [], 'global_top1_acc': [], 'global_top5_acc': [],
                        'base_loss_mean': [], 'base_loss_std': [],
                        'per_loss_mean': [], 'per_loss_std': [],
                        'base_top1_acc_mean': [], 'base_top1_acc_mean': [],
                        'per_top1_acc_mean': [], 'per_top1_acc_std': [],
                        'base_top5_acc_mean': [], 'base_top6_acc_mean': [],
                        'per_top5_acc_mean': [], 'per_top6_acc_std': [],
                        'base_ece_mean': [], 'base_ece_std': [],
                        'per_ece_mean': [], 'per_ece_std': []}
    
     def create_clients(self, local_datasets):
        """Initialize each client instance.
        """
        clients = []
        for k, (training_set, test_set) in tqdm(enumerate(local_datasets), desc='[INFO] ...enroll clients to the server!', leave=False):
            client = Client(args=self.args, client_id=k, training_set=training_set, test_set=test_set, device=self.device)
            client.model = copy.deepcopy(self.model)
            client.initialize_model()
            client.optimizer = copy.deepcopy(self.optimizer)
            client.criterion = copy.deepcopy(self.criterion)
            clients.append(client)
        else:
            message = f'[Round: {str(self._round).zfill(4)}] ...successfully created all {str(self.num_clients)} clients!'
            print(message); logging.info(message)
            del message; gc.collect()
        return clients
    
    def setup(self):
        """Set up all configuration for federated learning."""
        # valid only at the very first round
        assert self._round == 0
        
        # assign dataset to each client
        self.clients = self.create_clients(local_datasets)
        del self.client_datasets; gc.collect()
        
        # send the model skeleton to all clients
        self.transmit_model()

    def transmit_model(self, sampled_client_indices=None):
        if sampled_client_indices is None:
            # send the global model to all clients before the very first and after the last federated round
            assert (self._round == 0) or (self._round == self.num_rounds)

            for idx, client in tqdm(enumerate(self.clients), desc='[INFO] ...transmit global models to clients!', leave=False):
                client.model = copy.deepcopy(self.model)
                
            message = f'[Round: {str(self._round).zfill(4)}] ...successfully transmitted models to all {str(self.num_clients)} clients!'
            print(message); logging.info(message)
            del message; gc.collect()
        else:
            # send the global model to selected clients
            assert self._round > 0

            # only send a global model (alpha = 0)
            partial = {k: v for k, v in self.model.state_dict().items() if '_1' not in k}
            for idx in tqdm(sampled_client_indices, leave=False):
                state = self.clients[idx].model.state_dict()
                state.update(partial)
                self.clients[idx].model.load_state_dict(state)
            """
            for idx in tqdm(sampled_client_indices, leave=False):
                self.clients[idx].model = copy.deepcopy(self.model)
            """
            message = f"[Round: {str(self._round).zfill(4)}] ...successfully transmitted models to {str(len(sampled_client_indices))} selected clients!"
            print(message); logging.info(message)
            del message; gc.collect()

    def sample_clients(self):
        """Select some fraction of all clients."""
        # sample clients randommly
        message = f"[Round: {str(self._round).zfill(4)}] Select clients...!"
        print(message); logging.info(message)
        del message; gc.collect()

        num_sampled_clients = max(int(self.fraction * self.num_clients), 1)
        sampled_client_indices = sorted(np.random.choice(a=[i for i in range(self.num_clients)], size=num_sampled_clients, replace=False).tolist())

        return sampled_client_indices
    
    def update_selected_clients(self, sampled_client_indices, epoch=None, lr=None):
        """Call "client_update" function of each selected client."""
        if lr is None:
            lr = self.eta_max * 0.995**self._round
            message = f"[Round: {str(self._round).zfill(4)}] Start updating selected {len(sampled_client_indices)} clients...!"
        else:
            message = f"[Round: {str(self._round).zfill(4)}] Start personalizing selected {len(sampled_client_indices)} clients...!"
        print(message); logging.info(message)
        del message; gc.collect()
        
        # update selected clients
        selected_total_size = 0
        for idx in tqdm(sampled_client_indices, leave=False):
            self.clients[idx].client_update(lr, epoch)
            selected_total_size += len(self.clients[idx])

        message = f"[Round: {str(self._round).zfill(4)}] ...{len(sampled_client_indices)} clients are selected and updated (with total sample size: {str(selected_total_size)})!"
        print(message); logging.info(message)
        del message; gc.collect()

        return selected_total_size
    
    def mp_update_selected_clients(self, sampled_client_indices):
        """Multiprocessing-applied version of "update_selected_clients" method."""
        # update selected clients
        message = f"[Round: {str(self._round).zfill(4)}] Start updating selected client {str(self.clients[sampled_client_indices].id).zfill(4)}...!"
        print(message, flush=True); logging.info(message)
        del message; gc.collect()
        
        if self._round >= int(self.per_frac * self.num_rounds):
            start_local_training = True
        else:
            start_local_training = False
            
        self.clients[sampled_client_indices].client_update(lr=self.eta_max * 0.995**self._round, epoch=None, start_local_training=start_local_training)
        client_size = len(self.clients[sampled_client_indices])

        message = f"[Round: {str(self._round).zfill(4)}] ...client {str(self.clients[sampled_client_indices].id).zfill(4)} is selected and updated (with total sample size: {str(client_size)})!"
        print(message, flush=True); logging.info(message)
        del message; gc.collect()

        return client_size
    
    def average_model(self, sampled_client_indices, coefficients):
        """Average the updated and transmitted parameters from each selected client."""
        message = f"[Round: {str(self._round).zfill(4)}] Aggregate updated weights of {len(sampled_client_indices)} clients...!"
        print(message); logging.info(message)
        del message; gc.collect()
        
        averaged_weights = OrderedDict()
        for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
            local_weights = self.clients[idx].model.state_dict()
            for key in self.model.state_dict().keys():
                if it == 0:
                    averaged_weights[key] = coefficients[it] * local_weights[key]
                else:
                    averaged_weights[key] += coefficients[it] * local_weights[key]
        self.model.load_state_dict(averaged_weights)
                        
        message = f"[Round: {str(self._round).zfill(4)}] ...updated weights of {len(sampled_client_indices)} clients are successfully averaged!"
        print(message); logging.info(message)
        del message; gc.collect()
    
    def evaluate_selected_models(self, sampled_client_indices):
        """Call "client_evaluate" function of each selected client."""
        message = f"[Round: {str(self._round).zfill(4)}] Evaluate selected {str(len(sampled_client_indices))} clients' models...!"
        print(message); logging.info(message)
        del message; gc.collect()
        
        local_losses, local_accs, local_eces = [], [], []
        for idx in sampled_client_indices:
            loss, acc, ece = self.clients[idx].client_evaluate()
            local_losses.append(loss)
            local_accs.append(acc)
            local_eces.append(ece)
            
        message = f"[Round: {str(self._round).zfill(4)}] ...finished evaluation of {str(len(sampled_client_indices))} selected clients!"
        print(message); logging.info(message)
        del message; gc.collect()
        
        return local_losses, local_accs, local_eces

    def train_federated_model(self):
        """Do federated training."""
        
        """
        1. Sample clients
        """
        # select pre-defined fraction of clients randomly
        sampled_client_indices = self.sample_clients()
        
        
        
        """
        2. Broadcast model
        """
        # send global model to the selected clients
        self.transmit_model(sampled_client_indices)
        
        
        
        """
        3. Evaluate clients with global model
        """
        # evaluate selected clients with local dataset (i.e., initial performance)
        #if self._round == self.num_rounds:
        if self._round % 100 == 0:
            losses, accs, eces = self.evaluate_selected_models([i for i in range(self.num_clients)])
            #losses, accs, eces = self.evaluate_selected_models(sampled_client_indices)
            visualize_metrics(self.writer, self._round, losses, "initial loss")
            visualize_metrics(self.writer, self._round, accs, "initial accuracy")
            visualize_metrics(self.writer, self._round, eces, "expected calibration errors")
            
            loss_mean, loss_std = torch.Tensor(losses).mean(0).numpy()[::-1], torch.Tensor(losses).std(0).numpy()[::-1]
            acc_mean, acc_std = torch.Tensor(accs).mean(0).numpy()[::-1], torch.Tensor(accs).std(0).numpy()[::-1]
            ece_mean, ece_std = torch.Tensor(eces).mean(0).numpy()[::-1], torch.Tensor(eces).std(0).numpy()[::-1]
            
            self.results['init_loss_mean'].append(loss_mean)
            self.results['init_loss_std'].append(loss_std)
            self.results['init_acc_mean'].append(acc_mean)
            self.results['init_acc_std'].append(acc_std)
            self.results['init_ece_mean'].append(ece_mean)
            self.results['init_ece_std'].append(ece_std)
        
        """
        4. Update clients' model
        """
        # updated selected clients with local dataset
        with pool.ThreadPool(processes=cpu_count() - 1) as workhorse:
            selected_total_size = workhorse.map(self.mp_update_selected_clients, sampled_client_indices)
        selected_total_size = sum(selected_total_size)

        """
        5. Average model
        """
        # calculate averaging coefficient of weights
        mixing_coefficients = [len(self.clients[idx]) / selected_total_size for idx in sampled_client_indices]

        # average each updated model parameters of the selected clients and update the global model
        self.average_model(sampled_client_indices, mixing_coefficients)
        
        """
        6. Evaluate clients with personalized model after personalized update (not used for this setting)
        """
        if self._round > self.num_rounds:
            # personalized update
            _ = self.update_selected_clients([i for i in range(self.num_clients)], epoch=self.per_epoch, lr=self.per_lr)

            # evaluate selected clients with local dataset (i.e., personalized performance)
            losses, accs, eces = self.evaluate_selected_models([i for i in range(self.num_clients)])
            visualize_metrics(self.writer, self._round, losses, "personalized loss")
            visualize_metrics(self.writer, self._round, accs, "personalized accuracy")
            visualize_metrics(self.writer, self._round, eces, "expected calibration errors")

            loss_mean, loss_std = torch.Tensor(losses).mean(0).numpy()[::-1], torch.Tensor(losses).std(0).numpy()[::-1]
            acc_mean, acc_std = torch.Tensor(accs).mean(0).numpy()[::-1], torch.Tensor(accs).std(0).numpy()[::-1]
            ece_mean, ece_std = torch.Tensor(eces).mean(0).numpy()[::-1], torch.Tensor(eces).std(0).numpy()[::-1]
            
            self.results['per_loss_mean'].append(loss_mean)
            self.results['per_loss_std'].append(loss_std)
            self.results['per_acc_mean'].append(acc_mean)
            self.results['per_acc_std'].append(acc_std)
            self.results['per_ece_mean'].append(ece_mean)
            self.results['per_ece_std'].append(ece_std)
        
    def evaluate_global_model(self):
        """Evaluate the global model using the global holdout dataset (self.data)."""
        self.model.eval()
        self.model.to(self.device)
        DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        best_acc, best_loss, best_topk = 0, 1000, 0
        for alpha in [0.0]:
            # set alpha for inference
            for m in self.model.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d):
                    setattr(m, f"alpha", alpha)
            
            accs = []
            test_loss, correct = 0, 0
            with torch.no_grad():
                for data, labels in self.dataloader:
                    data, labels = data.float().to(self.device), labels.long().to(self.device)
                    outputs = self.model(data)
                    test_loss += torch.nn.CrossEntropyLoss()(outputs, labels).item()

                    predicted = outputs.argmax(dim=1, keepdim=True)
                    correct += predicted.eq(labels.view_as(predicted)).sum().item()

                    if self.device == "cuda": torch.cuda.empty_cache()
                    accs.append(accuracy(outputs, labels, (5,)))
            test_loss = test_loss / len(self.dataloader)
            test_accuracy = correct / len(self.data)
            print(f"[INFO] test_loss: {test_loss:.4f}, test_acc: {test_accuracy:.4f}")
            
            if test_accuracy > best_acc:
                best_acc = test_accuracy
            if test_loss < best_loss:
                best_loss = test_loss
            if torch.stack(accs).mean(0) > best_topk:
                best_topk = torch.stack(accs).mean(0).item()
        else:   
            for m in self.model.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d):
                    setattr(m, f"alpha", 0.0)
                    
            self.model.to("cpu")
        return best_loss, best_acc, best_topk

    def fit(self, exp_name):
        """Execute the whole process of the federated learning."""
        for r in range(self.num_rounds):
            self._round = r + 1
            
            self.train_federated_model()
            test_loss, test_accuracy, top_k_accs = self.evaluate_global_model()
            
            self.results['global_loss'].append(test_loss)
            self.results['global_acc'].append(test_accuracy)
            self.results['global_top_k_accs'].append(top_k_accs)

            self.writer.add_scalars(
                'Global Loss',
                {f"[{exp_name}] {self.dataset_name}_{self.model.name}_C_{self.fraction}, E_{self.local_epochs}, B_{self.batch_size}": test_loss},
                self._round
                )
            self.writer.add_scalars(
                'Global Accuracy', 
                {f"[{exp_name}] {self.dataset_name}_{self.model.name}_C_{self.fraction}, E_{self.local_epochs}, B_{self.batch_size}": test_accuracy},
                self._round
                )
            self.writer.add_scalars(
                'Global Top 5 Accuracy', 
                {f"[{exp_name}] {self.dataset_name}_{self.model.name}_C_{self.fraction}, E_{self.local_epochs}, B_{self.batch_size}": top_k_accs},
                self._round
                )
            
            message = f"[Round: {str(self._round).zfill(4)}] Evaluate global model's performance...!\
                \n\t[Server] ...finished evaluation!\
                \n\t=> Loss: {test_loss:.4f}\
                \n\t=> Accuracy: {100. * test_accuracy:.2f}%\n"            
            print(message); logging.info(message)
            del message; gc.collect()
        self.transmit_model()

