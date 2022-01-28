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

from .models import *
from .utils import *
from .client import Client

logger = logging.getLogger(__name__)


class Server(object):
    """Class for implementing center server orchestrating the whole process of federated learning
    
    At first, center server distribute model skeleton to all participating clients with configurations.
    While proceeding federated learning rounds, the center server samples some fraction of clients,
    receives locally updated parameters, averages them as a global parameter (model), and apply them to global model.
    In the next round, newly selected clients will recevie the updated global model as its local model.  
    
    Attributes:
        clients: List containing Client instances participating a federated learning.
        __round: Int for indcating the current federated round.
        writer: SummaryWriter instance to track a metric and a loss of the global model.
        model: torch.nn instance for a global model.
        seed: Int for random seed.
        device: Training machine indicator (e.g. "cpu", "cuda").
        mp_flag: Boolean indicator of the usage of multiprocessing for "client_update" and "client_evaluate" methods.
        data_path: Path to read data.
        dataset_name: Name of the dataset.
        num_shards: Number of shards for simulating non-IID data split (valid only when 'iid = False").
        iid: Boolean Indicator of how to split dataset (IID or non-IID).
        init_config: kwargs for the initialization of the model.
        fraction: Ratio for the number of clients selected in each federated round.
        num_clients: Total number of participating clients.
        local_epochs: Epochs required for client model update.
        batch_size: Batch size for updating/evaluating a client/global model.
        criterion: torch.nn instance for calculating loss.
        optimizer: torch.optim instance for updating parameters.
        optim_config: Kwargs provided for optimizer.
    """
    def __init__(self, writer, args):
        self.clients = None
        self._round = 0
        self.writer = writer
        
        self.model = eval(args.model_name)(args.model_name, args.in_channels, args.hidden_channels, args.num_hiddens, args.num_classes)
        
        self.seed = args.global_seed
        self.device = args.device
        self.eta_max = args.lr
        
        self.data_path = args.data_path
        self.dataset_name = args.dataset
        self.num_shards = args.num_shards
        self.iid = args.iid
        
        self.gpu_ids = [0] if self.device=="cuda" else []
        self.init_config = {"init_type": args.init_type, "init_gain": args.init_gain, "seeds": list(map(int, args.init_seed)), "gpu_ids": self.gpu_ids}

        self.fraction = args.C
        self.num_clients = args.K
        self.num_rounds = args.R
        self.local_epochs = args.E
        self.batch_size = args.B
        self.per_frac = args.L
        
        self.mu = args.mu
        self.beta = args.beta
        self.per_lr = args.p_lr
        self.per_epoch = args.p_e
        self.optimizer = torch.optim.SGD
        self.optim_config = {"momentum": 0.9}
        self.criterion = torch.nn.CrossEntropyLoss
        
        self.results = {"global_loss": [], "global_acc": [], 'global_top_k_accs': [],
                        "init_loss_mean": [], "init_loss_std": [],
                        "per_loss_mean": [], "per_loss_std": [],
                        "init_acc_mean": [], "init_acc_std": [],
                        "per_acc_mean": [], "per_acc_std": [],
                        'init_ece_mean': [], 'init_ece_std': [],
                        'per_ece_mean': [], 'per_ece_std': []}
        
    def setup(self, **init_kwargs):
        """Set up all configuration for federated learning."""
        # valid only before the very first round
        assert self._round == 0

        # initialize weights of the model
        torch.manual_seed(self.seed)
        init_net(self.model, **self.init_config)
        
        message = f"[Round: {str(self._round).zfill(4)}] ...successfully initialized model (# parameters: {str(sum(p.numel() for p in self.model.parameters()))})!"
        print(message); logging.info(message)
        del message; gc.collect()

        # split local dataset for each client
        local_datasets, test_dataset = create_datasets(self.data_path, self.dataset_name, self.num_clients, self.num_shards, self.iid)
        
        # assign dataset to each client
        self.clients = self.create_clients(local_datasets)

        # prepare hold-out dataset for evaluation
        self.data = test_dataset
        self.dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        
        # configure detailed settings for client upate and 
        self.setup_clients(
            batch_size=self.batch_size, num_local_epochs=self.local_epochs,
            optimizer=copy.deepcopy(self.optimizer), optim_config=self.optim_config, criterion=copy.deepcopy(self.criterion),
            mu=self.mu, beta=self.beta
            )
        
        # send the model skeleton to all clients
        self.transmit_model()
        
    def create_clients(self, local_datasets):
        """Initialize each Client instance."""
        clients = []
        for k, dataset in tqdm(enumerate(local_datasets), leave=False):
            client = Client(client_id=k, local_data=dataset, device=self.device)
            clients.append(client)

        message = f"[Round: {str(self._round).zfill(4)}] ...successfully created all {str(self.num_clients)} clients!"
        print(message); logging.info(message)
        del message; gc.collect()
        return clients

    def setup_clients(self, **client_config):
        """Set up each client."""
        for k, client in tqdm(enumerate(self.clients), leave=False):
            client.setup(**client_config)
        
        message = f"[Round: {str(self._round).zfill(4)}] ...successfully finished setup of all {str(self.num_clients)} clients!"
        print(message); logging.info(message)
        del message; gc.collect()

    def transmit_model(self, sampled_client_indices=None):
        """Send the updated global model to selected/all clients."""
        if sampled_client_indices is None:
            # send the global model to all clients before the very first and after the last federated round
            assert (self._round == 0) or (self._round == self.num_rounds)

            for idx, client in tqdm(enumerate(self.clients), leave=False):
                client.model = copy.deepcopy(self.model)
                
            message = f"[Round: {str(self._round).zfill(4)}] ...successfully transmitted models to all {str(self.num_clients)} clients!"
            print(message); logging.info(message)
            del message; gc.collect()
        else:
            # send the global model to selected clients
            assert self._round != 0

            # only send global model (alpha = 0)
            partial = {k: v for k, v in self.model.state_dict().items() if 'weight1' not in k}
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
                {f"[{exp_name}] {self.dataset_name}_{self.model.name}_C_{self.fraction}, E_{self.local_epochs}, B_{self.batch_size}, IID_{self.iid}": test_loss},
                self._round
                )
            self.writer.add_scalars(
                'Global Accuracy', 
                {f"[{exp_name}] {self.dataset_name}_{self.model.name}_C_{self.fraction}, E_{self.local_epochs}, B_{self.batch_size}, IID_{self.iid}": test_accuracy},
                self._round
                )
            self.writer.add_scalars(
                'Global Top 5 Accuracy', 
                {f"[{exp_name}] {self.dataset_name}_{self.model.name}_C_{self.fraction}, E_{self.local_epochs}, B_{self.batch_size}, IID_{self.iid}": top_k_accs},
                self._round
                )
            
            message = f"[Round: {str(self._round).zfill(4)}] Evaluate global model's performance...!\
                \n\t[Server] ...finished evaluation!\
                \n\t=> Loss: {test_loss:.4f}\
                \n\t=> Accuracy: {100. * test_accuracy:.2f}%\n"            
            print(message); logging.info(message)
            del message; gc.collect()
        self.transmit_model()

