import os
import gc
import copy
import torch
import numpy as np

from tqdm import tqdm, trange
from collections import OrderedDict, defaultdict
from multiprocessing import pool

from .client import Client
from .algorithm import *
from .utils import record_results, plot_by_lambda



class Server(object):
    """Central server orchestrating the whole process of a federated learning.
    
    Terms:
        Global model: model reside in the server
        Federated model: local version of a global model (i.e., global model broadcasted to & updated at clients)
        Local model: model reside only in the client (may or may not be the same in structure of the global model)
        Personalized model: (* model interpolation method only *) combinated model by mixing federated and local model
    """
    def __init__(self, args, writer, model, builder, block, server_testset, client_datasets):
        # default attributes
        self.args = args
        self.writer = writer
        self.global_model = model
        self.builder = builder
        self.block = block
        
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
        self.optimizer = getattr(torch.optim, args.optimizer)
        self.criterion = getattr(torch.nn, args.criterion)
        
        # result container
        self.results = defaultdict(list)
        
    def create_clients(self, k, training_set, test_set):
        """Initialize each client instance.
        """
        client = Client(args=self.args, client_id=k, training_set=training_set, test_set=test_set, device=self.args.device)
        if self.args.algorithm in ['superfed-mm', 'superfed-lm']:
            seed = self.args.global_seed[:]
            seed.append(k)
        elif self.args.algorithm in ['scaffold']:
            seed = self.args.global_seed[:]
            seed.append(-1)
        elif self.args.algorithm in ['apfl', 'pfedme', 'ditto', 'superfed-mm', 'superfed-lm']:
            seed = self.args.global_seed[:]
            seed.append(seed[-1])
        else:
            seed = self.args.global_seed[:]
        client.model = copy.deepcopy(self.global_model)(builder=self.builder(self.args), args=self.args, block=self.block, seed=seed)
        client.optimizer = copy.deepcopy(self.optimizer)
        client.criterion = copy.deepcopy(self.criterion)
        return client
    
    def setup(self):
        """Set up all configuration for federated learning.
        """
        # valid only at the very first round
        assert self._round == 0
        
        # setup clients (assign dataset, pass model, optimizer and criterion)
        with pool.ThreadPool(processes=min(self.args.n_jobs, self.args.K)) as workhorse:
            self.clients = workhorse.starmap(self.create_clients, [(k, training_set, test_set) for k, (training_set, test_set) in tqdm(enumerate(self.client_datasets), desc='[INFO] ...enroll clients to the server!')])
        del self.client_datasets; gc.collect()
        
        # sanity check
        self.num_clients = self.args.K = len(self.clients)
        
        # initialize server model
        if self.algorithm in ['scaffold']:
            seed = self.args.global_seed[:]; seed.append(-1)
            self.global_model = self.global_model(builder=self.builder(self.args), args=self.args, block=self.block, seed=seed)
        elif self.algorithm in ['pfedme']:
            seed = self.args.global_seed[:]; seed.append(seed[-1])
            self.global_model = self.global_model(builder=self.builder(self.args), args=self.args, block=self.block, seed=seed)
        else: # only need a global model
            server_args = copy.deepcopy(self.args)
            server_args.fc_type = 'StandardLinear'
            server_args.conv_type = 'StandardConv'
            server_args.bn_type = 'StandardBN'
            server_args.embedding_type = 'StandardEmbedding'
            server_args.lstm_type = 'StandardLSTM'
            self.global_model = self.global_model(builder=self.builder(server_args), args=server_args, block=self.block, seed=[server_args.global_seed[0]])
        
        # notice
        print(f'[INFO] ...successfully created all {str(self.num_clients)} clients!'); gc.collect()

    def sample_clients(self):
        """Select some fraction of all clients.
        """
        # sample clients randommly
        num_sampled_clients = max(int(self.fraction * self.num_clients), 1)
        sampled_client_indices = sorted(np.random.choice(a=[i for i in range(self.num_clients)], size=num_sampled_clients, replace=False).tolist())
        return sampled_client_indices
    
    def transmit_model(self, idx):
        """Transmit global model to clients.
        """
        # transmit all model parameters
        if self.algorithm in ['fedavg', 'fedprox', 'scaffold', 'pfedme']: 
            self.clients[idx]._model = copy.deepcopy(self.global_model)

        # transmit paramaeters attached to the penultimate layer
        elif self.algorithm in ['lg-fedavg']: 
            partial_model = {k: v for k, v in self.global_model.state_dict().items() if 'classifier' in k}
            federated_model_at_client = self.clients[idx]._model.state_dict()
            federated_model_at_client.update(partial_model)
            self.clients[idx]._model.load_state_dict(federated_model_at_client)
            
        # transmit paramaeters NOT attached to the penultimate layer
        elif self.algorithm in ['fedper', 'fedrep']:
            partial_model = {k: v for k, v in self.global_model.state_dict().items() if 'classifier' not in k}
            federated_model_at_client = self.clients[idx]._model.state_dict()
            federated_model_at_client.update(partial_model)
            self.clients[idx]._model.load_state_dict(federated_model_at_client)
        
        # transmit parameters of a federated model only (lambda = 0)
        elif self.algorithm in ['ditto', 'apfl', 'superfed-mm', 'superfed-lm']:
            federated_model_at_client = self.clients[idx]._model.state_dict()
            federated_model_at_client.update(self.global_model.state_dict())
            self.clients[idx]._model.load_state_dict(federated_model_at_client)

    def evaluate_clients(self, idx):
        """Call `client_evaluate` function of clients.
        """
        loss, acc1, acc5, ece, mce = self.clients[idx].client_evaluate(self._round)
        return loss, acc1, acc5, ece, mce
    
    def update_clients(self, idx):
        """Call `client_update` function of clients.
        """
        self.clients[idx].client_update(self._round)
        return len(self.clients[idx])
    
    def aggregate_model(self, sampled_client_indices, coefficients):
        """Aggregate the updated and transmitted parameters from each selected client.
        """
        # empty model container
        aggregated_weights = OrderedDict()
        
        # aggregate all model parameters
        if self.algorithm in ['fedavg', 'fedprox']: # aggregate all parameters
            for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
                local_weights = self.clients[idx]._model.state_dict()
                for key in self.global_model.state_dict().keys():
                    if 'weight' not in key:
                        aggregated_weights[key] = self.global_model.state_dict()[key].clone()
                    else:
                        if it == 0:
                            aggregated_weights[key] = coefficients[it] * local_weights[key] * self.args.beta + self.global_model.state_dict()[key].clone() * (1 - self.args.beta)
                        else:
                            aggregated_weights[key] += coefficients[it] * local_weights[key] * self.args.beta + self.global_model.state_dict()[key].clone() * (1 - self.args.beta)

        # aggregate global model parameters and control variates
        elif self.algorithm in ['scaffold']:
            for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
                local_weights = self.clients[idx]._model.state_dict()
                for key in self.global_model.state_dict().keys():
                    if 'weight' not in key:
                        aggregated_weights[key] = self.global_model.state_dict()[key].clone()
                    elif 'local' not in key: # update global model
                        if it == 0:
                            aggregated_weights[key] = coefficients[it] * local_weights[key] * self.args.beta + self.global_model.state_dict()[key].clone() * (1 - self.args.beta)
                        else:
                            aggregated_weights[key] += coefficients[it] * local_weights[key] * self.args.beta + self.global_model.state_dict()[key].clone() * (1 - self.args.beta)
                    else: # update server's control variates
                        if it == 0:
                            aggregated_weights[key] = self.global_model.state_dict()[key] + self.args.C * local_weights[key]
                        else:
                            aggregated_weights[key] += self.args.C * local_weights[key]
                        local_weights[key] += self.clients[idx].control_variates.state_dict()[key]
                else:
                    self.clients[idx]._model.load_state_dict(local_weights)
                            
        # aggregate paramaeters of head
        elif self.algorithm in ['lg-fedavg']: 
            for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
                local_weights = self.clients[idx]._model.state_dict()
                for key in self.global_model.state_dict().keys():
                    if 'weight' not in key:
                        aggregated_weights[key] = self.global_model.state_dict()[key].clone()
                    elif 'classifier' in key:
                        if it == 0:
                            aggregated_weights[key] = coefficients[it] * local_weights[key] * self.args.beta + self.global_model.state_dict()[key].clone() * (1 - self.args.beta)
                        else:
                            aggregated_weights[key] += coefficients[it] * local_weights[key] * self.args.beta + self.global_model.state_dict()[key].clone() * (1 - self.args.beta)
                    else:
                        aggregated_weights[key] = self.global_model.state_dict()[key]
        
        # aggregate paramaeters of body
        elif self.algorithm in ['fedper', 'fedrep']:
            for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
                local_weights = self.clients[idx]._model.state_dict()
                for key in self.global_model.state_dict().keys():
                    if 'weight' not in key:
                        aggregated_weights[key] = self.global_model.state_dict()[key].clone()
                    elif 'classifier' not in key:
                        if it == 0:
                            aggregated_weights[key] = coefficients[it] * local_weights[key] * self.args.beta + self.global_model.state_dict()[key].clone() * (1 - self.args.beta)
                        else:
                            aggregated_weights[key] += coefficients[it] * local_weights[key] * self.args.beta + self.global_model.state_dict()[key].clone() * (1 - self.args.beta)
                    else:
                        aggregated_weights[key] = self.global_model.state_dict()[key]
        
        # aggregate parameters of a federated model only (lambda = 0)
        elif self.algorithm in ['pfedme']:
            for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
                local_weights = self.clients[idx]._model.state_dict()
                for key in self.global_model.state_dict().keys():
                    if 'weight' not in key:
                        aggregated_weights[key] = self.global_model.state_dict()[key].clone()
                    elif 'local' not in key:
                        if it == 0:
                            aggregated_weights[key] = coefficients[it] * local_weights[key] * self.args.beta + self.global_model.state_dict()[key].clone() * (1 - self.args.beta)
                        else:
                            aggregated_weights[key] += coefficients[it] * local_weights[key] * self.args.beta + self.global_model.state_dict()[key].clone() * (1 - self.args.beta)
                else:
                    for key in self.global_model.state_dict().keys():
                        if 'local' in key:
                            aggregated_weights[key] = aggregated_weights[key.replace('_local', '')].clone()
                        
        # aggregate parameters of a federated model only (lambda = 0)
        elif self.algorithm in ['ditto', 'apfl', 'pfedme', 'superfed-mm', 'superfed-lm']:
            for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
                local_weights = self.clients[idx]._model.state_dict()
                for key in self.global_model.state_dict().keys():
                    if 'weight' not in key:
                        aggregated_weights[key] = self.global_model.state_dict()[key].clone()
                    elif 'local' not in key:
                        if it == 0:
                            aggregated_weights[key] = coefficients[it] * local_weights[key] * self.args.beta + self.global_model.state_dict()[key].clone() * (1 - self.args.beta)
                        else:
                            aggregated_weights[key] += coefficients[it] * local_weights[key] * self.args.beta + self.global_model.state_dict()[key].clone() * (1 - self.args.beta)
                    else:
                        aggregated_weights[key] = self.global_model.state_dict()[key]
        
        # replace the model in the server
        self.global_model.load_state_dict(aggregated_weights)

    def train_federated_model(self):
        """Do federated training.
        """
        # 1) sample clients
        sampled_client_indices = self.sample_clients()
        print(f'[INFO] [Round: {str(self._round).zfill(4)}] ...selected clients!'); gc.collect()
        
        # 2) broadcast a global model
        with pool.ThreadPool(processes=min(self.args.n_jobs, int(self.args.C * self.args.K))) as workhorse:
            workhorse.map(self.transmit_model, tqdm(sampled_client_indices, desc=f'[INFO] [Round: {str(self._round).zfill(4)}] ...transmit global models to clients!'))
        print(f'[INFO] [Round: {str(self._round).zfill(4)}] ...successfully transmitted models to {str(len(sampled_client_indices))} selected clients!'); gc.collect()
        
        # 3) update client models
        with pool.ThreadPool(processes=self.args.n_jobs) as workhorse:
            selected_sizes = workhorse.starmap(self.update_clients, [(idx,) for idx in tqdm(sampled_client_indices, desc=f'[INFO] [Round: {str(self._round).zfill(4)}] ...update models of selected clients!')])
        print(f'[INFO] [Round: {str(self._round).zfill(4)}] ...{len(sampled_client_indices)} clients are selected and updated!'); gc.collect()

        # 4) aggregate client models
        ## calculate averaging coefficient of weights
        mixing_coefficients = [len(self.clients[idx]) / sum(selected_sizes) for idx in sampled_client_indices]
        
        ## average each updated model parameters of the selected clients and update the global model
        self.aggregate_model(sampled_client_indices, mixing_coefficients)
        print(f'[INFO] [Round: {str(self._round).zfill(4)}] ...updated weights of {len(sampled_client_indices)} clients are successfully aggregated!'); gc.collect() 
        
        # 5) evaluate personalization performance of selected clients
        if self._round % self.args.eval_every == 0:
            with pool.ThreadPool(processes=min(self.args.n_jobs, int(self.args.C * self.args.K))) as workhorse:
                results = workhorse.map(self.evaluate_clients, tqdm(sampled_client_indices, desc=f'[INFO] [Round: {str(self._round).zfill(4)}] ...evaluate a global model in selected clients!'))
        
            ## record results
            if (self._round > int(self.args.L * self.args.R)) and (self.algorithm in ['superfed-mm', 'superfed-lm']):
                results_all = torch.stack([torch.stack(tensor) for tensor in results]) # args.K x 5 x 21
                self.results, per_loss, per_acc1, per_acc5, per_ece, per_mce = record_results(self.args, self.writer, self.results, 'selected', self._round, *torch.index_select(results_all, dim=2, index=results_all[:, 1, :].argmax(-1)).max(-1)[0].T)
            else:
                self.results, per_loss, per_acc1, per_acc5, per_ece, per_mce = record_results(self.args, self.writer, self.results, 'selected', self._round, *torch.tensor(results).T)
            print(f'[INFO] [Round: {str(self._round).zfill(4)}] ...finished evaluation of selected clients!'); gc.collect()
    
    def evaluate_personalized_model(self):
        """Evaluate the personalization performance of given algorithm using all of the client-side holdout dataset.
        """
        # 1) broadcast current global model to all clients
        with pool.ThreadPool(processes=min(self.args.n_jobs, int(self.args.C * self.args.K))) as workhorse:
            workhorse.map(self.transmit_model, tqdm(range(self.num_clients), desc=f'[INFO] [Round: {str(self._round).zfill(4)}] ...transmit global models to ALL clients!'))

        # 2) evaluate baseline performance of all clients
        with pool.ThreadPool(processes=min(self.args.n_jobs, int(self.args.C * self.args.K))) as workhorse:
            results = workhorse.map(self.evaluate_clients, tqdm(range(self.num_clients), desc=f'[INFO] [Round: {str(self._round).zfill(4)}] ...evaluate a final performance of ALL clients!'))

        ## record results
        if (self._round > int(self.args.L * self.args.R)) and (self.algorithm in ['superfed-mm', 'superfed-lm']):
            results_all = torch.stack([torch.stack(tensor) for tensor in results]) # args.K x 5 x 11
            self.results, base_loss, base_acc1, base_acc5, base_ece, base_mce = record_results(self.args, self.writer, self.results, 'baseline_all', self._round, *torch.index_select(results_all, dim=2, index=results_all[:, 1, :].argmax(-1)).max(-1)[0].T)
            plot_by_lambda(self.args, self._round // self.args.eval_every, results_all)
        else: # args.K x 5
            self.results, base_loss, base_acc1, base_acc5, base_ece, base_mce = record_results(self.args, self.writer, self.results, 'baseline_all', self._round // self.args.eval_every, *torch.tensor(results).T)
        print(f'[INFO] [Round: {str(self._round).zfill(4)}] ...finished final evaluation of ALL clients!'); gc.collect()
            
    def evaluate_global_model(self):
        """Evaluate the global model at the server using the server-side holdout dataset.
        (possible only if the algorithm supports an exchange of whole parameters)
        """
        try:
            assert self.server_testset is not None, '[ERROR] Server should have global testset!'
            assert self.algorithm in ['fedavg', 'fedprox', 'scaffold', 'ditto', 'apfl', 'pfedme', 'superfed-mm', 'superfed-lm'], '[ERROR] Algorithm should support an exchange of whole model parameters!'
        except BaseException: # skip global model evaluation
            return None
        return basic_evaluate(None, self.args, self.global_model, self.criterion, self.server_testset)
        
    def fit(self):
        """Execute the whole process of the federated learning.
        """
        for r in range(self.num_rounds):
            self._round = r + 1
            
            # do federated training and get the performance on selected clients in current rounds
            self.train_federated_model()
            
            # evaluate personalization performance on all clients
            if self._round == self.num_rounds:
                self.evaluate_personalized_model()

            # evaluate server-side model's performance using server-side holdout set if possible
            results = self.evaluate_global_model()
            if results is not None: self.results = record_results(self.args, self.writer, self.results, 'server', self._round, *torch.tensor(results).T)[0]    
