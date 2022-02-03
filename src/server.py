import os
import gc
import copy
import torch
import logging
import numpy as np

from tqdm import tqdm
from collections import OrderedDict, defaultdict
from joblib import Parallel, delayed

from .client import Client
from .algorithm import *
from .utils import record_results

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
        self.optimizer = torch.optim.SGD
        self.criterion = torch.nn.CrossEntropyLoss
        
        # result container
        self.results = defaultdict(list)
        
    def create_clients(self, k, training_set, test_set):
        """Initialize each client instance.
        """
        client = Client(args=self.args, client_id=k, training_set=training_set, test_set=test_set, device=self.args.device)
        client.model = copy.deepcopy(self.model)
        client.optimizer = copy.deepcopy(self.optimizer)
        client.criterion = copy.deepcopy(self.criterion)
        client.initialize_model()
        return client
    
    def setup(self):
        """Set up all configuration for federated learning.
        """
        # valid only at the very first round
        assert self._round == 0
        
        # setup clients (assign dataset, pass model, optimizer and criterion)
        self.clients = Parallel(n_jobs=os.cpu_count() - 1, prefer='threads')(delayed(self.create_clients)(k, training_set, test_set) for k, (training_set, test_set) in tqdm(enumerate(self.client_datasets), desc='[INFO] ...enroll clients to the server!'))
        del self.client_datasets; gc.collect()
        
        # notice
        message = f'[INFO] ...successfully created all {str(self.num_clients)} clients!'
        print(message); logging.info(message); del message; gc.collect()

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
        if self.algorithm in ['fedavg', 'fedprox']: 
            self.clients[idx].model = copy.deepcopy(self.model)
            
        # transmit paramaeters attached to the penultimate layer
        elif self.algorithm in ['lg-fedavg']: 
            partial_model = {k: v for k, v in self.model.state_dict().items() if 'classifier' in k}
            federated_model_at_client = self.clients[idx].model.state_dict()
            federated_model_at_client.update(partial_model)
            self.clients[idx].model.load_state_dict(federated_model_at_client)
            
        # transmit paramaeters NOT attached to the penultimate layer
        elif self.algorithm in ['fedper', 'fedrep']:
            partial_model = {k: v for k, v in self.model.state_dict().items() if 'classifier' not in k}
            federated_model_at_client = self.clients[idx].model.state_dict()
            federated_model_at_client.update(partial_model)
            self.clients[idx].model.load_state_dict(federated_model_at_client)
            
        # transmit parameters of a federated model only
        elif self.algorithm in ['ditto', 'apfl', 'pfedme', 'l2gd']:
            pass
        
        # transmit parameters of a federated model only (alpha = 0)
        elif self.algorithm in ['superfed-mm', 'superfed-lm']:
            global_model = {k: v for k, v in self.model.state_dict().items() if '_1' not in k}
            federated_model_at_client = copy.deepcopy(self.clients[idx].model.state_dict())
            federated_model_at_client.update(global_model)
            self.clients[idx].model.load_state_dict(federated_model_at_client)
    
    def evaluate_clients(self, idx, is_finetune):
        """Call `client_evaluate` function of clients.
        """
        loss, acc1, acc5, ece, mce = self.clients[idx].client_evaluate(is_finetune)
        return loss, acc1, acc5, ece, mce
    
    def update_clients(self, idx):
        """Call `client_update` function of clients.
        """
        self.clients[idx].client_update(self._round)
    
    def aggregate_model(self, sampled_client_indices, coefficients):
        """Aggregate the updated and transmitted parameters from each selected client.
        """
        # empty model container
        aggregated_weights = OrderedDict()
        
        # aggregate all model parameters
        if self.algorithm in ['fedavg', 'fedprox']: # aggregate all parameters
            for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
                local_weights = self.clients[idx].model.state_dict()
                for key in self.model.state_dict().keys():
                    if it == 0:
                        averaged_weights[key] = coefficients[it] * local_weights[key]
                    else:
                        averaged_weights[key] += coefficients[it] * local_weights[key]
                        
        # aggregate paramaeters attached to the penultimate layer
        elif self.algorithm in ['lg-fedavg']: 
            for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
                local_weights = self.clients[idx].model.state_dict()
                for key in self.model.state_dict().keys():
                    if 'classifier' in key:
                        if it == 0:
                            averaged_weights[key] = coefficients[it] * local_weights[key]
                        else:
                            averaged_weights[key] += coefficients[it] * local_weights[key]
                    else:
                        averaged_weights[key] = local_weights[key]
        
        # aggregate paramaeters NOT attached to the penultimate layer
        elif self.algorithm in ['fedper', 'fedrep']:
            for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
                local_weights = self.clients[idx].model.state_dict()
                for key in self.model.state_dict().keys():
                    if 'classifier' not in key:
                        if it == 0:
                            averaged_weights[key] = coefficients[it] * local_weights[key]
                        else:
                            averaged_weights[key] += coefficients[it] * local_weights[key]
                    else:
                        averaged_weights[key] = local_weights[key]
                        
        # aggregate parameters of a federated model only  
        elif self.algorithm in ['ditto', 'apfl', 'pfedme', 'l2gd']:
            pass
        
        # aggregate parameters of a federated model only (alpha = 0)
        elif self.algorithm in ['superfed-mm', 'superfed-lm']:
            for it, idx in tqdm(enumerate(sampled_client_indices), leave=False):
                local_weights = self.clients[idx].model.state_dict()
                for key in self.model.state_dict().keys():
                    if '_1' not in key:
                        if it == 0:
                            averaged_weights[key] = coefficients[it] * local_weights[key]
                        else:
                            averaged_weights[key] += coefficients[it] * local_weights[key]
                    else:
                        averaged_weights[key] = local_weights[key]
        
        # replace the model in the server
        self.model.load_state_dict(averaged_weights)

    def train_federated_model(self):
        """Do federated training.
        """
        # 1) sample clients
        sampled_client_indices = self.sample_clients()
 
        ## notice
        message = f'[INFO] [Round: {str(self._round).zfill(4)}] ...selected clients!'
        print(message); logging.info(message); del message; gc.collect()
        
        # 2) broadcast a global model
        _ = Parallel(n_jobs=os.cpu_count() - 1, prefer='threads')(delayed(self.transmit_model)(idx) for idx in tqdm(sampled_client_indices, desc='[INFO] ...transmit global models to clients!'))
        
        ## notice
        message = f'[INFO] [Round: {str(self._round).zfill(4)}] ...successfully transmitted models to {str(len(sampled_client_indices))} selected clients!'
        print(message); logging.info(message); del message; gc.collect()
        
        # 3) update client models
        _ = Parallel(n_jobs=os.cpu_count() - 1, prefer='threads')(delayed(self.update_clients)(idx) for idx in tqdm(sampled_client_indices, desc='[INFO] ...update models of selected clients!'))

        ## notice
        message = f'[INFO] [Round: {str(self._round).zfill(4)}] ...{len(sampled_client_indices)} clients are selected and updated!'
        print(message); logging.info(message); del message; gc.collect()
        
        # 4) aggregate client models
        ## calculate averaging coefficient of weights
        mixing_coefficients = [len(self.clients[idx]) / selected_total_size for idx in sampled_client_indices]

        ## average each updated model parameters of the selected clients and update the global model
        self.aggregate_model(sampled_client_indices, mixing_coefficients)

        ## notice
        message = f'[INFO] [Round: {str(self._round).zfill(4)}] ...updated weights of {len(sampled_client_indices)} clients are successfully aggregated!'
        print(message); logging.info(message); del message; gc.collect()
        
        # 5) evaluate personalization performance of selected clients
        per_loss, per_acc1, per_acc5, per_ece, per_mce = Parallel(n_jobs=os.cpu_count() - 1, prefer='threads')(delayed(self.evaluate_clients)(idx, False) for idx in tqdm(sampled_client_indices, desc='[INFO] ...evaluate a global model in selected clients!'))
        
        ## record results
        per_loss, per_acc1, per_acc5, per_ece, per_mce = record_results(self.args, self.writer, 'per_selected', self._round, per_loss, per_acc1, per_acc5, per_ece, per_mce)
   
        ## notice
        message = f'[INFO] [Round: {str(self._round).zfill(4)}] ...finished personalization evaluation of selected clients!'
        print(message); logging.info(message); del message; gc.collect()
        return per_loss, per_acc1, per_acc5, per_ece, per_mce
    
    def evaluate_personalized_model(self):
        """Evaluate the personalization performance of given algorithm using all of the client-side holdout dataset.
        """
        # 1) evaluate baseline performance of all clients
        base_loss, base_acc1, base_acc5, base_ece, base_mce = Parallel(n_jobs=os.cpu_count() - 1, prefer='threads')(delayed(self.evaluate_clients)(idx, False) for idx in tqdm(range(self.num_clients), desc='[INFO] ...evaluate a baseline performance in all clients!'))
        
        ## record results
        base_loss, base_acc1, base_acc5, base_ece, base_mce = record_results(self.args, self.writer, 'base_all', self._round // self.args.eval_every, base_loss, base_acc1, base_acc5, base_ece, base_mce)

        ## notice
        message = f'[INFO] [Round: {str(self._round).zfill(4)}] ...finished baseline evaluation of all clients!'
        print(message); logging.info(message); del message; gc.collect()
        
        # 2) evaluate personalization performance of all clients
        per_loss, per_acc1, per_acc5, per_ece, per_mce = Parallel(n_jobs=os.cpu_count() - 1, prefer='threads')(delayed(self.evaluate_clients)(idx, True) for idx in tqdm(range(self.num_clients), desc='[INFO] ...evaluate a personalization performance in all clients!'))
        
        ## record results
        per_loss, per_acc1, per_acc5, per_ece, per_mce = record_results(self.args, self.writer, 'per_all', self._round // self.args.eval_every, per_loss, per_acc1, per_acc5, per_ece, per_mce)
        
        ## notice
        message = f'[INFO] [Round: {str(self._round).zfill(4)}] ...finished personalization evaluation of all clients!'
        print(message); logging.info(message); del message; gc.collect()
        
        # 3) calculate delta metrics 
        loss_delta = per_loss - base_loss
        acc1_delta = per_acc1 - base_acc1
        acc5_delta = per_acc5 - base_acc5
        ece_delta = per_ece - base_ece
        mce_delta = per_mce - base_mce

        ## notice
        message = f'[INFO] [Round: {str(self._round).zfill(4)}] ...finished personalization evaluation of all clients!'
        print(message); logging.info(message); del message; gc.collect()
        return loss_delta, acc1_delta, acc5_delta, ece_delta, mce_delta
    
    def evaluate_global_model(self):
        """Evaluate the global model at the server using the server-side holdout dataset.
        (Possible only if the algorithm supports an exchange of whole parameters)
        """
        try:
            assert self.server_testset is not None, '[ERROR] Server should have global testset!'
            assert self.algorithm in ['fedavg', 'fedprox', 'ditto', 'apfl', 'pfedme', 'l2gd', 'superfed-mm', 'superfed-lm'], '[ERROR] Algorithm should support an exchange of whole model parameters!'
        except:
            return None, None, None, None, None
        
        # algorithm-specific evaluation is needed
        if self.algorithm in ['fedavg', 'fedprox']:
            return basic_evaluate(self.args, self.model, self.criterion, self.server_testset)
        elif self.algorithm in ['ditto', 'apfl', 'pfedme', 'l2gd']:
            return global_evaluate(self.args, self.model, self.criterion, self.server_testset)
        elif self.algorithm in ['superfed-mm', 'superfed-lm']:
            return superfed_evaluate(self.args, self.model, self.criterion, self.server_testset)
        
    def fit(self):
        """Execute the whole process of the federated learning.
        """
        for r in range(self.num_rounds):
            self._round = r + 1
            
            # do federated training and get the performance on selected clients in current rounds
            fed_loss, fed_acc1, fed_acc5, fed_ece, fed_mce = self.train_federated_model()
            
            # evaluate personalization performance on all clients
            if (self._round % self.args.eval_every == 0) or (self._round == self.num_rounds):
                loss_delta, acc1_delta, acc5_delta, ece_delta, mce_delta = self.evaluate_personalized_model()
                # TODO: plot histogram of delta-metric
                
            # evaluate server-side model's performance using server-side holdout set if possible
            test_loss, test_acc1, test_acc5, test_ece, test_mce = self.evaluate_global_model()
            
            # record server-side performance if possible
            if test_loss is not None:
                _ = record_results(self.args, self.writer, 'server', self._round, test_loss, test_acc1, test_acc5, test_ece, test_mce)
            
