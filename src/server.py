import os
import gc
import copy
import torch
import logging
import numpy as np

from tqdm import tqdm
from collections import OrderedDict, defaultdict
from joblib import Parallel, delayed

from .utils import *
from .client import Client

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
        """
        {'global_loss': [], 'global_top1_acc': [], 'global_top5_acc': [],
        'base_loss_mean': [], 'base_loss_std': [],
        'per_loss_mean': [], 'per_loss_std': [],
        'base_top1_acc_mean': [], 'base_top1_acc_mean': [],
        'per_top1_acc_mean': [], 'per_top1_acc_std': [],
        'base_top5_acc_mean': [], 'base_top6_acc_mean': [],
        'per_top5_acc_mean': [], 'per_top6_acc_std': [],
        'base_ece_mean': [], 'base_ece_std': [],
        'per_ece_mean': [], 'per_ece_std': []}
        """
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
        elif self.algorithm in ['ditto', 'apfl', 'pfedme']:
            pass
        
        # transmit parameters of a federated model only (alpha = 0)
        elif self.algorithm in ['superfed-mm', 'superfed-lm']:
            global_model = {k: v for k, v in self.model.state_dict().items() if '_1' not in k}
            federated_model_at_client = copy.deepcopy(self.clients[idx].model.state_dict())
            federated_model_at_client.update(global_model)
            self.clients[idx].model.load_state_dict(federated_model_at_client)
    
    def evaluate_clients(self, idx, is_finetune):
        """Call `client_evaluate` function of clients."""
        loss, acc1, acc5, ece, mce, bri = self.clients[idx].client_evaluate(is_finetune)
        return loss, acc1, acc5, ece, mce, bri
    
    def update_clients(self, idx):
        """Call `client_update` function of clients."""
        self.clients[idx].client_update(self._round)
    
    def aggregate_model(self, sampled_client_indices, coefficients):
        """Aggregate the updated and transmitted parameters from each selected client."""
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
        elif self.algorithm in ['ditto', 'apfl', 'pfedme']:
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
        """Do federated training."""
        #####################
        # 1. sample clients #
        #####################
        sampled_client_indices = self.sample_clients()
        
        # notice
        message = f'[INFO] [Round: {str(self._round).zfill(4)}] ...selected clients!'
        print(message); logging.info(message); del message; gc.collect()
        
        
        
        ###############################
        # 2. broadcast a global model #
        ###############################
        _ = Parallel(n_jobs=os.cpu_count() - 1, prefer='threads')(delayed(self.transmit_model)(idx) for idx in tqdm(sampled_client_indices, desc='[INFO] ...transmit global models to clients!'))
        
        # notice
        message = f'[INFO] [Round: {str(self._round).zfill(4)}] ...successfully transmitted models to {str(len(sampled_client_indices))} selected clients!'
        print(message); logging.info(message); del message; gc.collect()
        
        
        
        ##################################################################
        # 3. evaluate global model's baseline performance at each client #
        ##################################################################
        if (self._round % self.args.eval_every == 0) or (self._round == self.num_rounds):
            base_loss, base_acc1, base_acc5, base_ece, base_mce, base_bri = Parallel(n_jobs=os.cpu_count() - 1, prefer='threads')(delayed(self.evaluate_clients)(idx, False) for idx in tqdm(range(len(self.num_clients)), desc='[INFO] ...evaluate a global model in all clients!'))
            
            # record metrics and loss
            if base_loss[0] is not None: 
                base_loss = torch.tensor(base_loss)
                self.results['base_eval_loss_mean'].append(base_loss.mean()); self.results['base_eval_loss_std'].append(base_loss.std())
                self.writer.add_scalars(
                    'Loss',
                    {f'[{self.args.exp_name}] {self.algorithm}_{self.args.dataset}_{self.model.__class__.__name__}_base_loss_mean': base_loss.mean()},
                    self._round // self.args.eval_every
                )
            if base_acc1[0] is not None: 
                base_acc1 = torch.tensor(base_acc1)
                self.results['base_eval_acc1_mean'].append(base_acc1.mean()); self.results['base_eval_acc1_std'].append(base_acc1.std())
                self.writer.add_scalars(
                    'Accuracy',
                    {f'[{self.args.exp_name}] {self.algorithm}_{self.args.dataset}_{self.model.__class__.__name__}_base_top1_acc_mean': base_acc1.mean()},
                    self._round // self.args.eval_every
                )
            if base_acc5[0] is not None: 
                base_acc5 = torch.tensor(base_acc5)
                self.results['base_eval_acc5_mean'].append(base_acc5.mean()); self.results['base_eval_acc5_std'].append(base_acc5.std())
                self.writer.add_scalars(
                    'Accuracy',
                    {f'[{self.args.exp_name}] {self.algorithm}_{self.args.dataset}_{self.model.__class__.__name__}_base_top5_acc_mean': base_acc5.mean()},
                    self._round // self.args.eval_every
                )
            if base_ece[0] is not None: 
                base_ece = torch.tensor(base_ece)
                self.results['base_eval_ece_mean'].append(base_ece.mean()); self.results['base_eval_ece_std'].append(base_ece.std())
                self.writer.add_scalars(
                    'Expected Calibration Error',
                    {f'[{self.args.exp_name}] {self.algorithm}_{self.args.dataset}_{self.model.__class__.__name__}_base_ece_mean': base_ece.mean()},
                    self._round // self.args.eval_every
                )
            if base_mce[0] is not None: 
                base_mce = torch.tensor(base_mce)
                self.results['base_eval_mce_mean'].append(base_mce.mean()); self.results['base_eval_mce_std'].append(base_mce.std())
                self.writer.add_scalars(
                    'Maximum Calibration Error',
                    {f'[{self.args.exp_name}] {self.algorithm}_{self.args.dataset}_{self.model.__class__.__name__}_base_mce_mean': base_mce.mean()},
                    self._round // self.args.eval_every
                )
            if base_bri[0] is not None: 
                base_bri = torch.tensor(base_bri)
                self.results['base_eval_bri_mean'].append(base_bri.mean()); self.results['base_eval_bri_std'].append(base_bri.std())
                self.writer.add_scalars(
                    'Brier Score',
                    {f'[{self.args.exp_name}] {self.algorithm}_{self.args.dataset}_{self.model.__class__.__name__}_base_bri_mean': base_bri.mean()},
                    self._round // self.args.eval_every
                )
            
            # notice
            message = f'[INFO] [Round: {str(self._round).zfill(4)}] ...finished baseline evaluation of all clients!'
            print(message); logging.info(message); del message; gc.collect()
        
        
        
        ###########################
        # 4. update client models #
        ###########################
        _ = Parallel(n_jobs=os.cpu_count() - 1, prefer='threads')(delayed(self.update_clients)(idx) for idx in tqdm(sampled_client_indices, desc='[INFO] ...update models of selected clients!'))
        
        # notice
        message = f'[INFO] [Round: {str(self._round).zfill(4)}] ...{len(sampled_client_indices)} clients are selected and updated!'
        print(message); logging.info(message); del message; gc.collect()
        
        
        
        ##############################
        # 5. aggregate client models #
        #############################
        # calculate averaging coefficient of weights
        mixing_coefficients = [len(self.clients[idx]) / selected_total_size for idx in sampled_client_indices]

        # average each updated model parameters of the selected clients and update the global model
        self.average_model(sampled_client_indices, mixing_coefficients)
        
        # notice
        message = f'[INFO] [Round: {str(self._round).zfill(4)}] ...updated weights of {len(sampled_client_indices)} clients are successfully averaged!'
        print(message); logging.info(message); del message; gc.collect()
        
        
        
        #########################################################################
        # 6. evaluate global model's personalization performance at each client #
        #########################################################################
        if (self._round % self.args.eval_every == 0) or (self._round == self.num_rounds):
            # record metrics and loss
            per_loss, per_acc1, per_acc5, per_ece, per_mce, per_bri = Parallel(n_jobs=os.cpu_count() - 1, prefer='threads')(delayed(self.evaluate_clients)(idx, True) for idx in tqdm(range(len(self.num_clients)), desc='[INFO] ...evaluate a global model in all clients!'))
            
            # record metrics and loss
            if per_loss[0] is not None:
                per_loss = torch.tensor(per_loss)
                self.results['per_eval_loss_mean'].append(per_loss.mean()); self.results['per_eval_loss_std'].append(per_loss.std())
                self.writer.add_scalars(
                    'Loss',
                    {f'[{self.args.exp_name}] {self.algorithm}_{self.args.dataset}_{self.model.__class__.__name__}_per_loss_mean': per_loss.mean()},
                    self._round // self.args.eval_every
                )
            if per_acc1[0] is not None: 
                per_acc = torch.tensor(per_acc)
                self.results['per_eval_acc1_mean'].append(per_acc1.mean()); self.results['per_eval_acc1_std'].append(per_acc1.std())
                self.writer.add_scalars(
                    'Accuracy',
                    {f'[{self.args.exp_name}] {self.algorithm}_{self.args.dataset}_{self.model.__class__.__name__}_per_top1_acc_mean': per_acc1.mean()},
                    self._round // self.args.eval_every
                )
            if per_acc5[0] is not None: 
                per_acc5 = torch.tensor(per_acc5)
                self.results['per_eval_acc5_mean'].append(per_acc5.mean()); self.results['per_eval_acc5_std'].append(per_acc5.std())
                self.writer.add_scalars(
                    'Accuracy',
                    {f'[{self.args.exp_name}] {self.algorithm}_{self.args.dataset}_{self.model.__class__.__name__}_per_top5_acc_mean': per_acc5.mean()},
                    self._round // self.args.eval_every
                )
            if per_ece[0] is not None: 
                per_ece = torch.tesnor(per_ece)
                self.results['per_eval_ece_mean'].append(per_ece.mean()); self.results['per_eval_ece_std'].append(per_ece.std())
                self.writer.add_scalars(
                    'Expected Calibration Error',
                    {f'[{self.args.exp_name}] {self.algorithm}_{self.args.dataset}_{self.model.__class__.__name__}_per_ece_mean': per_ece.mean()},
                    self._round // self.args.eval_every
                )
            if per_mce[0] is not None: 
                per_mce = torch.tensor(per_mce)
                self.results['per_eval_mce_mean'].append(per_mce.mean()); self.results['per_eval_mce_std'].append(per_mce.std())
                self.writer.add_scalars(
                    'Maximum Calibration Error',
                    {f'[{self.args.exp_name}] {self.algorithm}_{self.args.dataset}_{self.model.__class__.__name__}_per_mce_mean': per_mce.mean()},
                    self._round // self.args.eval_every
                )
            if per_bri[0] is not None: 
                per_bri = torch.tensor(per_bri)
                self.results['per_eval_bri_mean'].append(per_bri.mean()); self.results['per_eval_bri_std'].append(per_bri.std())
                self.writer.add_scalars(
                    'Brier Score',
                    {f'[{self.args.exp_name}] {self.algorithm}_{self.args.dataset}_{self.model.__class__.__name__}_per_bri_mean': per_bri.mean()},
                    self._round // self.args.eval_every
                )
            
            # notice
            message = f'[INFO] [Round: {str(self._round).zfill(4)}] ...finished personalization evaluation of all clients!'
            print(message); logging.info(message); del message; gc.collect()
            
            # calculate delta metric
            
            return 
            
    @torch.no_grad()
    def evaluate_global_model(self):
        """Evaluate the global model at the server using the global holdout dataset if possible."""
        try:
            assert self.server_testset is not None
        except:
            return None
        
        self.model.eval()
        self.model.to(self.args.device)
        
        torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
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

    def fit(self):
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

