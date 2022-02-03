import os
import logging
import torch
import torchvision
import numpy as np

from tqdm import tqdm
from joblib import Parallel, delayed
from .dataset import MNISTDataset, CIFARDataset, NoisyMNISTDataset, NoisyCIFARDataset, TinyImageNetDataset, FEMNISTDataset, ShakespeareDataset

logger = logging.getLogger(__name__)



#######################
# TensorBaord setting #
#######################
def launch_tensor_board(log_path, port, host):
    """Function for initiating TensorBoard.
    
    Args:
        log_path: Path where the log is stored.
        port: Port number used for launching TensorBoard.
        host: Address used for launching TensorBoard.
    """
    os.system(f"tensorboard --logdir={log_path} --port={port} --host={host}")
    return True



#######################
# Datset manipulation #
#######################
# split data suited to a federated learning
def split_data(args, raw_train):
    """Split data indices using labels.
    
    Args:
        args (argparser): arguments
        raw_train (dataset): raw dataset object to parse 
        
    Returns:
        split_map (dict): dictionary with key is a client index (~args.K) and a corresponding value is a list of indice array
    """
    # IID split (i.e., statistical homogeneity)
    if args.split_type == 'iid':
        # randomly shuffle label indices
        shuffled_indices = np.random.permutation(len(raw_train))
        
        # split shuffled indices by the number of clients
        split_indices = np.array_split(shuffled_indices, args.K)
        
        # construct a hashmap
        split_map = {i: split_indices[i] for i in range(args.K)}
        return split_map
    # Non-IID split proposed in McMahan et al., 2016 (i.e., each client has samples from at least two different classes)
    elif args.split_type == 'pathological':
        assert args.dataset in ['MNIST', 'CIFAR10'], '[ERROR] `pathological non-IID setting` is supported only for `MNIST` or `CIFAR10` dataset!'
        assert len(raw_train) / args.shard_size / args.K == 2, '[ERROR] each client should have samples from class at least 2 different classes!'
        
        # sort data by labels
        sorted_indices = np.argsort(np.array(raw_train.targets))
        shard_indices = np.array_split(sorted_indices, len(raw_train) // args.shard_size)

        # sort the list to conveniently assign samples to each clients from at least two~ classes
        split_indices = [[] for _ in range(args.K)]
        
        # retrieve each shard in order to each client
        for idx, shard in enumerate(shard_indices):
            split_indices[idx % args.K].extend(shard)
        
        # construct a hashmap
        split_map = {i: split_indices[i] for i in range(args.K)}
        return split_map
    # Non-IID split proposed in Hsu et al., 2019 (i.e., using Dirichlet distribution to simulate non-IID split)
    # https://github.com/QinbinLi/FedKT/blob/0bb9a89ea266c057990a4a326b586ed3d2fb2df8/experiments.py
    elif args.split_type == 'dirichlet':
        # container
        client_indices_list = [[] for _ in range(args.K)]

        # iterate through all classes
        for c in range(args.num_classes):
            # get corresponding class indices
            target_class_indices = np.where(np.array(raw_train.targets) == c)[0]

            # shuffle class indices
            np.random.seed(args.global_seed); np.random.shuffle(target_class_indices)

            # get label retrieval probability per each client based on a Dirichlet distribution
            proportions = np.random.dirichlet(np.repeat(args.alpha, args.K))
            proportions = np.array([p * (len(idx) < len(raw_train) / args.K) for p, idx in zip(proportions, client_indices_list)])

            # normalize
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(target_class_indices)).astype(int)[:-1]

            # split class indices by proportions
            idx_split = np.array_split(target_class_indices, proportions)
            client_indices_list = [j + idx.tolist() for j, idx in zip(client_indices_list, idx_split)]

        # shuffle finally
        for j in range(args.K):
            np.random.seed(args.global_seed); np.random.shuffle(client_indices_list[j])
        
        # construct a hashmap
        split_map = dict(zip([i for i in range(args.K)], client_indices_list))
        split_map = {k: v for k, v in split_map.items() if len(v) > 0}
        return split_map
    # LEAF benchmark dataset
    elif args.split_type == 'realistic':
        return print('[INFO] No need to split... use LEAF parser directly!')

# parser object for LEAF benchmark datsaet
class LEAFParser:
    def __init__(self, args):
        self.root = args.data_path
        self.dataset_name = args.dataset.lower()
        
        # declare appropriate dataset class
        if 'femnist' in self.dataset_name:
            self.dataset_class = FEMNISTDataset
        elif 'shakespeare' in self.dataset_name:
            self.dataset_class = ShakespeareDataset
        else:
            raise NotImplementedError(f'[ERROR] {self.dataset_name} is not supported yet!')
                                      
        # set path
        self.train_root = f'{self.root}/{self.dataset_name.lower()}/data/train'
        self.test_root = f'{self.root}/{self.dataset_name}/data/test'
        
        # get raw data
        self.raw_train = self._parse_data(self.train_root, 'train')
        self.raw_test = self._parse_data(self.test_root, 'test')
        
        # merge raw data
        self.merged_train = self._merge_raw_data(self.raw_train, 'train')
        self.merged_test = self._merge_raw_data(self.raw_test, 'test')
        del self.raw_train, self.raw_test; gc.collect()
        
        # make dataset for each client
        self.split_map, self.datasets = self._convert_to_dataset(self.merged_train, self.merged_test)
        del self.merged_train, self.merged_test; gc.collect()
        
    def _parse_data(self, root, mode):
        raw_all = []
        for file in tqdm(os.listdir(root), desc=f'[INFO] ...parsing {mode} data (LEAF - {self.dataset_name.upper()})'):
            with open(f'{root}/{file}') as raw_files:
                for raw_file in raw_files:
                    raw_all.append(json.loads(raw_file))
        return raw_all
    
    def _merge_raw_data(self, data, mode):
        merged_raw_data = {'users': list(), 'num_samples': list(), 'user_data': dict()}
        for raw_data in tqdm(data, desc=f'[INFO] ...merging raw {mode} data (LEAF - {self.dataset_name.upper()})'):
            merged_raw_data['users'].extend(raw_data['users'])
            merged_raw_data['num_samples'].extend(raw_data['num_samples'])
            merged_raw_data['user_data'] = {**merged_raw_data['user_data'], **raw_data['user_data']}
        return merged_raw_data
    
    def _convert_to_dataset(self, merged_train, merged_test):
        """
        Returns:
            [tuple(local_training_set[indices_1], local_test_set[indices_1]), tuple(local_training_set[indices_2], local_test_set[indices_2]), ...]
        """
        def construct_leaf(idx, user):
            # copy dataset class prototype for each training set and test set
            tr_dset, te_dset = copy.deepcopy(self.dataset_class)(), copy.deepcopy(self.dataset_class)()
            setattr(tr_dset, 'train', True); setattr(te_dset, 'train', False)
            
            # set essential attributes
            tr_dset.identifier = user; te_dset.identifier = user
            tr_dset.data = merged_train['user_data'][user]; te_dset.data = merged_test['user_data'][user]
            tr_dset.num_samples = merged_train['num_samples'][idx]; te_dset.num_samples = merged_test['num_samples'][idx]
            tr_dset._make_dataset(); te_dset._make_dataset()
            return (tr_dset, te_dset)
        datasets = Parallel(n_jobs=os.cpu_count() - 1, prefer='threads')(delayed(construct_leaf)(idx, user) for idx, user in tqdm(enumerate(merged_train['users']), desc=f'[INFO] ...create datasets [LEAF - {self.dataset_name.upper()}]!'))
        split_map = dict(zip([i for i in range(len(merged_train['user_data']))], list(map(sum, zip(merged_train['num_samples'], merged_test['num_samples'])))))
        return split_map, datasets
    
    def get_datasets(self):
        assert self.datasets is not None, '[ERROR] dataset is not constructed internally!'
        return self.split_map, self.datasets
    
# construct a client dataset (training & test set)
def get_dataset(args):
    """
    Retrieve requested datasets.
    
    Args:
        args (argparser): arguments
        
    Returns:
        metadata: {0: [indices_1], 1: [indices_2], ... , K: [indices_K]}
        global_testset: (optional) test set located at the central server, 
        client datasets: [tuple(local_training_set[indices_1], local_test_set[indices_1]), tuple(local_training_set[indices_2], local_test_set[indices_2]), ...]
    """
    client_datasets = []
    if args.dataset in ['MNIST', 'CIFAR10', 'CIFAR100']:
        # call raw datasets
        raw_train = torchvision.datasets.__dict__[args.dataset](root=args.data_path, train=True, transform=torchvision.transforms.ToTensor(), download=True)
        raw_test = torchvision.datasets.__dict__[args.dataset](root=args.data_path, train=False, transform=torchvision.transforms.ToTensor(), download=True)
        
        # get split indices
        split_map = split_data(args, raw_train)

        if args.dataset == 'MNIST':
            # construct client datasets
            def construct_mnist(indices):
                if args.label_noise:
                    return (
                        NoisyMNISTDataset(args, indices=indices[:int(len(indices) * (1. - args.test_fraction))], train=True, transform=torchvision.transforms.ToTensor()),
                        MNISTDataset(args, indices=indices[int(len(indices) * (1. - args.test_fraction)):], train=True, transform=torchvision.transforms.ToTensor())
                    )
                else:
                    return (
                        MNISTDataset(args, indices=indices[:int(len(indices) * (1. - args.test_fraction))], train=True, transform=torchvision.transforms.ToTensor()),
                        MNISTDataset(args, indices=indices[int(len(indices) * (1. - args.test_fraction)):], train=True, transform=torchvision.transforms.ToTensor())
                    )
            client_datasets = Parallel(n_jobs=os.cpu_count() - 1, prefer='threads')(delayed(construct_mnist)(indices) for _, indices in tqdm(split_map.items(), desc=f'[INFO] ...create datasets [{args.dataset}]!'))
                
            # construct server datasets
            server_testset = MNISTDataset(args, train=False, transform=torchvision.transforms.ToTensor())
            split_map = {k: len(v) for k, v in split_map.items()}
            return split_map, server_testset, client_datasets
        
        elif args.dataset in ['CIFAR10', 'CIFAR100']:
            # construct client datasets
            def construct_cifar(indices):
                if args.label_noise:
                    return (
                        NoisyCIFARDataset(args, indices=indices[:int(len(indices) * (1. - args.test_fraction))], train=True, transform=torchvision.transforms.ToTensor()),
                        CIFARDataset(args, indices=indices[int(len(indices) * (1. - args.test_fraction)):], train=True, transform=torchvision.transforms.ToTensor())
                    )
                else:
                    return (
                        CIFARDataset(args, indices=indices[:int(len(indices) * (1. - args.test_fraction))], train=True, transform=torchvision.transforms.ToTensor()),
                        CIFARDataset(args, indices=indices[int(len(indices) * (1. - args.test_fraction)):], train=True, transform=torchvision.transforms.ToTensor())
                    )
            client_datasets = Parallel(n_jobs=os.cpu_count() - 1, prefer='threads')(delayed(construct_cifar)(indices) for _, indices in tqdm(split_map.items(), desc=f'[INFO] ...create datasets [{args.dataset}]!'))

            # construct server datasets
            server_testset = CIFARDataset(args, train=False, transform=torchvision.transforms.ToTensor())
            split_map = {k: len(v) for k, v in split_map.items()}
            return split_map, server_testset, client_datasets
        
    elif args.dataset == 'TinyImageNet':
        # call raw dataset
        raw_train = TinyImageNet(root=args.data_path, split='train', transform=torchvision.transforms.ToTensor(), download=True)
        raw_test = TinyImageNet(root=args.data_path, split='val', transform=torchvision.transforms.ToTensor(), download=True)
        
        # get split indices
        split_map = split_data(args, raw_train)
        
        # construct client datasets
        def construct_tinyimagenet(indices):
            return (
                TinyImageNetDataset(args, indices=indices[:int(len(indices) * (1. - args.test_fraction))], train=True, transform=torchvision.transforms.ToTensor()),
                TinyImageNetDataset(args, indices=indices[int(len(indices) * (1. - args.test_fraction)):], train=True, transform=torchvision.transforms.ToTensor())
            )
        client_datasets = Parallel(n_jobs=os.cpu_count() - 1, prefer='threads')(delayed(construct_tinyimagenet)(indices) for _, indices in tqdm(split_map.items(), desc=f'[INFO] ...create datasets [{args.dataset}]!'))

        # construct server datasets
        server_testset = TinyImageNetDataset(args, train=False, transform=torchvision.transforms.ToTensor())
        return split_map, server_testset, client_datasets
    
    elif args.dataset in ['FEMNIST', 'Shakespeare']:
        assert args.split_type == 'realistic', '[ERROR] LEAF benchmark dataset is only supported for `realistic` split scenario!'
        
        # parse dataset
        parser = LEAFParser(args)
        
        # construct client datasets
        split_map, client_datasets = parser.get_datasets()
        return split_map, None, client_datasets


###################
# Model initation #
###################
def init_weights(model, init_type, init_gain, seeds):
    """Initialize network weights.

    Args:
        model (torch.nn.Module): network to be initialized
        init_type (string): the name of an initialization method: normal | xavier | xavier_uniform | kaiming | orthogonal | none
        init_gain (float): scaling factor for normal, xavier and orthogonal
        seeds (list): list of seeds used for an initialization

    Returns:
        model (torch.nn.Module): initialized model with `init_type` and `init_gain`
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if classname.find('BatchNorm2d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                torch.manual_seed(seeds[0]); torch.nn.init.normal_(m.weight.data, 1.0, init_gain)
                if len(seeds) == 2: torch.manual_seed(seeds[-1]); torch.nn.init.normal_(m.weight_1.data, 1.0, init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                torch.manual_seed(seeds[0]); torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
                if len(seeds) == 2: torch.manual_seed(seeds[-1]); torch.nn.init.normal_(m.weight_1.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.manual_seed(seeds[0]); torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
                if len(seeds) == 2: torch.manual_seed(seeds[-1]); torch.nn.init.xavier_normal_(m.weight_1.data, gain=init_gain)
            elif init_type == 'xavier_uniform':
                torch.manual_seed(seeds[0]); torch.nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                if len(seeds) == 2: torch.manual_seed(seeds[-1]); torch.nn.init.xavier_uniform_(m.weight_1.data, gain=1.0)
            elif init_type == 'kaiming':
                torch.manual_seed(seeds[0]); torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                if len(seeds) == 2: torch.manual_seed(seeds[-1]); torch.nn.init.kaiming_normal_(m.weight_1.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.manual_seed(seeds[0]); torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
                if len(seeds) == 2: torch.manual_seed(seeds[-1]); torch.nn.init.orthogonal_(m.weight_1.data, gain=init_gain)
            elif init_type == 'none':  # uses pytorch's default init method
                m.reset_parameters()
            else:
                raise NotImplementedError(f'[ERROR] Initialization method {init_type} is not implemented!')
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('LSTM') != -1:
            for l in range(m.num_layers):
                if init_type == 'normal':
                    torch.manual_seed(seeds[0]); torch.nn.init.normal_(getattr(m, f'weight_hh_l{l}'), 0.0, init_gain); torch.nn.init.normal_(getattr(m, f'weight_ih_l{l}'), 0.0, init_gain)
                    if len(seeds) == 2: torch.manual_seed(seeds[-1]); torch.nn.init.normal_(getattr(m, f'weight_hh_l{l}_1'), 0.0, init_gain); torch.nn.init.normal_(getattr(m, f'weight_ih_l{l}_1'), 0.0, init_gain)
                elif init_type == 'xavier':
                    torch.manual_seed(seeds[0]); torch.nn.init.xavier_normal_(getattr(m, f'weight_hh_l{l}'), gain=init_gain); torch.nn.init.xavier_normal_(getattr(m, f'weight_ih_l{l}'), gain=init_gain)
                    if len(seeds) == 2: torch.manual_seed(seeds[-1]); torch.nn.init.xavier_normal_(getattr(m, f'weight_hh_l{l}_1'), gain=init_gain); torch.nn.init.xavier_normal_(getattr(m, f'weight_ih_l{l}_1'), gain=init_gain)
                elif init_type == 'xavier_uniform':
                    torch.manual_seed(seeds[0]); torch.nn.init.xavier_uniform_(getattr(m, f'weight_hh_l{l}'), gain=1.0); torch.nn.init.xavier_uniform_(getattr(m, f'weight_ih_l{l}'), gain=1.0)
                    if len(seeds) == 2: torch.manual_seed(seeds[-1]); torch.nn.init.xavier_uniform_(getattr(m, f'weight_hh_l{l}_1'), gain=1.0); torch.nn.init.xavier_uniform_(getattr(m, f'weight_ih_l{l}_1'), gain=1.0)
                elif init_type == 'kaiming':
                    torch.manual_seed(seeds[0]); torch.nn.init.kaiming_normal_(getattr(m, f'weight_hh_l{l}'), a=0, mode='fan_in'); torch.nn.init.kaiming_normal_(getattr(m, f'weight_ih_l{l}'), a=0, mode='fan_in')
                    if len(seeds) == 2: torch.manual_seed(seeds[-1]); torch.nn.init.kaiming_normal_(getattr(m, f'weight_hh_l{l}_1'), a=0, mode='fan_in'); torch.nn.init.kaiming_normal_(getattr(m, f'weight_ih_l{l}_1'), a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    torch.manual_seed(seeds[0]); torch.nn.init.orthogonal_(getattr(m, f'weight_hh_l{l}'), gain=init_gain); torch.nn.init.orthogonal_(getattr(m, f'weight_ih_l{l}'), gain=init_gain)
                    if len(seeds) == 2: torch.manual_seed(seeds[-1]); torch.nn.init.orthogonal_(getattr(m, f'weight_hh_l{l}_1'), gain=init_gain); torch.nn.init.orthogonal_(getattr(m, f'weight_ih_l{l}_1'), gain=init_gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(f'[ERROR] Initialization method {init_type} is not implemented!')
                if m.bias is True:
                    torch.nn.init.constant_(getattr(m, f'bias_hh_l{l}'), 0.0); torch.nn.init.constant_(getattr(m, f'bias_ih_l{l}'), 0.0)
                    if len(seeds) == 2: torch.nn.init.constant_(getattr(m, f'bias_hh_l{l}_1'), 0.0); torch.nn.init.constant_(getattr(m, f'bias_ih_l{l}_1'), 0.0)
    model.apply(init_func)
    return model

def initiate_model(model, args):
    """Initiate model instance; use multi-GPU if available.
    
    Args:
        model (nn.Module): model instance to initiate
        args (argument): parsed arguments
    
    Returns:
        model: (nn.Module) initiated instance
    """    
    # GPU setting
    if 'cuda' in args.device:
        if torch.cuda.device_count() > 1:
            model_instance = torch.nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])
        else:
            model_instance = model
    # CPU setting
    else:
        model_instance = model
    model_instance = model_instance.to(args.device)
    return model_instance




###########
# Metrics #
###########
# top-k accuracy
def get_accuracy(output, target, topk=(1, 5)):
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.

    ref:
    - https://pytorch.org/docs/stable/generated/torch.topk.html
    - https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840
    - https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
    - https://discuss.pytorch.org/t/top-k-error-calculation/48815/2
    - https://stackoverflow.com/questions/59474987/how-to-get-top-k-accuracy-in-semantic-segmentation-using-pytorch

    :param output: output is the prediction of the model e.g. scores, logits, raw y_pred before normalization or getting classes
    :param target: target is the truth
    :param topk: tuple of topk's to compute e.g. (1, 2, 5) computes top 1, top 2 and top 5.
    e.g. in top 2 it means you get a +1 if your models's top 2 predictions are in the right label.
    So if your model predicts cat, dog (0, 1) and the true label was bird (3) you get zero
    but if it were either cat or dog you'd accumulate +1 for that example.
    :return: list of topk accuracy [top1st, top2nd, ...] depending on your topk input
    """
    with torch.no_grad():
        # ---- get the topk most likely labels according to your model
        # get the largest k \in [n_classes] (i.e. the number of most likely probabilities we will use)
        maxk = max(topk)  # max number labels we will consider in the right choices for out model
        batch_size = target.size(0)

        # get top maxk indicies that correspond to the most likely probability scores
        # (note _ means we don't care about the actual top maxk scores just their corresponding indicies/labels)
        _, y_pred = output.topk(k=maxk, dim=1)  # _, [B, n_classes] -> [B, maxk]
        y_pred = y_pred.t()  # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.

        # - get the credit for each example if the models predictions is in maxk values (main crux of code)
        # for any example, the model will get credit if it's prediction matches the ground truth
        # for each example we compare if the model's best prediction matches the truth. If yes we get an entry of 1.
        # if the k'th top answer of the model matches the truth we get 1.
        # Note: this for any example in batch we can only ever get 1 match (so we never overestimate accuracy <1)
        target_reshaped = target.view(1, -1).expand_as(y_pred)  # [B] -> [B, 1] -> [maxk, B]
        # compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
        correct = (y_pred == target_reshaped)  # [maxk, B] were for each example we know which topk prediction matched truth
        # original: correct = pred.eq(target.view(1, -1).expand_as(pred))

        # -- get topk accuracy
        list_topk_accs = []  # idx is topk1, topk2, ... etc
        for k in topk:
            # get tensor of which topk answer was right
            ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
            # flatten it to help compute if we got it correct for each example in batch
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()  # [k, B] -> [kB]
            # get if we got it right for any of our top k prediction for each example in batch
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)  # [kB] -> [1]
            # compute topk accuracy - the accuracy of the mode's ability to get it right within it's top k guesses/preds
            topk_acc = tot_correct_topk / batch_size  # topk accuracy for entire batch
            list_topk_accs.append(topk_acc.detach().cpu())
        return torch.stack(list_topk_accs)  # list of topk accuracies for entire batch [topk1, topk2, ... etc]

# calibration error
class CalibrationError(torch.nn.Module):
    """
    Credits to: https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
    Calculates the Expected Calibration Error & Maximum Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(CalibrationError, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        mce = []
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                mce.append(torch.abs(avg_confidence_in_bin - accuracy_in_bin))
        else:
            mce = torch.stack(mce).max()
        return ece, mce

    
    
###########################
# Record metrics & losses #
###########################
def record_results(args, writer, mode, step, loss, acc1, acc5, ece, mce):
    # convert to tensors
    loss = torch.tensor(loss)
    acc1 = torch.tensor(acc1)
    acc5 = torch.tensor(acc5)
    ece = torch.tesnor(ece)
    mce = torch.tensor(mce)

    # save as results
    self.results[f'{mode}_loss_mean'].append(loss.mean()); self.results[f'{mode}_loss_std'].append(loss.std())
    self.results[f'{mode}_acc1_mean'].append(acc1.mean()); self.results[f'{mode}_acc1_std'].append(acc1.std())
    self.results[f'{mode}_acc5_mean'].append(acc5.mean()); self.results[f'{mode}_acc5_std'].append(acc5.std())
    self.results[f'{mode}_ece_mean'].append(ece.mean()); self.results[f'{mode}_ece_std'].append(ece.std())
    self.results[f'{mode}_mce_mean'].append(per_mce.mean()); self.results[f'{mode}_mce_std'].append(per_mce.std())

    # record metrics and loss
    writer.add_scalars(
        'Loss',
        {f'[{args.exp_name}] {args.algorithm}_{args.dataset}_{args.model_name}_{mode}_loss_mean': loss.mean()},
        step
    )
    writer.add_scalars(
        'Accuracy',
        {f'[{args.exp_name}] {args.algorithm}_{args.dataset}_{args.model_name}_{mode}_top1_acc_mean': acc1.mean()},
        step
    )
    writer.add_scalars(
        'Accuracy',
        {f'[{args.exp_name}] {args.algorithm}_{args.dataset}_{args.model_name}_{mode}_top5_acc_mean': acc5.mean()},
        step
    )
    writer.add_scalars(
        'Expected Calibration Error',
        {f'[{args.exp_name}] {args.algorithm}_{args.dataset}_{args.model_name}_{mode}_ece_mean': ece.mean()},
        step
    )
    writer.add_scalars(
        'Maximum Calibration Error',
        {f'[{args.exp_name}] {args.algorithm}_{args.dataset}_{args.model_name}_{mode}_mce_mean': mce.mean()},
        step
    )
    return loss, acc1, acc5, ece, mce



######################
# Plot delta metrics #
######################
def plot_delta(delta):
    pass