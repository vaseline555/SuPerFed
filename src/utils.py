import os
import gc
import logging
import io
import PIL.Image
import shutil

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.utils.data import Dataset, TensorDataset, ConcatDataset, Subset
from torchvision import datasets, transforms

from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets.utils import download_and_extract_archive

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

def visualize_metrics(writer, step, metrics, tag):
    """
    Visualization of confusion matrix

    Parameters:
        writer (tensorboard.SummaryWriter): TensorBoard SummaryWriter instance.
        step (int): Counter usually specifying steps/epochs/time.
        matrix (numpy.array): Square-shaped array of size class x class.
            Should specify cross-class accuracies/confusion in percent
            values (range 0-1).
        class_dict (dict): Dictionary specifying class names as keys and
            corresponding integer labels/targets as values.
    """

    # Create the figure
    fig = plt.figure()
    x = np.arange(0, 1.1, 0.1)
    y = torch.Tensor(metrics).mean(0).numpy()[::-1]
    yerr = np.where(torch.Tensor(metrics).std(0) == 0, np.nan, torch.Tensor(metrics).std(0))[::-1]

    ax = fig.add_subplot(111)
    ax.errorbar(x, y, yerr=yerr, marker='o')
    plt.title(f"Round{str(step).zfill(4)}_{tag}")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    
    image = PIL.Image.open(buf)
    image = transforms.ToTensor()(image)
    writer.add_image(tag, image, step)
    
    plt.close(fig)
    
    del fig
    gc.collect()
    

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
                if len(seeds) == 2: torch.manual_seed(seeds[-1]); torch.nn.init.normal_(m.weight1.data, 1.0, init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                torch.manual_seed(seeds[0]); torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
                if len(seeds) == 2: torch.manual_seed(seeds[-1]); torch.nn.init.normal_(m.weight1.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.manual_seed(seeds[0]); torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
                if len(seeds) == 2: torch.manual_seed(seeds[-1]); torch.nn.init.xavier_normal_(m.weight1.data, gain=init_gain)
            elif init_type == 'xavier_uniform':
                torch.manual_seed(seeds[0]); torch.nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                if len(seeds) == 2: torch.manual_seed(seeds[-1]); torch.nn.init.xavier_uniform_(m.weight1.data, gain=1.0)
            elif init_type == 'kaiming':
                torch.manual_seed(seeds[0]); torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                if len(seeds) == 2: torch.manual_seed(seeds[-1]); torch.nn.init.kaiming_normal_(m.weight1.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.manual_seed(seeds[0]); torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
                if len(seeds) == 2: torch.manual_seed(seeds[-1]); torch.nn.init.orthogonal_(m.weight1.data, gain=init_gain)
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
    # initialize model
    model = init_weights(model, args.init_type, args.init_gain, args.init_seed)
    
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

################
# TinyImageNet #
################
def normalize_tin_val_folder_structure(path,
                                       images_folder='images',
                                       annotations_file='val_annotations.txt'):
    # Check if files/annotations are still there to see
    # if we already run reorganize the folder structure.
    images_folder = os.path.join(path, images_folder)
    annotations_file = os.path.join(path, annotations_file)

    # Exists
    if not os.path.exists(images_folder) \
       and not os.path.exists(annotations_file):
        if not os.listdir(path):
            raise RuntimeError('Validation folder is empty.')
        return

    # Parse the annotations
    with open(annotations_file) as f:
        for line in f:
            values = line.split()
            img = values[0]
            label = values[1]
            img_file = os.path.join(images_folder, values[0])
            label_folder = os.path.join(path, label)
            os.makedirs(label_folder, exist_ok=True)
            try:
                shutil.move(img_file, os.path.join(label_folder, img))
            except FileNotFoundError:
                continue
                
    os.sync()
    assert not os.listdir(images_folder)
    shutil.rmtree(images_folder)
    os.remove(annotations_file)
    os.sync()


class TinyImageNet(ImageFolder):
    """Dataset for TinyImageNet-200"""
    base_folder = 'tiny-imagenet-200'
    zip_md5 = '90528d7ca1a48142e341f4ef8d21d0de'
    splits = ('train', 'val')
    filename = 'tiny-imagenet-200.zip'
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'

    def __init__(self, root, split='train', download=False, **kwargs):
        self.data_root = os.path.expanduser(root)
        self.split = verify_str_arg(split, "split", self.splits)

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')
        super().__init__(self.split_folder, **kwargs)
        
    @property
    def dataset_folder(self):
        return os.path.join(self.data_root, self.base_folder)

    @property
    def split_folder(self):
        return os.path.join(self.dataset_folder, self.split)

    def _check_exists(self):
        return os.path.exists(self.split_folder)

    def extra_repr(self):
        return "Split: {split}".format(**self.__dict__)

    def download(self):
        if self._check_exists():
            return
        download_and_extract_archive(
            self.url, self.data_root, filename=self.filename,
            remove_finished=True, md5=self.zip_md5)
        assert 'val' in self.splits
        normalize_tin_val_folder_structure(
            os.path.join(self.dataset_folder, 'val'))


#################
# Dataset split #
#################
class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms."""
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]
        if self.transform:
            x = self.transform(x.numpy().astype(np.uint8))
        return x, y

    def __len__(self):
        return self.tensors[0].size(0)
    
def create_datasets(data_path, dataset_name, num_clients, num_shards, iid):
    """Split the whole dataset in IID or non-IID manner for distributing to clients."""
    # get dataset from torchvision.datasets if exists
    if hasattr(datasets, dataset_name):
        # set transformation differently per dataset
        if dataset_name in ["CIFAR10", "CIFAR100", "MNIST", "KMNIST"]:
            transform = transforms.ToTensor()
        
            # prepare raw training & test datasets
            training_dataset = datasets.__dict__[dataset_name](
                root=data_path,
                train=True,
                download=True,
                transform=transform
            )
            test_dataset = datasets.__dict__[dataset_name](
                root=data_path,
                train=False,
                download=True,
                transform=transform
            )
        elif dataset_name in ["EMNIST"]:
            transform = transforms.ToTensor()
        
            # prepare raw training & test datasets
            training_dataset = datasets.__dict__[dataset_name](
                root=data_path,
                train=True,
                download=True,
                transform=transform,
                split='byclass'
            )
            test_dataset = datasets.__dict__[dataset_name](
                root=data_path,
                train=False,
                download=True,
                transform=transform,
                split='byclass'
            )
    else:
        if dataset_name == "TinyImageNet":
            # set transformation
            transform = transforms.ToTensor()

            # prepare raw training & test datasets
            training_dataset = TinyImageNet(
                    root=data_path,
                    split='train',
                    download=True,
                    transform=transform
                )
            test_dataset = TinyImageNet(
                    root=data_path,
                    split='val',
                    download=True,
                    transform=transform
            )
        else:
            # dataset not found exception
            error_message = f"...dataset \"{dataset_name}\" is not supported or cannot be found!"
            logging.error(error_message)
            raise AttributeError(error_message)

    # split dataset according to iid flag
    if dataset_name in ["MNIST", "KMNIST", "CIFAR10", "CIFAR100"]:
        # unsqueeze channel dimension for grayscale image datasets
        if training_dataset.data.ndim == 3: # convert to NxHxW -> NxHxWx1
            training_dataset.data.unsqueeze_(3)
        num_categories = np.unique(training_dataset.targets).shape[0]
        
        # sanity check
        if "ndarray" not in str(type(training_dataset.data)):
            training_dataset.data = np.asarray(training_dataset.data)
        if "list" not in str(type(training_dataset.targets)):
            training_dataset.targets = training_dataset.targets.tolist()
        
        ## split scenario!
        if iid:
            # shuffle data
            shuffled_indices = torch.randperm(len(training_dataset))
            training_inputs = training_dataset.data[shuffled_indices]
            training_labels = torch.Tensor(training_dataset.targets)[shuffled_indices]

            # partition data into num_clients
            split_size = len(training_dataset) // num_clients
            split_datasets = list(
                zip(
                    torch.split(torch.Tensor(training_inputs), split_size),
                    torch.split(torch.Tensor(training_labels), split_size)
                )
            )

            # finalize bunches of local datasets
            local_datasets = [
                CustomTensorDataset(local_dataset, transform=transform)
                for local_dataset in split_datasets
                ]
        else:
            # sort data by labels
            sorted_indices = torch.argsort(torch.Tensor(training_dataset.targets))
            training_inputs = training_dataset.data[sorted_indices]
            training_labels = torch.Tensor(training_dataset.targets)[sorted_indices]

            # partition data into shards first
            shard_size = len(training_dataset) // num_shards #300
            split_datasets = list(
                zip(
                    torch.split(torch.Tensor(training_inputs), shard_size),
                    torch.split(torch.Tensor(training_labels), shard_size)
                )
            )

            # store temoporary dataset object for storing shards into a list
            shard_datasets = [
                CustomTensorDataset(local_dataset, transform=transform)
                for local_dataset in split_datasets]

            # sort the list to conveniently assign samples to each clients from at least two~ classes
            shard_sorted = []
            for i in range(num_shards // num_categories):
                for j in range(0, ((num_shards // num_categories) * num_categories), (num_shards // num_categories)):
                    shard_sorted.append(shard_datasets[i + j])

            # finalize local datasets by assigning shards to each client
            shards_per_clients = num_shards // num_clients
            local_datasets = [
                ConcatDataset(shard_sorted[i:i + shards_per_clients]) 
                for i in range(0, len(shard_sorted), shards_per_clients)
                ]
    elif dataset_name == "EMNIST":
        # unsqueeze channel dimension for grayscale image datasets
        if training_dataset.data.ndim == 3: # convert to NxHxW -> NxHxWx1
            training_dataset.data.unsqueeze_(3)
        num_categories = np.unique(training_dataset.targets).shape[0]
        
        # sanity check
        if "ndarray" not in str(type(training_dataset.data)):
            training_dataset.data = np.asarray(training_dataset.data)
        if "list" not in str(type(training_dataset.targets)):
            training_dataset.targets = training_dataset.targets.tolist()
        
        ## split scenario!
        if iid:
            # argument `num_shards` is ignored in this block...
            # shuffle data
            shuffled_indices = torch.randperm(len(training_dataset))
            training_inputs = training_dataset.data[shuffled_indices]
            training_labels = torch.Tensor(training_dataset.targets)[shuffled_indices]

            # partition data into num_clients
            split_size = len(training_dataset) // num_clients
            split_datasets = list(
                zip(
                    torch.split(torch.Tensor(training_inputs), split_size),
                    torch.split(torch.Tensor(training_labels), split_size)
                )
            )

            # finalize bunches of local datasets
            local_datasets = [
                CustomTensorDataset(local_dataset, transform=transform)
                for local_dataset in split_datasets
                ]
        else:
            # argument `num_clients` and `num_shards` are ignored in this block...
            """
            As a result, 1543 clients are generated:
            - avg. sample size: 452.31
            - std. sample size: 203.24
            """ 
            # randomly shuffle and split labels into 5 groups by its semantic 
            digit_labels = list(torch.split(torch.randperm(10), 2))
            uppercase_labels = list(torch.split(torch.randperm(26) + 10, 5))
            lowercase_labels = list(torch.split(torch.randperm(26) + 36, 5))
            
            lowercase_labels[-2] = torch.cat((lowercase_labels[-2], lowercase_labels[-1]))
            del lowercase_labels[-1]
            
            uppercase_labels[-2] = torch.cat((uppercase_labels[-2], uppercase_labels[-1]))
            del uppercase_labels[-1]
            
            # make label groups
            groups = []
            for d, l, u in zip(digit_labels, lowercase_labels, uppercase_labels):
                groups.append(torch.cat((d, l, u)))
            del digit_labels, lowercase_labels, uppercase_labels
            gc.collect()
            
            # make latent component indices by labels
            indices = []
            for group in groups:
                indices.append(
                    (torch.Tensor(training_dataset.targets)[..., None] == group).any(-1).nonzero().squeeze()
                )
            del groups
            gc.collect()
            
            # split dataset by indices
            input_sets, target_sets = [], []
            for idx in indices:
                input_sets.append(
                    torch.Tensor(training_dataset.data)[idx]
                )
                target_sets.append(
                    torch.Tensor(training_dataset.targets)[idx]
                )
            del indices
            gc.collect()

            # create clients by randomly splitting each dset
            local_datasets = []
            for ds, ts in zip(input_sets, target_sets):
                remaining_size = 1e5
                dset_to_split = CustomTensorDataset((ds, ts), transform=transform)
                while remaining_size > 100:
                    size = torch.randint(low=100, high=800, size=(1,)).item()
                    remaining_size = len(ds) - size
                    if remaining_size < 0:
                        break
                    split_dset, remaining_dset = torch.utils.data.random_split(
                        dset_to_split,
                        [size, remaining_size]
                        
                    )
                    local_datasets.append(split_dset)
                    dset_to_split = remaining_dset
                    ds = remaining_dset
            
            del input_sets, target_sets
            gc.collect()
    elif dataset_name == "TinyImageNet":
        ## split scenario!
        if iid:
            raise NotImplementedError("...this will be implemented in later version!")
        else:
            # argument `num_clients` and `num_shards` are ignored in this block...
            """
            As a result, 390 clients are generated:
            - avg. sample size: 251.98
            - std. sample size: 87.56
            """ 
            # initialize label index: files are stored in the increasing order with the same size (500 images * 200 classes)
            label_indices = torch.split(
                torch.arange(len(training_dataset)),
                500
            )
            label_indices = torch.stack(list(label_indices), dim=0)
            
            # shuffle labels
            label_indices = label_indices[torch.randperm(len(label_indices)), :]
            
            # make each client has images from at most 20 classes: 10 groups are generated
            split_indices = torch.stack(list(torch.split(label_indices, 20)), dim=0)
            del label_indices
            gc.collect()

            # for each group, randomly split into clients
            final_indices = []
            for group in split_indices:
                # mix internally
                mixed_group = group.flatten()[torch.randperm(len(group.flatten()))]
                while len(mixed_group) > 100:
                    size = torch.randint(low=100, high=400, size=(1,)).item()
                    remaining_size = len(mixed_group) - size
                    if remaining_size < 0:
                        break
                    split_group, remaining_group = torch.split(
                        mixed_group,
                        [size, remaining_size]
                    )
                    final_indices.append(split_group)
                    mixed_group = remaining_group
            del split_indices
            gc.collect()
                
            # split dataset by indices
            local_datasets = []
            for indices in final_indices:
                #print(indices)
                local_datasets.append(
                    Subset(training_dataset, indices=indices)
                )
            
    return local_datasets, test_dataset

##############################
# top K accuracy calculation #
##############################
def accuracy(output, target, topk=(1,)):
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

##############################
# Expected Calibration Error #
##############################
class ECELoss(nn.Module):
    """
    Credits to: https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
    Calculates the Expected Calibration Error of a model.
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
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece
