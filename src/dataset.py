import os
import gc
import PIL
import shutil
import string
import torch
import torchvision
import numpy as np


    
########################
# TinyImageNet dataset #
########################
# TinyImageNet
def normalize_tin_val_folder_structure(path, images_folder='images', annotations_file='val_annotations.txt'):
    # Check if files/annotations are still there to see if we already run reorganize the folder structure.
    images_folder = os.path.join(path, images_folder)
    annotations_file = os.path.join(path, annotations_file)

    # Exists
    if not os.path.exists(images_folder) \
       and not os.path.exists(annotations_file):
        if not os.listdir(path):
            raise RuntimeError('[ERROR] validation folder is empty!')
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

class TinyImageNetDataset(torchvision.datasets.ImageFolder):
    """Dataset for TinyImageNet-200"""
    base_folder = 'tiny-imagenet-200'
    zip_md5 = '90528d7ca1a48142e341f4ef8d21d0de'
    splits = ('train', 'val')
    filename = 'tiny-imagenet-200.zip'
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'

    def __init__(self, args, split='train', download=False, **kwargs):
        self.data_root = os.path.expanduser(args.data_path)
        self.split = torchvision.datasets.utils.verify_str_arg(split, 'split', self.splits)

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('[ERROR] dataset not found! You can use `download=True` to download it.')
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
        if self._check_exists(): return
        torchvision.datasets.utils.download_and_extract_archive(self.url, self.data_root, filename=self.filename, remove_finished=True, md5=self.zip_md5)
        assert 'val' in self.splits
        normalize_tin_val_folder_structure(os.path.join(self.dataset_folder, 'val'))
    

    
########################################
# Dataset prepared in `LEAF` benchmark #
########################################
# base dataset class for LEAF benchmark dataset
class LEAFDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(LEAFDataset, self).__init__()
        self._id = None
        self._data = None
        self._num_samples = 0

    @property
    def identifier(self):
        return self._id
    
    @identifier.setter
    def identifier(self, identifier):
        self._id = identifier  
        
    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data  
        
    def __getitem__(self, index):
        raise NotImplementedError('[ERROR] subclass should implement this!')

    def __len__(self):
        raise NotImplementedError('[ERROR] subclass should implement this!')

# LEAF - FEMNIST
class FEMNISTDataset(LEAFDataset):
    def __init__(self, train=None):
        super(FEMNISTDataset, self).__init__()
        self.train = train

    def _make_dataset(self):
        self.inputs, self.targets = self.data['x'], self.data['y']
        
    def __getitem__(self, index):
        # get corresponding inputs & targets pair
        inputs, targets = self.inputs[index], self.targets[index]
        inputs = np.array(inputs).reshape(-1, 28, 28).astype(np.float32)
        return inputs, targets
    
    def __len__(self):
        return len(self.inputs)
    
# LEAF - Shakespeare
class ShakespeareDataset(LEAFDataset):
    def __init__(self, train=None):
        super(ShakespeareDataset, self).__init__()
        self.train = train
        self.all_characters = string.printable
        self.vocab_size = len(self.all_characters)
        
    def _make_dataset(self):
        self.inputs, self.targets = self.data['x'], self.data['y']
        self._build_mapping()
        self._tokenize()
        
    def _build_mapping(self):
        self.char_to_idx = dict()
        for idx, char in enumerate(self.all_characters):
            self.char_to_idx[char] = idx
        
    def _tokenize(self):
        for idx in range(len(self.inputs)):
            self.inputs[idx] = [self.char_to_idx[char] for char in self.inputs[idx]]
        
        for idx in range(len(self.targets)):
            self.targets[idx] = [self.char_to_idx[char] for char in self.targets[idx]]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return np.array(self.inputs)[index], np.array(self.targets)[index][0]



########################################
# Dataset for simulating a label noise #
########################################
# for simulating a label noise
# https://github.com/UCSC-REAL/cores/blob/main/data/utils.py
def multiclass_noisify(targets, P, seed):
    assert P.shape[0] == P.shape[1]
    assert np.max(targets) < P.shape[0]
    np.testing.assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    noisy_targets = targets.copy()
    flipper = np.random.RandomState(seed)

    for idx in np.arange(len(targets)):
        i = targets[idx]
        flipped = flipper.multinomial(n=1, pvals=P[i, :], size=1)[0]
        noisy_targets[idx] = np.where(flipped == 1)[0]
    return noisy_targets

def multiclass_pair_noisify(targets, noise_rate, seed, num_classes):
    P = np.eye(num_classes)

    if noise_rate > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1. - noise_rate, noise_rate
        for i in range(1, num_classes-1):
            P[i, i], P[i, i + 1] = 1. - noise_rate, noise_rate
        P[num_classes - 1, num_classes - 1], P[num_classes - 1, 0] = 1. - noise_rate, noise_rate
 
        noisy_targets = multiclass_noisify(targets, P=P, seed=seed)
        actual_noise = (np.array(noisy_targets).flatten() != np.array(targets).flatten()).mean()
        assert actual_noise > 0.0
    return np.array(noisy_targets).flatten(), actual_noise
        
def multiclass_symmetric_noisify(targets, noise_rate, seed, num_classes):
    P = np.ones((num_classes, num_classes))
    P *= (noise_rate / (num_classes - 1)) 

    if noise_rate > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - noise_rate
        for i in range(1, num_classes-1):
            P[i, i] = 1. - noise_rate
        P[num_classes - 1, num_classes - 1] = 1. - noise_rate

        noisy_targets = multiclass_noisify(targets, P, seed)
        actual_noise = (np.array(noisy_targets).flatten() != np.array(targets).flatten()).mean()
        assert actual_noise > 0.0
    return np.array(noisy_targets).flatten(), actual_noise

class LabelNoiseDataset(torch.utils.data.Dataset):
    def __init__(self, args, dataset, transform):
        super(LabelNoiseDataset, self).__init__()
        self.dataset = args.dataset
        self.transform = transform
        self.noise_rate = args.noise_rate
        self.inputs, self.targets = np.array(dataset.data), np.array(dataset.targets)
        del dataset; gc.collect()
        
        # inject label noise
        if args.noise_type == 'pair':
            noisy_targets, actual_noise_rate = multiclass_symmetric_noisify(targets=self.targets, noise_rate=args.noise_rate, seed=args.global_seed, num_classes=args.num_classes)
        elif args.noise_type == 'symmetric':
            noisy_targets, actual_noise_rate = multiclass_pair_noisify(targets=self.targets, noise_rate=args.noise_rate, seed=args.global_seed, num_classes=args.num_classes)
        self.noise_mask = np.transpose(noisy_targets) != np.transpose(self.targets)
        self.targets, self.original_targets = noisy_targets, self.targets
    
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        inputs, targets = self.inputs[index], self.targets[index]
        if self.dataset == 'MNIST':
            inputs = PIL.Image.fromarray(inputs, mode='L')
        else:
            inputs = PIL.Image.fromarray(inputs, mode='RGB')
        if self.transform is not None:
            inputs = self.transform(inputs)
        return inputs, targets
    
    def get_noise_mask(self):
        return self.noise_mask
    
    def get_original_targets(self):
        return self.original_targets
    
    