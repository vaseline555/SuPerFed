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
# https://github.com/luoyan407/congruency/blob/master/train_timgnet.py
def parse_classes(file):
    classes = []
    filenames = []
    with open(file) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    for x in range(len(lines)):
        tokens = lines[x].split()
        classes.append(tokens[1])
        filenames.append(tokens[0])
    return filenames, classes

class TinyImageNetDataset(torch.utils.data.Dataset):
    """Dataset wrapping images and ground truths."""
    def __init__(self, img_path, gt_path, class_to_idx=None, transform=None):
        self.img_path = img_path
        self.transform = transform
        self.gt_path = gt_path
        self.class_to_idx = class_to_idx
        self.classidx = []
        self.imgs, self.classnames = parse_classes(gt_path)
        for classname in self.classnames:
            self.classidx.append(self.class_to_idx[classname])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, y) where y is the label of the image.
        """
        img = None
        with open(os.path.join(self.img_path, self.imgs[index]), 'rb') as f:
            img = PIL.Image.open(f).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
        y = self.classidx[index]
        return img, y

    def __len__(self):
        return len(self.imgs)
    
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
        self.inputs = torch.tensor(self.inputs)
        self.targets = torch.tensor(self.targets).long()
        
    def __getitem__(self, index):
        # get corresponding inputs & targets pair
        inputs, targets = self.inputs[index].reshape(-1, 28, 28), self.targets[index]
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
        self.inputs = torch.tensor(self.inputs).long()
        self.targets = torch.tensor(self.targets).long()
        
    def _build_mapping(self):
        self.char_to_idx = dict()
        for idx, char in enumerate(self.all_characters):
            self.char_to_idx[char] = idx
        
    def _tokenize(self):
        for idx in range(len(self.inputs)):
            self.inputs[idx] = [self.char_to_idx[char] for char in self.inputs[idx]]
        
        for idx in range(len(self.targets)):
            self.targets[idx] = [self.char_to_idx[char] for char in self.targets[idx]]

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index][0]

    def __len__(self):
        return len(self.inputs)



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
    
    