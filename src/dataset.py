import os
import gc
import PIL
import string
import torch
import torchvision
import numpy as np


    
########################
# TinyImageNet dataset #
########################
# TinyImageNet
class TinyImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, args, train=True, transform=None):
        super(TinyImageNetDataset, self).__init__()
        self.root = args.data_path
        self.dataset_name = args.dataset
        self.train = train
        self.transform = transform
        self.root = os.path.join(self.root, 'tiny-imagenet-200')
        
        # create index dictionary
        if self.train:
            self._create_class_idx_dict_train()
        else:
            self._create_class_idx_dict_val()
        
        # retrieve inputs and targets
        self.inputs, self.targets = self._make_dataset()
        
        # set ids
        words_file, wnids_file = os.path.join(self.root, 'words.txt'), os.path.join(self.root, 'wnids.txt')
        self.set_nids = set()
        with open(wnids_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                self.set_nids.add(entry.strip('\n'))

        self.class_to_label = {}
        with open(words_file, 'r') as fo:
            data = fo.readlines()
            for entry in data:
                words = entry.split('\t')
                if words[0] in self.set_nids:
                    self.class_to_label[words[0]] = (words[1].strip('\n').split(','))[0]

    def _create_class_idx_dict_train(self):
        classes = [d for d in os.listdir(os.path.join(self.root, 'train')) if os.path.isdir(os.path.join(os.path.join(self.root, 'train'), d))]
        classes = sorted(classes)
        num_images = 0
        for root, dirs, files in os.walk(os.path.join(self.root, 'train')):
            for f in files:
                if f.endswith('.JPEG'):
                    num_images = num_images + 1

        self.target_idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.class_to_target_idx = {classes[i]: i for i in range(len(classes))}

    def _create_class_idx_dict_val(self):
        val_image_dir = os.path.join(os.path.join(self.root, 'val'), 'images')
        images = [d for d in os.listdir(os.path.join(self.root, 'val')) if os.path.isfile(os.path.join(os.path.join(self.root, 'train'), d))]
        val_annotations_file = os.path.join(os.path.join(self.root, 'val'), 'val_annotations.txt')
        
        self.val_img_to_class, set_of_classes = dict(), set()
        with open(val_annotations_file, 'r') as fo:
            entry = fo.readlines()
            for data in entry:
                words = data.split('\t')
                self.val_img_to_class[words[0]] = words[1]
                set_of_classes.add(words[1])
        
        classes = sorted(list(set_of_classes))
        self.class_to_target_idx = {classes[i]: i for i in range(len(sorted(list(set_of_classes))))}
        self.target_idx_to_class = {i: classes[i] for i in range(len(sorted(list(set_of_classes))))}

    def _make_dataset(self):
        inputs, targets = [], []
        if self.train:
            img_root_dir = os.path.join(self.root, 'train')
            list_of_dirs = [target for target in self.class_to_target_idx.keys()]
        else:
            img_root_dir = os.path.join(self.root, 'val')
            list_of_dirs = [target for target in self.class_to_target_idx.keys()]

        for target in list_of_dirs:
            dirs = os.path.join(img_root_dir, target)
            if not os.path.isdir(dirs): continue
            for root, _, files in sorted(os.walk(dirs)):
                for file_name in sorted(files):
                    if file_name.endswith('.JPEG'):
                        inputs.append(os.path.join(root, file_name))
                        if self.train:
                            targets.append(self.class_to_target_idx[target])
                        else:
                            targets.append(self.class_to_target_idx[self.val_img_to_class[file_name]])
        return inputs, targets
    
    def return_label(self, idx):
        return [self.class_to_label[self.target_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        # get corresponding inputs & targets pair
        inputs, targets = self.inputs[index], self.targets[index]
        inputs = PIL.Image.open(inputs).convert('RGB')
        
        # apply transformation
        if self.transform is not None:
            inputs = self.transform(inputs)
        return inputs, targets
    

    
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
    def __init__(self, train=None, transform=torchvision.transforms.ToTensor()):
        super(FEMNISTDataset, self).__init__()
        self.train = train
        self.transform = transform
    
    def _make_dataset(self):
        self.inputs, self.targets = self.data['x'], self.data['y']
        
    def __getitem__(self, index):
        # get corresponding inputs & targets pair
        inputs, targets = self.inputs[index], self.targets[index]
        inputs = PIL.Image.fromarray(np.array(inputs).reshape(28, 28), mode='L')
        
        # apply transformation
        if self.transform is not None:
            inputs = self.transform(inputs)
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
    
    