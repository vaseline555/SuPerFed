import os
import gc
import PIL
import copy
import string
import torch
import torchvision

from utils import multiclass_symmetric_noisify

##############################################
# Dataset prepared in `torchvision.datasets` #
##############################################
class SplitDatset(torch.utils.data.Dataset):
    def __init__(self, args, indices=None, train=True, transform=None, target_transform=None, download=False):
        self.root = args.data_path
        self.dataset_name = args.dataset
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.indices = indices
        self.inputs, self.targets = self._make_dataset()

    def _make_dataset(self):
        # retrieve data
        raw = torchvision.datasets.__dict__[self.dataset_name](root=self.root, train=self.train, transform=self.transform, target_transform=self.target_transform, download=self.download)
        
        # get inputs and targets
        inputs, targets = raw.data, raw.targets
        if self.indices is not None:
            inputs, targets = inputs[self.indices], targets[self.indices]
        return inputs, targets

    def __getitem__(self, index):
        raise NotImplementedError('[ERROR] subclass should implement this!')

    def __len__(self):
        raise NotImplementedError('[ERROR] subclass should implement this!')

# MNIST
class MNISTDataset(SplitDatset):
    def __getitem__(self, index):
        # get corresponding inputs & targets pair
        inputs, targets = self.inputs[index], self.targets[index]
        inputs = PIL.Image.fromarray(inputs.numpy(), mode='L')
        
        # apply transformation
        if self.transform is not None:
            inputs = self.transform(inputs)

        if self.target_transform is not None:
            targets = self.target_transform(targets)
        return inputs, targets
    
    def __len__(self):
        return len(self.inputs)

# EMNIST
class EMNISTDataset(SplitDatset):
    def __init__(self, split='byclass', **kwargs):
        self.split = split
        super(EMNISTDataset, self).__init__(**kwargs)     

    def _make_dataset(self):
        # retrieve data
        raw = torchvision.datasets.__dict__[self.dataset_name](root=self.root, train=self.train, split=self.split, transform=self.transform, target_transform=self.target_transform, download=self.download)
        
        # get inputs and targets
        inputs, targets = raw.data, raw.targets
        if self.indices is not None:
            inputs, targets = inputs[self.indices], targets[self.indices]
        return inputs, targets
    
    def __getitem__(self, index):
        # get corresponding inputs & targets pair
        inputs, targets = self.inputs[index], self.targets[index]
        inputs = PIL.Image.fromarray(inputs.numpy(), mode='L')
        
        # apply transformation
        if self.transform is not None:
            inputs = self.transform(inputs)

        if self.target_transform is not None:
            targets = self.target_transform(targets)
        return inputs, targets
    
    def __len__(self):
        return len(self.inputs)

# CIFAR10, CIFAR100
class CIFARDataset(SplitDatset):
    def __init__(self, **kwargs):
        super(CIFARDataset, self).__init__(**kwargs)     
        
    def __getitem__(self, index):
        # get corresponding inputs & targets pair
        inputs, targets = self.inputs[index], self.targets[index]
        inputs = PIL.Image.fromarray(inputs, mode='RGB')
        
        # apply transformation
        if self.transform is not None:
            inputs = self.transform(inputs)

        if self.target_transform is not None:
            targets = self.target_transform(targets)
        return inputs, targets
    
    def __len__(self):
        return len(self.inputs)

# TinyImageNet prototype
class TinyImageNet(torchvision.datasets.ImageFolder):
    """Dataset for TinyImageNet-200.
    """
    base_folder = 'tiny-imagenet-200'
    zip_md5 = '90528d7ca1a48142e341f4ef8d21d0de'
    splits = ('train', 'val')
    filename = 'tiny-imagenet-200.zip'
    url = f'http://cs231n.stanford.edu/{filename}'
    
    def __init__(self, root, split='train', download=False, **kwargs):
        self.root = os.path.expanduser(root)
        self.split = torchvision.datasets.utils.verify_str_arg(split, "split", self.splits)

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')
        super().__init__(self.split_folder, **kwargs)
        
    @property
    def dataset_folder(self):
        return os.path.join(self.root, self.base_folder)

    @property
    def split_folder(self):
        return os.path.join(self.dataset_folder, self.split)

    def _check_exists(self):
        return os.path.exists(self.split_folder)

    def extra_repr(self):
        return "Split: {split}".format(**self.__dict__)
    
    def normalize_tin_val_folder_structure(self, path, images_folder='images', annotations_file='val_annotations.txt'):
        images_folder, annotations_file = os.path.join(path, images_folder), os.path.join(path, annotations_file)

        # exists
        if not os.path.exists(images_folder) and not os.path.exists(annotations_file):
            if not os.listdir(path):
                raise RuntimeError('Validation folder is empty.')
            return

        # parse the annotations
        with open(annotations_file) as file:
            for line in file:
                values = line.split()
                img, label = values[0], values[1]
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
        os.remove(annotations_file); os.sync()
    
    def download(self):
        if self._check_exists(): return
        torchvision.datasets.utils.download_and_extract_archive(self.url, self.root, filename=self.filename, remove_finished=True, md5=self.zip_md5)
        assert 'val' in self.splits
        self.normalize_tin_val_folder_structure(os.path.join(self.dataset_folder, 'val'))

# TinyImageNet
class TinyImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, args, indices=None, train=True, transform=None, target_transform=None, download=False):
        self.root = args.data_path
        self.dataset_name = args.dataset
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.indices = indices
        raw = TinyImageNet(root=self.root, split='train' if self.train else 'val', download=download)
        self.root = os.path.join(self.root, raw.base_folder)
            
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

        self.len = num_images
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
        
        self.len = len(list(self.val_img_to_class.keys()))
        self.class_to_target_idx = {classes[i]: i for i in range(len(sorted(list(set_of_classes))))}
        self.target_idx_to_class = {i: classes[i] for i in range(len(sorted(list(set_of_classes))))}

    def _make_dataset(self):
        inputs, targets = [], []
        if self.train:
            img_root_dir = os.path.join(self.root, 'train')
            list_of_dirs = [target for target in self.class_to_target_idx.keys()]
        else:
            img_root_dir = os.path.join(self.root, 'val')
            list_of_dirs = ['images']

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
                            
        if self.indices is not None:
            inputs, targets = inputs[self.indices], targets[self.indices]
        return inputs, targets
    
    def return_label(self, idx):
        return [self.class_to_label[self.target_idx_to_class[i.item()]] for i in idx]

    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        # get corresponding inputs & targets pair
        inputs, targets = self.inputs[index], self.targets[index]
        inputs = PIL.Image.open(inputs).convert('RGB')
        
        # apply transformation
        if self.transform is not None:
            inputs = self.transform(inputs)

        if self.target_transform is not None:
            targets = self.target_transform(targets)
        return inputs, targets
########################################
# Dataset prepared in `LEAF` benchmark #
########################################
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
        self.datasets = self._convert_to_dataset(self.merged_train, self.merged_test)
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
            [(train_dataset1, test_dataset1), (train_dataset2, test_dataset2), ...]
        """
        datasets = []
        for idx, user in tqdm(enumerate(merged_train['users']), desc=f'[INFO] ...make dataset (LEAF - {self.dataset_name.upper()})'):
            # copy dataset class prototype for each training set and test set
            tr_dset, te_dset = copy.deepcopy(self.dataset_class)(), copy.deepcopy(self.dataset_class)()
            setattr(tr_dset, 'train', True); setattr(te_dset, 'train', False)
            
            # set essential attributes
            tr_dset.identifier = user; te_dset.identifier = user
            tr_dset.data = merged_train['user_data'][user]; te_dset.data = merged_test['user_data'][user]
            tr_dset._make_dataset(); te_dset._make_dataset()
                                     
            # store as tuples
            datasets.append(tuple((tr_dset, te_dset)))
        return datasets
    
    def get_datasets(self):
        assert self.datasets is not None, '[ERROR] dataset is not constructed internally!'
        return self.datasets

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
            self.targets[idx] = [self.char_to_idx[char] for char in self.targets[idx]]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return np.array(self.inputs[index]), np.array(self.targets[index]), index

########################################
# Dataset for simulating a label noise #
########################################   
# EMNIST
class NoisyEMNISTDataset(EMNISTDataset):
    def __init__(self, args, noise_rate=0.2, **kwargs):
        super(EMNISTDataset, self).__init__(args, **kwargs)     
        self.noise_rate = noise_rate
        if self.train:
            self.noisy_targets, self.actual_noise_rate = multiclass_symmetric_noisify(targets=self.targets, noise_rate=self.noise_rate, seed=args.global_seed, num_classes=args.num_classes)
            self.noise_mask = np.transpose(self.noisy_targets) != np.transpose(self.targets)
        self.targets, self.original_targets = self.noisy_targets, self.targets

# CIFAR100
class NoisyCIFARDataset(CIFARDataset):
    def __init__(self, args, noise_rate=0.2, **kwargs):
        super(CIFARDataset, self).__init__(args, **kwargs)     
        self.noise_rate = noise_rate
        if self.train:
            self.noisy_targets, self.actual_noise_rate = multiclass_symmetric_noisify(targets=self.targets, noise_rate=self.noise_rate, seed=args.global_seed, num_classes=args.num_classes)
            self.noise_mask = np.transpose(self.noisy_targets) != np.transpose(self.targets)
        self.targets, self.original_targets = self.noisy_targets, self.targets