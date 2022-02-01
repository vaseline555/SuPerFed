import torchvision
import torch
import PIL


##############################################
# Dataset prepared in `torchvision.datasets` #
##############################################
class VisionDataset(torch.utils.data.Dataset):
    def __init__(self, args, indices=None, train=True, transform=None, target_transform=None, download=False):
        self.root = args.data_path
        self.dataset_name = args.dataset
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.indices = indices
        self.inputs, self.targets = self._get_data()

    def _get_data(self):
        # retrieve data
        raw = torchvision.datasets.__dict__[self.dataset_name](root=self.root, train=self.train, transform=self.transform, target_transform=self.target_transform, download=self.download)
        
        # get inputs and targets
        inputs, targets = raw.data, raw.targets
        if self.indices is not None:
            inputs, targets = inputs[self.indices], targets[self.indices]
        return inputs, targets

    def __getitem__(self, index):
        raise NotImplementedError('[ERROR] Subclass should implement this!')

    def __len__(self):
        raise NotImplementedError('[ERROR] Subclass should implement this!')

# MNIST
class MNISTDataset(VisionDataset):
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
class EMNISTDataset(VisionDataset):
    def __init__(self, split='byclass', **kwargs):
        self.split = split
        super(EMNISTDataset, self).__init__(**kwargs)     

    def _get_data(self):
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
class CIFARDataset(VisionDataset):
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

########################################
# Dataset prepared in `LEAF` benchmark #
########################################


########################################
# Dataset for simulating a label noise #
########################################
# CIFAR100
class NoisyCIFARDataset(CIFARDataset):
    def __init__(self, args, noise_rate=0.2, **kwargs):
        super(CIFARDataset, self).__init__(args, **kwargs)     
        self.noise_rate = noise_rate
        if self.train:
            self.noisy_targets, self.actual_noise_rate = self.multiclass_symmetric_noisify(targets=self.targets, noise_rate=self.noise_rate, seed=args.global_seed, num_classes=args.num_classes)
            self.noise_mask = np.transpose(self.noisy_targets) != np.transpose(self.targets)
        self.targets, self.original_targets = self.noisy_targets, self.targets

    def multiclass_noisify(self, targets, P, seed):
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

    def multiclass_symmetric_noisify(self, targets, noise_rate, seed, num_classes):
        P = np.ones((num_classes, num_classes))
        P *= (noise_rate / (num_classes - 1)) 

        if noise_rate > 0.0:
            # 0 -> 1
            P[0, 0] = 1. - noise_rate
            for i in range(1, num_classes-1):
                P[i, i] = 1. - noise_rate
            P[num_classes - 1, num_classes - 1] = 1. - noise_rate

            noisy_targets = self.multiclass_noisify(targets, P, seed)
            actual_noise = (np.array(noisy_targets).flatten() != np.array(targets).flatten()).mean()
            assert actual_noise > 0.0
        return np.array(noisy_targets).flatten(), actual_noise