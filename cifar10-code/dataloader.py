from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import pdb
import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity


class CIFAR10(data.Dataset):

    def __init__(self, root, train=True,
                 transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.train_data = []
        self.train_label = []
        self.test_data = []
        self.test_label = []
        # now load the picked numpy arrays
        if self.train:
            for i in range(5):
                file = os.path.join(self.root, "data_batch_" + str(i+1))
                with open(file, 'rb') as fo:
                    dic = pickle.load(fo, encoding='bytes')
                    self.train_data.append(dic[b'data'])
                    self.train_label += dic[b'labels']
            self.train_data = np.vstack(self.train_data).reshape(-1, 3, 32, 32)
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
        else:
            file = os.path.join(self.root, "test_batch")
            with open(file, 'rb') as fo:
                dic = pickle.load(fo, encoding='bytes')
                self.test_data.append(dic[b'data'])
                self.test_label += dic[b'labels']
            self.test_data = np.vstack(self.test_data).reshape(-1, 3, 32, 32)
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):

        if self.train:
            img = self.train_data[index]
            target = self.train_label[index]
        else:
            img = self.test_data[index]
            target = self.test_label[index]

        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_label)
        else:
            return len(self.test_label)


    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
