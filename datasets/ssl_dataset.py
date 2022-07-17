import torch

from .data_utils import split_ssl_data, sample_labeled_data
from .dataset import BasicDataset, BaseDataset
from collections import Counter
import torchvision
import numpy as np
from torchvision import transforms
import json
import os
from .augmentation.randaugment import RandAugment

from torch.utils.data import sampler, DataLoader
from torch.utils.data.sampler import BatchSampler
import torch.distributed as dist

import copy


mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]


class TransformMultipleWithOriginal(object):
    def __init__(self, args, mean, std, crop_size, augmented_num=10):
        self.args = args
        self.augmented_num = augmented_num

        if self.args.adjust_augmentation == "no":

            self.weak = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.RandomCrop(
                                                crop_size, padding=4, padding_mode='reflect'),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean, std)])

            self.strong = copy.deepcopy(self.weak)
            # self.strong.transforms.insert(0, RandAugment(3, 5))
            self.strong.transforms.insert(0, RandAugment(2, 10))

        elif self.args.adjust_augmentation == "yes":
            self.weak = transforms.Compose([transforms.RandomHorizontalFlip(),
                                            transforms.RandomCrop(
                                                crop_size, padding=4, padding_mode='reflect'),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean, std)])

            self.strong = copy.deepcopy(self.weak)
            self.strong.transforms.insert(
                0, RandAugment(args.randaug_n, args.randaug_m))

        self.original = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean, std)])

    def __call__(self, x):
        return [self.weak(x) for _ in range(self.augmented_num)], [self.strong(x) for _ in range(self.augmented_num)], self.original(x)


class SSLDataset:
    """
    updated by Allen :D
    SSL_Dataset class gets dataset from torchvision.datasets,
    separates labeled and unlabeled data,
    and return BasicDataset: torch.utils.data.Dataset (see datasets.dataset.py)
    """

    def __init__(self,
                 args,
                 alg='fixmatch',
                 name='cifar10',
                 num_classes=10,
                 data_dir='./data'):
        """
        Args
            alg: SSL algorithms
            name: name of dataset in torchvision.datasets (cifar10, cifar100, svhn, stl10)
            train: True means the dataset is training dataset (default=True)
            num_classes: number of label classes
            data_dir: path of directory, where data is downloaed or stored.
        """
        self.args = args
        self.alg = alg
        self.name = name

        self.num_classes = num_classes
        self.data_dir = data_dir
        # crop_size = 96 if self.name.upper() == 'STL10' else 224 if self.name.upper() == 'IMAGENET' else 32
        crop_size = 32
        self.train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                   transforms.RandomCrop(
            crop_size, padding=4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
        self.weak_transform = self.train_transform
        self.test_transform = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize(mean, std)])
        self.strong_transform = copy.deepcopy(self.train_transform)
        # self.strong_transform.transforms.insert(0, RandAugment(3, 5))
        self.strong_transform.transforms.insert(0, RandAugment(2, 10))
        self.attack_transform = TransformMultipleWithOriginal(
            args, mean=mean, std=std, crop_size=crop_size, augmented_num=args.augmented_num)

    def get_data(self):
        """
        get_data returns data (images) and targets (labels)
        shape of data: B, H, W, C
        shape of labels: B,
        """

        if self.name.upper() == "CIFAR10":
            d_train = torchvision.datasets.CIFAR10(root=self.data_dir,
                                                   train=True,
                                                   transform=None,
                                                   download=True)
            d_test = torchvision.datasets.CIFAR10(root=self.data_dir,
                                                  train=False,
                                                  transform=None,
                                                  download=True)
            dataset = d_train + d_test

        elif self.name.upper() == "CIFAR100":
            d_train = torchvision.datasets.CIFAR100(root=self.data_dir,
                                                    train=True,
                                                    transform=None,
                                                    download=True)
            d_test = torchvision.datasets.CIFAR100(root=self.data_dir,
                                                   train=False,
                                                   transform=None,
                                                   download=True)
            dataset = d_train + d_test

        elif self.name.upper() == "SVHN":
            d_train = torchvision.datasets.SVHN(root=self.data_dir,
                                                split="train",
                                                transform=None,
                                                download=True)
            d_test = torchvision.datasets.SVHN(root=self.data_dir,
                                               split="test",
                                               transform=None,
                                               download=True)
            dataset = d_train + d_test

        target_train, target_train_labeled, target_test, shadow_train, shadow_train_labeled, shadow_test = self.prepare_dataset(
            dataset, self.args.num_labels)
        dataset_list = [target_train, target_train_labeled,
                        target_test, shadow_train, shadow_train_labeled, shadow_test]
        return dataset_list

    def transform_dataset_list(self, dataset_list):
        target_train, target_train_labeled, target_test, shadow_train, shadow_train_labeled, shadow_test = dataset_list
        num_classes = self.num_classes
        if self.alg == "fullysupervised":
            target_train_lb = BaseDataset(self.alg, target_train + target_train_labeled, num_classes, transform=self.weak_transform,
                                          is_ulb=False, strong_transform=self.strong_transform, onehot=False)
            target_test = BaseDataset(self.alg, target_test, num_classes, transform=self.test_transform,
                                      is_ulb=False, strong_transform=self.strong_transform, onehot=False)
            shadow_train_lb = BaseDataset(self.alg, shadow_train + shadow_train_labeled, num_classes, transform=self.weak_transform,
                                          is_ulb=False, strong_transform=self.strong_transform, onehot=False)
            shadow_test = BaseDataset(self.alg, shadow_test, num_classes, transform=self.test_transform,
                                      is_ulb=False, strong_transform=self.strong_transform, onehot=False)
            transform_dataset_list = [
                None, target_train_lb, target_test, None, shadow_train_lb, shadow_test]
        elif self.alg == "attack":
            target_train_ulb = BaseDataset(self.alg, target_train, num_classes, transform=self.attack_transform,
                                           is_ulb=False, strong_transform=self.strong_transform, onehot=False)
            target_train_lb = BaseDataset(self.alg, target_train_labeled, num_classes, transform=self.attack_transform,
                                          is_ulb=False, strong_transform=self.strong_transform, onehot=False)
            target_test = BaseDataset(self.alg, target_test, num_classes, transform=self.attack_transform,
                                      is_ulb=False, strong_transform=self.strong_transform, onehot=False)
            shadow_train_ulb = BaseDataset(self.alg, shadow_train, num_classes, transform=self.attack_transform,
                                           is_ulb=False, strong_transform=self.strong_transform, onehot=False)
            shadow_train_lb = BaseDataset(self.alg, shadow_train_labeled, num_classes, transform=self.attack_transform,
                                          is_ulb=False, strong_transform=self.strong_transform, onehot=False)
            shadow_test = BaseDataset(self.alg, shadow_test, num_classes, transform=self.attack_transform,
                                      is_ulb=False, strong_transform=self.strong_transform, onehot=False)
            transform_dataset_list = [target_train_ulb, target_train_lb,
                                      target_test, shadow_train_ulb, shadow_train_lb, shadow_test]
        else:
            target_train_ulb = BaseDataset(self.alg, target_train, num_classes, transform=self.weak_transform,
                                           is_ulb=True, strong_transform=self.strong_transform, onehot=False)
            target_train_lb = BaseDataset(self.alg, target_train_labeled, num_classes, transform=self.weak_transform,
                                          is_ulb=False, strong_transform=self.strong_transform, onehot=False)
            target_test = BaseDataset(self.alg, target_test, num_classes, transform=self.test_transform,
                                      is_ulb=False, strong_transform=self.strong_transform, onehot=False)
            shadow_train_ulb = BaseDataset(self.alg, shadow_train, num_classes, transform=self.weak_transform,
                                           is_ulb=True, strong_transform=self.strong_transform, onehot=False)
            shadow_train_lb = BaseDataset(self.alg, shadow_train_labeled, num_classes, transform=self.weak_transform,
                                          is_ulb=False, strong_transform=self.strong_transform, onehot=False)
            shadow_test = BaseDataset(self.alg, shadow_test, num_classes, transform=self.test_transform,
                                      is_ulb=False, strong_transform=self.strong_transform, onehot=False)
            transform_dataset_list = [target_train_ulb, target_train_lb,
                                      target_test, shadow_train_ulb, shadow_train_lb, shadow_test]
        return transform_dataset_list

    def prepare_dataset(self, dataset, n_labeled):
        length = len(dataset)
        each_length = length//4
        torch.manual_seed(0)
        target_train, target_train_labeled, target_test, shadow_train, shadow_train_labeled, shadow_test, _ = torch.utils.data.random_split(
            dataset, [each_length - n_labeled, n_labeled, each_length, each_length - n_labeled, n_labeled, each_length, len(dataset) - (each_length*4)])

        # print(target_train)
        # for i in range(10):
        #     print(target_train[i][1],)
        # print(len(target_train), len(target_train_labeled))
        return target_train, target_train_labeled, target_test, shadow_train, shadow_train_labeled, shadow_test
