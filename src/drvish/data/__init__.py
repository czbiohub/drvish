#!/usr/bin/env python

import numpy as np

import torch

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler, Sampler


class TensorTargetDataset(TensorDataset):
    def __init__(self, *tensors: torch.Tensor):
        super().__init__(*tensors[:-1])
        self.class_idx = torch.arange(tensors[0].size(1), requires_grad=False)
        self.target = tensors[-1]

    def __getitem__(self, index):
        return tuple(tensor[index, self.class_idx, ...] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(1)


class StratifiedSubset2DSampler(Sampler):
    def __init__(self, indices, n_classes: int):
        super().__init__(None)
        self.indices = indices
        self.n_classes = n_classes

    def __iter__(self):
        return zip(
            *(
                tuple(self.indices[i] for i in torch.randperm(len(self.indices)))
                for _ in range(self.n_classes)
            )
        )

    def __len__(self):
        return len(self.indices)


class DataLoader2D(DataLoader):
    def __iter__(self):
        for indices in iter(self.batch_sampler):
            batch = self.collate_fn([self.dataset[i] for i in indices])
            batch.append(self.dataset.target)

            yield batch


def split_dataset(
    *xs: torch.Tensor, batch_size: int, train_p: float, use_cuda: bool = False
):
    n_cells = xs[0].shape[0]

    example_indices = np.random.permutation(n_cells)
    n_train = int(train_p * n_cells)

    dataset = TensorDataset(*xs)

    data_loader_train = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        pin_memory=use_cuda,
        sampler=SubsetRandomSampler(example_indices[:n_train]),
    )

    data_loader_test = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        pin_memory=use_cuda,
        sampler=SubsetRandomSampler(example_indices[n_train:]),
    )

    return data_loader_train, data_loader_test


def split_2d_dataset(
    *xs: torch.Tensor, batch_size: int, train_p: float, use_cuda: bool = False
):
    n_cells_per_class = xs[0].shape[0]
    n_classes = xs[0].shape[1]

    example_indices = np.random.permutation(n_cells_per_class)
    n_train = int(train_p * n_cells_per_class)

    dataset = TensorTargetDataset(*xs)

    data_loader_train = DataLoader2D(
        dataset=dataset,
        batch_size=batch_size,
        pin_memory=use_cuda,
        sampler=StratifiedSubset2DSampler(example_indices[:n_train], n_classes),
    )

    data_loader_test = DataLoader2D(
        dataset=dataset,
        batch_size=batch_size,
        pin_memory=use_cuda,
        sampler=StratifiedSubset2DSampler(example_indices[n_train:], n_classes),
    )

    return data_loader_train, data_loader_test
