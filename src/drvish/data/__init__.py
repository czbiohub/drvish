#!/usr/bin/env python

import numpy as np

import torch

from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler, BatchSampler


class TensorTargetDataset(TensorDataset):
    def __init__(self, *tensors: torch.Tensor):
        super().__init__(*tensors[:-1])
        self.target = tensors[-1]


class StratifiedSubsetSampler(BatchSampler):
    """
    A batch sampler that generates class-balanced mini-batches of samples, based on the
    input labels. This sampler generates each batch randomly, and does not guarantee
    that a given sample will appear only once per iteration, or will appear at all. For
    """

    def __init__(
        self, indices: np.ndarray, class_vector: np.ndarray, batch_size: int,
    ):
        """
        :param indices: indices to provide samples from
        :param class_vector: a vector of class labels
        :param batch_size: number of samples to provide per iteration
        """
        super().__init__(None, batch_size, False)

        assert batch_size < len(class_vector)

        self.indices = indices
        self.n_splits = class_vector.shape[0] // batch_size
        self.class_vector = class_vector

    def __iter__(self):
        splitter = StratifiedShuffleSplit(
            n_splits=self.n_splits, test_size=self.batch_size
        )
        for _, test_index in splitter.split(self.indices, self.class_vector):
            yield self.indices[test_index]

    def __len__(self):
        return self.n_splits


class DataTargetLoader(DataLoader):
    def __iter__(self):
        for indices in iter(self.batch_sampler):
            batch = self.collate_fn([self.dataset[i] for i in indices])
            batch.append(self.dataset.target)

            yield batch


def split_dataset(
    *xs: torch.Tensor,
    batch_size: int,
    train_p: float,
    dataloader_cls: Type[DataLoader] = DataLoader,
):
    """
    Split a dataset of tensors into training and validation sets with a given split.

    :param xs: tensor(s) of data to split into two parts
    :param batch_size: number of samples to provide in a single iteration
    :param train_p: proportion of the data to put in the training set
    :param dataloader_cls: class constructor for data loader
    :return: two DataLoaders, one for training and another for validation
    """
    n_cells = xs[0].shape[0]

    example_indices = np.random.permutation(n_cells)
    n_train = int(train_p * n_cells)

    dataset = TensorDataset(*xs)

    data_loader_train = dataloader_cls(
        dataset=dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(example_indices[:n_train]),
    )

    data_loader_validation = dataloader_cls(
        dataset=dataset,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(example_indices[n_train:]),
    )

    return data_loader_train, data_loader_validation


def split_labeled_dataset(
    *xs: torch.Tensor,
    labels: np.ndarray,
    target: torch.Tensor,
    batch_size: int,
    train_p: float,
):
    """
    Split a labeled dataset of tensors into training and validation sets, and provide
    stratified samples over the two sets so that they are always class-balanced.

    :param xs: tensor(s) of data to split into two parts
    :param labels: a label (int) for each sample that indicates the class
    :param target: tensor of response data
    :param batch_size: number of samples to provide in a single iteration
    :param train_p: proportion of the data to put in the training set
    :return: two DataLoaders, one for training and another for validation
    """
    n_cells = xs[0].shape[0]

    example_indices = np.random.permutation(n_cells)
    example_labels = labels[example_indices]
    n_train = int(train_p * n_cells)

    dataset = TensorTargetDataset(*xs, torch.from_numpy(labels), target)

    data_loader_train = DataTargetLoader(
        dataset=dataset,
        batch_sampler=StratifiedSubsetSampler(
            example_indices[:n_train], example_labels[:n_train], batch_size
        ),
    )

    data_loader_validation = DataTargetLoader(
        dataset=dataset,
        batch_sampler=StratifiedSubsetSampler(
            example_indices[n_train:], example_labels[n_train:], batch_size
        ),
    )

    return data_loader_train, data_loader_validation
