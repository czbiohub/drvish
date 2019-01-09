#!/usr/bin/env python

import numpy as np

import torch

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler

import pyro.infer as pi


__all__ = ["split_dataset", "train", "evaluate"]


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


def train(
    svi: pi.SVI,
    train_loader: DataLoader,
    n_train: int,
    annealing_factor: float = 1.0,
    use_cuda: bool = False,
):
    af = torch.tensor(annealing_factor, requires_grad=False)
    if use_cuda:
        af = af.cuda()

    # initialize loss accumulator
    epoch_loss = 0.

    # do a training epoch over each mini-batch x returned by the data loader
    for i, xs in enumerate(train_loader):
        if use_cuda:
            xs = tuple(x.cuda() for x in xs)

        # do ELBO gradient and accumulate loss
        loss = svi.step(*xs, af)
        epoch_loss += loss

    # return epoch loss
    total_epoch_loss_train = epoch_loss / n_train

    return total_epoch_loss_train


def evaluate(svi: pi.SVI, test_loader: DataLoader, n_test: int, use_cuda: bool = False):
    af = torch.tensor(1.0, requires_grad=False)
    if use_cuda:
        af = af.cuda()

    # initialize loss accumulator
    test_loss = 0.

    # compute the loss over the entire test set
    for _, xs in enumerate(test_loader):
        if use_cuda:
            xs = tuple(x.cuda() for x in xs)

        # compute ELBO estimate and accumulate loss
        test_loss += svi.evaluate_loss(*xs, af)

    total_epoch_loss_test = test_loss / n_test

    return total_epoch_loss_test
