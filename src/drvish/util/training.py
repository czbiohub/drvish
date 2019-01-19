#!/usr/bin/env python

import typing

import numpy as np

import torch

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler

import pyro.infer as pi


__all__ = ["split_dataset", "evaluate_step", "train", "evaluate"]


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


def evaluate_step(
    eval_fn: typing.Callable,
    data_loader: DataLoader,
    loss_scaling_factor: int,
    annealing_factor: float = 1.0,
    use_cuda: bool = False,
):
    """
    Go through a dataset and return the total loss

    :param eval_fn: typically either SVI.step (for training)
                    or SVI.evaluate_loss (for testing)
    :param data_loader:
    :param loss_scaling_factor:
    :param annealing_factor:
    :param use_cuda:
    :return:
    """
    af = torch.tensor(annealing_factor, requires_grad=False)
    if use_cuda:
        af = af.cuda()

    # initialize loss accumulator
    epoch_loss = 0.

    # do a training epoch over each mini-batch x returned by the data loader
    for _, xs in enumerate(data_loader):
        if use_cuda:
            xs = tuple(x.cuda() for x in xs)

        # do ELBO gradient and accumulate loss
        epoch_loss += eval_fn(*xs, af)

    # return epoch loss
    total_epoch_loss = epoch_loss / loss_scaling_factor

    return total_epoch_loss


def train(
    svi: pi.SVI,
    train_loader: DataLoader,
    n_train: int,
    annealing_factor: float = 1.0,
    use_cuda: bool = False,
):
    return evaluate_step(
        svi.step,
        data_loader=train_loader,
        loss_scaling_factor=n_train,
        annealing_factor=annealing_factor,
        use_cuda=use_cuda,
    )


def evaluate(svi: pi.SVI, test_loader: DataLoader, n_test: int, use_cuda: bool = False):
    return evaluate_step(
        svi.evaluate_loss,
        data_loader=test_loader,
        loss_scaling_factor=n_test,
        annealing_factor=1.0,
        use_cuda=use_cuda,
    )
