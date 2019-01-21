#!/usr/bin/env python

import typing

import torch

from torch.utils.data import DataLoader

import pyro.infer as pi


def cos_annealing_factor(epoch: int, max_epoch: int, eta_min: float = 1e-4):
    return eta_min + (1. - eta_min) * (1. - np.cos(np.pi * epoch / max_epoch)) / 2.


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
