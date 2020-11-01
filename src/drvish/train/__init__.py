#!/usr/bin/env python

import itertools
from typing import Callable, List, Tuple, Union

import numpy as np

import torch
from torch.utils.data import DataLoader

import pyro.infer as pi
import pyro.optim as po

from drvish.train.aggmo import AggMo


PyroAggMo = (
    lambda _Optim: lambda optim_args, clip_args=None: po.PyroOptim(
        _Optim, optim_args, clip_args
    )
)(AggMo)


def evaluate_epoch(
    eval_fn: Callable, data_loader: DataLoader, use_cuda: bool = False,
) -> float:
    """
    Go through a dataset and return the total loss

    :param eval_fn: typically either SVI.step (for training)
                    or SVI.evaluate_loss (for testing)
    :param data_loader: the dataset to iterate through and evaluate
    :param use_cuda: if compute is on GPU
    :return:
    """

    # initialize loss accumulator
    epoch_loss = 0.0

    # do a training epoch over each mini-batch x returned by the data loader
    for xs in data_loader:
        if use_cuda:
            xs = (x.cuda() for x in xs)

        # do ELBO gradient and accumulate loss
        epoch_loss += eval_fn(*xs)

    # return epoch loss
    total_epoch_loss = epoch_loss / len(data_loader)

    return total_epoch_loss


def train_until_plateau(
    svi: pi.SVI,
    scheduler: po.PyroLRScheduler,
    training_data: DataLoader,
    validation_data: DataLoader,
    min_cycles: int = 3,
    threshold: float = 0.01,
    use_cuda: bool = False,
    verbose: bool = False,
) -> Tuple[List[float], List[float]]:
    """Train a model with cosine scheduling until validation loss stabilizes. This
    function uses CosineWithRestarts to train until the learning rate stops improving.

    :param svi: pyro SVI instance, that is using a LR scheduler
    :param scheduler: the pyro LRScheduler instance
    :param training_data: Training dataset. Should produce tuples of Tensors, all but
                          the last are considered to be input and the last is the target
    :param validation_data: Validation dataset in the same format
    :param min_cycles: Minimum number of cycles to run before checking for convergence
    :param threshold: Tolerance threshold for calling convergence
    :param use_cuda: Whether to use the GPU for training
    :param verbose: Print training progress to stdout
    :return: Lists of training and validation loss and correlation values
    """

    assert 0.0 <= threshold < 1.0

    train_loss = []
    val_loss = []

    best = np.inf
    rel_epsilon = 1.0 - threshold
    neg_epsilon = 1.0 + threshold
    cycle = 0

    for epoch in itertools.count():
        train_loss.append(evaluate_epoch(svi.step, training_data, use_cuda=use_cuda))
        val_loss.append(
            evaluate_epoch(svi.evaluate_loss, validation_data, use_cuda=use_cuda)
        )

        scheduler.step(epoch)
        if any(opt.T_cur == opt.T_0 - 1 for opt in scheduler.optim_objs.values()):
            if verbose:
                print(
                    f"[epoch {epoch:03d}]  average training loss: {train_loss[-1]:.5f}"
                )
            cycle += 1

            if 0 <= val_loss[-1] < best * rel_epsilon:
                best = val_loss[-1]
            elif 0 > val_loss[-1] and val_loss[-1] < best * neg_epsilon:
                best = val_loss[-1]
            elif cycle >= min_cycles:
                break

    return train_loss, val_loss
