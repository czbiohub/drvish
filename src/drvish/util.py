#!/usr/bin/env python

import torch
import umap

import altair as alt
import numpy as np
import pandas as pd


def cos_annealing_factor(epoch, max_epoch, eta_min=1e-4):
    return eta_min + (1.0 - eta_min) * (1 - np.cos(np.pi * epoch / max_epoch)) / 2


def train(svi, train_loader, n_train, annealing_factor=1.0, use_cuda=False):
    af = torch.tensor(annealing_factor, dtype=torch.float, requires_grad=False)
    if use_cuda:
        af = af.cuda()

    # initialize loss accumulator
    epoch_loss = 0.

    # do a training epoch over each mini-batch x returned
    # by the data loader
    for i, xs in enumerate(train_loader):
        if use_cuda:
            xs = tuple(x.cuda() for x in xs)

        # do ELBO gradient and accumulate loss
        loss = svi.step(*xs, af)
        epoch_loss += loss

    # return epoch loss
    total_epoch_loss_train = epoch_loss / n_train

    return total_epoch_loss_train


def evaluate(svi, test_loader, n_test, use_cuda=False):
    af = torch.tensor(1.0, dtype=torch.float, requires_grad=False)
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


def plot_llk(train_elbo, test_elbo, test_int):
    x = np.arange(len(train_elbo))

    d = pd.concat(
        [
            pd.DataFrame({"x": x, "y": train_elbo, "run": "train"}),
            pd.DataFrame({"x": x[::test_int], "y": test_elbo, "run": "test"}),
        ]
    )

    return (
        alt.Chart(d)
        .encode(x="x:Q", y="y:Q", color="run:N")
        .mark_line(point=True)
        .properties(height=240, width=240)
    )


def plot_umap(z, d, lbls, n_neighbors=8):
    u = umap.UMAP(n_neighbors=n_neighbors, metric="cosine").fit_transform(z)

    log_d = np.log1p(d.sum(1))
    bot_d, top_d = np.percentile(log_d, (2.5, 97.5))

    c = alt.Chart(
        pd.DataFrame({"x": u[:, 0], "y": u[:, 1], "c": lbls, "log_d": log_d})
    ).properties(height=300, width=300)

    return alt.hconcat(
        c.mark_point(opacity=0.3).encode(
            x="x:Q", y="y:Q", color=alt.Color("c:N", legend=None)
        ),
        c.mark_point(opacity=0.8).encode(
            x="x:Q",
            y="y:Q",
            color=alt.Color(
                "log_d:Q",
                scale=alt.Scale(
                    scheme="viridis", clamp=True, nice=True, domain=(bot_d, top_d)
                ),
            ),
        ),
    )
