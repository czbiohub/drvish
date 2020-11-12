#!/usr/bin/env python

import collections
from typing import Sequence, Tuple

import torch
import torch.nn as nn
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroParam, PyroSample


def make_fc(dims: Sequence[int]):
    layers = []

    for in_dim, out_dim in zip(dims, dims[1:]):
        layers.append(nn.BatchNorm1d(in_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(in_dim, out_dim))

    return nn.Sequential(*layers)


# Splits a tensor in half along the final dimension
def split_in_half(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return t.reshape(t.shape[:-1] + (2, -1)).unbind(-2)


class Encoder(nn.Module):
    r"""Encodes data of ``n_input`` dimensions into a latent space of ``n_output``
    dimensions using a fully-connected neural network of ``layers``.

    :param n_input: The dimensionality of the input (data space)
    :param n_output: The dimensionality of the output (latent space)
    :param layers: The number and size of fully-connected hidden layers
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(self, n_input: int, n_output: int, layers: Sequence[int]):
        super().__init__()

        self.fc = make_fc([n_input] + list(layers) + [2 * n_output])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""The forward computation for a single sample.
         #. Transform input counts with sqrt
         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\) (clamped to \\( [-5, 5] \\))
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim N(q_m, \\mathbf{I}q_v) \\)

        :param x: tensor of counts with shape (n_input,)
        :return: tensors of shape ``(n_latent,)`` for mean and var, and sample
        """
        q_m, q_v = split_in_half(self.fc(torch.sqrt(x)))
        q_v = nn.functional.softplus(q_v)

        return q_m, q_v


class NBDecoder(nn.Module):
    r"""Decodes data from latent space of ``n_input`` dimensions to ``n_output``
    dimensions using a fully-connected neural network of ``layers``.

    :param n_input: The dimensionality of the input (latent space)
    :param n_output: The dimensionality of the output (data space)
    :param layers: Size and number of fully-connected hidden layers
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(self, n_input: int, n_output: int, layers: Sequence[int]):
        super().__init__()

        self.fc = make_fc([n_input] + list(layers) + [2 * n_output])

    def forward(
        self, z: torch.Tensor, library: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression

        *Note* This parameterization is different from the one in the scVI package

        :param z: tensor with shape ``(n_input,)``
        :param library: library size
        :return: parameters for the NB distribution of expression
        """

        # The decoder returns values for the parameters of the NB distribution
        scale, log_r = split_in_half(self.fc(z))
        scale = nn.functional.log_softmax(scale, dim=-1)

        logit = library + scale - log_r

        return log_r, logit


class LinearMultiBias(PyroModule):
    """Module that computes a multi-target logistic function based on a learnable
    vector of weights on ``n_input`` features and a vector of biases for different
    conditions. Note that everything is computed in log space so the result is a linear
    module.

    :param n_input: The dimensionality of the input (latent space)
    :param n_targets: The dimensionality of the output (e.g. number of drugs)
    :param n_conditions: The dimensionality of the bias vector (e.g. number of doses)
    :param lam_scale: Scale for the Laplace prior on weights
    :param bias_scale: Scale for the Normal prior on biases
    """

    def __init__(
        self,
        n_input: int,
        n_targets: int,
        n_conditions: int,
        lam_scale: float,
        bias_scale: float,
    ):
        super().__init__()

        # priors
        self.weight = PyroSample(
            dist.Laplace(0.0, lam_scale).expand([n_input, n_targets]).to_event(2)
        )
        self.bias = PyroSample(
            dist.Normal(0.0, bias_scale).expand([n_conditions, n_targets]).to_event(2)
        )

        # parameters for guide
        self.weight_loc = nn.Parameter(
            torch.nn.init.xavier_uniform_(torch.Tensor(n_input, n_targets))
        )
        self.weight_scale = PyroParam(
            torch.ones(n_input, n_targets), constraint=constraints.positive
        )
        self.bias_loc = nn.Parameter(
            torch.nn.init.xavier_uniform_(torch.Tensor(n_conditions, n_targets))
        )
        self.bias_scale = PyroParam(
            torch.ones(n_conditions, n_targets), constraint=constraints.positive
        )

    @staticmethod
    def logit_mean_sigmoid(
        x: torch.Tensor, weights: torch.Tensor, bias: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Computes a logistic regression on the input, averages over the class labels,
        and then converts the results back into log space with logit function.

        :param x: input tensor (n_cells x n_latent)
        :param weights: regression weights (n_latent, n_targets)
        :param bias: offsets for each condition (n_conditions, n_targets)
        :param labels: label for each cell (n_cells,)
        :return: average dose response in log space (n_classes, n_conditions, n_targets)
        """
        response = torch.sigmoid((x @ weights)[..., None, :] + bias)

        labels = labels.view(-1, 1, 1).expand(-1, *response.size()[1:])
        unique_labels, labels_count = labels.unique(dim=0, return_counts=True)

        res = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(
            0, labels, response
        )
        return torch.logit(res / labels_count.float().view(-1, 1, 1), eps=1e-5)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.logit_mean_sigmoid(x, self.weight, self.bias, labels)

    def calc_response(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """This is the same as the forward method but will not sample weights"""
        return self.logit_mean_sigmoid(x, self.weight_loc, self.bias_loc, labels)

    def sample_guide(self):
        pyro.sample(
            "weight", dist.Normal(self.weight_loc, self.weight_scale).to_event(2)
        )
        pyro.sample(
            "bias", dist.Normal(self.bias_loc, self.bias_scale).to_event(2)
        )
