#!/usr/bin/env python

import collections
from typing import Sequence, Tuple

import torch
import torch.nn as nn

import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample


def _normal_prior(loc: float, scale: float, *sizes: int, use_cuda: bool = False):
    """Helper function to make a normal distribution on the correct device"""

    loc = loc * torch.ones(*sizes, requires_grad=False)
    scale = scale * torch.ones(*sizes, requires_grad=False)

    if use_cuda:
        loc = loc.cuda()
        scale = scale.cuda()

    return dist.Normal(loc, scale)


class FCLayers(nn.Module):
    """A helper class to build fully-connected layers for a neural network.

    :param n_input: The dimensionality of the input
    :param layers: Size of the intermediate layers
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self, layers: Sequence[int], dropout_rate: float = 0.1, use_bias: bool = True
    ):
        super().__init__()
        self.fc_layers = nn.Sequential(
            *(
                nn.Sequential(
                    nn.BatchNorm1d(n_in, eps=1e-3, momentum=0.01),
                    nn.ReLU(),
                    nn.Linear(n_in, n_out, bias=use_bias),
                    nn.Dropout(p=dropout_rate),
                )
                for n_in, n_out in zip(layers[:-1], layers[1:])
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward computation on ``x``.

        :param x: tensor of values with shape ``(n_input,)``
        :return: output of fully-connected layers
        """

        for layers in self.fc_layers:
            for layer in layers:
                if isinstance(layer, nn.BatchNorm1d) and x.dim() == 3:
                    x = torch.cat(
                        [(layer(slice_x)).unsqueeze(0) for slice_x in x], dim=0
                    )
                else:
                    x = layer(x)
        return x


class Encoder(nn.Module):
    r"""Encodes data of ``n_input`` dimensions into a latent space of ``n_output``
    dimensions using a fully-connected neural network of ``layers``.

    :param n_input: The dimensionality of the input (data space)
    :param n_output: The dimensionality of the output (latent space)
    :param layers: The number and size of fully-connected hidden layers
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self, n_input: int, n_output: int, layers: Sequence[int], dropout_rate: float
    ):
        super().__init__()

        self.encoder = FCLayers([n_input] + list(layers), dropout_rate=dropout_rate)
        self.mean_encoder = FCLayers([layers[-1], n_output], dropout_rate=0.0)
        self.var_encoder = FCLayers([layers[-1], n_output], dropout_rate=0.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""The forward computation for a single sample.
         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\) (clamped to \\( [-5, 5] \\))
         #. Samples a new value from an i.i.d. multivariate normal \\( \\sim N(q_m, \\mathbf{I}q_v) \\)

        :param x: tensor with shape (n_input,)
        :return: tensors of shape ``(n_latent,)`` for mean and var, and sample
        """
        # Parameters for latent distribution
        q = self.encoder(x)
        q_m = self.mean_encoder(q)

        # computational stability safeguard
        q_v = torch.exp(torch.clamp(self.var_encoder(q), -5, 5))

        return q_m, q_v


class NBDecoder(nn.Module):
    r"""Decodes data from latent space of ``n_input`` dimensions to ``n_output``
    dimensions using a fully-connected neural network of ``layers``.

    :param n_input: The dimensionality of the input (latent space)
    :param n_output: The dimensionality of the output (data space)
    :param layers: Size and number of fully-connected hidden layers
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self, n_input: int, n_output: int, layers: Sequence[int], dropout_rate: float
    ):
        super().__init__()
        self.px_decoder = FCLayers(
            layers=[n_input] + list(layers), dropout_rate=dropout_rate
        )

        # mean gamma
        self.scale_decoder = nn.Sequential(
            FCLayers(layers=[layers[-1], n_output], dropout_rate=0.0),
            nn.LogSoftmax(dim=-1),
        )

        # dispersion: here we only deal with gene-cell dispersion case
        self.r_decoder = FCLayers(layers=[layers[-1], n_output], dropout_rate=0.0)

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
        px = self.px_decoder(z)
        scale = self.scale_decoder(px)

        log_r = self.r_decoder(px)

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
        self.linear = PyroModule[nn.Linear](n_input, n_targets, bias=False)
        self.linear.weight = PyroSample(
            dist.Laplace(0.0, lam_scale).expand([n_targets, n_input]).to_event(2)
        )
        self.biases = PyroSample(
            dist.Normal(0.0, bias_scale).expand([n_conditions, n_targets]).to_event(2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.linear(x)[..., None, :] + self.biases).mean(0)
