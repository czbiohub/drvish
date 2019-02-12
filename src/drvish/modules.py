#!/usr/bin/env python


import collections

from typing import Sequence, Tuple

import torch
import torch.nn as nn

from torch.nn.functional import softplus
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine


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

    def __init__(self, n_input: int, layers: Sequence[int], dropout_rate: float = 0.1):
        super(FCLayers, self).__init__()
        layers_dim = [n_input] + layers
        self.fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "Layer_{}".format(i),
                        nn.Sequential(
                            nn.Linear(n_in, n_out),
                            nn.BatchNorm1d(n_out, eps=1e-3, momentum=0.01),
                            nn.ReLU(),
                            nn.Dropout(p=dropout_rate),
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(
                        zip(layers_dim[:-1], layers_dim[1:])
                    )
                ]
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
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (data space)
    :param n_output: The dimensionality of the output (latent space)
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
    ):
        super(Encoder, self).__init__()
        self.encoder = FCLayers(
            n_input=n_input, layers=[n_hidden] * n_layers, dropout_rate=dropout_rate
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

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
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (latent space)
    :param n_output: The dimensionality of the output (data space)
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :param dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
    ):
        super(NBDecoder, self).__init__()
        self.px_decoder = FCLayers(
            n_input=n_input, layers=[n_hidden] * n_layers, dropout_rate=dropout_rate
        )

        # mean gamma
        self.scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output), nn.LogSoftmax(dim=-1)
        )

        # dispersion: here we only deal with gene-cell dispersion case
        self.r_decoder = nn.Linear(n_hidden, n_output)

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


class LinearMultiBias(nn.Module):
    """Module that compute a multi-target logistic function based on a learnable
    vector of weights on ``n_input`` features and a vector of biases for different
    conditions.

    :param n_input: The dimensionality of the input (latent space)
    :param n_drugs: The dimensionality of the output (number of targets)
    :param n_conditions: The dimensionality of the bias vector)
    """

    def __init__(self, n_input: int, n_drugs: int, n_conditions: int):
        super(LinearMultiBias, self).__init__()
        self.linear = nn.Linear(n_input, n_drugs, bias=False)
        self.biases = nn.Parameter(torch.Tensor(n_conditions, n_drugs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.linear(x)[..., None, :] + self.biases).mean(0)


class NBVAE(nn.Module):
    r"""Variational auto-encoder model with negative binomial loss.

    :param n_input: Number of input genes
    :param n_latent: Dimensionality of the latent space
    :param n_layers: Number of hidden layers used for encoder and decoder NNs
    :param n_hidden: Number of nodes per hidden layer
    :param lib_loc: Mean for prior distribution on library scaling factor
    :param lib_scale: Scale for prior distribution on library scaling factor
    :param dropout_rate: Dropout rate for neural networks
    :param use_cuda: if True, copy parameters into GPU memory
    :param eps: value to add to NB count parameter for numerical stability
    """

    def __init__(
        self,
        *,
        n_input: int,
        n_latent: int = 8,
        n_layers: int = 3,
        n_hidden: int = 64,
        lib_loc: float = 7.5,
        lib_scale: float = 0.5,
        dropout_rate: float = 0.1,
        use_cuda: bool = False,
        eps: float = 1e-6,
    ):
        super(NBVAE, self).__init__()
        self.encoder = Encoder(n_input, n_latent, n_layers, n_hidden, dropout_rate)
        self.l_encoder = Encoder(n_input, 1, n_layers, n_hidden, dropout_rate)
        self.decoder = NBDecoder(n_latent, n_input, n_layers, n_hidden, dropout_rate)

        self.eps = torch.tensor(eps, requires_grad=False)

        self.z_prior = _normal_prior(0.0, 1.0, 1, n_latent, use_cuda=use_cuda)
        self.l_prior = _normal_prior(lib_loc, lib_scale, 1, 1, use_cuda=use_cuda)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()

        self.use_cuda = use_cuda
        self.n_latent = n_latent

    def model(self, x: torch.Tensor, af: torch.Tensor):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)

        with pyro.plate("data"):
            with poutine.scale(scale=af):
                z = pyro.sample(
                    "latent",
                    self.z_prior.expand(
                        torch.Size((x.size(0), self.n_latent))
                    ).to_event(1),
                )
                l = pyro.sample(
                    "library",
                    self.l_prior.expand(torch.Size((x.size(0), 1))).to_event(1),
                )

            log_r, logit = self.decoder.forward(z, l)

            pyro.sample(
                "obs",
                dist.NegativeBinomial(
                    total_count=torch.exp(log_r) + self.eps,
                    logits=logit,
                    validate_args=True,
                ).to_event(1),
                obs=x,
            )

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x: torch.Tensor, af: torch.Tensor):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        pyro.module("l_encoder", self.l_encoder)

        with pyro.plate("data"):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder.forward(x)
            l_loc, l_scale = self.l_encoder.forward(x)

            # sample the latent code z
            with poutine.scale(scale=af):
                pyro.sample(
                    "latent",
                    dist.Normal(z_loc, z_scale, validate_args=True).to_event(1),
                )
                pyro.sample(
                    "library",
                    dist.Normal(l_loc, l_scale, validate_args=True).to_event(1),
                )


class DRNBVAE(nn.Module):
    """A variational auto-encoder module with negative binomial loss and a dose
    reponse module attached.

    :param n_input: Number of input genes
    :param n_classes: Number of different cell types used for drug response
    :param n_drugs: Number of different drug targets in response data
    :param n_conditions: Number of conditions for each target
    :param n_latent: Dimensionality of the latent space
    :param n_layers: Number of hidden layers used for encoder and decoder NNs
    :param n_hidden: Number of nodes per hidden layer
    :param lib_loc: Mean for prior distribution on library scaling factor
    :param lib_scale: Scale for prior distribution on library scaling factor
    :param lam_scale: Scaling factor for prior on regression weights
    :param bias_scale: Scaling factor for prior on regression biases
    :param dropout_rate: Dropout rate for neural networks
    :param use_cuda: if True, copy parameters into GPU memory
    :param eps: value to add for numerical stability
    """

    def __init__(
        self,
        *,
        n_input: int,
        n_classes: int,
        n_drugs: int,
        n_conditions: int,
        n_latent: int = 8,
        n_layers: int = 3,
        n_hidden: int = 64,
        lib_loc: float = 7.5,
        lib_scale: float = 0.5,
        lam_scale: float = 5.0,
        bias_scale: float = 10.0,
        dropout_rate: float = 0.1,
        use_cuda: bool = False,
        eps: float = 1e-6,
    ):
        super(DRNBVAE, self).__init__()
        # create the encoder and decoder networks
        self.encoder = Encoder(n_input, n_latent, n_layers, n_hidden, dropout_rate)
        self.l_encoder = Encoder(n_input, 1, n_layers, n_hidden, dropout_rate)
        self.decoder = NBDecoder(n_latent, n_input, n_layers, n_hidden, dropout_rate)

        self.lmb = LinearMultiBias(n_latent, n_drugs, n_conditions)

        self.eps = torch.tensor(eps, requires_grad=False)

        self.z_prior = _normal_prior(0.0, 1.0, 1, n_latent, use_cuda=use_cuda)
        self.l_prior = _normal_prior(lib_loc, lib_scale, 1, 1, use_cuda=use_cuda)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()

        self.use_cuda = use_cuda
        self.n_latent = n_latent
        self.n_classes = n_classes
        self.n_drugs = n_drugs
        self.n_conditions = n_conditions

        # to get a tensor on the right device
        x = self.lmb.biases

        self.prior = {
            "linear.weight": dist.Laplace(
                x.new_zeros((n_drugs, n_latent)),
                lam_scale * x.new_ones((n_drugs, n_latent)),
            ).to_event(2),
            "biases": dist.Normal(
                x.new_zeros(n_conditions, n_drugs),
                bias_scale * x.new_ones(n_conditions, n_drugs),
            ).to_event(2),
        }

    # define the model p(x|z)p(z)
    def model(self, x: torch.Tensor, y: torch.Tensor, af: torch.Tensor):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)

        r_module = pyro.random_module("lmb", nn_module=self.lmb, prior=self.prior)()
        if self.use_cuda:
            r_module.cuda()

        with pyro.plate("data"):
            with poutine.scale(scale=af):
                z = pyro.sample(
                    "latent",
                    self.z_prior.expand(
                        torch.Size((x.size(0), self.n_classes, self.n_latent))
                    ).to_event(2),
                )
                l = pyro.sample(
                    "library",
                    self.l_prior.expand(
                        torch.Size((x.size(0), self.n_classes, 1))
                    ).to_event(2),
                )

            log_r, logit = self.decoder.forward(z, l)

            pyro.sample(
                "obs",
                dist.NegativeBinomial(
                    total_count=torch.exp(log_r) + self.eps,
                    logits=logit,
                    validate_args=True,
                ).to_event(2),
                obs=x,
            )

        mean_dr_logit = r_module.forward(z)

        pyro.sample(
            "drs",
            dist.Normal(
                loc=mean_dr_logit,
                scale=torch.ones_like(mean_dr_logit),
                validate_args=True,
            ).to_event(3),
            obs=y,
        )

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x: torch.Tensor, y: torch.Tensor, af: torch.Tensor):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        pyro.module("l_encoder", self.l_encoder)

        # register variational parameters with pyro
        a_loc = pyro.param("alpha_loc", x.new_zeros((self.n_drugs, self.n_latent)))
        a_scale = pyro.param(
            "alpha_scale",
            x.new_ones((self.n_drugs, self.n_latent)),
            constraint=constraints.positive,
        )

        b_loc = pyro.param("beta_loc", x.new_zeros((self.n_conditions, self.n_drugs)))
        b_scale = pyro.param(
            "beta_scale",
            x.new_ones((self.n_conditions, self.n_drugs)),
            constraint=constraints.positive,
        )

        prior = {
            "linear.weight": dist.Laplace(
                a_loc, a_scale + self.eps, validate_args=True
            ).to_event(2),
            "biases": dist.Normal(
                b_loc, b_scale + self.eps, validate_args=True
            ).to_event(2),
        }

        pyro.random_module("lmb", nn_module=self.lmb, prior=prior)()

        with pyro.plate("data"):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder.forward(x)
            l_loc, l_scale = self.l_encoder.forward(x)

            # sample the latent code z
            with poutine.scale(scale=af):
                pyro.sample(
                    "latent",
                    dist.Normal(z_loc, z_scale, validate_args=True).to_event(2),
                )
                pyro.sample(
                    "library",
                    dist.Normal(l_loc, l_scale, validate_args=True).to_event(2),
                )
