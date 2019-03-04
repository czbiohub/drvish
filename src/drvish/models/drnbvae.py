#!/usr/bin/env python


from typing import Sequence

import pyro
import torch
from pyro import distributions as dist, poutine as poutine
from torch import nn as nn
from torch.distributions import constraints

from drvish.models.modules import Encoder, NBDecoder, LinearMultiBias, _normal_prior


class DRNBVAE(nn.Module):
    """A variational auto-encoder module with negative binomial loss and a dose
    reponse module attached.

    :param n_input: Number of input genes
    :param n_classes: Number of different cell types used for drug response
    :param n_drugs: Number of different drug targets in response data
    :param n_conditions: Number of conditions for each target
    :param n_latent: Dimensionality of the latent space
    :param layers: Number and size of hidden layers used for encoder and decoder NNs
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
        layers: Sequence[int] = (64, 64, 64),
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
        self.encoder = Encoder(n_input, n_latent, layers, dropout_rate)
        self.l_encoder = Encoder(n_input, 1, layers, dropout_rate)
        self.decoder = NBDecoder(n_latent, n_input, layers[::-1], dropout_rate)

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
