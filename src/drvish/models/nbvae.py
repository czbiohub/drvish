#!/usr/bin/env python


from typing import Sequence

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import torch
import torch.nn as nn

from drvish.models.modules import Encoder, NBDecoder


class NBVAE(nn.Module):
    r"""Variational auto-encoder model with negative binomial loss.

    :param n_input: Number of input genes
    :param n_latent: Dimensionality of the latent space
    :param layers: Number and size of hidden layers used for encoder and decoder NNs
    :param lib_loc: Mean for prior distribution on library scaling factor
    :param lib_scale: Scale for prior distribution on library scaling factor
    :param dropout_rate: Dropout rate for neural networks
    :param scale_factor: For adjusting the ELBO loss
    :param use_cuda: if True, copy parameters into GPU memory
    :param eps: value to add to NB count parameter for numerical stability
    """

    def __init__(
        self,
        *,
        n_input: int,
        n_latent: int = 8,
        layers: Sequence[int] = (64, 64, 64),
        lib_loc: float = 7.5,
        lib_scale: float = 0.5,
        dropout_rate: float = 0.1,
        scale_factor: float = 1.0,
        use_cuda: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.encoder = Encoder(n_input, n_latent, layers, dropout_rate)
        self.l_encoder = Encoder(n_input, 1, layers, dropout_rate)
        self.decoder = NBDecoder(n_latent, n_input, layers[::-1], dropout_rate)

        self.eps = torch.tensor(eps, requires_grad=False)

        self.l_loc = lib_loc
        self.l_scale = lib_scale

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()

        self.use_cuda = use_cuda
        self.n_latent = n_latent
        self.scale_factor = scale_factor

    def model(self, x: torch.Tensor):
        # register modules with Pyro
        pyro.module("nbvae", self)

        with pyro.plate("data", len(x)), poutine.scale(scale=self.scale_factor):
            z = pyro.sample(
                "latent", dist.Normal(0, x.new_ones(self.n_latent)).to_event(1)
            )

            lib_scale = self.l_scale * x.new_ones(1)
            l = pyro.sample("library", dist.Normal(self.l_loc, lib_scale).to_event(1),)

            log_r, logit = self.decoder(z, l)

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
    def guide(self, x: torch.Tensor):
        pyro.module("nbvae", self)

        with pyro.plate("data", len(x)), poutine.scale(scale=self.scale_factor):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder(x)
            l_loc, l_scale = self.l_encoder(x)

            # sample the latent code z
            pyro.sample(
                "latent", dist.Normal(z_loc, z_scale, validate_args=True).to_event(1),
            )
            pyro.sample(
                "library", dist.Normal(l_loc, l_scale, validate_args=True).to_event(1),
            )
