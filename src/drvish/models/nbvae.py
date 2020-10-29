#!/usr/bin/env python


from typing import Sequence

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import torch
import torch.nn as nn

from drvish.models.modules import Encoder, NBDecoder, _normal_prior


class NBVAE(nn.Module):
    r"""Variational auto-encoder model with negative binomial loss.

    :param n_input: Number of input genes
    :param n_latent: Dimensionality of the latent space
    :param layers: Number and size of hidden layers used for encoder and decoder NNs
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
        layers: Sequence[int] = (64, 64, 64),
        lib_loc: float = 7.5,
        lib_scale: float = 0.5,
        dropout_rate: float = 0.1,
        use_cuda: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.encoder = Encoder(n_input, n_latent, layers, dropout_rate)
        self.l_encoder = Encoder(n_input, 1, layers, dropout_rate)
        self.decoder = NBDecoder(n_latent, n_input, layers[::-1], dropout_rate)

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
        # register modules with Pyro
        pyro.module("nbvae", self)

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
    def guide(self, x: torch.Tensor, af: torch.Tensor):
        pyro.module("nbvae", self)

        with pyro.plate("data"):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder(x)
            l_loc, l_scale = self.l_encoder(x)

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
