#!/usr/bin/env python


from typing import Sequence

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import torch
import torch.nn as nn

from drvish.models.modules import Encoder, NBDecoder


class MCVNBVAE(nn.Module):
    r"""Variational auto-encoder model with negative binomial loss.

    :param n_input: Number of input genes
    :param n_latent: Dimensionality of the latent space
    :param layers: Number and size of hidden layers used for encoder and decoder NNs
    :param lib_loc: Mean for prior distribution on library scaling factor
    :param lib_scale: Scale for prior distribution on library scaling factor
    :param scale_factor: For adjusting the ELBO loss
    :param epsilon: Value to add to NB count parameter for numerical stability
    :param device: If not None, tensors will be copied to cuda device
    """

    def __init__(
        self,
        *,
        n_input: int,
        n_latent: int = 8,
        layers: Sequence[int] = (64, 64, 64),
        lib_loc: float = 7.5,
        lib_scale: float = 0.5,
        scale_factor: float = 1.0,
        epsilon: float = 1e-6,
        device: torch.device = None,
    ):
        super().__init__()
        self.encoder = Encoder(n_input, n_latent, layers)
        self.decoder = NBDecoder(n_latent, n_input, layers[::-1])

        self.epsilon = torch.tensor(epsilon).to(device)

        self.l_loc = torch.tensor(lib_loc).to(device)
        self.l_scale = lib_scale * torch.ones(1, device=device)

        if device is not None:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda(device)

        self.n_latent = n_latent
        self.scale_factor = scale_factor

    def model(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        log_data_split: torch.Tensor,
        log_data_split_complement: torch.Tensor,
    ):
        # register modules with Pyro
        pyro.module("mcv_nbvae", self)

        with pyro.plate("data", len(x0)), poutine.scale(scale=self.scale_factor):
            z = pyro.sample(
                "latent", dist.Normal(0, x0.new_ones(self.n_latent)).to_event(1)
            )

            l = pyro.sample(
                "library", dist.Normal(self.l_loc, self.l_scale).to_event(1)
            )

            log_r, logit = self.decoder(z, l)

            # adjust for data split
            log_r += log_data_split_complement - log_data_split

            pyro.sample(
                "obs",
                dist.NegativeBinomial(
                    total_count=torch.exp(log_r) + self.epsilon, logits=logit
                ).to_event(1),
                obs=x1,
            )

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        log_data_split: torch.Tensor,
        log_data_split_complement: torch.Tensor,
    ):
        pyro.module("mcv_nbvae", self)

        with pyro.plate("data", len(x0)), poutine.scale(scale=self.scale_factor):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale, l_loc, l_scale = self.encoder(x0)

            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            pyro.sample("library", dist.Normal(l_loc, l_scale).to_event(1))
