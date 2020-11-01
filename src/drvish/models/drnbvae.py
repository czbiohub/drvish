from typing import Sequence

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
import torch
import torch.nn as nn
from torch.distributions import constraints

from drvish.models.modules import Encoder, LinearMultiBias, NBDecoder


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
    :param scale_factor: For adjusting the ELBO loss
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
        scale_factor: float = 1.0,
        use_cuda: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        # create the encoder and decoder networks
        self.encoder = Encoder(n_input, n_latent, layers, dropout_rate)
        self.l_encoder = Encoder(n_input, 1, layers, dropout_rate)
        self.decoder = NBDecoder(n_latent, n_input, layers[::-1], dropout_rate)

        self.lmb = LinearMultiBias(
            n_latent, n_drugs, n_conditions, lam_scale, bias_scale
        )

        self.eps = torch.tensor(eps, requires_grad=False)

        self.l_loc = lib_loc
        self.l_scale = lib_scale

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()

        self.n_latent = n_latent
        self.n_classes = n_classes
        self.scale_factor = scale_factor

    # define the model p(x|z)p(z)
    def model(self, x: torch.Tensor, labels: torch.Tensor, y: torch.Tensor):
        # register modules with Pyro
        pyro.module("drnbvae", self)

        with pyro.plate("data", len(x)), poutine.scale(scale=self.scale_factor):
            z = pyro.sample(
                "latent", dist.Normal(0, x.new_ones(self.n_latent)).to_event(1)
            )

            lib_scale = self.l_scale * x.new_ones(1)
            l = pyro.sample("library", dist.Normal(self.l_loc, lib_scale).to_event(1))

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

        mean_dr_logit = self.lmb.calc_response(z, labels)

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
    def guide(self, x: torch.Tensor, labels: torch.Tensor, y: torch.Tensor):
        # register modules with Pyro
        pyro.module("drnbvae", self)

        with pyro.plate("data", len(x)), poutine.scale(scale=self.scale_factor):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder(x)
            l_loc, l_scale = self.l_encoder(x)

            # sample the latent code z
            pyro.sample(
                "latent", dist.Normal(z_loc, z_scale, validate_args=True).to_event(1)
            )
            pyro.sample(
                "library", dist.Normal(l_loc, l_scale, validate_args=True).to_event(1)
            )

            self.lmb(z_loc, labels)
