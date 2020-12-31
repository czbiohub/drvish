from typing import List, Sequence

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.nn import PyroModule

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
    :param lam_scale: Scale for prior on regression weights
    :param bias_scale: Scale for prior on regression biases
    :param sigma_scale: Scale for drug response observations
    :param scale_factor: For adjusting the ELBO loss
    :param epsilon: Value to add for numerical stability
    :param device: If not None, tensors will be copied to cuda device
    """

    def __init__(
        self,
        *,
        n_input: int,
        n_classes: int,
        n_drugs: int,
        n_conditions: Sequence[int],
        n_latent: int = 8,
        layers: Sequence[int] = (64, 64, 64),
        lib_loc: float = 7.5,
        lib_scale: float = 0.5,
        lam_scale: float = 5.0,
        bias_scale: float = 10.0,
        sigma_scale: float = 1.0,
        scale_factor: float = 1.0,
        epsilon: float = 1e-6,
        device: torch.device = None,
    ):
        super().__init__()
        # create the encoder and decoder networks
        self.encoder = Encoder(n_input, n_latent, layers)
        self.decoder = NBDecoder(n_latent, n_input, layers[::-1])

        self.lmb_list = PyroModule[nn.ModuleList](
            [
                LinearMultiBias(
                    n_latent,
                    1,
                    n_conditions[i],
                    torch.tensor(lam_scale).to(device),
                    torch.tensor(bias_scale).to(device),
                )
                for i in range(n_drugs)
            ]
        )

        self.epsilon = torch.tensor(epsilon).to(device)
        self.sigma_scale = torch.tensor(sigma_scale).to(device)

        self.l_loc = torch.tensor(lib_loc).to(device)
        self.l_scale = lib_scale * torch.ones(1, device=device)

        if device is not None:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda(device)

        self.n_latent = n_latent
        self.n_classes = n_classes
        self.scale_factor = scale_factor

    # define the model p(x|z)p(z)
    def model(self, x: torch.Tensor, labels: torch.Tensor, y: List[torch.Tensor]):
        # register modules with Pyro
        pyro.module("drnbvae", self)

        with pyro.plate("data", len(x)), poutine.scale(scale=self.scale_factor):
            z = pyro.sample(
                "latent", dist.Normal(0, x.new_ones(self.n_latent)).to_event(1)
            )

            l = pyro.sample(
                "library", dist.Normal(self.l_loc, self.l_scale).to_event(1)
            )

            log_r, logit = self.decoder(z, l)

            pyro.sample(
                "obs",
                dist.NegativeBinomial(
                    total_count=torch.exp(log_r) + self.epsilon, logits=logit
                ).to_event(1),
                obs=x,
            )

        for i, lmb in enumerate(self.lmb_list):
            mean_dr_logit = lmb(z, labels)
            pyro.sample(
                f"drs_{i}",
                dist.Normal(loc=mean_dr_logit, scale=self.sigma_scale).to_event(2),
                obs=y[i],
            )

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x: torch.Tensor, labels: torch.Tensor, y: List[torch.Tensor]):
        # register modules with Pyro
        pyro.module("drnbvae", self)

        with pyro.plate("data", len(x)), poutine.scale(scale=self.scale_factor):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale, l_loc, l_scale = self.encoder(x)

            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            pyro.sample("library", dist.Normal(l_loc, l_scale).to_event(1))

        for lmb in self.lmb_list:
            lmb.sample_guide()
