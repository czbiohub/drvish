#!/usr/bin/env python


import collections

import numpy as np
import torch
import torch.nn as nn

import pyro
import pyro.distributions as dist
import pyro.poutine as poutine


# class for creating a bunch of fully connected layers
class FCLayers(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
    ):
        super(FCLayers, self).__init__()
        layers_dim = [n_input] + (n_layers - 1) * [n_hidden] + [n_output]
        self.fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        "Layer {}".format(i),
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

    def forward(self, x: torch.Tensor):
        for layers in self.fc_layers:
            for layer in layers:
                if isinstance(layer, nn.BatchNorm1d) and x.dim() == 3:
                    x = torch.cat(
                        [(layer(slice_x)).unsqueeze(0) for slice_x in x], dim=0
                    )
                else:
                    x = layer(x)
        return x


# the Encoder takes data and encodes it in the latent space
# this is scvi.models.modules.Encoder


class Encoder(nn.Module):
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
            n_input=n_input,
            n_output=n_hidden,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor):
        # Parameters for latent distribution
        q = self.encoder(x)
        q_m = self.mean_encoder(q)

        # computational stability safeguard
        q_v = torch.exp(torch.clamp(self.var_encoder(q), -5, 5))

        return q_m, q_v


softplus = nn.Softplus()


class NBDecoder(nn.Module):
    r"""Decodes data from latent space of ``n_input`` dimensions ``n_output``
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
            n_input=n_input,
            n_output=n_hidden,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )

        # mean gamma
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output), nn.Softmax(dim=-1)
        )

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(n_hidden, n_output)

    def forward(self, z: torch.Tensor, library: torch.Tensor):
        r"""The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``

        :param z: tensor with shape ``(n_input,)``
        :param library: library size
        :return: parameters for the NB distribution of expression
        :rtype: 4-tuple of :py:class:`torch.Tensor`
        """

        # The decoder returns values for the parameters of the NB distribution
        px = self.px_decoder(z)
        px_scale = self.px_scale_decoder(px)
        px_rate = softplus(torch.exp(library) * px_scale)
        px_r = self.px_r_decoder(px)
        return px_r, px_rate


class PoissonDecoder(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
    ):
        super(PoissonDecoder, self).__init__()
        self.decoder = FCLayers(
            n_input=n_input,
            n_output=n_hidden,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output), nn.Softmax(dim=-1)
        )

    def forward(self, z, library):
        px_rate = softplus(torch.exp(library) * self.px_scale_decoder(self.decoder(z)))
        return px_rate


class BinomDecoder(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
    ):
        super(BinomDecoder, self).__init__()
        self.decoder = FCLayers(
            n_input=n_input,
            n_output=n_hidden,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
        )
        self.px_logit_decoder = nn.Linear(n_hidden, n_output)

    def forward(self, z):
        return self.px_logit_decoder(self.decoder(z))


class NBVAE(nn.Module):
    def __init__(
        self,
        *,
        n_input: int,
        z_dim: int = 8,
        n_layers: int = 3,
        n_hidden: int = 64,
        use_cuda: bool = False,
        eps: float = 1e-8,
    ):
        super(NBVAE, self).__init__()
        self.encoder = Encoder(n_input, z_dim, n_layers=n_layers, n_hidden=n_hidden)
        self.l_encoder = Encoder(n_input, 1, n_layers=1, n_hidden=n_hidden)
        self.decoder = NBDecoder(z_dim, n_input, n_layers=n_layers, n_hidden=n_hidden)

        self.eps = torch.tensor(eps)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()

        self.use_cuda = use_cuda
        self.z_dim = z_dim

    def model(self, x, af):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)

        with pyro.plate("data"):
            # setup hyperparameters for prior p(z)
            z_loc = x.new_zeros(torch.Size((x.size(0), self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.size(0), self.z_dim)))

            l_loc = x.new_zeros(torch.Size((x.size(0), 1)))
            l_scale = x.new_ones(torch.Size((x.size(0), 1)))

            with poutine.scale(scale=af):
                z = pyro.sample(
                    "latent",
                    dist.Normal(z_loc, z_scale, validate_args=True).to_event(1),
                )
                l = pyro.sample(
                    "library",
                    dist.Normal(l_loc, l_scale, validate_args=True).to_event(1),
                )

                px_r, px_rate = self.decoder.forward(z, l)
                px_logit = torch.log(px_rate) - px_r

                pyro.sample(
                    "obs",
                    dist.NegativeBinomial(
                        total_count=torch.exp(px_r) + self.eps,
                        logits=px_logit,
                        validate_args=True,
                    ).to_event(1),
                    obs=x,
                )

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x, af):
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


class BinomVAE(nn.Module):
    def __init__(
        self,
        *,
        n_input: int,
        z_dim: int = 8,
        n_layers: int = 3,
        n_hidden: int = 64,
        use_cuda: bool = False,
    ):
        super(BinomVAE, self).__init__()
        self.encoder = Encoder(n_input, z_dim, n_layers=n_layers, n_hidden=n_hidden)
        self.decoder = BinomDecoder(
            z_dim, n_input, n_layers=n_layers, n_hidden=n_hidden
        )

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()

        self.use_cuda = use_cuda
        self.z_dim = z_dim

    def model(self, x, af):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)

        with pyro.plate("data"):
            z_loc = x.new_zeros(torch.Size((x.size(0), self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.size(0), self.z_dim)))

            with poutine.scale(scale=af):
                # sample from prior (value will be sampled by guide when computing the ELBO)
                z = pyro.sample(
                    "latent",
                    dist.Normal(z_loc, z_scale, validate_args=True).to_event(1),
                )

                px_logit = self.decoder.forward(z)
                x_count = torch.sum(x, -1, keepdim=True)

                pyro.sample(
                    "obs",
                    dist.Binomial(
                        total_count=x_count, logits=px_logits, validate_args=True
                    ).to_event(1),
                    obs=x,
                )

    def guide(self, x, af):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)

        with pyro.plate("data"):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder.forward(x)

            # sample the latent code z
            with poutine.scale(scale=af):
                pyro.sample(
                    "latent",
                    dist.Normal(z_loc, z_scale, validate_args=True).to_event(1),
                )
