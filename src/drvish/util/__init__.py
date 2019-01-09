#!/usr/bin/env python


from typing import Union

import torch

import numpy as np

import drvish.util.plot

from .training import split_dataset, train, evaluate


def cos_annealing_factor(epoch: int, max_epoch: int, eta_min: float = 1e-4):
    return eta_min + (1. - eta_min) * (1. - np.cos(np.pi * epoch / max_epoch)) / 2.


def average_response(x: torch.Tensor, c: Union[torch.Tensor, np.ndarray]):
    return torch.stack([x[c == i, ...].mean(0) for i in c])


def build_dr_dataset(n_classes: int,
                     n_latent: int,
                     n_cells: int,
                     n_features: int,
                     n_drugs: int,
                     n_conditions: int,
                     prog_sparsity: float = 0.5,
                     class_sparsity: float = 0.5,
                     drug_sparsity: float = 0.5,
                     scale: Union[int, float] = 3):
    assert n_cells // n_classes == n_cells / n_classes

    n_cells_per_class = n_cells // n_classes

    programs = latent.gen_programs(n_latent, n_features, prog_sparsity, scale)

    classes = latent.gen_classes(n_latent, n_classes, class_sparsity, scale)

    class_labels = np.random.permutation(
        np.repeat(np.arange(n_classes), n_cells_per_class)
    )

    latent_exp = np.empty((n_cells, n_latent))
    for i in np.arange(n_classes):
        latent_exp[class_labels == i, :] = latent.gen_class_samples(n_cells_per_class,
                                                                    classes[i, :])

    exp = np.dot(latent_exp, programs)

    z_weights = []
    doses = []
    dr = []

    for i in range(n_drugs):
        z_weights.append(
            drug.drug_projection(n_latent, scale=scale, sparsity=drug_sparsity))
        doses.append(drug.drug_doses(n_latent, scale, n_conditions))
        dr.append(drug.drug_response(latent_exp, z_weights[-1], doses[-1]))

    lib_size = sequencing.library_size(n_cells, loc=8.5, scale=0.5)

    umis = sequencing.umi_counts(np.exp(exp), lib_size=lib_size)

    return latent_exp, class_labels, programs, z_weights, doses, dr, lib_size, umis

