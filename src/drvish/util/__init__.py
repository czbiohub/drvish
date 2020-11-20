#!/usr/bin/env python


import numpy as np

import simscity

import drvish.util.plot


def build_dr_dataset(
    n_classes: int,
    n_latent: int,
    n_cells_per_class: int,
    n_features: int,
    n_drugs: int,
    n_conditions: int,
    class_cov: np.ndarray = None,
    prog_kw: dict = None,
    class_kw: dict = None,
    drug_kw: dict = None,
    library_kw: dict = None,
):
    prog_kw = (
        dict(scale=1.0 / np.sqrt(n_features), sparsity=1.0)
        if prog_kw is None
        else prog_kw.copy()
    )
    class_kw = (
        dict(scale=1.0 / np.sqrt(n_latent), sparsity=1.0)
        if class_kw is None
        else class_kw.copy()
    )
    drug_kw = (
        dict(scale=1.0 / np.sqrt(n_drugs), sparsity=0.5)
        if drug_kw is None
        else drug_kw.copy()
    )
    library_kw = (
        dict(loc=np.log(0.1 * n_features), scale=0.5)
        if library_kw is None
        else library_kw.copy()
    )

    programs = simscity.latent.gen_programs(n_latent, n_features, **prog_kw)

    classes = simscity.latent.gen_classes(n_latent, n_classes, **class_kw)

    class_labels = np.tile(np.arange(n_classes), n_cells_per_class)

    latent_exp = np.empty((n_cells_per_class, n_classes, n_latent))
    for i in np.arange(n_classes):
        latent_exp[:, i, :] = simscity.latent.gen_class_samples(
            n_cells_per_class, classes[i, :], cov=class_cov,
        )

    exp = np.dot(latent_exp, programs)

    z_weights = []
    doses = []
    responses = []

    for i in range(n_drugs):
        z_weights.append(simscity.drug.projection(n_latent, **drug_kw))

        doses.append(
            simscity.drug.doses(np.sqrt(n_latent) * drug_kw["scale"], n_conditions)
        )
        responses.append(simscity.drug.response(latent_exp, z_weights[-1], doses[-1]))

    lib_size = simscity.sequencing.library_size(
        (n_cells_per_class, n_classes), **library_kw
    )

    umis = simscity.sequencing.umi_counts(np.exp(exp), lib_size=lib_size)

    return (
        latent_exp,
        class_labels,
        programs,
        z_weights,
        doses,
        responses,
        lib_size,
        umis,
    )
