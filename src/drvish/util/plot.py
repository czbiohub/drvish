#!/usr/bin/env python

import itertools

import altair as alt
import numpy as np
import pandas as pd

from umap import UMAP

from typing import Sequence


def make_grid(*charts: alt.Chart, n_cols: int) -> alt.Chart:
    """Arrange a series of charts into a grid.

    :param charts: the plots to arrange into a grid
    :param n_cols: number of columns to create
    """
    it = iter(charts)
    return alt.vconcat(
        *(
            alt.hconcat(*filter(None, c))
            for c in itertools.zip_longest(*((it,) * n_cols))
        )
    )


def loss(
    train_loss: Sequence[float], test_loss: Sequence[float], test_int: int
) -> alt.Chart:
    """Plot the loss over epochs for training and testing datasets.

    :param train_loss: Loss per epoch on training data
    :param test_loss: Loss per epoch on test data
    :param test_int: Interval if testing was not performed every epoch
    """
    x = np.arange(len(train_loss))

    d = pd.concat(
        [
            pd.DataFrame({"epoch": x, "loss": train_loss, "run": "train"}),
            pd.DataFrame({"epoch": x[::test_int], "loss": test_loss, "run": "test"}),
        ]
    )

    return (
        alt.Chart(d)
        .encode(x="epoch:Q", y="loss:Q", color="run:N")
        .mark_line(point=True)
        .properties(height=240, width=240)
    )


def umap(z: np.ndarray, d: np.ndarray, lbls: np.ndarray, n_neighbors: int = 8):
    u = UMAP(n_neighbors=n_neighbors, metric="cosine").fit_transform(z)

    log_d = np.log1p(d.sum(1))
    bot_d, top_d = np.percentile(log_d, (2.5, 97.5))

    c = alt.Chart(
        pd.DataFrame({"x": u[:, 0], "y": u[:, 1], "c": lbls, "log_d": log_d})
    ).properties(height=300, width=300)

    return alt.hconcat(
        c.mark_point(opacity=0.3).encode(
            x="x:Q", y="y:Q", color=alt.Color("c:N", legend=None)
        ),
        c.mark_point(opacity=0.8).encode(
            x="x:Q",
            y="y:Q",
            color=alt.Color(
                "log_d:Q",
                scale=alt.Scale(
                    scheme="viridis", clamp=True, nice=True, domain=(bot_d, top_d)
                ),
            ),
        ),
    )


def drug_response(dr: np.ndarray, doses: np.ndarray, class_labels: np.ndarray):
    drs = np.array(
        [dr[class_labels == i, :].mean(0).squeeze() for i in np.unique(class_labels)]
    )

    n_classes = len(np.unique(class_labels))
    n_conditions = doses.shape[0]

    df = pd.DataFrame(
        {
            "Dose": np.tile(doses, n_classes),
            "% Inhibition": drs.flatten(),
            "class": np.repeat(np.arange(n_classes), n_conditions),
        }
    )

    return (
        alt.Chart(data=df)
        .mark_line()
        .encode(
            x="Dose",
            y=alt.Y("% Inhibition", type="quantitative", axis=alt.Axis(format="%")),
            color="class:N",
        )
    )


def dose_curve_comparison(dr_curve: np.ndarray, dr_curve2: np.ndarray):
    return (
        alt.Chart(
            pd.DataFrame(
                {
                    "x": np.tile(np.arange(dr_curve.shape[0]), 2),
                    "y": np.hstack((dr_curve, dr_curve2)),
                    "c": np.repeat(("True", "Predicted"), dr_curve.shape[0]),
                }
            )
        )
        .mark_line()
        .encode(
            x=alt.X(
                "x",
                type="quantitative",
                axis=alt.Axis(title="Drug Dose", ticks=False, labels=False),
            ),
            y=alt.Y(
                "y",
                type="quantitative",
                axis=alt.Axis(title="% Inhibition", format="%"),
            ),
            color=alt.Color("c:N", sort=["True", "Predicted"]),
        )
        .properties(height=100, width=100)
    )


def drug_combo(combo_data: np.ndarray):
    # Compute x^2 + y^2 across a 2D grid
    n, m = combo_data.shape
    x, y = np.mgrid[0:n, 0:m]

    # Convert this grid to columnar data expected by Altair
    data = pd.DataFrame({"x": x.ravel(), "y": y.ravel(), "z": combo_data.ravel()})

    return (
        alt.Chart(data)
        .mark_rect()
        .encode(
            x=alt.X(
                "x",
                type="ordinal",
                axis=alt.Axis(
                    title="Drug 1",
                    ticks=False,
                    labels=False,
                    orient=alt.AxisOrient("top"),
                ),
            ),
            y=alt.Y(
                "y",
                type="ordinal",
                axis=alt.Axis(title="Drug 2", ticks=False, labels=False),
                sort="descending",
            ),
            color=alt.Color("z:Q", scale=alt.Scale(scheme="viridis", domain=[1, 0])),
        )
        .properties(width=250, height=250)
    )
