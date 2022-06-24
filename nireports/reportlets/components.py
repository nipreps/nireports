"""
Foundational tooling for the :mod:`~nireports.reportlets` module.
"""
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import gridspec as mgs


def confoundplot(
    time_series,
    subplot_spec: mgs.SubplotSpec,
    dist_subplot_spec: mgs.SubplotSpec = None,
    name: str = None,
    units: str = None,
    tr: float = None,
    hide_x: bool = True,
    color: str = "b",
    thresholds: Iterable[float] = None,
    ylims: Tuple[float, float] = None,
):

    # Define TR and number of frames
    tr = 1.0 if tr is None else tr
    n_timepoints = len(time_series)
    time_series = np.array(time_series)

    # Define nested GridSpec
    gs = mgs.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=subplot_spec, width_ratios=[1, 100], wspace=0.0
    )

    ax_ts = plt.subplot(gs[1])
    ax_ts.grid(False)

    # Set 10 frame markers in X axis
    interval = max((n_timepoints // 10, n_timepoints // 5, 1))
    xticks = np.arange(0, n_timepoints, interval)
    ax_ts.set_xticks(xticks)

    if not hide_x:
        if tr is None:
            ax_ts.set_xlabel("time (frame #)")
        else:
            ax_ts.set_xlabel("time (s)")
            labels = tr * np.array(xticks)
            xticklabels = [f"{t:.2f}" for t in labels.tolist()]
            ax_ts.set_xticklabels(xticklabels)
    else:
        ax_ts.set_xticklabels([])

    if name is not None:
        if units is not None:
            name += " [%s]" % units

        ax_ts.annotate(
            name,
            xy=(0.0, 0.7),
            xytext=(0, 0),
            xycoords="axes fraction",
            textcoords="offset points",
            va="center",
            ha="left",
            color=color,
            size=8,
            bbox={
                "boxstyle": "round",
                "fc": "w",
                "ec": "none",
                "color": "none",
                "lw": 0,
                "alpha": 0.8,
            },
        )

    for side in ["top", "right"]:
        ax_ts.spines[side].set_color("none")
        ax_ts.spines[side].set_visible(False)

    if not hide_x:
        ax_ts.spines["bottom"].set_position(("outward", 20))
        ax_ts.xaxis.set_ticks_position("bottom")
    else:
        ax_ts.spines["bottom"].set_color("none")
        ax_ts.spines["bottom"].set_visible(False)

    # ax_ts.spines["left"].set_position(('outward', 30))
    ax_ts.spines["left"].set_color("none")
    ax_ts.spines["left"].set_visible(False)
    # ax_ts.yaxis.set_ticks_position('left')

    ax_ts.set_yticks([])
    ax_ts.set_yticklabels([])

    nonnan = time_series[~np.isnan(time_series)]
    if nonnan.size > 0:
        # Calculate Y limits
        valrange = nonnan.max() - nonnan.min()
        def_ylims = [
            nonnan.min() - 0.1 * valrange,
            nonnan.max() + 0.1 * valrange,
        ]
        if ylims is not None:
            if ylims[0] is not None:
                def_ylims[0] = min([def_ylims[0], ylims[0]])
            if ylims[1] is not None:
                def_ylims[1] = max([def_ylims[1], ylims[1]])

        # Add space for plot title and mean/SD annotation
        def_ylims[0] -= 0.1 * (def_ylims[1] - def_ylims[0])

        ax_ts.set_ylim(def_ylims)

        # Annotate stats
        maxv = nonnan.max()
        mean = nonnan.mean()
        stdv = nonnan.std()
        p95 = np.percentile(nonnan, 95.0)
    else:
        maxv = 0
        mean = 0
        stdv = 0
        p95 = 0

    stats_label = (
        r"max: {max:.3f}{units} $\bullet$ mean: {mean:.3f}{units} "
        r"$\bullet$ $\sigma$: {sigma:.3f}"
    ).format(max=maxv, mean=mean, units=units or "", sigma=stdv)
    ax_ts.annotate(
        stats_label,
        xy=(0.98, 0.7),
        xycoords="axes fraction",
        xytext=(0, 0),
        textcoords="offset points",
        va="center",
        ha="right",
        color=color,
        size=4,
        bbox={
            "boxstyle": "round",
            "fc": "w",
            "ec": "none",
            "color": "none",
            "lw": 0,
            "alpha": 0.8,
        },
    )

    # Annotate percentile 95
    ax_ts.plot((0, n_timepoints - 1), [p95] * 2, linewidth=0.1, color="lightgray")
    ax_ts.annotate(
        "%.2f" % p95,
        xy=(0, p95),
        xytext=(-1, 0),
        textcoords="offset points",
        va="center",
        ha="right",
        color="lightgray",
        size=3,
    )

    if thresholds is None:
        thresholds = []

    for i, thr in enumerate(thresholds):
        ax_ts.plot((0, n_timepoints - 1), [thr] * 2, linewidth=0.2, color="dimgray")

        ax_ts.annotate(
            "%.2f" % thr,
            xy=(0, thr),
            xytext=(-1, 0),
            textcoords="offset points",
            va="center",
            ha="right",
            color="dimgray",
            size=3,
        )

    ax_ts.plot(time_series, color=color, linewidth=0.8)
    ax_ts.set_xlim((0, n_timepoints - 1))

    if dist_subplot_spec is not None:
        ax_dist = plt.subplot(dist_subplot_spec)
        sns.displot(time_series, vertical=True, ax=ax_dist)
        ax_dist.set_xlabel("Timesteps")
        ax_dist.set_ylim(ax_ts.get_ylim())
        ax_dist.set_yticklabels([])

        return [ax_ts, ax_dist], gs
    return ax_ts, gs
