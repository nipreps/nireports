"""
Foundational tooling for the :mod:`~nireports.reportlets` module.
"""
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpecFromSubplotSpec, SubplotSpec

NAME_ANNOTATION: Dict[str, Any] = {
    "xy": (0.0, 0.7),
    "xytext": (0, 0),
    "xycoords": "axes fraction",
    "textcoords": "offset points",
    "va": "center",
    "ha": "left",
    "size": 8,
    "bbox": {
        "boxstyle": "round",
        "fc": "w",
        "ec": "none",
        "color": "none",
        "lw": 0,
        "alpha": 0.8,
    },
}
STATS_ANNOTATION: Dict[str, Any] = {
    "xy": (0.98, 0.7),
    "xycoords": "axes fraction",
    "xytext": (0, 0),
    "textcoords": "offset points",
    "va": "center",
    "ha": "right",
    "size": 4,
    "bbox": {
        "boxstyle": "round",
        "fc": "w",
        "ec": "none",
        "color": "none",
        "lw": 0,
        "alpha": 0.8,
    },
}
PERCENTILE_ANNOTATION: Dict[str, Any] = {
    "xytext": (-1, 0),
    "textcoords": "offset points",
    "va": "center",
    "ha": "right",
    "color": "lightgray",
    "size": 3,
}
THRESHOLD_ANNOTATION: Dict[str, Any] = {
    "xytext": (-1, 0),
    "textcoords": "offset points",
    "va": "center",
    "ha": "right",
    "color": "dimgray",
    "size": 3,
}
STATS_LABEL: str = (
    r"max: {max_value:.3f}{units} $\bullet$ mean: {mean_value:.3f}{units} "
    r"$\bullet$ $\sigma$: {std_value:.3f}"
)
SPINELESS_SIDES: List[str] = ["top", "right", "left"]
DIST_KWARGS: Dict[str, Any] = {"xlabel": "Timesteps", "yticklabels": []}
THRESHOLD_PLOT: Dict[str, Any] = {"linewidth": 0.2, "color": "dimgray"}
PERCENTILE_PLOT: Dict[str, Any] = {"linewidth": 0.1, "color": "lightgray"}
TIME_SERIES_PLOT: Dict[str, Any] = {"linewidth": 0.8}


def confoundplot(
    time_series,
    subplot_spec: SubplotSpec,
    dist_subplot_spec: SubplotSpec = None,
    name: str = None,
    units: str = None,
    tr: float = None,
    hide_x: bool = True,
    color: str = "b",
    thresholds: Iterable[float] = None,
    ylims: Tuple[float, float] = None,
    annotation_percentile: float = 95.0,
):
    # Mediate input.
    thresholds = [] if thresholds is None else thresholds
    units = units or ""

    # Define TR and number of frames.
    tr = 1.0 if tr is None else tr
    n_timepoints = len(time_series)
    time_series = np.array(time_series)

    # Define nested GridSpec.
    grid_spec = GridSpecFromSubplotSpec(
        1, 2, subplot_spec=subplot_spec, width_ratios=[1, 100], wspace=0.0
    )

    ax = plt.subplot(grid_spec[1])
    ax.grid(False)
    ax.set(yticks=[], yticklabels=[])

    # Set X axis frame markers.
    interval = max((n_timepoints // 10, n_timepoints // 5, 1))
    xticks = np.arange(0, n_timepoints, interval)
    ax.set_xticks(xticks)

    # Set X axis label and tick labels.
    if not hide_x:
        xlabel = "Frame (index)"
        if tr is not None:
            xlabel = "Time (seconds)"
            labels = tr * np.array2string(xticks, precision=2)
            ax.set_xticklabels(labels)
        ax.set_xlabel(xlabel)
    else:
        ax.set_xticklabels([])

    # Annotate name and units.
    if name is not None:
        if units != "":
            name += f" [{units}]"
        ax.annotate(
            name,
            color=color,
            **NAME_ANNOTATION,
        )

    # Manage spines.
    sides = SPINELESS_SIDES.copy()
    if hide_x:
        sides.append("bottom")
    else:
        ax.spines["bottom"].set_position(("outward", 20))
        ax.xaxis.set_ticks_position("bottom")
    for side in sides:
        ax.spines[side].set_color("none")
        ax.spines[side].set_visible(False)

    # Calculate and set y-axis limits.
    max_value = mean_value = std_value = percentile_value = 0
    nonnan = time_series[~np.isnan(time_series)]
    if nonnan.size > 0:
        # Calculate y-axis limits.
        value_range = nonnan.max() - nonnan.min()
        def_ylims = [
            nonnan.min() - 0.1 * value_range,
            nonnan.max() + 0.1 * value_range,
        ]
        # Choose maximal value against *ylims* parameter.
        if ylims is not None:
            if ylims[0] is not None:
                def_ylims[0] = min([def_ylims[0], ylims[0]])
            if ylims[1] is not None:
                def_ylims[1] = max([def_ylims[1], ylims[1]])

        # Add space for plot title and mean/SD annotation.
        def_ylims[0] -= 0.1 * (def_ylims[1] - def_ylims[0])

        # Set y-axis limits.
        ax.set_ylim(def_ylims)

        # Calculate basic statistical properties.
        max_value = nonnan.max()
        mean_value = nonnan.mean()
        std_value = nonnan.std()
        percentile_value = np.percentile(nonnan, annotation_percentile)

    # Annotate basic statistical properties.
    stats_label = STATS_LABEL.format(
        max_value=max_value,
        mean_value=mean_value,
        units=units,
        std_value=std_value,
    )
    ax.annotate(stats_label, color=color, **STATS_ANNOTATION)

    # Annotate chosen percentile.
    xlim = (0, n_timepoints - 1)
    ax.plot(xlim, [percentile_value] * 2, **PERCENTILE_PLOT)
    ax.annotate(
        f"{percentile_value:.2f}",
        xy=(0, percentile_value),
        **PERCENTILE_ANNOTATION,
    )

    # Annotate thresholds.
    for threshold in enumerate(thresholds):
        # Plot.
        y = [threshold] * 2
        ax.plot(xlim, y, **THRESHOLD_PLOT)
        # Annotate.
        xy = (0, threshold)
        ax.annotate(f"{threshold:.2f}", xy=xy, **THRESHOLD_ANNOTATION)

    ax.plot(time_series, color=color, **TIME_SERIES_PLOT)
    ax.set_xlim(xlim)

    # Plot distribution.
    if dist_subplot_spec is not None:
        ax_dist = plt.subplot(dist_subplot_spec)
        sns.displot(time_series, vertical=True, ax=ax_dist)
        ax_dist.set(ylim=ax.get_ylim(), **DIST_KWARGS)
        return [ax, ax_dist], grid_spec

    return ax, grid_spec
