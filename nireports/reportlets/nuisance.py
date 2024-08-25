# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2023 The NiPreps Developers <nipreps@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# We support and encourage derived works from this project, please read
# about our expectations at
#
#     https://www.nipreps.org/community/licensing/
#
# STATEMENT OF CHANGES: This file was ported carrying over full git history from
# other NiPreps projects licensed under the Apache-2.0 terms.
"""Plotting distributions."""

import math
import operator
import os.path as op

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import FigureCanvasPdf as FigureCanvas
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

from nireports.tools.ndimage import _get_values_inside_a_mask

DEFAULT_DPI = 300
DINA4_LANDSCAPE = (11.69, 8.27)
DINA4_PORTRAIT = (8.27, 11.69)


def plot_fd(fd_file, fd_radius, mean_fd_dist=None, figsize=DINA4_LANDSCAPE):
    fd_power = _calc_fd(fd_file, fd_radius)

    fig = plt.Figure(figsize=figsize)
    FigureCanvas(fig)

    if mean_fd_dist:
        grid = GridSpec(2, 4)
    else:
        grid = GridSpec(1, 2, width_ratios=[3, 1])
        grid.update(hspace=1.0, right=0.95, left=0.1, bottom=0.2)

    ax = fig.add_subplot(grid[0, :-1])
    ax.plot(fd_power)
    ax.set_xlim((0, len(fd_power)))
    ax.set_ylabel("Frame Displacement [mm]")
    ax.set_xlabel("Frame number")
    ylim = ax.get_ylim()

    ax = fig.add_subplot(grid[0, -1])
    sns.distplot(fd_power, vertical=True, ax=ax)
    ax.set_ylim(ylim)

    if mean_fd_dist:
        ax = fig.add_subplot(grid[1, :])
        sns.distplot(mean_fd_dist, ax=ax)
        ax.set_xlabel("Mean Frame Displacement (over all subjects) [mm]")
        mean_fd = fd_power.mean()
        label = rf"$\overline{{\text{{FD}}}}$ = {mean_fd:g}"
        plot_vline(mean_fd, label, ax=ax)

    return fig


def plot_dist(
    main_file,
    mask_file,
    xlabel,
    distribution=None,
    xlabel2=None,
    figsize=DINA4_LANDSCAPE,
):
    data = _get_values_inside_a_mask(main_file, mask_file)

    fig = plt.Figure(figsize=figsize)
    FigureCanvas(fig)

    gsp = GridSpec(2, 1)
    ax = fig.add_subplot(gsp[0, 0])
    sns.distplot(data.astype(np.double), kde=False, bins=100, ax=ax)
    ax.set_xlabel(xlabel)

    ax = fig.add_subplot(gsp[1, 0])
    sns.distplot(np.array(distribution).astype(np.double), ax=ax)
    cur_val = np.median(data)
    label = f"{cur_val:g}"
    plot_vline(cur_val, label, ax=ax)
    ax.set_xlabel(xlabel2)

    return fig


def plot_vline(cur_val, label, ax):
    ax.axvline(cur_val)
    ylim = ax.get_ylim()
    vloc = (ylim[0] + ylim[1]) / 2.0
    xlim = ax.get_xlim()
    pad = (xlim[0] + xlim[1]) / 100.0
    ax.text(
        cur_val - pad,
        vloc,
        label,
        color="blue",
        rotation=90,
        verticalalignment="center",
        horizontalalignment="right",
    )


def _calc_rows_columns(ratio, n_images):
    rows = 2
    for _ in range(100):
        columns = math.floor(ratio * rows)
        total = (rows - 1) * columns
        if total > n_images:
            rows = np.ceil(n_images / columns) + 1
            break
        rows += 1
    return int(rows), int(columns)


def _calc_fd(fd_file, fd_radius):
    from math import pi

    lines = open(fd_file, "r").readlines()
    rows = [[float(x) for x in line.split()] for line in lines]
    cols = np.array([list(col) for col in zip(*rows)])

    translations = np.transpose(np.abs(np.diff(cols[0:3, :])))
    rotations = np.transpose(np.abs(np.diff(cols[3:6, :])))

    fd_power = np.sum(translations, axis=1) + (fd_radius * pi / 180) * np.sum(rotations, axis=1)

    # FD is zero for the first time point
    fd_power = np.insert(fd_power, 0, 0)

    return fd_power


def _get_mean_fd_distribution(fd_files, fd_radius):
    mean_fds = []
    max_fds = []
    for fd_file in fd_files:
        fd_power = _calc_fd(fd_file, fd_radius)
        mean_fds.append(fd_power.mean())
        max_fds.append(fd_power.max())

    return mean_fds, max_fds


def plot_qi2(x_grid, ref_pdf, fit_pdf, ref_data, cutoff_idx, out_file=None):
    fig, ax = plt.subplots()

    ax.plot(
        x_grid,
        ref_pdf,
        linewidth=2,
        alpha=0.5,
        label="background",
        color="dodgerblue",
    )

    refmax = np.percentile(ref_data, 99.95)
    x_max = x_grid[-1]

    ax.hist(
        ref_data,
        40 * max(int(refmax / x_max), 1),
        fc="dodgerblue",
        histtype="stepfilled",
        alpha=0.2,
        density=True,
    )
    fit_pdf[fit_pdf > 1.0] = np.nan
    ax.plot(
        x_grid,
        fit_pdf,
        linewidth=2,
        alpha=0.5,
        label="chi2",
        color="darkorange",
    )

    ylims = ax.get_ylim()
    ax.axvline(
        x_grid[-cutoff_idx],
        ymax=ref_pdf[-cutoff_idx] / ylims[1],
        color="dodgerblue",
    )
    plt.xlabel('Intensity within "hat" mask')
    plt.ylabel("Frequency")
    ax.set_xlim([0, x_max])
    plt.legend()

    if out_file is None:
        out_file = op.abspath("qi2_plot.svg")

    fig.savefig(out_file, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close(fig=fig)
    return out_file


def plot_carpet(
    data,
    segments=None,
    cmap=None,
    tr=None,
    detrend=True,
    subplot=None,
    title=None,
    output_file=None,
    size=(900, 1200),
    sort_rows="ward",
    drop_trs=0,
    legend=True,
    fontsize=None,
):
    """
    Plot an image representation of voxel intensities across time.

    This kind of plot is known as "carpet plot" or "Power plot".
    See Jonathan Power Neuroimage 2017 Jul 1; 154:150-158.

    Parameters
    ----------
    data : N x T :obj:`numpy.array`
        The functional data to be plotted (*N* sampling locations by *T* timepoints).
    segments: :obj:`dict`, optional
        A mapping between segment labels (e.g., `"Left Cortex"`) and list of indexes
        in the data array.
    cmap : colormap
        Overrides the generation of an automated colormap.
    tr : float , optional
        Specify the TR, if specified it uses this value. If left as None,
        # of frames is plotted instead of time.
    detrend : :obj:`bool`, optional
        Detrend and standardize the data prior to plotting.
    subplot : matplotlib subplot, optional
        Subplot to plot figure on.
    title : string, optional
        The title displayed on the figure.
    output_file : string, or None, optional
        The name of an image file to export the plot to. Valid extensions
        are .png, .pdf, .svg. If output_file is not None, the plot
        is saved to a file, and the display is closed.
    size : :obj:`tuple`
        Maximum number of samples to plot (voxels, timepoints)
    sort_rows : :obj:`str` or :obj:`False` or :obj:`None`
        Apply a clustering algorithm to reorganize the rows of the carpet.
        ``""``, ``False``, and ``None`` skip clustering sorting.
        ``"linkage"`` uses linkage hierarchical clustering
        :obj:`scipy.cluster.hierarchy.linkage`.
        Any other value that Python evaluates to ``True`` will use the
        default clustering, which is :obj:`sklearn.cluster.ward_tree`.

    """
    if segments is None:
        segments = {"whole brain (voxels)": list(range(data.shape[0]))}

    nsegments = len(segments)
    if nsegments == 1:
        legend = False

    if cmap is None:
        colors = mpl.colormaps["tab10"].colors
    elif cmap == "paired":
        colors = list(mpl.colormaps["Paired"].colors)
        colors[0], colors[1] = colors[1], colors[0]
        colors[2], colors[7] = colors[7], colors[2]

    if detrend:
        from nilearn.signal import clean

        data = clean(data.T, t_r=tr, filter=False).T

    # We want all subplots to have the same dynamic range
    vminmax = (np.percentile(data, 2), np.percentile(data, 98))

    # Decimate number of time-series before clustering
    n_dec = int((1.8 * data.shape[0]) // size[0])
    if n_dec > 1:
        segments = {
            lab: idx[::n_dec] for lab, idx in segments.items() if np.array(idx).shape >= (1,)
        }

    # Cluster segments (if argument enabled)
    if sort_rows:
        from scipy.cluster.hierarchy import dendrogram, linkage
        from sklearn.cluster import ward_tree

        for seg_label, seg_idx in segments.items():
            # In debugging cases, we might have ROIs too small to have enough rows to sort
            if len(seg_idx) < 2:
                continue
            roi_data = data[seg_idx]
            if isinstance(sort_rows, str) and sort_rows.lower() == "linkage":
                linkage_matrix = linkage(
                    roi_data, method="average", metric="euclidean", optimal_ordering=True
                )
            else:
                children, _, n_leaves, _, distances = ward_tree(roi_data, return_distance=True)
                linkage_matrix = _ward_to_linkage(children, n_leaves, distances)

            dn = dendrogram(linkage_matrix, no_plot=True)
            # Override the ordering of the indices in this segment
            segments[seg_label] = np.array(seg_idx)[np.array(dn["leaves"])]

    # If subplot is not defined
    if subplot is None:
        figure, allaxes = plt.subplots(figsize=(19.2, 10))
        allaxes.spines[:].set_visible(False)
        allaxes.spines[:].set_color("none")
        allaxes.get_xaxis().set_visible(False)
        allaxes.get_yaxis().set_visible(False)
        subplot = allaxes.get_subplotspec()
        fontsize = fontsize or 24
    else:
        figure = plt.gcf()

    # Length before decimation
    n_trs = data.shape[-1] - drop_trs

    # Calculate time decimation factor
    t_dec = max(int((1.8 * n_trs) // size[1]), 1)
    data = data[:, drop_trs::t_dec]

    # Define nested GridSpec
    gs = GridSpecFromSubplotSpec(
        nsegments,
        1,
        subplot_spec=subplot,
        hspace=0.05,
        height_ratios=[len(v) for v in segments.values()],
    )

    for i, (_, indices) in enumerate(segments.items()):
        # Carpet plot
        ax = plt.subplot(gs[i])

        ax.imshow(
            data[indices, :],
            interpolation="nearest",
            aspect="auto",
            cmap="gray",
            vmin=vminmax[0],
            vmax=vminmax[1],
        )

        # Toggle the spine objects
        ax.spines["top"].set_color("none")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_color("none")
        ax.spines["right"].set_visible(False)

        # Make colored left axis
        ax.spines["left"].set_linewidth(3)
        ax.spines["left"].set_color(colors[i])
        ax.spines["left"].set_capstyle("butt")
        ax.spines["left"].set_position(("outward", 2))

        # Make all subplots have same xticks
        xticks = np.linspace(0, data.shape[-1], endpoint=True, num=7)
        ax.set_xticks(xticks)
        ax.set_yticks([])
        ax.grid(False)

        if i == (nsegments - 1):
            xlabel = "time-points (index)"
            xticklabels = (xticks * n_trs / data.shape[-1]).astype("uint32") + drop_trs
            if tr is not None:
                xlabel = "time (mm:ss)"
                xticklabels = [
                    f"{int(t // 60):02d}:{(t % 60).round(0).astype(int):02d}"
                    for t in (tr * xticklabels)
                ]

            ax.set_xlabel(xlabel)
            ax.set_xticklabels(xticklabels)
            ax.spines["bottom"].set_position(("outward", 5))
            ax.spines["bottom"].set_color("k")
            ax.spines["bottom"].set_linewidth(0.8)
        else:
            ax.set_xticklabels([])
            ax.set_xticks([])
            ax.spines["bottom"].set_color("none")
            ax.spines["bottom"].set_visible(False)

        if title and i == 0:
            ax.set_title(title)

    if nsegments == 1:
        ax.set_ylabel(list(segments.keys())[0])

    if legend:
        from matplotlib.patches import Patch
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        axlegend = inset_axes(
            ax,
            width="100%",
            height=0.01,
            loc="lower center",
            borderpad=-4.1,
        )
        axlegend.grid(False)
        axlegend.set_xticks([])
        axlegend.set_yticks([])
        axlegend.patch.set_alpha(0.0)
        for loc in ("top", "bottom", "left", "right"):
            axlegend.spines[loc].set_color("none")
            axlegend.spines[loc].set_visible(False)

        axlegend.legend(
            handles=[
                Patch(color=colors[i], label=_label) for i, _label in enumerate(segments.keys())
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, 0),
            shadow=False,
            fancybox=False,
            ncol=min(len(segments.keys()), 5),
            frameon=False,
            prop={"size": fontsize} if fontsize else {},
        )

    if fontsize is not None:
        ax.xaxis.label.set_fontsize(max(8, fontsize * 0.75))
        # Change font size according to figure size
        for item in [ax.title, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels():
            item.set_fontsize(fontsize)

    if output_file is not None:
        figure.savefig(output_file, bbox_inches="tight")
        plt.close(figure)
        figure = None
        return output_file

    return gs


def spikesplot(
    ts_z,
    outer_gs=None,
    tr=None,
    zscored=True,
    spike_thresh=6.0,
    title="Spike plot",
    ax=None,
    cmap="viridis",
    hide_x=True,
    nskip=0,
):
    """
    A spikes plot. Thanks to Bob Dogherty (this docstring needs be improved with proper ack)
    """

    if ax is None:
        ax = plt.gca()

    if outer_gs is not None:
        gs = GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer_gs, width_ratios=[1, 100], wspace=0.0
        )
        ax = plt.subplot(gs[1])

    # Define TR and number of frames
    if tr is None:
        tr = 1.0

    # Load timeseries, zscored slice-wise
    nslices = ts_z.shape[0]
    ntsteps = ts_z.shape[1]

    # Load a colormap
    my_cmap = mpl.colormaps[cmap]
    norm = Normalize(vmin=0, vmax=float(nslices - 1))
    colors = [my_cmap(norm(sl)) for sl in range(nslices)]

    stem = len(np.unique(ts_z).tolist()) == 2
    # Plot one line per axial slice timeseries
    for sl in range(nslices):
        if not stem:
            ax.plot(ts_z[sl, :], color=colors[sl], lw=0.5)
        else:
            markerline, stemlines, baseline = ax.stem(ts_z[sl, :])
            plt.setp(markerline, "markerfacecolor", colors[sl])
            plt.setp(baseline, "color", colors[sl], "linewidth", 1)
            plt.setp(stemlines, "color", colors[sl], "linewidth", 1)

    # Handle X, Y axes
    ax.grid(False)

    # Handle X axis
    last = ntsteps - 1
    ax.set_xlim(0, last)
    xticks = list(range(0, last)[::20]) + [last] if not hide_x else []
    ax.set_xticks(xticks)

    if not hide_x:
        if tr is None:
            ax.set_xlabel("time (frame #)")
        else:
            ax.set_xlabel("time (s)")
            ax.set_xticklabels(["%.02f" % t for t in (tr * np.array(xticks)).tolist()])

    # Handle Y axis
    ylabel = "slice-wise noise average on background"
    if zscored:
        ylabel += " (z-scored)"
        zs_max = np.abs(ts_z).max()
        ax.set_ylim(
            (
                -(np.abs(ts_z[:, nskip:]).max()) * 1.05,
                (np.abs(ts_z[:, nskip:]).max()) * 1.05,
            )
        )

        ytick_vals = np.arange(0.0, zs_max, float(np.floor(zs_max / 2.0)))
        yticks = list(reversed((-1.0 * ytick_vals[ytick_vals > 0]).tolist())) + ytick_vals.tolist()

        # TODO plot min/max or mark spikes
        # yticks.insert(0, ts_z.min())
        # yticks += [ts_z.max()]
        for val in ytick_vals:
            ax.plot((0, ntsteps - 1), (-val, -val), "k:", alpha=0.2)
            ax.plot((0, ntsteps - 1), (val, val), "k:", alpha=0.2)

        # Plot spike threshold
        if zs_max < spike_thresh:
            ax.plot((0, ntsteps - 1), (-spike_thresh, -spike_thresh), "k:")
            ax.plot((0, ntsteps - 1), (spike_thresh, spike_thresh), "k:")
    else:
        yticks = [
            ts_z[:, nskip:].min(),
            np.median(ts_z[:, nskip:]),
            ts_z[:, nskip:].max(),
        ]
        ax.set_ylim(0, max(yticks[-1] * 1.05, (yticks[-1] - yticks[0]) * 2.0 + yticks[-1]))
        # ax.set_ylim(ts_z[:, nskip:].min() * 0.95,
        #             ts_z[:, nskip:].max() * 1.05)

    ax.annotate(
        ylabel,
        xy=(0.0, 0.7),
        xycoords="axes fraction",
        xytext=(0, 0),
        textcoords="offset points",
        va="center",
        ha="left",
        color="gray",
        size=plt.rcParams["font.size"] * 0.5,
        bbox={
            "boxstyle": "round",
            "fc": "w",
            "ec": "none",
            "color": "none",
            "lw": 0,
            "alpha": 0.8,
        },
    )
    ax.set_yticks([])
    ax.set_yticklabels([])

    # if yticks:
    #     # ax.set_yticks(yticks)
    #     # ax.set_yticklabels(['%.02f' % y for y in yticks])
    #     # Plot maximum and minimum horizontal lines
    #     ax.plot((0, ntsteps - 1), (yticks[0], yticks[0]), 'k:')
    #     ax.plot((0, ntsteps - 1), (yticks[-1], yticks[-1]), 'k:')

    for side in ["top", "right"]:
        ax.spines[side].set_color("none")
        ax.spines[side].set_visible(False)

    if not hide_x:
        ax.spines["bottom"].set_position(("outward", 10))
        ax.xaxis.set_ticks_position("bottom")
    else:
        ax.spines["bottom"].set_color("none")
        ax.spines["bottom"].set_visible(False)

    # ax.spines["left"].set_position(('outward', 30))
    # ax.yaxis.set_ticks_position('left')
    ax.spines["left"].set_visible(False)
    ax.spines["left"].set_color(None)

    # labels = [label for label in ax.yaxis.get_ticklabels()]
    # labels[0].set_weight('bold')
    # labels[-1].set_weight('bold')
    if title:
        ax.set_title(title)
    return ax


def spikesplot_cb(position, cmap="viridis", fig=None):
    # Add colorbar
    if fig is None:
        fig = plt.gcf()

    cax = fig.add_axes(position)
    cb = ColorbarBase(
        cax,
        cmap=mpl.colormaps[cmap],
        spacing="proportional",
        orientation="horizontal",
        drawedges=False,
    )
    cb.set_ticks([0, 0.5, 1.0])
    cb.set_ticklabels(["Inferior", "(axial slice)", "Superior"])
    cb.outline.set_linewidth(0)
    cb.ax.xaxis.set_tick_params(width=0)
    return cax


def confoundplot(
    tseries,
    gs_ts,
    gs_dist=None,
    name=None,
    units=None,
    tr=None,
    hide_x=True,
    color="b",
    nskip=0,
    cutoff=None,
    ylims=None,
):
    # Define TR and number of frames
    notr = False
    if tr is None:
        notr = True
        tr = 1.0
    ntsteps = len(tseries)
    tseries = np.array(tseries)

    # Define nested GridSpec
    gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_ts, width_ratios=[1, 100], wspace=0.0)

    ax_ts = plt.subplot(gs[1])
    ax_ts.grid(False)

    # Set 10 frame markers in X axis
    interval = max((ntsteps // 10, ntsteps // 5, 1))
    xticks = list(np.arange(0, ntsteps)[::interval])
    ax_ts.set_xticks(xticks)

    if not hide_x:
        if notr:
            ax_ts.set_xlabel("time (frame #)")
        else:
            ax_ts.set_xlabel("time (s)")
            labels = tr * np.array(xticks)
            ax_ts.set_xticklabels([f"{t:.02f}" for t in labels.tolist()])
    else:
        ax_ts.set_xticklabels([])

    fontsize = plt.rcParams["font.size"]
    if name is not None:
        if units is not None:
            name += f" [{units}]"

        ax_ts.annotate(
            name,
            xy=(0.0, 0.2),
            xytext=(0, 0),
            xycoords="axes fraction",
            textcoords="offset points",
            va="top",
            ha="left",
            color=color,
            size=fontsize * 0.45,
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

    nonnan = tseries[~np.isnan(tseries)]
    if nonnan.size > 0:
        # Calculate Y limits
        valrange = nonnan.max() - nonnan.min()
        def_ylims = [nonnan.min() - 0.1 * valrange, nonnan.max() + 0.1 * valrange]
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

    units = units or ""
    stats_label = (
        rf"max: {maxv:.3f}{units} $\bullet$ mean: {mean:.3f}{units} "
        rf"$\bullet$ $\sigma$: {stdv:.3f}"
    )
    ax_ts.annotate(
        stats_label,
        xy=(0.98, 0.1),
        xycoords="axes fraction",
        xytext=(0, 0),
        textcoords="offset points",
        va="top",
        ha="right",
        color=color,
        size=fontsize * 0.5,
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
    ax_ts.plot((0, ntsteps - 1), [p95] * 2, linewidth=0.1, color="lightgray")
    ax_ts.annotate(
        "%.2f" % p95,
        xy=(0, p95),
        xytext=(-1, 0),
        textcoords="offset points",
        va="center",
        ha="right",
        color="lightgray",
        size=fontsize * 0.4,
    )

    if cutoff is None:
        cutoff = []

    for thr in cutoff:
        ax_ts.plot((0, ntsteps - 1), [thr] * 2, linewidth=0.2, color="dimgray")

        ax_ts.annotate(
            "%.2f" % thr,
            xy=(0, thr),
            xytext=(-1, 0),
            textcoords="offset points",
            va="center",
            ha="right",
            color="dimgray",
            size=fontsize * 0.4,
        )

    ax_ts.plot(tseries, color=color, linewidth=0.8)
    ax_ts.set_xlim((0, ntsteps - 1))

    if gs_dist is not None:
        ax_dist = plt.subplot(gs_dist)
        sns.displot(tseries, vertical=True, ax=ax_dist)
        ax_dist.set_xlabel("Timesteps")
        ax_dist.set_ylim(ax_ts.get_ylim())
        ax_dist.set_yticklabels([])

        return [ax_ts, ax_dist], gs
    return ax_ts, gs


def _ward_to_linkage(children, n_leaves, distances):
    """Create linkage matrix from the output of Ward clustering."""
    # create the counts of samples under each node
    counts = np.zeros(children.shape[0])
    n_samples = n_leaves
    for i, merge in enumerate(children):
        current_count = 0
        for child_idx in merge:
            current_count += 1 if child_idx < n_samples else counts[child_idx - n_samples]
        counts[i] = current_count

    return np.column_stack([children, distances, counts]).astype(float)


def confounds_correlation_plot(
    confounds_file,
    columns=None,
    figure=None,
    max_dim=20,
    output_file=None,
    reference="global_signal",
    ignore_initial_volumes=0,
):
    """
    Generate a bar plot with the correlation of confounds.

    Parameters
    ----------
    confounds_file: :obj:`str`
        File containing all confound regressors to be included in the
        correlation plot.
    figure: figure or None
        Existing figure on which to plot.
    columns: :obj:`list` or :obj:`None`.
        Select a list of columns from the dataset.
    max_dim: :obj:`int`
        The maximum number of regressors to be included in the output plot.
        Reductions (e.g., CompCor) of high-dimensional data can yield so many
        regressors that the correlation structure becomes obfuscated. This
        criterion selects the ``max_dim`` regressors that have the largest
        correlation magnitude with ``reference`` for inclusion in the plot.
    output_file: :obj:`str` or :obj:`None`
        Path where the output figure should be saved. If this is not defined,
        then the plotting axes will be returned instead of the saved figure
        path.
    reference: :obj:`str`
        ``confounds_correlation_plot`` prepares a bar plot of the correlations
        of each confound regressor with a reference column. By default, this
        is the global signal (so that collinearities with the global signal
        can readily be assessed).
    ignore_initial_volumes : :obj:`int`
        Number of non-steady-state volumes at the beginning of the scan to ignore.

    Returns
    -------
    axes and gridspec
        Plotting axes and gridspec. Returned only if ``output_file`` is ``None``.
    output_file: :obj:`str`
        The file where the figure is saved.
    """

    confounds_data = pd.read_table(confounds_file)

    if columns:
        columns = dict.fromkeys(columns)  # Drop duplicates
        columns[reference] = None  # Make sure the reference is included
        confounds_data = confounds_data[list(columns)]

    confounds_data = confounds_data.loc[
        ignore_initial_volumes:,
        np.logical_not(np.isclose(confounds_data.var(skipna=True), 0)),
    ]
    corr = confounds_data.corr()

    gscorr = corr.copy()
    gscorr["index"] = gscorr.index
    gscorr[reference] = np.abs(gscorr[reference])
    gs_descending = gscorr.sort_values(by=reference, ascending=False)["index"]
    n_vars = corr.shape[0]
    max_dim = min(n_vars, max_dim)

    gs_descending = gs_descending[:max_dim]
    features = [p for p in corr.columns if p in gs_descending]
    corr = corr.loc[features, features]
    np.fill_diagonal(corr.values, 0)

    if figure is None:
        plt.figure(figsize=(15, 5))
    gs = GridSpec(1, 21)
    ax0 = plt.subplot(gs[0, :10])
    ax1 = plt.subplot(gs[0, 11:])

    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, linewidths=0.5, cmap="coolwarm", center=0, square=True, ax=ax0)
    ax0.tick_params(axis="both", which="both", width=0)

    for label in ax0.xaxis.get_majorticklabels():
        label.set_fontsize("small")
    for label in ax0.yaxis.get_majorticklabels():
        label.set_fontsize("small")
    sns.barplot(
        data=gscorr,
        x="index",
        y=reference,
        hue="index",
        ax=ax1,
        order=gs_descending,
        palette="Reds_d",
        saturation=0.5,
        legend=False,
    )

    ax1.set_xlabel("Confound time series")
    ax1.set_ylabel(f"Magnitude of correlation with {reference}")
    ax1.tick_params(axis="x", which="both", width=0)
    ax1.tick_params(axis="y", which="both", width=5, length=5)

    for label in ax1.xaxis.get_majorticklabels():
        label.set_fontsize("small")
        label.set_rotation("vertical")
    for label in ax1.yaxis.get_majorticklabels():
        label.set_fontsize("small")
    for side in ["top", "right", "left"]:
        ax1.spines[side].set_color("none")
        ax1.spines[side].set_visible(False)

    if output_file is not None:
        figure = plt.gcf()
        figure.savefig(output_file, bbox_inches="tight")
        plt.close(figure)
        figure = None
        return output_file
    return [ax0, ax1], gs


def _plot_density(x, y, df, group_name, palette, orient):
    ax = sns.violinplot(
        x=x,
        y=y,
        data=df,
        hue=group_name,
        dodge=False,
        palette=palette,
        density_norm="width",
        inner=None,
        orient=orient,
    )

    # Cut half of the violins
    for violin in ax.collections:
        bbox = violin.get_paths()[0].get_extents()
        x0, y0, width, height = bbox.bounds
        width_denom = 2
        height_denom = 1
        if orient == "h":
            width_denom = 1
            height_denom = 2
        violin.set_clip_path(
            plt.Rectangle(
                (x0, y0), width / width_denom, height / height_denom, transform=ax.transData
            )
        )

    return ax


def _jitter_data_points(old_len_collections, orient, width, ax):
    offset = np.array([width, 0])
    if orient == "h":
        offset = np.array([0, width])
    for dots in ax.collections[old_len_collections:]:
        dots.set_offsets(dots.get_offsets() + offset)


def _plot_nans(df, x, y, color, orient, ax):
    df_nans = df[df.isna().any(axis=1)]
    sns.stripplot(
        x=x,
        y=y,
        data=df_nans,
        color=color,
        orient=orient,
        ax=ax,
    )


def _plot_out_of_range(
    df,
    x,
    feature,
    orient,
    limit_offset,
    limit_value,
    limit_color,
    limit_name,
    color_vble_name,
    _op,
    ax,
):
    if limit_color is None:
        raise ValueError(
            f"``{color_vble_name}`` must be provided if ``{limit_name}`` is provided."
        )
    if limit_offset is None:
        raise ValueError(f"``limit_offset`` must be provided if ``{limit_name}`` is provided.")
    if _op == operator.gt:
        arithm = operator.add
    elif _op == operator.lt:
        arithm = operator.sub
    else:
        raise ValueError(f"``{_op}`` must be either ``gt`` or ``lt``.")

    df_overflow = df[_op(df[feature], limit_value)]
    sns.stripplot(
        x=x,
        y=arithm(limit_value, limit_offset),
        data=df_overflow,
        color=limit_color,
        orient=orient,
        ax=ax,
    )


def plot_raincloud(
    data_file,
    group_name,
    feature,
    palette="Set2",
    orient="v",
    density=True,
    upper_limit_value=None,
    upper_limit_color="gray",
    lower_limit_value=None,
    lower_limit_color="gray",
    limit_offset=None,
    mark_nans=True,
    nans_value=None,
    nans_color="black",
    figure=None,
    output_file=None,
):
    """
    Generate a raincloud plot with the data points corresponding to the
    ``feature`` value contained in the data file. If ``upper_limit_value`` or
    ``lower_limit_value`` is provided, the values outside that range are
    clipped. Thus, a large density around those values, together with the values
    plot with the distinctive ``upper_limit_color`` and ``lower_limit_color``
    styles may be indicative of unexpected values in the data. Similarly, NaN
    values, if present, will be marked with the distinctive ``nans_color``
    style, and may again be indicative of unexpected values in the data.

    Parameters
    ----------
    data_file : :obj:`str`
        File containing the data of interest.
    figure : :obj:`matplotlib.pyplot.figure` or None
        Existing figure on which to plot.
    group_name : :obj:`str`
        The group name of interest to be plot.
    feature : :obj:`str`
        The feature of interest to be plot.
    palette : :obj:`str`, optional
        Color palette name provided to :func:`sns.stripplot`.
    orient : :obj:`str`, optional
        Plot orientation (``v`` or ``h``).
    density : :obj:`bool`, optional
        ``True`` to plot the density of the data points.
    upper_limit_value : :obj:`float`, optional
        Upper limit value over which any value in the data will be styled with a
        different style.
    upper_limit_color : :obj:`str`, optional
        Color name to represent values over ``upper_limit_value``.
    lower_limit_value : :obj:`float`, optional
        Lower limit value under which any value in the data will be styled with
        a different style.
    lower_limit_color : :obj:`str`, optional
        Color name to represent values under ``lower_limit_value``.
    limit_offset : :obj:`float`, optional
        Offset to plot the values over/under the upper/lower limit values.
    mark_nans : :obj:`bool`, optional
        ``True`` to plot NaNs as dots. ``nans_values`` must be provided if True.
    nans_value : :obj:`float`, optional
        Value to use for NaN values.
    nans_color : :obj:`str`, optional
        Color name to represent NaN values.
    output_file : :obj:`str` or :obj:`None`
        Path where the output figure should be saved. If this is not defined,
        then the plotting axes will be returned instead of the saved figure
        path.

    Returns
    -------
    axes and gridspec
        Plotting axes and gridspec. Returned only if ``output_file`` is ``None``.
    output_file : :obj:`str`
        The file where the figure is saved.
    """

    df = pd.read_csv(data_file, sep=r"[\t\s]+", engine="python")

    df_clip = df.copy(deep=True)
    df_clip[feature] = df[feature].clip(lower=lower_limit_value, upper=upper_limit_value)

    if figure is None:
        plt.figure(figsize=(7, 5))

    gs = GridSpec(1, 1)
    ax = plt.subplot(gs[0, 0])

    sns.set(style="white", font_scale=2)

    x = feature
    y = group_name
    # Swap x/y if the requested orientation is vertical
    if orient == "v":
        x = group_name
        y = feature

    # Plot the density
    if density:
        ax = _plot_density(x, y, df_clip, group_name, palette, orient)

    # Add boxplots
    width = 0.15
    sns.boxplot(
        x=x,
        y=y,
        data=df_clip,
        color="black",
        width=width,
        zorder=10,
        showcaps=True,
        boxprops={"facecolor": "none", "zorder": 10},
        showfliers=True,
        whiskerprops={"linewidth": 2, "zorder": 10},
        saturation=1,
        orient=orient,
        ax=ax,
    )

    old_len_collections = len(ax.collections)

    # Plot the data points as dots
    sns.stripplot(
        x=x,
        y=y,
        hue=group_name,
        data=df_clip,
        palette=palette,
        edgecolor="white",
        size=3,
        jitter=0.1,
        zorder=0,
        orient=orient,
        ax=ax,
    )

    # Offset the dots that would be otherwise shadowed by the violins
    if density:
        _jitter_data_points(old_len_collections, orient, width, ax)

    # Draw nans if any
    if mark_nans:
        if nans_value is None:
            raise ValueError("``nans_value`` must be provided if ``mark_nans`` is True.")
        _plot_nans(df, x, nans_value, nans_color, orient, ax)

    # If upper/lower limits are provided, draw the points with a different color
    if upper_limit_value is not None:
        _plot_out_of_range(
            df,
            x,
            feature,
            orient,
            limit_offset,
            upper_limit_value,
            upper_limit_color,
            "upper_limit_value",
            "upper_limit_color",
            operator.gt,
            ax,
        )

    if lower_limit_value is not None:
        _plot_out_of_range(
            df,
            x,
            feature,
            orient,
            limit_offset,
            lower_limit_value,
            lower_limit_color,
            "lower_limit_value",
            "lower_limit_color",
            operator.lt,
            ax,
        )

    if output_file is not None:
        figure = plt.gcf()
        plt.tight_layout()
        figure.savefig(output_file, bbox_inches="tight")
        plt.close(figure)
        figure = None
        return output_file

    return ax, gs
