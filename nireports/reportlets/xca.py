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
# STATEMENT OF CHANGES: This file was ported carrying over full git history from niworkflows,
# another NiPreps project licensed under the Apache-2.0 terms, and has been changed since.
"""Plotting results of component decompositions (xCA -- P/I-CA)."""

import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np
import pandas as pd
from nilearn.plotting.cm import cold_white_hot

from nireports.reportlets.utils import transform_to_2d

DINA4_LANDSCAPE = (11.69, 8.27)


def plot_melodic_components(
    melodic_dir,
    in_file,
    tr=None,
    out_file="melodic_reportlet.svg",
    compress="auto",
    report_mask=None,
    noise_components_file=None,
):
    """
    Plots the spatiotemporal components extracted by FSL MELODIC
    from functional MRI data.

    Parameters
    ----------
    melodic_dir : str
        Path pointing to the outputs of MELODIC
    in_file :  str
        Path pointing to the reference fMRI dataset. This file
        will be used to extract the TR value, if the ``tr`` argument
        is not set. This file will be used to calculate a mask
        if ``report_mask`` is not provided.
    tr : float
        Repetition time in seconds
    out_file : str
        Path where the resulting SVG file will be stored
    compress : ``'auto'`` or bool
        Whether SVG should be compressed. If ``'auto'``, compression
        will be executed if dependencies are installed (SVGO)
    report_mask : str
        Path to a brain mask corresponding to ``in_file``
    noise_components_file : str
        A CSV file listing the indexes of components classified as noise
        by some manual or automated (e.g. ICA-AROMA) procedure. If a
        ``noise_components_file`` is provided, then components will be
        plotted with red/green colors (correspondingly to whether they
        are in the file -noise components, red-, or not -signal, green-).
        When all or none of the components are in the file, a warning
        is printed at the top.

    """
    import os

    import numpy as np
    import pylab as plt
    import seaborn as sns
    from matplotlib.gridspec import GridSpec
    from nilearn.image import index_img, iter_img

    try:
        from nilearn.maskers import NiftiMasker
    except ImportError:  # nilearn < 0.9
        from nilearn.input_data import NiftiMasker

    sns.set_style("white")
    current_palette = sns.color_palette()
    in_nii = nb.load(in_file)
    if not tr:
        tr = in_nii.header.get_zooms()[3]
        units = in_nii.header.get_xyzt_units()
        if units and units in ("msec", "usec"):
            tr = tr / (1000.0 if units[-1] == "msec" else 1000000.0)

    if not report_mask:
        nifti_masker = NiftiMasker(mask_strategy="epi")
        nifti_masker.fit(index_img(in_nii, range(2)))
        mask_img = nifti_masker.mask_img_
    else:
        mask_img = nb.load(report_mask)

    mask_sl = []
    for j in range(3):
        mask_sl.append(transform_to_2d(mask_img.get_fdata(), j))

    timeseries = np.loadtxt(os.path.join(melodic_dir, "melodic_mix"))
    power = np.loadtxt(os.path.join(melodic_dir, "melodic_FTmix"))
    stats = np.loadtxt(os.path.join(melodic_dir, "melodic_ICstats"))
    n_components = stats.shape[0]
    Fs = 1.0 / tr
    Ny = Fs / 2
    f = Ny * (np.array(list(range(1, power.shape[0] + 1)))) / (power.shape[0])

    # Set default colors
    color_title = "k"
    color_time = current_palette[0]
    color_power = current_palette[1]
    classified_colors = None

    warning_row = 0  # Do not allocate warning row
    # Only if the components file has been provided, a warning banner will
    # be issued if all or none of the components were classified as noise
    if noise_components_file:
        noise_components = np.loadtxt(noise_components_file, dtype=int, delimiter=",", ndmin=1)
        # Activate warning row if pertinent
        warning_row = int(noise_components.size in (0, n_components))
        classified_colors = {True: "r", False: "g"}

    n_rows = int((n_components + (n_components % 2)) / 2)
    fig = plt.figure(figsize=(6.5 * 1.5, (n_rows + warning_row) * 0.85))
    gs = GridSpec(
        n_rows * 2 + warning_row,
        9,
        width_ratios=[1, 1, 1, 4, 0.001, 1, 1, 1, 4],
        height_ratios=[5] * warning_row + [1.1, 1] * n_rows,
    )

    if warning_row:
        ax = fig.add_subplot(gs[0, :])
        ncomps = "NONE of the"
        if noise_components.size == n_components:
            ncomps = "ALL"
        ax.annotate(
            f"WARNING: {ncomps} components were classified as noise",
            xy=(0.0, 0.5),
            xycoords="axes fraction",
            xytext=(0.01, 0.5),
            textcoords="axes fraction",
            size=12,
            color="#ea8800",
            bbox={"boxstyle": "round", "fc": "#f7dcb7", "ec": "#FC990E"},
        )
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

    titlefmt = "C{id:d}{noise}: Tot. var. expl. {var:.2g}%".format
    ICs = nb.load(os.path.join(melodic_dir, "melodic_IC.nii.gz"))
    # Ensure 4D
    if ICs.ndim == 3:
        ICs = ICs.slicer[..., None]
    for i, img in enumerate(iter_img(ICs)):
        col = i % 2
        row = i // 2
        l_row = row * 2 + warning_row
        is_noise = False

        if classified_colors:
            # If a noise components list is provided, assign red/green
            is_noise = (i + 1) in noise_components
            color_title = color_time = color_power = classified_colors[is_noise]

        data = img.get_fdata()
        for j in range(3):
            ax1 = fig.add_subplot(gs[l_row : l_row + 2, j + col * 5])
            sl = transform_to_2d(data, j)
            m = np.abs(sl).max()
            ax1.imshow(sl, vmin=-m, vmax=+m, cmap=cold_white_hot, interpolation="nearest")
            ax1.contour(mask_sl[j], levels=[0.5], colors="k", linewidths=0.5)
            plt.axis("off")
            ax1.autoscale_view("tight")
            if j == 0:
                ax1.set_title(
                    titlefmt(id=i + 1, noise=" [noise]" * is_noise, var=stats[i, 1]),
                    x=0,
                    y=1.18,
                    fontsize=7,
                    horizontalalignment="left",
                    verticalalignment="top",
                    color=color_title,
                )

        ax2 = fig.add_subplot(gs[l_row, 3 + col * 5])
        ax3 = fig.add_subplot(gs[l_row + 1, 3 + col * 5])

        ax2.plot(
            np.arange(len(timeseries[:, i])) * tr,
            timeseries[:, i],
            linewidth=min(200 / len(timeseries[:, i]), 1.0),
            color=color_time,
        )
        ax2.set_xlim([0, len(timeseries[:, i]) * tr])
        ax2.axes.get_yaxis().set_visible(False)
        ax2.autoscale_view("tight")
        ax2.tick_params(axis="both", which="major", pad=0)
        sns.despine(left=True, bottom=True)
        for label in ax2.xaxis.get_majorticklabels():
            label.set_fontsize(6)
            label.set_color(color_time)

        ax3.plot(
            f[0:],
            power[0:, i],
            color=color_power,
            linewidth=min(100 / len(power[0:, i]), 1.0),
        )
        ax3.set_xlim([f[0], f.max()])
        ax3.axes.get_yaxis().set_visible(False)
        ax3.autoscale_view("tight")
        ax3.tick_params(axis="both", which="major", pad=0)
        for label in ax3.xaxis.get_majorticklabels():
            label.set_fontsize(6)
            label.set_color(color_power)
        sns.despine(left=True, bottom=True)

    plt.subplots_adjust(hspace=0.5)
    fig.savefig(
        out_file,
        dpi=300,
        format="svg",
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.01,
    )
    fig.clf()


def compcor_variance_plot(
    metadata_files,
    metadata_sources=None,
    output_file=None,
    varexp_thresh=(0.5, 0.7, 0.9),
    fig=None,
):
    """
    Parameters
    ----------
    metadata_files: list
        List of paths to files containing component metadata. If more than one
        decomposition has been performed (e.g., anatomical and temporal
        CompCor decompositions), then all metadata files can be provided in
        the list. However, each metadata file should have a corresponding
        entry in `metadata_sources`.
    metadata_sources: list or None
        List of source names (e.g., ['aCompCor']) for decompositions. This
        list should be of the same length as `metadata_files`.
    output_file: str or None
        Path where the output figure should be saved. If this is not defined,
        then the plotting axes will be returned instead of the saved figure
        path.
    varexp_thresh: tuple
        Set of variance thresholds to include in the plot (default 0.5, 0.7,
        0.9).
    fig: figure or None
        Existing figure on which to plot.

    Returns
    -------
    ax: axes
        Plotting axes. Returned only if the `output_file` parameter is None.
    output_file: str
        The file where the figure is saved.
    """
    metadata = {}
    if metadata_sources is None:
        if len(metadata_files) == 1:
            metadata_sources = ["CompCor"]
        else:
            metadata_sources = [f"Decomposition {i:d}" for i in range(len(metadata_files))]
    for file, source in zip(metadata_files, metadata_sources):
        metadata[source] = pd.read_csv(str(file), sep=r"\s+")
        metadata[source]["source"] = source
    metadata = pd.concat(list(metadata.values()))
    bbox_txt = {
        "boxstyle": "round",
        "fc": "white",
        "ec": "none",
        "color": "none",
        "linewidth": 0,
        "alpha": 0.8,
    }

    decompositions = []
    data_sources = list(metadata.groupby(["source", "mask"]).groups.keys())
    for source, mask in data_sources:
        if not np.isnan(
            metadata.loc[(metadata["source"] == source) & (metadata["mask"] == mask)][
                "singular_value"
            ].values[0]
        ):
            decompositions.append((source, mask))

    if fig is not None:
        ax = [fig.add_subplot(1, len(decompositions), i + 1) for i in range(len(decompositions))]
    elif len(decompositions) > 1:
        fig, ax = plt.subplots(1, len(decompositions), figsize=(5 * len(decompositions), 5))
    else:
        ax = [plt.axes()]

    for m, (source, mask) in enumerate(decompositions):
        components = metadata[(metadata["mask"] == mask) & (metadata["source"] == source)]
        if len([m for s, m in decompositions if s == source]) > 1:
            title_mask = f" ({mask} mask)"
        else:
            title_mask = ""
        fig_title = f"{source}{title_mask}"

        ax[m].plot(
            np.arange(components.shape[0] + 1),
            [0] + list(100 * components["cumulative_variance_explained"]),
            color="purple",
            linewidth=2.5,
        )
        ax[m].grid(False)
        ax[m].set_xlabel("number of components in model")
        ax[m].set_ylabel("cumulative variance explained (%)")
        ax[m].set_title(fig_title)

        varexp = {}

        for i, thr in enumerate(varexp_thresh):
            varexp[thr] = (
                np.atleast_1d(np.searchsorted(components["cumulative_variance_explained"], thr))
                + 1
            )
            ax[m].axhline(y=100 * thr, color="lightgrey", linewidth=0.25)
            ax[m].axvline(x=varexp[thr], color=f"C{i}", linewidth=2, linestyle=":")
            ax[m].text(
                0,
                100 * thr,
                "{:.0f}".format(100 * thr),
                fontsize="x-small",
                bbox=bbox_txt,
            )
            ax[m].text(
                varexp[thr][0],
                25,
                "{} components explain\n{:.0f}% of variance".format(varexp[thr][0], 100 * thr),
                rotation=90,
                horizontalalignment="center",
                fontsize="xx-small",
                bbox=bbox_txt,
            )

        ax[m].set_yticks([])
        ax[m].set_yticklabels([])
        for label in ax[m].xaxis.get_majorticklabels():
            label.set_fontsize("x-small")
            label.set_rotation("vertical")
        for side in ["top", "right", "left"]:
            ax[m].spines[side].set_color("none")
            ax[m].spines[side].set_visible(False)

    if output_file is not None:
        if fig is None:
            fig = plt.gcf()
        fig.savefig(output_file, bbox_inches="tight")
        fig.clf()
        fig = None
        return output_file
    return ax
