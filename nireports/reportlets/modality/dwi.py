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
"""Visualizations for diffusion MRI data."""
import numpy as np
from matplotlib import pyplot as plt


def plot_heatmap(
    data,
    b_indices,
    bvals,
    mask,
    scalar,
    scalar_label="DWI-derived scalar (a.u.)",
    bins=(150, 11),
    imax=None,
    sub_size=100000,
    cmap="YlGn",
    sigma=None,
):
    """Create heatmap plot."""
    # Round scalar to a single-digit decimal
    scalar = np.clip(np.round(scalar + 0.005, 1), 0, 1)

    # Prepare data in shells
    shells = [np.rint(data[mask][..., idx].reshape(-1)) for idx in b_indices]

    # Maximum intensity level to be plotted
    if imax is None:
        # If not provided, set 75th percentile of lowest b-value.
        imax = np.percentile(shells[0], 75)

    fig, axs = plt.subplots(
        len(b_indices) + 1,
        sharex=True,
        figsize=(20, 1.6 * (len(b_indices) + 1)),
    )
    axs[-1].spines[:].set_visible(False)
    axs[-1].grid(which="minor", color="w", linestyle='-', linewidth=1)
    for i, shelldata in enumerate(shells):
        x = shelldata[shelldata < imax]
        y = np.array([scalar[mask]] * len(b_indices[i])).reshape(-1)[shelldata < imax]

        if sub_size is not None:
            choice = np.random.choice(range(x.size), size=sub_size)
            x = x[choice]
            y = y[choice]

        histdata, _, _ = np.histogram2d(x, y, bins=bins, range=((0, int(imax)), (0, 1)))
        axs[i].imshow(
            histdata.T,
            interpolation='nearest',
            origin='lower',
            aspect="auto",
            cmap=cmap,
        )

        # Show all ticks and label them with the respective list entries.
        axs[i].set_yticks(
            [0.5, 5, 9.5],
            labels=["0.0", "0.5", "1.0"],
        )

        # Turn spines off and create white grid.
        axs[i].spines[:].set_visible(False)

        # axs[i].set_xticks(np.arange(bins[0] + 1) - .5, minor=True)
        axs[i].set_yticks(np.arange(bins[1] + 1) - 0.5, minor=True)
        axs[i].grid(which="minor", color="w", linestyle='-', linewidth=1)
        axs[i].tick_params(which="minor", bottom=False, left=False)
        axs[i].set_ylabel(f"$b$ = {bvals[i]}\n($n$ = {len(b_indices[i])})", fontsize=15)

        marginal_H, edges = np.histogram(x, bins=bins[0], range=(0, int(imax)), density=True)
        axs[-1].bar(
            np.linspace(0, bins[0], num=bins[0], endpoint=False, dtype=int),
            marginal_H,
            alpha=0.4,
        )

    if sigma is not None:
        max_snr = imax / sigma
        labels_bins = [1.937, 5.0, 8.0, round(max_snr, 1)]
        labels_bins_position = bins[0] * np.array(labels_bins) / max_snr

    else:
        labels_bins_position = np.arange(bins[0], step=20) + 0.5
        labels_bins = (labels_bins_position - 0.5) * (imax / bins[0])

    axs[-1].set_xticks(
        labels_bins_position,
        labels=labels_bins_position
        if sigma is None
        else [f"{v}\n[{round(10 * np.log(v), 0):.0f} dB]" for v in labels_bins],
        fontsize=14,
    )
    axs[-1].legend([f"{b}" for b in bvals], ncol=len(bvals), title="$b$ value")
    axs[-1].set_yticks([], labels=[])
    axs[-1].set_xlabel(
        f"SNR [noise floor estimated at {sigma:0.2f}]" if sigma is not None
        else "DWI intensity",
        fontsize=20,
    )
    fig.supylabel(scalar_label, fontsize=20, y=0.65)
    fig.tight_layout(rect=[0.02, 0, 1, 1])

    return fig
