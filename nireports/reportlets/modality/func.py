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
"""Visualizations specific to functional imaging."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from nireports.reportlets.nuisance import confoundplot, plot_carpet, spikesplot


class fMRIPlot:
    """Generates the fMRI Summary Plot."""

    __slots__ = (
        "timeseries",
        "segments",
        "tr",
        "confounds",
        "spikes",
        "nskip",
        "sort_carpet",
        "paired_carpet",
    )

    def __init__(
        self,
        timeseries,
        segments,
        confounds=None,
        conf_file=None,
        tr=None,
        usecols=None,
        units=None,
        vlines=None,
        spikes_files=None,
        nskip=0,
        sort_carpet=True,
        paired_carpet=False,
    ):
        self.timeseries = timeseries
        self.segments = segments
        self.tr = tr
        self.nskip = nskip
        self.sort_carpet = sort_carpet
        self.paired_carpet = paired_carpet

        if units is None:
            units = {}
        if vlines is None:
            vlines = {}
        self.confounds = {}
        if confounds is None and conf_file:
            confounds = pd.read_csv(conf_file, sep=r"[\t\s]+", usecols=usecols, index_col=False)

        if confounds is not None:
            for name in confounds.columns:
                self.confounds[name] = {
                    "values": confounds[[name]].values.squeeze().tolist(),
                    "units": units.get(name),
                    "cutoff": vlines.get(name),
                }

        self.spikes = []
        if spikes_files:
            for sp_file in spikes_files:
                self.spikes.append((np.loadtxt(sp_file), None, False))

    def plot(self, figure=None, out_file=None, fontsize=24):
        """Main plotter"""

        plt.rcParams.update({"font.size": fontsize})

        nconfounds = len(self.confounds)
        nspikes = len(self.spikes)
        nrows = 1 + nconfounds + nspikes

        # Calculate height ratios in figure points
        height_ratios = [0.8] * (nconfounds + nspikes) + [10]

        if figure is None:
            figure = plt.figure(figsize=(19.2, sum(height_ratios)))

        # Create grid
        grid = GridSpec(
            nrows,
            1,
            figure=figure,
            wspace=0.0,
            hspace=0.05,
            height_ratios=height_ratios,
        )

        grid_id = 0
        for tsz, name, iszs in self.spikes:
            spikesplot(tsz, title=name, outer_gs=grid[grid_id], tr=self.tr, zscored=iszs)
            grid_id += 1

        if self.confounds:
            from seaborn import color_palette

            palette = color_palette("husl", nconfounds)

        for i, (name, kwargs) in enumerate(self.confounds.items()):
            tseries = kwargs.pop("values")
            confoundplot(tseries, grid[grid_id], tr=self.tr, color=palette[i], name=name, **kwargs)
            grid_id += 1

        plot_carpet(
            self.timeseries,
            segments=self.segments,
            subplot=grid[-1],
            tr=self.tr,
            sort_rows=self.sort_carpet,
            drop_trs=self.nskip,
            cmap="paired" if self.paired_carpet else None,
        )

        if out_file is not None:
            figure.savefig(out_file, bbox_inches="tight")
            plt.close(figure)
            figure = None
            return out_file

        return figure
