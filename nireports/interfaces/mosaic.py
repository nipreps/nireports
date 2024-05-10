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
# STATEMENT OF CHANGES: This file was ported carrying over full git history from MRIQC,
# another NiPreps project licensed under the Apache-2.0 terms, and has been changed since.
# The original file this work derives from is found at:
# https://github.com/nipreps/mriqc/blob/1ffd4c8d1a20b44ebfea648a7b12bb32a425d4ec/
# mriqc/interfaces/viz.py
"""Visualization of n-D images with mosaics cutting through planes."""

from pathlib import Path

import numpy as np
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)

from nireports.interfaces.base import _PlotBaseInputSpec
from nireports.reportlets.mosaic import plot_mosaic, plot_segmentation, plot_spikes


class _PlotContoursInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="File to be plotted")
    in_contours = File(exists=True, mandatory=True, desc="file to pick the contours from")
    cut_coords = traits.Int(8, usedefault=True, desc="number of slices")
    levels = traits.List([0.5], traits.Float, usedefault=True, desc="add a contour per level")
    colors = traits.List(
        ["r"],
        traits.Str,
        usedefault=True,
        desc="colors to be used for contours",
    )
    display_mode = traits.Enum(
        "ortho",
        "x",
        "y",
        "z",
        "yx",
        "xz",
        "yz",
        usedefault=True,
        desc="visualization mode",
    )
    saturate = traits.Bool(False, usedefault=True, desc="saturate background")
    out_file = traits.File(exists=False, desc="output file name")
    vmin = traits.Float(desc="minimum intensity")
    vmax = traits.Float(desc="maximum intensity")


class _PlotContoursOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="output svg file")


class PlotContours(SimpleInterface):
    """Plot contours"""

    input_spec = _PlotContoursInputSpec
    output_spec = _PlotContoursOutputSpec

    def _run_interface(self, runtime):
        in_file_ref = Path(self.inputs.in_file)

        if isdefined(self.inputs.out_file):
            in_file_ref = Path(self.inputs.out_file)

        fname = in_file_ref.name.rstrip("".join(in_file_ref.suffixes))
        out_file = (Path(runtime.cwd) / ("plot_%s_contours.svg" % fname)).resolve()
        self._results["out_file"] = str(out_file)

        vmax = None if not isdefined(self.inputs.vmax) else self.inputs.vmax
        vmin = None if not isdefined(self.inputs.vmin) else self.inputs.vmin

        plot_segmentation(
            self.inputs.in_file,
            self.inputs.in_contours,
            out_file=str(out_file),
            cut_coords=self.inputs.cut_coords,
            display_mode=self.inputs.display_mode,
            levels=self.inputs.levels,
            colors=self.inputs.colors,
            saturate=self.inputs.saturate,
            vmin=vmin,
            vmax=vmax,
        )

        return runtime


class _PlotMosaicInputSpec(_PlotBaseInputSpec):
    bbox_mask_file = File(exists=True, desc="brain mask")
    only_noise = traits.Bool(False, usedefault=True, desc="plot only noise")
    view = traits.List(
        traits.Enum("axial", "sagittal", "coronal"),
        value=["axial", "sagittal"],
        minlen=1,
        maxlen=3,
        help="Sequence of views to plot (up to three)",
        usedefault=True,
    )


class _PlotMosaicOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="output pdf file")


class PlotMosaic(SimpleInterface):
    """
    Plots slices of a 3D volume into a pdf file
    """

    input_spec = _PlotMosaicInputSpec
    output_spec = _PlotMosaicOutputSpec

    def _run_interface(self, runtime):
        mask = self.inputs.bbox_mask_file if isdefined(self.inputs.bbox_mask_file) else None

        title = self.inputs.title if isdefined(self.inputs.title) else None

        plot_mosaic(
            self.inputs.in_file,
            out_file=self.inputs.out_file,
            title=title,
            only_plot_noise=self.inputs.only_noise,
            bbox_mask_file=mask,
            cmap=self.inputs.cmap,
            annotate=self.inputs.annotate,
            views=self.inputs.view,
        )
        self._results["out_file"] = str((Path(runtime.cwd) / self.inputs.out_file).resolve())
        return runtime


class _PlotSpikesInputSpec(_PlotBaseInputSpec):
    in_spikes = File(exists=True, mandatory=True, desc="tsv file of spikes")
    in_fft = File(exists=True, mandatory=True, desc="nifti file with the 4D FFT")


class _PlotSpikesOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="output svg file")


class PlotSpikes(SimpleInterface):
    """Plot slices of a dataset with spikes."""

    input_spec = _PlotSpikesInputSpec
    output_spec = _PlotSpikesOutputSpec

    def _run_interface(self, runtime):
        out_file = str((Path(runtime.cwd) / self.inputs.out_file).resolve())
        self._results["out_file"] = out_file

        spikes_list = np.loadtxt(self.inputs.in_spikes, dtype=int).tolist()
        # No spikes
        if not spikes_list:
            Path(out_file).write_text("<p>No high-frequency spikes were found in this dataset</p>")
            return runtime

        spikes_list = [tuple(i) for i in np.atleast_2d(spikes_list).tolist()]
        plot_spikes(
            self.inputs.in_file,
            self.inputs.in_fft,
            spikes_list,
            out_file=out_file,
        )
        return runtime
