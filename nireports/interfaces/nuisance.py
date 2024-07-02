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
"""Screening nuisance signals."""

from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.utils.filemanip import fname_presuffix

from nireports.reportlets.nuisance import confounds_correlation_plot, plot_raincloud
from nireports.reportlets.xca import compcor_variance_plot


class _CompCorVariancePlotInputSpec(BaseInterfaceInputSpec):
    metadata_files = traits.List(
        File(exists=True),
        mandatory=True,
        desc="List of files containing component metadata",
    )
    metadata_sources = traits.List(
        traits.Str,
        desc="List of names of decompositions "
        "(e.g., aCompCor, tCompCor) yielding "
        "the arguments in `metadata_files`",
    )
    variance_thresholds = traits.Tuple(
        traits.Float(0.5),
        traits.Float(0.7),
        traits.Float(0.9),
        usedefault=True,
        desc="Levels of explained variance to include in plot",
    )
    out_file = traits.Either(None, File, value=None, usedefault=True, desc="Path to save plot")


class _CompCorVariancePlotOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="Path to saved plot")


class CompCorVariancePlot(SimpleInterface):
    """Plot the number of components necessary to explain the specified levels of variance."""

    input_spec = _CompCorVariancePlotInputSpec
    output_spec = _CompCorVariancePlotOutputSpec

    def _run_interface(self, runtime):
        if self.inputs.out_file is None:
            self._results["out_file"] = fname_presuffix(
                self.inputs.metadata_files[0],
                suffix="_compcor.svg",
                use_ext=False,
                newpath=runtime.cwd,
            )
        else:
            self._results["out_file"] = self.inputs.out_file
        compcor_variance_plot(
            metadata_files=self.inputs.metadata_files,
            metadata_sources=self.inputs.metadata_sources,
            output_file=self._results["out_file"],
            varexp_thresh=self.inputs.variance_thresholds,
        )
        return runtime


class _ConfoundsCorrelationPlotInputSpec(BaseInterfaceInputSpec):
    confounds_file = File(exists=True, mandatory=True, desc="File containing confound regressors")
    out_file = traits.Either(None, File, value=None, usedefault=True, desc="Path to save plot")
    reference_column = traits.Str(
        "global_signal",
        usedefault=True,
        desc="Column in the confound file for "
        "which all correlation magnitudes "
        "should be ranked and plotted",
    )
    columns = traits.List(traits.Str, desc="Filter out all regressors not found in this list.")
    max_dim = traits.Int(
        20,
        usedefault=True,
        desc="Maximum number of regressors to include in "
        "plot. Regressors with highest magnitude of "
        "correlation with `reference_column` will be "
        "selected.",
    )
    ignore_initial_volumes = traits.Int(
        0,
        usedefault=True,
        desc="Number of non-steady-state volumes at the beginning of the scan " "to ignore.",
    )


class _ConfoundsCorrelationPlotOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="Path to saved plot")


class ConfoundsCorrelationPlot(SimpleInterface):
    """Plot the correlation among confound regressors."""

    input_spec = _ConfoundsCorrelationPlotInputSpec
    output_spec = _ConfoundsCorrelationPlotOutputSpec

    def _run_interface(self, runtime):
        if self.inputs.out_file is None:
            self._results["out_file"] = fname_presuffix(
                self.inputs.confounds_file,
                suffix="_confoundCorrelation.svg",
                use_ext=False,
                newpath=runtime.cwd,
            )
        else:
            self._results["out_file"] = self.inputs.out_file
        confounds_correlation_plot(
            confounds_file=self.inputs.confounds_file,
            columns=self.inputs.columns if isdefined(self.inputs.columns) else None,
            max_dim=self.inputs.max_dim,
            output_file=self._results["out_file"],
            reference=self.inputs.reference_column,
            ignore_initial_volumes=self.inputs.ignore_initial_volumes,
        )
        return runtime


class _RaincloudPlotInputSpec(BaseInterfaceInputSpec):
    data_file = File(exists=True, mandatory=True, desc="File containing the data")
    out_file = traits.Either(None, File, value=None, usedefault=True, desc="Path to save plot")
    group_name = traits.Str(
        "group_name",
        mandatory=True,
        desc="Group name of interest",
    )
    feature = traits.Str(
        "feature",
        mandatory=True,
        desc="Feature of interest",
    )
    palette = traits.Str(
        "Set2",
        usedefault=True,
        desc="Color palette name",
    )
    orient = traits.Str(
        "v",
        usedefault=True,
        desc="Orientation",
    )
    density = traits.Bool(
        True,
        usedefault=True,
        desc="``True`` to plot the density",
    )
    upper_limit_value = traits.Float(
        None,
        usedefault=True,
        desc="Upper limit value over which any value in the data will be styled "
        "with a different style",
    )
    upper_limit_color = traits.Str(
        "gray",
        usedefault=True,
        desc="Lower limit value under which any value in the data will be styled "
        "with a different style",
    )
    lower_limit_value = traits.Float(
        None,
        usedefault=True,
        desc="",
    )
    lower_limit_color = traits.Str(
        "gray",
        usedefault=True,
        desc="Color name to represent values under ``lower_limit_value``",
    )
    limit_offset = traits.Float(
        None,
        usedefault=True,
        desc="Offset to plot the values over/under the upper/lower limit values",
    )
    mark_nans = traits.Bool(
        True,
        usedefault=True,
        desc="``True`` to plot NaNs as dots. ``nans_values`` must be provided if True",
    )
    nans_value = traits.Float(
        None,
        usedefault=True,
        desc="Value to use for NaN values`",
    )
    nans_color = traits.Str(
        "black",
        usedefault=True,
        desc="Color name to represent NaN values",
    )


class _RaincloudPlotOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="Path to saved plot")


class RaincloudPlot(SimpleInterface):
    """Plot a raincloud of values."""

    input_spec = _RaincloudPlotInputSpec
    output_spec = _RaincloudPlotOutputSpec

    def _run_interface(self, runtime, **kwargs):
        if self.inputs.out_file is None:
            self._results["out_file"] = fname_presuffix(
                self.inputs.data_file,
                suffix="_raincloud.svg",
                use_ext=False,
                newpath=runtime.cwd,
            )
        else:
            self._results["out_file"] = self.inputs.out_file
        plot_raincloud(
            data_file=self.inputs.data_file,
            group_name=self.inputs.group_name,
            feature=self.inputs.feature,
            palette=self.inputs.palette,
            orient=self.inputs.orient,
            density=self.inputs.density,
            mark_nans=self.inputs.mark_nans,
            output_file=self._results["out_file"],
            **kwargs,
        )
        return runtime
