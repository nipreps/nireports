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
"""Functional MRI -specific visualization."""

import nibabel as nb
import numpy as np
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    SimpleInterface,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.utils.filemanip import fname_presuffix

from nireports.reportlets.modality.func import fMRIPlot
from nireports.tools.timeseries import cifti_timeseries, get_tr, nifti_timeseries


class _FMRISummaryInputSpec(BaseInterfaceInputSpec):
    in_func = File(exists=True, mandatory=True, desc="")
    in_spikes_bg = File(exists=True, desc="")
    fd = File(exists=True, desc="")
    dvars = File(exists=True, desc="")
    outliers = File(exists=True, desc="")
    in_segm = File(exists=True, desc="")
    tr = traits.Either(None, traits.Float, usedefault=True, desc="the TR")
    fd_thres = traits.Float(0.2, usedefault=True, desc="")
    drop_trs = traits.Int(0, usedefault=True, desc="dummy scans")


class _FMRISummaryOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="written file path")


class FMRISummary(SimpleInterface):
    """Prepare an fMRI summary plot for the report."""

    input_spec = _FMRISummaryInputSpec
    output_spec = _FMRISummaryOutputSpec

    def _run_interface(self, runtime):
        import pandas as pd

        self._results["out_file"] = fname_presuffix(
            self.inputs.in_func,
            suffix="_fmriplot.svg",
            use_ext=False,
            newpath=runtime.cwd,
        )

        dataframe = (
            pd.DataFrame(
                {
                    "outliers": np.loadtxt(self.inputs.outliers, usecols=[0]).tolist(),
                    # Pick non-standardize dvars (col 1)
                    # First timepoint is NaN (difference)
                    "DVARS": [np.nan]
                    + np.loadtxt(self.inputs.dvars, skiprows=1, usecols=[1]).tolist(),
                    # First timepoint is zero (reference volume)
                    "FD": [0.0] + np.loadtxt(self.inputs.fd, skiprows=1, usecols=[0]).tolist(),
                }
            )
            if (
                isdefined(self.inputs.outliers)
                and isdefined(self.inputs.dvars)
                and isdefined(self.inputs.fd)
            )
            else None
        )

        input_data = nb.load(self.inputs.in_func)
        seg_file = self.inputs.in_segm if isdefined(self.inputs.in_segm) else None
        dataset, segments = (
            cifti_timeseries(input_data)
            if isinstance(input_data, nb.Cifti2Image)
            else nifti_timeseries(input_data, seg_file)
        )

        fMRIPlot(
            dataset,
            segments=segments,
            spikes_files=(
                [self.inputs.in_spikes_bg] if isdefined(self.inputs.in_spikes_bg) else None
            ),
            tr=(self.inputs.tr if isdefined(self.inputs.tr) else get_tr(input_data)),
            confounds=dataframe,
            units={"outliers": "%", "FD": "mm"},
            vlines={"FD": [self.inputs.fd_thres]},
            nskip=self.inputs.drop_trs,
        ).plot(out_file=self._results["out_file"])

        return runtime
