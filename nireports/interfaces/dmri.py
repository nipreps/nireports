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
"""Diffusion MRI -specific visualization."""

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

from nireports.reportlets.modality.dwi import plot_heatmap


class _DWIHeatmapInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="a DWI file")
    scalarmap = File(exists=True, mandatory=True, desc="a DWI-derived scalar map")
    mask_file = File(exists=True, mandatory=True, desc="a 3D mask")
    b_indices = traits.Dict(
        traits.Int, traits.List(traits.Int), mandatory=True, desc="index of b values"
    )
    threshold = traits.Float(desc="a DWI intensity threshold")
    subsample = traits.Int(desc="a number of samples if subsampling the data")
    sigma = traits.Float(desc="standard deviation of the noise (for SNR conversions)")
    colormap = traits.Str("YlGn", usedefault=True, desc="colormap of the heatmap")
    scalarmap_label = traits.Str(desc="set the label designating the scalar map")
    bins = traits.Tuple((150, 11), traits.Int, traits.Int, usedefault=True, desc="bins")


class _DWIHeatmapOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc="written file path")


class DWIHeatmap(SimpleInterface):
    """Prepare an dMRI summary plot for the report."""

    input_spec = _DWIHeatmapInputSpec
    output_spec = _DWIHeatmapOutputSpec

    def _run_interface(self, runtime):
        gtable = self.inputs.b_indices
        threshold = None if not isdefined(self.inputs.threshold) else self.inputs.threshold
        subsample = None if not isdefined(self.inputs.subsample) else self.inputs.subsample
        sigma = None if not isdefined(self.inputs.sigma) else self.inputs.sigma
        scalarmap_label = (
            None if not isdefined(self.inputs.scalarmap_label) else self.inputs.scalarmap_label
        )

        out_figure = plot_heatmap(
            nb.load(self.inputs.in_file).get_fdata(dtype="float32"),
            list(gtable.values()),
            list(gtable.keys()),
            np.abs(np.asanyarray(nb.load(self.inputs.mask_file).dataobj) - 1) < 1e-3,
            nb.load(self.inputs.scalarmap).get_fdata(dtype="float32"),
            scalar_label=scalarmap_label,
            imax=threshold,
            sub_size=subsample,
            sigma=sigma,
            cmap=self.inputs.colormap,
            bins=self.inputs.bins,
        )

        self._results["out_file"] = fname_presuffix(
            self.inputs.in_file,
            newpath=runtime.cwd,
            suffix="heatmap.svg",
            use_ext=False,
        )

        out_figure.savefig(self._results["out_file"], format="svg", dpi=300)
        out_figure.clf()
        out_figure = None

        return runtime
