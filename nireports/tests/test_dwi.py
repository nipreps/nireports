# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2024 The NiPreps Developers <nipreps@gmail.com>
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
"""Test DWI reportlets."""

import nibabel as nb
import numpy as np
import pytest
from matplotlib import pyplot as plt

import os.path as op
from nireports.reportlets.modality.dwi \
    import plot_dwi, plot_gradients, plot_carpet


def test_plot_dwi(tmp_path, testdata_path, outdir):
    """Check the plot of DWI data."""

    stem = "ds000114_sub-01_ses-test_desc-trunc_dwi"
    dwi_img = nb.load(testdata_path / f"{stem}.nii.gz")
    affine = dwi_img.affine

    bvecs = np.loadtxt(testdata_path / f"{stem}.bvec").T
    bvals = np.loadtxt(testdata_path / f"{stem}.bval")

    gradients = np.hstack([bvecs, bvals[:, None]])

    # Pick a random volume to show
    rng = np.random.default_rng(1234)
    idx = rng.integers(low=0, high=dwi_img.shape[-1], size=1).item()

    _ = plot_dwi(dwi_img.get_fdata()[..., idx], affine, gradient=gradients[idx])

    if outdir is not None:
        plt.savefig(outdir / f"{stem}.svg", bbox_inches="tight")


@pytest.mark.parametrize(
    "dwi_btable",
    ["ds000114_singleshell", "hcph_multishell", "ds004737_dsi"],
)
def test_plot_gradients(tmp_path, testdata_path, dwi_btable, outdir):
    """Check the plot of DWI gradients."""

    bvecs = np.loadtxt(testdata_path / f"{dwi_btable}.bvec").T
    bvals = np.loadtxt(testdata_path / f"{dwi_btable}.bval")

    b0s_mask = bvals < 50

    gradients = np.hstack([bvecs[~b0s_mask], bvals[~b0s_mask, None]])
    _ = plot_gradients(gradients)

    if outdir is not None:
        plt.savefig(outdir / f"{dwi_btable}.svg", bbox_inches="tight")


def test_plot_carpet(tmp_path, testdata_path, outdir):
    """Check the carpet plot"""

    testdata_name = "ds000114_sub-01_ses-test_desc-trunc_dwi"

    nii = nb.load(testdata_path / f'{testdata_name}.nii.gz')
    bvals = np.loadtxt(testdata_path / f'{testdata_name}.bval')

    nii_data = nii.get_fdata()
    segmentation = nii_data > 3000
    segment_labels = {"<3000": [0], ">3000": [1]}

    image_path = None

    if outdir is not None:
        image_path = outdir / f'{testdata_name}_carpet.svg'

    plot_carpet(nii,
                bvals=bvals,
                segmentation=segmentation,
                segment_labels=segment_labels,
                output_file=image_path
                )
