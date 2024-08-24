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

from nireports.reportlets.modality.dwi import (
    get_segment_labels,
    nii_to_carpetplot_data,
    plot_dwi,
    plot_gradients,
    plot_tissue_values,
)
from nireports.reportlets.nuisance import plot_carpet
from nireports.tests.utils import _generate_raincloud_random_data


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
        plt.close(plt.gcf())


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
        plt.close(plt.gcf())


def test_plot_tissue_values(tmp_path):
    features_label = "fa"
    group_label = "tissue"
    group_names = ["CSF", "GM", "WM"]
    min_val_csf = 0.0
    max_val_csf = 0.2
    min_max_csf = (min_val_csf, max_val_csf)
    min_val_gm = 0.0
    max_val_gm = 0.6
    min_max_gm = (min_val_gm, max_val_gm)
    min_val_wm = 0.3
    max_val_wm = 1.0
    min_max_wm = (min_val_wm, max_val_wm)
    min_max = [min_max_csf, min_max_gm, min_max_wm]
    n_grp_samples = 250
    data_file = tmp_path / "tissue_fa.tsv"

    _generate_raincloud_random_data(
        min_max, n_grp_samples, features_label, group_label, group_names, data_file
    )

    palette = "Set2"
    orient = "v"
    density = True
    output_file = tmp_path / "tissue_fa.png"
    mark_nans = False

    plot_tissue_values(
        data_file,
        group_label,
        features_label,
        palette=palette,
        orient=orient,
        density=density,
        mark_nans=mark_nans,
        output_file=output_file,
    )


def test_nii_to_carpetplot_data(tmp_path, testdata_path, outdir):
    """Check the nii to carpet plot data function"""

    testdata_name = "ds000114_sub-01_ses-test_desc-trunc_dwi"

    nii = nb.load(testdata_path / f"{testdata_name}.nii.gz")
    bvals = np.loadtxt(testdata_path / f"{testdata_name}.bval")

    mask_data = np.round(82 * np.random.rand(nii.shape[0], nii.shape[1], nii.shape[2]))

    mask_nii = nb.Nifti1Image(mask_data, np.eye(4))

    filepath = testdata_path / "aseg.auto_noCCseg.label_intensities.txt"
    keywords = ["Cerebral_White_Matter", "Cerebral_Cortex", "Ventricle"]

    segment_labels = get_segment_labels(filepath, keywords)

    image_path = None

    if outdir is not None:
        image_path = outdir / f"{testdata_name}_nii_to_carpet.svg"

    data, segments = nii_to_carpetplot_data(
        nii, bvals=bvals, mask_nii=mask_nii, segment_labels=segment_labels
    )

    plot_carpet(data, segments, output_file=image_path)


def test_get_segment_labels(tmp_path, testdata_path):
    """Check the segment label function"""

    testdata_name = "aseg.auto_noCCseg.label_intensities.txt"

    filepath = testdata_path / testdata_name
    keywords = ["Cerebral_White_Matter", "Cerebral_Cortex", "Ventricle"]

    segment_labels = get_segment_labels(filepath, keywords)

    assert segment_labels is not None
