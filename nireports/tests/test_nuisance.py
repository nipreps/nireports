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

import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np
import pytest
from nilearn.masking import compute_epi_mask
from scipy.ndimage import shift

from nireports.reportlets.nuisance import (
    ORIENTATIONS,
    plot_motion_overlay,
    plot_volumewise_motion,
)


@pytest.fixture
def motion_data(test_data_package):
    """A DWI volume, its brain mask, and the relative difference after a simulated shift."""
    img = nb.load(test_data_package / "ds000114_sub-01_ses-test_desc-trunc_dwi.nii.gz")
    b0 = img.slicer[..., 1]
    data = b0.get_fdata()
    mask = compute_epi_mask(b0).get_fdata().astype(bool)

    moved = shift(data, (0.5, -0.3, 0), order=1)
    rel_diff = np.zeros_like(data)
    valid = mask & (data > 1e-5)
    rel_diff[valid] = 100 * (moved[valid] - data[valid]) / data[valid]
    return data, mask, np.clip(rel_diff, -10, 10)


def test_plot_volumewise_motion(request, outdir):
    rng = request.node.rng
    n_frames = 100
    # Random-walk translations (mm) and rotations (deg)
    motion_params = np.hstack(
        [
            rng.standard_normal((n_frames, 3)).cumsum(axis=0) * 0.2,
            rng.standard_normal((n_frames, 3)).cumsum(axis=0) * 0.1,
        ]
    )

    ax = plot_volumewise_motion(np.arange(n_frames), motion_params)

    assert [line.get_label() for line in ax[0].get_lines()] == ["x", "y", "z"]
    assert [line.get_label() for line in ax[1].get_lines()] == ["Rx", "Ry", "Rz"]
    np.testing.assert_array_equal(ax[1].get_lines()[2].get_ydata(), motion_params[:, 5])

    if outdir is not None:
        ax[0].figure.savefig(outdir / "volumewise_motion.svg", bbox_inches="tight")


@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_plot_motion_overlay(motion_data, orientation, outdir):
    data, mask, rel_diff = motion_data
    slice_idx = data.shape[ORIENTATIONS.index(orientation)] // 2

    ax = plot_motion_overlay(rel_diff, data, mask, orientation, slice_idx)

    _, overlay = ax.get_images()
    vmin, vmax = overlay.get_clim()
    assert vmin == -vmax

    if outdir is not None:
        ax.figure.savefig(outdir / f"motion_overlay_{orientation}.svg", bbox_inches="tight")


def test_motion_summary_composite(request, motion_data, outdir):
    rng = request.node.rng
    data, mask, rel_diff = motion_data
    motion_params = rng.standard_normal((50, 6)).cumsum(axis=0) * 0.1

    fig, axes = plt.subplot_mosaic(
        [["trans", *ORIENTATIONS], ["rot", *ORIENTATIONS]],
        figsize=(16, 5),
        constrained_layout=True,
    )
    plot_volumewise_motion(np.arange(50), motion_params, ax=[axes["trans"], axes["rot"]])
    for orientation in ORIENTATIONS:
        slice_idx = data.shape[ORIENTATIONS.index(orientation)] // 2
        plot_motion_overlay(
            rel_diff,
            data,
            mask,
            orientation,
            slice_idx,
            smooth=False,
            colorbar=orientation == ORIENTATIONS[-1],
            ax=axes[orientation],
        )

    # Unsmoothed axial overlay is the masked input slice, unrotated
    axial_slice = data.shape[2] // 2
    np.testing.assert_array_equal(
        axes["axial"].get_images()[1].get_array().filled(np.nan),
        np.where(mask, rel_diff, np.nan)[..., axial_slice],
    )
    assert axes["axial"].get_title() == "Relative Difference Overlay"
    # A single colorbar is shared by all overlays
    assert len(fig.axes) == len(axes) + 1

    if outdir is not None:
        fig.savefig(outdir / "motion_summary.svg", bbox_inches="tight")
