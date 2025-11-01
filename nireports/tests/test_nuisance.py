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

import nibabel as nb
import numpy as np
import pytest
from matplotlib.figure import Figure
from scipy.ndimage import affine_transform

from nireports.reportlets.nuisance import (
    ORIENTATIONS,
    plot_motion_overlay,
    plot_volumewise_motion,
)


def test_plot_volumewise_motion(request, tmp_path):
    rng = request.node.rng

    # Simulate motion for a given number of frames
    n_frames = 100
    frames = np.arange(n_frames)

    # Simulated translations (in mm)
    translations = rng.standard_normal((n_frames, 3)).cumsum(axis=0) * 0.2

    # Simulated rotations (in degrees)
    rotations = rng.standard_normal((n_frames, 3)).cumsum(axis=0) * 0.1

    # Combine into one motion matrix: shape (n_frames, 6)
    motion_params = np.hstack([translations, rotations])

    ax = plot_volumewise_motion(frames, motion_params)
    assert isinstance(ax[0].figure, Figure)
    fig = ax[0].figure
    out_svg = tmp_path / "volumewise_motion.svg"
    fig.savefig(out_svg, format="svg")


@pytest.mark.parametrize("orientation", ["axial", "coronal", "sagittal"])
def test_plot_motion_overlay(tmp_path, orientation, test_data_package):
    import scipy.ndimage as ndi

    def compute_brain_mask_from_b0(_img_data, _vol_idx=0, _threshold_percentile=20):
        b0 = img_data[..., _vol_idx] if img_data.ndim == 4 else img_data

        positive = b0[b0 > 0]
        thr = np.percentile(positive, _threshold_percentile) if positive.size else 0.0
        _brain_mask = b0 > thr

        labels, nlab = ndi.label(_brain_mask)
        if nlab > 0:
            sizes = ndi.sum(_brain_mask, labels, index=np.arange(1, nlab + 1))
            keep = 1 + np.argmax(sizes)
            _brain_mask = labels == keep

        _brain_mask = ndi.binary_fill_holes(_brain_mask)
        return _brain_mask

    def _compute_percentage_change(reference, test, mask):
        # Avoid divide-by-zero errors
        eps = 1e-5
        _rel_diff = np.zeros_like(reference)
        mask = mask.copy()
        mask[reference <= eps] = False
        _rel_diff[mask] = 100 * (test[mask] - reference[mask]) / reference[mask]

        return _rel_diff

    dwi_img = nb.load(test_data_package / "ds000114_sub-01_ses-test_desc-trunc_dwi.nii.gz")
    img_data = dwi_img.get_fdata()

    brain_mask = compute_brain_mask_from_b0(img_data, _vol_idx=0)

    # Create an affine transformation (rotation + translation)
    theta = np.deg2rad(5)  # 5 degree rotation
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # Rotate around Z axis and shift by +5 in x and -3 in y
    rotation_matrix = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]])
    translation = np.array([5, -3, 0])  # in voxel units

    # Transform matrix
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = translation

    # Inverse affine because scipy applies the inverse
    inv_transform = np.linalg.inv(transform)

    dwi_dir_data = img_data[..., 1]

    # Apply the transformation
    shifted = affine_transform(
        dwi_dir_data, inv_transform[:3, :3], offset=inv_transform[:3, 3], order=1
    )

    # Compute relative difference
    rel_diff = _compute_percentage_change(dwi_dir_data, shifted, brain_mask)

    # Clip values for visualization purposes
    rel_diff = np.clip(rel_diff, -10, 10)

    smooth = True
    axis = ORIENTATIONS.index(orientation)
    slice_idx = img_data.shape[axis] // 2

    with pytest.raises(IndexError):
        _ = plot_motion_overlay(
            rel_diff[..., np.newaxis],
            dwi_dir_data,
            brain_mask,
            orientation,
            slice_idx,
            smooth=smooth,
        )

    _slice_idx = img_data.shape[axis]
    with pytest.raises(IndexError):
        _ = plot_motion_overlay(
            rel_diff,
            dwi_dir_data,
            brain_mask,
            orientation,
            _slice_idx,
            smooth=smooth,
        )

    ax = plot_motion_overlay(
        rel_diff, dwi_dir_data, brain_mask, orientation, slice_idx, smooth=smooth
    )
    assert isinstance(ax.figure, Figure)
    fig = ax.figure
    out_svg = tmp_path / "motion_overlay.svg"
    fig.savefig(out_svg, format="svg")
