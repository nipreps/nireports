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

import nibabel as nb
import numpy as np
import pytest

from nireports.reportlets.utils import (
    _compute_crop_slices,
    _largest_connected_component,
    compute_common_display_params,
    compute_display_params,
    crop_img,
    load_framewise_displacement,
    merge_crop_slices,
    render_comparison_frames,
)
from nireports.tests.utils import _write_image
from nireports.tools.ndimage import load_api


def test_largest_connected_component_selects_largest():
    mask = np.zeros((3, 3, 3), dtype=bool)
    mask[0, 0, 0] = True
    mask[1:3, 1:3, 1] = True

    largest = _largest_connected_component(mask)

    assert largest.sum() == 4
    assert largest[0, 0, 0] == 0


def test_compute_crop_slices_returns_none_without_positive(tmp_path, monkeypatch):
    img_path = tmp_path / "zeros.nii.gz"
    img = nb.Nifti1Image(np.zeros((4, 4, 4), dtype=float), np.eye(4))
    img.to_filename(img_path)

    def raise_error(_img):
        raise RuntimeError

    monkeypatch.setattr("nireports.reportlets.utils.compute_epi_mask", raise_error)

    result = _compute_crop_slices(nb.load(str(img_path)))

    assert result is None


def test_crop_img_adjusts_affine():
    data = np.ones((4, 4, 4), dtype=float)
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    img = nb.Nifti1Image(data, affine)

    cropped = crop_img(img, (slice(1, 3), slice(0, 2), slice(2, 4)))

    assert np.allclose(cropped.affine[:3, 3], [2.0, 0.0, 8.0])


def test_crop_img_adjusts_affine_for_oriented_image():
    data = np.ones((4, 4, 4), dtype=float)
    affine = np.array(
        [
            [0.0, -2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 3.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    img = nb.Nifti1Image(data, affine)

    cropped = crop_img(img, (slice(1, 3), slice(0, 2), slice(2, 4)))

    assert np.allclose(cropped.affine[:3, 3], [0.0, 2.0, 6.0])


def test_merge_crop_slices_uses_union():
    merged = merge_crop_slices(
        (slice(2, 8), slice(4, 9), slice(1, 5)),
        (slice(1, 6), slice(5, 10), slice(0, 7)),
    )

    assert merged == (slice(1, 8), slice(4, 10), slice(0, 7))


def test_compute_display_params_handles_single_frame(tmp_path):
    img_path = _write_image(tmp_path / "single.nii.gz", (5, 5, 5))
    img = load_api(img_path, nb.spatialimages.SpatialImage)
    mid_img, cut_coords, vmin, vmax, _ = compute_display_params(img)

    assert mid_img.ndim == 3
    assert len(cut_coords) == 3
    assert vmin <= vmax


def test_compute_common_display_params_calls_helpers(monkeypatch):

    uncorr_img = object()
    corr_img = object()

    def fake_compute_display_params(img, _crop_slices=None):
        # First pass: uncorr then corr (without crop_slices)
        # Second pass: uncorr then corr (with crop_slices)
        if _crop_slices is None:
            if img is uncorr_img:
                return None, None, 1.0, 9.0, (slice(0, 5), slice(0, 5), slice(0, 5))
            return None, None, 2.0, 8.0, (slice(1, 6), slice(1, 6), slice(1, 6))
        # With shared crop_slices, return cut coords
        if img is uncorr_img:
            return None, (10.0, 11.0, 12.0), None, None, None
        return None, (20.0, 21.0, 22.0), None, None, None

    def fake_merge_crop_slices(a, b):
        assert a == (slice(0, 5), slice(0, 5), slice(0, 5))
        assert b == (slice(1, 6), slice(1, 6), slice(1, 6))
        return (slice(0, 6), slice(0, 6), slice(0, 6))

    monkeypatch.setattr(
        "nireports.reportlets.utils.compute_display_params", fake_compute_display_params
    )
    monkeypatch.setattr("nireports.reportlets.utils.merge_crop_slices", fake_merge_crop_slices)

    vmin, vmax, cut_uncorr, cut_corr, crop_slices = compute_common_display_params(
        uncorr_img, corr_img
    )

    assert vmin == 1.0
    assert vmax == 9.0
    assert cut_uncorr == (10.0, 11.0, 12.0)
    assert cut_corr == (20.0, 21.0, 22.0)
    assert crop_slices == (slice(0, 6), slice(0, 6), slice(0, 6))


def test_render_comparison_frames(monkeypatch):
    n_frames = 2
    uncorr_img = object()
    corr_img = object()

    # Capture calls to ensure plotting invoked per frame and side
    plot_calls = []

    # We don't care about actual files; just return arrays based on filename pattern
    def fake_imread(path):
        p = str(path)
        # uncorr smaller height than corr -> should be padded before concat
        if "uncorr_" in p:
            return np.zeros((8, 5, 3), dtype=np.uint8) + 10
        if "corr_" in p:
            return np.zeros((10, 7, 3), dtype=np.uint8) + 20
        raise AssertionError(f"Unexpected path: {p}")

    def fake_plot_epi(frame, **kwargs):
        plot_calls.append(
            {
                "frame": frame,
                "output_file": kwargs.get("output_file"),
                "title": kwargs.get("title"),
                "vmin": kwargs.get("vmin"),
                "vmax": kwargs.get("vmax"),
                "cut_coords": kwargs.get("cut_coords"),
            }
        )

    # index_img returns frame token we can inspect later
    def fake_index_img(img, idx):
        return f"indexed-{id(img)}-{idx}"

    # crop_img wraps token so we can verify it's being called
    def fake_crop_img(indexed, crop_slices):
        return f"cropped({indexed})"

    monkeypatch.setattr("nireports.reportlets.utils.iio.imread", fake_imread)
    monkeypatch.setattr("nireports.reportlets.utils.plot_epi", fake_plot_epi)
    monkeypatch.setattr("nireports.reportlets.utils.nlimage.index_img", fake_index_img)
    monkeypatch.setattr("nireports.reportlets.utils.crop_img", fake_crop_img)

    frames = render_comparison_frames(
        uncorr_img,
        corr_img,
        n_frames,
        1.0,
        9.0,
        (1.0, 2.0, 3.0),
        (4.0, 5.0, 6.0),
        (slice(0, 5), slice(0, 5), slice(0, 5)),
    )

    # One combined frame per idx
    assert len(frames) == n_frames

    # uncorr: (8,5,3), corr: (10,7,3) => pad uncorr to height 10, concat width 12
    for combined in frames:
        assert combined.shape == (10, 12, 3)
        assert combined.dtype == np.uint8
        # left block from uncorr
        assert np.all(combined[:, :5, :] >= 0)
        # right block from corr
        assert np.all(combined[:, 5:, :] >= 0)

    # 2 plot calls per frame (uncorr + corr)
    assert len(plot_calls) == n_frames * 2

    # Spot-check call metadata
    assert any("Before motion correction | Frame 1" in c["title"] for c in plot_calls)
    assert any("After motion correction | Frame 1" in c["title"] for c in plot_calls)
    assert all(c["vmin"] == 1.0 for c in plot_calls)
    assert all(c["vmax"] == 9.0 for c in plot_calls)


def test_load_framewise_displacement(tmp_path):
    fd_path = tmp_path / "fd.tsv"

    fd_path.write_text("other\n1.0\n")
    with pytest.raises(ValueError):
        load_framewise_displacement(str(fd_path))

    fd_path.write_text("framewise_displacement\n0.1\n0.2\n")
    values = load_framewise_displacement(str(fd_path))
    assert np.allclose(values, [0.1, 0.2])

    fd_path.write_text("FD\n0.0\n")
    values = load_framewise_displacement(str(fd_path))
    assert np.allclose(values, [0.0])
