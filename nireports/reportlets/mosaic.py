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
# STATEMENT OF CHANGES: This file was ported carrying over full git history from niworkflows,
# another NiPreps project licensed under the Apache-2.0 terms, and has been changed since.
# The original file this work derives from is found at:
# https://github.com/nipreps/niworkflows/blob/fa273d004c362d9562616253180e95694f07be3b/
# niworkflows/viz/plots.py
"""Base components to generate mosaic-like reportlets."""
from uuid import uuid4
import numpy as np
import nibabel as nb
from svgutils.transform import fromstring
from nilearn.plotting import plot_anat
from nilearn import image as nlimage

from nireports.reportlets.misc import rotate_affine, rotation2canonical
from nireports.reportlets.utils import (
    _3d_in_file,
    cuts_from_bbox,
    extract_svg,
    robust_set_limits,
)


def plot_segs(
    image_nii,
    seg_niis,
    out_file,
    bbox_nii=None,
    masked=False,
    colors=None,
    compress="auto",
    **plot_params
):
    """
    Generate a static mosaic with ROIs represented by their delimiting contour.

    Plot segmentation as contours over the image (e.g. anatomical).
    seg_niis should be a list of files. mask_nii helps determine the cut
    coordinates. plot_params will be passed on to nilearn plot_* functions. If
    seg_niis is a list of size one, it behaves as if it was plotting the mask.
    """

    plot_params = {} if plot_params is None else plot_params

    image_nii = _3d_in_file(image_nii)
    canonical_r = rotation2canonical(image_nii)
    image_nii = rotate_affine(image_nii, rot=canonical_r)
    seg_niis = [rotate_affine(_3d_in_file(f), rot=canonical_r) for f in seg_niis]
    data = image_nii.get_fdata()

    plot_params = robust_set_limits(data, plot_params)

    bbox_nii = (
        image_nii if bbox_nii is None
        else rotate_affine(_3d_in_file(bbox_nii), rot=canonical_r)
    )

    if masked:
        bbox_nii = nlimage.threshold_img(bbox_nii, 1e-3)

    cuts = cuts_from_bbox(bbox_nii, cuts=7)
    plot_params["colors"] = colors or plot_params.get("colors", None)
    out_files = []
    for d in plot_params.pop("dimensions", ("z", "x", "y")):
        plot_params["display_mode"] = d
        plot_params["cut_coords"] = cuts[d]
        svg = _plot_anat_with_contours(
            image_nii, segs=seg_niis, compress=compress, **plot_params
        )
        # Find and replace the figure_1 id.
        svg = svg.replace("figure_1", "segmentation-%s-%s" % (d, uuid4()), 1)
        out_files.append(fromstring(svg))

    return out_files


def plot_registration(
    anat_nii,
    div_id,
    plot_params=None,
    order=("z", "x", "y"),
    cuts=None,
    estimate_brightness=False,
    label=None,
    contour=None,
    compress="auto",
    dismiss_affine=False,
):
    """
    Plots the foreground and background views
    Default order is: axial, coronal, sagittal
    """

    plot_params = {} if plot_params is None else plot_params

    # Use default MNI cuts if none defined
    if cuts is None:
        raise NotImplementedError  # TODO

    # nilearn 0.10.0 uses Nifti-specific methods
    anat_nii = nb.Nifti1Image.from_image(anat_nii)

    out_files = []
    if estimate_brightness:
        plot_params = robust_set_limits(anat_nii.get_fdata().reshape(-1), plot_params)

    # FreeSurfer ribbon.mgz
    if contour:
        contour = nb.Nifti1Image.from_image(anat_nii)

    ribbon = contour is not None and np.array_equal(
        np.unique(contour.get_fdata()), [0, 2, 3, 41, 42]
    )

    if ribbon:
        contour_data = contour.get_fdata() % 39
        white = nlimage.new_img_like(contour, contour_data == 2)
        pial = nlimage.new_img_like(contour, contour_data >= 2)

    if dismiss_affine:
        canonical_r = rotation2canonical(anat_nii)
        anat_nii = rotate_affine(anat_nii, rot=canonical_r)
        if ribbon:
            white = rotate_affine(white, rot=canonical_r)
            pial = rotate_affine(pial, rot=canonical_r)
        if contour:
            contour = rotate_affine(contour, rot=canonical_r)

    # Plot each cut axis
    for i, mode in enumerate(list(order)):
        plot_params["display_mode"] = mode
        plot_params["cut_coords"] = cuts[mode]
        if i == 0:
            plot_params["title"] = label
        else:
            plot_params["title"] = None

        # Generate nilearn figure
        display = plot_anat(anat_nii, **plot_params)
        if ribbon:
            kwargs = {"levels": [0.5], "linewidths": 0.5}
            display.add_contours(white, colors="b", **kwargs)
            display.add_contours(pial, colors="r", **kwargs)
        elif contour is not None:
            display.add_contours(contour, colors="r", levels=[0.5], linewidths=0.5)

        svg = extract_svg(display, compress=compress)
        display.close()

        # Find and replace the figure_1 id.
        svg = svg.replace("figure_1", "%s-%s-%s" % (div_id, mode, uuid4()), 1)
        out_files.append(fromstring(svg))

    return out_files


def _plot_anat_with_contours(image, segs=None, compress="auto", **plot_params):

    nsegs = len(segs or [])
    plot_params = plot_params or {}
    # plot_params' values can be None, however they MUST NOT
    # be None for colors and levels from this point on.
    colors = plot_params.pop("colors", None) or []
    levels = plot_params.pop("levels", None) or []
    missing = nsegs - len(colors)
    if missing > 0:  # missing may be negative
        from seaborn import color_palette

        colors = colors + color_palette("husl", missing)

    colors = [[c] if not isinstance(c, list) else c for c in colors]

    if not levels:
        levels = [[0.5]] * nsegs

    # anatomical
    display = plot_anat(image, **plot_params)

    # remove plot_anat -specific parameters
    plot_params.pop("display_mode")
    plot_params.pop("cut_coords")

    plot_params["linewidths"] = 0.5
    for i in reversed(range(nsegs)):
        plot_params["colors"] = colors[i]
        display.add_contours(segs[i], levels=levels[i], **plot_params)

    svg = extract_svg(display, compress=compress)
    display.close()
    return svg
