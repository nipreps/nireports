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
# niworkflows/utils/images.py
"""Tooling to manipulate n-dimensional images."""

import os
import typing as ty

import nibabel as nb
import numpy as np
import numpy.typing as npt
from nibabel.spatialimages import SpatialImage

ImgT = ty.TypeVar("ImgT", bound=nb.filebasedimages.FileBasedImage)
SpatImgT = ty.TypeVar("SpatImgT", bound=SpatialImage)
Mat = npt.NDArray[np.float64]


def rotation2canonical(img: SpatialImage) -> ty.Union[Mat, None]:
    """Calculate the rotation w.r.t. cardinal axes of input image."""
    img = nb.as_closest_canonical(img)
    # XXX: SpatialImage.affine needs to be typed
    affine: Mat = img.affine
    newaff = np.diag(img.header.get_zooms()[:3])
    r = newaff @ np.linalg.pinv(affine[:3, :3])
    if np.allclose(r, np.eye(3)):
        return None
    return r


def rotate_affine(img: SpatImgT, rot: ty.Union[Mat, None] = None) -> SpatImgT:
    """Rewrite the affine of a spatial image."""
    if rot is None:
        return img

    img = nb.as_closest_canonical(img)
    affine = np.eye(4)
    affine[:3] = rot @ img.affine[:3]
    return img.__class__(img.dataobj, affine, img.header)


def load_api(path: ty.Union[str, os.PathLike[str]], api: type[ImgT]) -> ImgT:
    img = nb.load(path)
    if not isinstance(img, api):
        raise TypeError(f"File {path} does not implement {api} interface")
    return img


def _get_values_inside_a_mask(
    main_file: ty.Union[str, os.PathLike[str]],
    mask_file: ty.Union[str, os.PathLike[str]],
) -> npt.NDArray[np.float64]:
    main_nii = load_api(main_file, SpatialImage)
    main_data = main_nii.get_fdata()
    nan_mask = np.logical_not(np.isnan(main_data))
    mask = load_api(mask_file, SpatialImage).get_fdata() > 0

    data = main_data[np.logical_and(nan_mask, mask)]
    return data
