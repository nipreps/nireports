# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2022 The NiPreps Developers <nipreps@gmail.com>
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
# niworkflows/utils/timeseries.py
"""Extracting signals from NIfTI and CIFTI2 files."""

import typing as ty
from collections.abc import Sequence

import nibabel as nb
import numpy as np
import numpy.typing as npt

from .ndimage import load_api


def get_tr(img: ty.Union[nb.Nifti1Image, nb.Cifti2Image]) -> float:
    """
    Attempt to extract repetition time from NIfTI/CIFTI header.

    Examples
    --------
    >>> get_tr(nb.load(
    ...     test_data_package
    ...     / 'sub-ds205s03_task-functionallocalizer_run-01_bold_volreg.nii.gz'
    ... ))
    2.2
    >>> get_tr(nb.load(
    ...     test_data_package
    ...     / 'sub-01_task-mixedgamblestask_run-02_space-fsLR_den-91k_bold.dtseries.nii'
    ... ))
    2.0

    """
    if isinstance(img, nb.Cifti2Image):
        return img.header.matrix.get_index_map(0).series_step
    else:
        return img.header.get_zooms()[-1]
    raise RuntimeError("Could not extract TR - unknown data structure type")


def cifti_timeseries(
    dataset: ty.Union[str, nb.Cifti2Image],
) -> tuple[npt.NDArray[np.float32], dict[str, list[int]]]:
    """Extract timeseries from CIFTI2 dataset."""
    dataset = load_api(dataset, nb.Cifti2Image) if isinstance(dataset, str) else dataset

    if dataset.nifti_header.get_intent()[0] != "ConnDenseSeries":
        raise ValueError("Not a dense timeseries")

    matrix = dataset.header.matrix
    labels = {
        "CIFTI_STRUCTURE_CORTEX_LEFT": "CtxL",
        "CIFTI_STRUCTURE_CORTEX_RIGHT": "CtxR",
        "CIFTI_STRUCTURE_CEREBELLUM_LEFT": "CbL",
        "CIFTI_STRUCTURE_CEREBELLUM_RIGHT": "CbR",
    }
    seg: dict[str, list[int]] = {label: [] for label in list(labels.values()) + ["Other"]}
    for bm in matrix.get_index_map(1).brain_models:
        label = labels.get(bm.brain_structure, "Other")
        seg[label] += list(range(bm.index_offset, bm.index_offset + bm.index_count))

    return dataset.get_fdata(dtype=np.float32).T, seg


def nifti_timeseries(
    dataset: ty.Union[str, nb.Nifti1Image],
    segmentation: ty.Union[str, nb.Nifti1Image, None] = None,
    labels: Sequence[str] = ("Ctx GM", "dGM", "WM+CSF", "Cb", "Crown"),
    remap_rois: bool = False,
    lut: ty.Union[npt.NDArray[np.uint8], None] = None,
) -> tuple[npt.NDArray[np.float32], ty.Union[dict[str, list[int]], None]]:
    """Extract timeseries from NIfTI1/2 datasets."""
    dataset = load_api(dataset, nb.Nifti1Image) if isinstance(dataset, str) else dataset
    data: npt.NDArray[np.float32] = dataset.get_fdata(dtype="float32").reshape(
        (-1, dataset.shape[-1])
    )

    if segmentation is None:
        return data, None

    # Open NIfTI and extract numpy array
    segmentation = (
        load_api(segmentation, nb.Nifti1Image) if isinstance(segmentation, str) else segmentation
    )
    seg_data = np.asanyarray(segmentation.dataobj, dtype=int).reshape(-1)

    remap_rois = remap_rois or (len(np.unique(seg_data[seg_data > 0])) > len(labels))

    # Map segmentation
    if remap_rois or lut is not None:
        if lut is None:
            lut = np.zeros((256,), dtype="uint8")
            lut[100:201] = 1  # Ctx GM
            lut[30:99] = 2  # dGM
            lut[1:11] = 3  # WM+CSF
            lut[255] = 4  # Cerebellum
        # Apply lookup table
        seg_data = lut[seg_data]

    fgmask = seg_data > 0
    seg_values = seg_data[fgmask]
    seg_dict: dict[str, list[int]] = {}
    for i in np.unique(seg_values):
        seg_dict[labels[i - 1]] = list(np.argwhere(seg_values == i).squeeze())

    return data[fgmask], seg_dict
