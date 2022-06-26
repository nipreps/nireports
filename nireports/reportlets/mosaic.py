# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2021 The NiPreps Developers <nipreps@gmail.com>
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
"""
Static segmentation plots with contours.
"""
import math
import os.path as op

import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np

def plot_slice(
    dslice,
    spacing=None,
    cmap="Greys_r",
    label=None,
    ax=None,
    vmax=None,
    vmin=None,
    annotate=False,
):
    from matplotlib.cm import get_cmap

    if isinstance(cmap, (str, bytes)):
        cmap = get_cmap(cmap)

    est_vmin, est_vmax = _get_limits(dslice)
    if not vmin:
        vmin = est_vmin
    if not vmax:
        vmax = est_vmax

    if ax is None:
        ax = plt.gca()

    if spacing is None:
        spacing = [1.0, 1.0]

    phys_sp = np.array(spacing) * dslice.shape
    ax.imshow(
        np.swapaxes(dslice, 0, 1),
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        interpolation="nearest",
        origin="lower",
        extent=[0, phys_sp[0], 0, phys_sp[1]],
    )
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(False)
    ax.axis("off")

    bgcolor = cmap(min(vmin, 0.0))
    fgcolor = cmap(vmax)

    if annotate:
        ax.text(
            0.95,
            0.95,
            "R",
            color=fgcolor,
            transform=ax.transAxes,
            horizontalalignment="center",
            verticalalignment="top",
            size=18,
            bbox=dict(boxstyle="square,pad=0", ec=bgcolor, fc=bgcolor),
        )
        ax.text(
            0.05,
            0.95,
            "L",
            color=fgcolor,
            transform=ax.transAxes,
            horizontalalignment="center",
            verticalalignment="top",
            size=18,
            bbox=dict(boxstyle="square,pad=0", ec=bgcolor, fc=bgcolor),
        )

    if label is not None:
        ax.text(
            0.98,
            0.01,
            label,
            color=fgcolor,
            transform=ax.transAxes,
            horizontalalignment="right",
            verticalalignment="bottom",
            size=18,
            bbox=dict(boxstyle="square,pad=0", ec=bgcolor, fc=bgcolor),
        )

    return ax


def plot_slice_tern(
    dslice,
    prev=None,
    post=None,
    spacing=None,
    cmap="Greys_r",
    label=None,
    ax=None,
    vmax=None,
    vmin=None,
):
    from matplotlib.cm import get_cmap

    if isinstance(cmap, (str, bytes)):
        cmap = get_cmap(cmap)

    est_vmin, est_vmax = _get_limits(dslice)
    if not vmin:
        vmin = est_vmin
    if not vmax:
        vmax = est_vmax

    if ax is None:
        ax = plt.gca()

    if spacing is None:
        spacing = [1.0, 1.0]
    else:
        spacing = [spacing[1], spacing[0]]

    phys_sp = np.array(spacing) * dslice.shape

    if prev is None:
        prev = np.ones_like(dslice)
    if post is None:
        post = np.ones_like(dslice)

    combined = np.swapaxes(np.vstack((prev, dslice, post)), 0, 1)
    ax.imshow(
        combined,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        interpolation="nearest",
        origin="lower",
        extent=[0, phys_sp[1] * 3, 0, phys_sp[0]],
    )
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(False)

    if label is not None:
        ax.text(
            0.5,
            0.05,
            label,
            transform=ax.transAxes,
            horizontalalignment="center",
            verticalalignment="top",
            size=14,
            bbox=dict(boxstyle="square,pad=0", ec="k", fc="k"),
            color="w",
        )

def plot_mosaic(
    img,
    out_file=None,
    ncols=8,
    title=None,
    overlay_mask=None,
    bbox_mask_file=None,
    only_plot_noise=False,
    annotate=True,
    vmin=None,
    vmax=None,
    cmap="Greys_r",
    plot_sagittal=True,
    fig=None,
    zmax=128,
):

    if isinstance(img, (str, bytes)):
        nii = nb.as_closest_canonical(nb.load(img))
        img_data = nii.get_fdata()
        zooms = nii.header.get_zooms()
    else:
        img_data = img
        zooms = [1.0, 1.0, 1.0]
        out_file = "mosaic.svg"

    # Remove extra dimensions
    img_data = np.squeeze(img_data)

    if img_data.shape[2] > zmax and bbox_mask_file is None:
        lowthres = np.percentile(img_data, 5)
        mask_file = np.ones_like(img_data)
        mask_file[img_data <= lowthres] = 0
        img_data = _bbox(img_data, mask_file)

    if bbox_mask_file is not None:
        bbox_data = nb.as_closest_canonical(nb.load(bbox_mask_file)).get_fdata()
        img_data = _bbox(img_data, bbox_data)

    z_vals = np.array(list(range(0, img_data.shape[2])))

    # Reduce the number of slices shown
    if len(z_vals) > zmax:
        rem = 15
        # Crop inferior and posterior
        if not bbox_mask_file:
            # img_data = img_data[..., rem:-rem]
            z_vals = z_vals[rem:-rem]
        else:
            # img_data = img_data[..., 2 * rem:]
            start_index = 2 * rem
            z_vals = z_vals[start_index:]

    while len(z_vals) > zmax:
        # Discard one every two slices
        # img_data = img_data[..., ::2]
        z_vals = z_vals[::2]

    n_images = len(z_vals)
    nrows = math.ceil(n_images / ncols)
    if plot_sagittal:
        nrows += 1

    if overlay_mask:
        overlay_data = nb.as_closest_canonical(nb.load(overlay_mask)).get_fdata()

    # create figures
    if fig is None:
        fig = plt.figure(figsize=(22, nrows * 3))

    est_vmin, est_vmax = _get_limits(img_data, only_plot_noise=only_plot_noise)
    if not vmin:
        vmin = est_vmin
    if not vmax:
        vmax = est_vmax

    naxis = 1
    for z_val in z_vals:
        ax = fig.add_subplot(nrows, ncols, naxis)

        if overlay_mask:
            ax.set_rasterized(True)
        plot_slice(
            img_data[:, :, z_val],
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            ax=ax,
            spacing=zooms[:2],
            label="%d" % z_val,
            annotate=annotate,
        )

        if overlay_mask:
            from matplotlib import cm

            msk_cmap = cm.Reds  # @UndefinedVariable
            msk_cmap._init()
            alphas = np.linspace(0, 0.75, msk_cmap.N + 3)
            msk_cmap._lut[:, -1] = alphas
            plot_slice(
                overlay_data[:, :, z_val],
                vmin=0,
                vmax=1,
                cmap=msk_cmap,
                ax=ax,
                spacing=zooms[:2],
            )
        naxis += 1

    if plot_sagittal:
        naxis = ncols * (nrows - 1) + 1

        step = int(img_data.shape[0] / (ncols + 1))
        start = step
        stop = img_data.shape[0] - step

        if step == 0:
            step = 1

        for x_val in list(range(start, stop, step))[:ncols]:
            ax = fig.add_subplot(nrows, ncols, naxis)

            plot_slice(
                img_data[x_val, ...],
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                ax=ax,
                label="%d" % x_val,
                spacing=[zooms[0], zooms[2]],
            )
            naxis += 1

    fig.subplots_adjust(
        left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.05
    )

    if title:
        fig.suptitle(title, fontsize="10")
    fig.subplots_adjust(wspace=0.002, hspace=0.002)

    if out_file is None:
        fname, ext = op.splitext(op.basename(img))
        if ext == ".gz":
            fname, _ = op.splitext(fname)
        out_file = op.abspath(fname + "_mosaic.svg")

    fig.savefig(out_file, format="svg", dpi=300, bbox_inches="tight")
    return out_file

def plot_segmentation(anat_file, segmentation, out_file, **kwargs):
    from nilearn.plotting import plot_anat
    from nitransforms.io.afni import _dicom_real_to_card

    vmax = kwargs.get("vmax")
    vmin = kwargs.get("vmin")

    anat_ras = nb.as_closest_canonical(nb.load(anat_file))
    anat_ras_plumb = anat_ras.__class__(
        anat_ras.dataobj,
        _dicom_real_to_card(anat_ras.affine),
        anat_ras.header
    )

    seg_ras = nb.as_closest_canonical(nb.load(segmentation))
    seg_ras_plumb = seg_ras.__class__(
        seg_ras.dataobj,
        _dicom_real_to_card(seg_ras.affine),
        seg_ras.header
    )

    if kwargs.get("saturate", False):
        vmax = np.percentile(anat_ras.get_fdata().reshape(-1), 70)

    if vmax is None and vmin is None:
        vmin = np.percentile(anat_ras.get_fdata().reshape(-1), 10)
        vmax = np.percentile(anat_ras.get_fdata().reshape(-1), 99)

    disp = plot_anat(
        anat_ras_plumb,
        display_mode=kwargs.get("display_mode", "ortho"),
        cut_coords=kwargs.get("cut_coords", 8),
        title=kwargs.get("title"),
        vmax=vmax,
        vmin=vmin,
    )
    disp.add_contours(
        seg_ras_plumb,
        levels=kwargs.get("levels", [1]),
        colors=kwargs.get("colors", "r"),
    )
    disp.savefig(out_file)
    disp.close()
    disp = None
    return out_file

def _get_limits(nifti_file, only_plot_noise=False):
    if isinstance(nifti_file, str):
        nii = nb.as_closest_canonical(nb.load(nifti_file))
        data = nii.get_fdata()
    else:
        data = nifti_file

    data_mask = np.logical_not(np.isnan(data))

    if only_plot_noise:
        data_mask = np.logical_and(data_mask, data != 0)
        vmin = np.percentile(data[data_mask], 0)
        vmax = np.percentile(data[data_mask], 61)
    else:
        vmin = np.percentile(data[data_mask], 0.5)
        vmax = np.percentile(data[data_mask], 99.5)

    return vmin, vmax

def _bbox(img_data, bbox_data):
    B = np.argwhere(bbox_data)
    (ystart, xstart, zstart), (ystop, xstop, zstop) = B.min(0), B.max(0) + 1
    return img_data[ystart:ystop, xstart:xstop, zstart:zstop]

