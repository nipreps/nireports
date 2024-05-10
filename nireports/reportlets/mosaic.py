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
# NiPreps projects licensed under the Apache-2.0 terms.
"""Base components to generate mosaic-like reportlets."""

import math
from os import path as op
from uuid import uuid4
from warnings import warn

import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np
from matplotlib.gridspec import GridSpec
from nilearn import image as nlimage
from nilearn.plotting import plot_anat
from svgutils.transform import fromstring

from nireports.reportlets.utils import (
    _3d_in_file,
    _bbox,
    _get_limits,
    cuts_from_bbox,
    extract_svg,
    get_parula,
    robust_set_limits,
)
from nireports.tools.ndimage import rotate_affine, rotation2canonical


def plot_segs(
    image_nii,
    seg_niis,
    out_file,
    bbox_nii=None,
    masked=False,
    colors=None,
    compress="auto",
    **plot_params,
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
        image_nii if bbox_nii is None else rotate_affine(_3d_in_file(bbox_nii), rot=canonical_r)
    )

    if masked:
        bbox_nii = nlimage.threshold_img(bbox_nii, 1e-3)

    cuts = cuts_from_bbox(bbox_nii, cuts=7)
    plot_params["colors"] = colors or plot_params.get("colors", None)
    out_files = []
    for d in plot_params.pop("dimensions", ("z", "x", "y")):
        plot_params["display_mode"] = d
        plot_params["cut_coords"] = cuts[d]
        svg = _plot_anat_with_contours(image_nii, segs=seg_niis, compress=compress, **plot_params)
        # Find and replace the figure_1 id.
        svg = svg.replace("figure_1", f"segmentation-{d}-{uuid4()}", 1)
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
        contour = nb.Nifti1Image.from_image(contour)

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
        svg = svg.replace("figure_1", f"{div_id}-{mode}-{uuid4()}", 1)
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


def plot_segmentation(anat_file, segmentation, out_file, **kwargs):
    """Plot a segmentation (from MRIQC)."""
    from nilearn.plotting import plot_anat
    from nitransforms.io.afni import _dicom_real_to_card

    vmax = kwargs.get("vmax")
    vmin = kwargs.get("vmin")

    anat_ras = nb.as_closest_canonical(nb.load(anat_file))
    anat_ras_plumb = anat_ras.__class__(
        anat_ras.dataobj, _dicom_real_to_card(anat_ras.affine), anat_ras.header
    )

    seg_ras = nb.as_closest_canonical(nb.load(segmentation))
    seg_ras_plumb = seg_ras.__class__(
        seg_ras.dataobj, _dicom_real_to_card(seg_ras.affine), seg_ras.header
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


def plot_slice(
    dslice,
    spacing=None,
    cmap="Greys_r",
    label=None,
    ax=None,
    vmax=None,
    vmin=None,
    annotate=None,
):
    if isinstance(cmap, (str, bytes)):
        cmap = mpl.colormaps[cmap]

    est_vmin, est_vmax = _get_limits(dslice)
    if not vmin:
        vmin = est_vmin
    if not vmax:
        vmax = est_vmax

    if ax is None:
        ax = plt.gca()

    if spacing is None:
        spacing = [1.0, 1.0]

    # Always swap axes because imshow defines the image as (M, N)
    # where M are rows (i.e., Y axis) and N are columns (X axis)
    dslice = np.swapaxes(dslice, 0, 1)
    spacing = (spacing[1], spacing[0])

    ax.imshow(
        dslice,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        extent=[0, dslice.shape[1] * spacing[1], 0, dslice.shape[0] * spacing[0]],
        interpolation="none",
        origin="lower",
    )
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(False)
    ax.axis("off")

    bgcolor = cmap(min(vmin, 0.0))
    fgcolor = cmap(vmax)

    if annotate is not None:
        ax.text(
            0.95,
            0.95,
            annotate[0],
            color=fgcolor,
            transform=ax.transAxes,
            horizontalalignment="center",
            verticalalignment="top",
            size=14,
            bbox={"boxstyle": "square,pad=0", "ec": bgcolor, "fc": bgcolor},
        )
        ax.text(
            0.05,
            0.95,
            annotate[1],
            color=fgcolor,
            transform=ax.transAxes,
            horizontalalignment="center",
            verticalalignment="top",
            size=14,
            bbox={"boxstyle": "square,pad=0", "ec": bgcolor, "fc": bgcolor},
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
            size=14,
            bbox={"boxstyle": "square,pad=0", "ec": bgcolor, "fc": bgcolor},
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
    if isinstance(cmap, (str, bytes)):
        cmap = mpl.colormaps[cmap]

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
            bbox={"boxstyle": "square,pad=0", "ec": "k", "fc": "k"},
            color="w",
        )


def plot_spikes(
    in_file,
    in_fft,
    spikes_list,
    cols=3,
    labelfmt="t={0:.3f}s (z={1:d})",
    out_file=None,
):
    """Plot a mosaic enhancing EM spikes."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    nii = nb.as_closest_canonical(nb.load(in_file))
    fft = nb.load(in_fft).get_fdata()

    data = nii.get_fdata()
    zooms = nii.header.get_zooms()[:2]
    tstep = nii.header.get_zooms()[-1]
    ntpoints = data.shape[-1]

    if len(spikes_list) > cols * 7:
        cols += 1

    nspikes = len(spikes_list)
    rows = 1
    if nspikes > cols:
        rows = math.ceil(nspikes / cols)

    fig = plt.figure(figsize=(7 * cols, 5 * rows))

    for i, (t, z) in enumerate(spikes_list):
        prev = None
        pvft = None
        if t > 0:
            prev = data[..., z, t - 1]
            pvft = fft[..., z, t - 1]

        post = None
        psft = None
        if t < (ntpoints - 1):
            post = data[..., z, t + 1]
            psft = fft[..., z, t + 1]

        ax1 = fig.add_subplot(rows, cols, i + 1)
        divider = make_axes_locatable(ax1)
        ax2 = divider.new_vertical(size="100%", pad=0.1)
        fig.add_axes(ax2)

        plot_slice_tern(
            data[..., z, t],
            prev=prev,
            post=post,
            spacing=zooms,
            ax=ax2,
            label=labelfmt.format(t * tstep, z),
        )

        plot_slice_tern(
            fft[..., z, t],
            prev=pvft,
            post=psft,
            vmin=-5,
            vmax=5,
            cmap=get_parula(),
            ax=ax1,
        )

    plt.tight_layout()
    if out_file is None:
        fname, ext = op.splitext(op.basename(in_file))
        if ext == ".gz":
            fname, _ = op.splitext(fname)
        out_file = op.abspath("%s.svg" % fname)

    fig.savefig(out_file, format="svg", dpi=300, bbox_inches="tight")
    return out_file


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
    plot_sagittal=False,
    fig=None,
    maxrows=16,
    views=("axial", "sagittal", None),
):
    """Plot a mosaic of 2D cuts."""

    VIEW_AXES_ORDER = {
        "axial": (0, 1, 2),
        "sagittal": (2, 0, 1),
        "coronal": (0, 2, 1),
    }

    if isinstance(views, str):
        views = (views, None, None)

    if len(views) != 3:
        _views = [None, None, None]

        for ii, vv in enumerate(views):
            _views[ii] = vv

        views = tuple(_views)

    # Error with inconsistent views input
    if views[0] is None or ((views[1] is None) and (views[2] is not None)):
        raise RuntimeError("First view must not be None")

    if not hasattr(img, "shape"):
        nii = nb.as_closest_canonical(nb.load(img))
        img_data = nii.get_fdata()
        zooms = nii.header.get_zooms()[:3]
    else:
        img_data = img
        zooms = [1.0, 1.0, 1.0]
        out_file = "mosaic.svg"

    if plot_sagittal and views[1] is None and views[0] != "sagittal":
        warn("Argument ``plot_sagittal`` for plot_mosaic() should not be used.", stacklevel=2)
        views = (views[0], "sagittal", None)

    # Create mask for bounding box
    bbox_data = None
    if bbox_mask_file is not None:
        bbox_data = np.asanyarray(nb.as_closest_canonical(nb.load(bbox_mask_file)).dataobj) > 1e-3
    elif img_data.shape[-1] > (ncols * maxrows):
        lowthres = np.percentile(img_data, 5)
        bbox_data = np.ones_like(img_data)
        bbox_data[img_data <= lowthres] = 0

    if bbox_data is not None:
        img_data = _bbox(img_data, bbox_data)

    shape = np.array(img_data.shape[:3])
    extents = shape * zooms

    view_x = {"axial": 0, "coronal": 0, "sagittal": 1}
    view_y = {"axial": 1, "coronal": 2, "sagittal": 2}
    axannotation = {"axial": "RL", "coronal": "RL", "sagittal": "AP"}

    nrows = min((shape[-1] + 1) // ncols, maxrows)

    # create figures
    if fig is None:
        fig = plt.figure(layout=None)

    # Load overlay if present
    if overlay_mask:
        overlay_data = nb.as_closest_canonical(nb.load(overlay_mask)).get_fdata()

        if bbox_data is not None:
            overlay_data = _bbox(overlay_data, bbox_data)

    # Decimate if too many values
    z_vals = np.unique(
        np.linspace(
            0,
            shape[-1] - 1,
            num=(ncols * nrows),
            dtype=int,
            endpoint=True,
        )
    )
    n_gs = sum(bool(v) for v in views)

    main_mosaic_idx = np.full((nrows * ncols,), -1, dtype=int)
    main_mosaic_idx[: len(z_vals)] = z_vals
    main_mosaic_idx = main_mosaic_idx.reshape(nrows, ncols)

    fig_height = []
    panel_width = []
    ncols = [ncols]
    view_spacing = []
    for ii, vv in enumerate(views):
        if vv is None:
            break

        axis_x = view_x[vv]
        axis_y = view_y[vv]
        view_spacing.append((zooms[axis_x], zooms[axis_y]))

        view_rows = nrows
        if ii > 0:
            ncols.append(int(panel_width[0] // extents[axis_x]))
            view_rows = 1

        fig_height.append(extents[axis_y] * view_rows)
        panel_width.append(extents[axis_x] * ncols[-1])

    fig_ratio = sum(fig_height) / panel_width[0]
    fig.set_size_inches(20, 20 * fig_ratio)

    subfigs = GridSpec(
        nrows=n_gs,
        ncols=1,
        # top=0.96,
        # bottom=0.01,
        hspace=0.001,
        height_ratios=np.array(fig_height) / fig_height[0],
    )

    est_vmin, est_vmax = _get_limits(img_data, only_plot_noise=only_plot_noise)
    if not vmin:
        vmin = est_vmin
    if not vmax:
        vmax = est_vmax

    # Fill in the main mosaic panel
    panel_axs = subfigs[0].subgridspec(
        nrows,
        ncols[0],
        hspace=0.0001,
        wspace=0.0001,
    )

    # Mosaic view 1: nrows x ncols (main view)
    view_data = np.moveaxis(
        np.squeeze(img_data),
        (0, 1, 2),
        VIEW_AXES_ORDER[views[0]],
    )

    if overlay_mask:
        view_overlay_data = np.moveaxis(
            overlay_data,
            (0, 1, 2),
            VIEW_AXES_ORDER[views[0]],
        )

    for ii, row_slices in enumerate(main_mosaic_idx):
        for jj, z_val in enumerate(row_slices):
            if z_val < 0:
                break

            ax = fig.add_subplot(panel_axs[ii, jj])
            if overlay_mask:
                panel_axs[ii, jj].set_rasterized(True)

            plot_slice(
                view_data[:, :, z_val],
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                ax=ax,
                spacing=view_spacing[0],
                label=f"{z_val:d}",
                annotate=axannotation[views[0]] if annotate else None,
            )

            if overlay_mask:
                msk_cmap = mpl.colormaps["Reds"]
                msk_cmap._init()
                alphas = np.linspace(0, 0.75, msk_cmap.N + 3)
                msk_cmap._lut[:, -1] = alphas
                plot_slice(
                    view_overlay_data[:, :, z_val],
                    vmin=0,
                    vmax=1,
                    cmap=msk_cmap,
                    ax=panel_axs[ii, jj],
                    spacing=view_spacing[0],
                )

    if views[1] is not None:
        # Mosaic view 2
        view_data = np.moveaxis(
            np.squeeze(img_data),
            (0, 1, 2),
            VIEW_AXES_ORDER[views[1]],
        )

        if overlay_mask:
            view_overlay_data = np.moveaxis(
                overlay_data,
                (0, 1, 2),
                VIEW_AXES_ORDER[views[1]],
            )
        step = max(int(view_data.shape[2] / (ncols[1] + 1)), 1)
        start = step
        stop = view_data.shape[2] - step
        panel_axs = subfigs[1].subgridspec(1, ncols[1], wspace=0.0001)

        y_vals = np.linspace(start, stop, num=ncols[1], dtype=int, endpoint=True)
        for jj, slice_val in enumerate(y_vals):
            ax = fig.add_subplot(panel_axs[jj])
            plot_slice(
                view_data[:, :, slice_val],
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                ax=ax,
                label=f"{slice_val:d}",
                spacing=view_spacing[1],
                annotate=axannotation[views[1]] if annotate else None,
            )

    if views[1] is not None and views[2] is not None:
        # Mosaic view 2
        view_data = np.moveaxis(
            np.squeeze(img_data),
            (0, 1, 2),
            VIEW_AXES_ORDER[views[2]],
        )

        if overlay_mask:
            view_overlay_data = np.moveaxis(
                overlay_data,
                (0, 1, 2),
                VIEW_AXES_ORDER[views[2]],
            )

        step = max(int(view_data.shape[2] / (ncols[2] + 1)), 1)
        start = step
        stop = view_data.shape[2] - step
        panel_axs = subfigs[2].subgridspec(1, ncols[2], wspace=0.0001)

        x_vals = np.linspace(start, stop, num=ncols[2], dtype=int, endpoint=True)
        for jj, slice_val in enumerate(x_vals):
            ax = fig.add_subplot(panel_axs[jj])
            plot_slice(
                view_data[:, :, slice_val],
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                ax=ax,
                label=f"{slice_val:d}",
                spacing=view_spacing[2],
                annotate=axannotation[views[2]] if annotate else None,
            )

    if title:
        fig.suptitle(title, fontsize="10")

    # fig.subplots_adjust(wspace=0.002, hspace=0.002)

    if out_file is None:
        fname, ext = op.splitext(op.basename(img))
        if ext == ".gz":
            fname, _ = op.splitext(fname)
        out_file = op.abspath(fname + "_mosaic.svg")

    fig.savefig(out_file, format="svg", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_file
