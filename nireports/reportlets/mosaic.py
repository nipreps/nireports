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
from uuid import uuid4
from os import path as op
import math
import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.cm import get_cmap
from svgutils.transform import fromstring
from nilearn.plotting import plot_anat
from nilearn import image as nlimage

from nireports.tools.ndimage import rotate_affine, rotation2canonical
from nireports.reportlets.utils import (
    _3d_in_file,
    _bbox,
    _get_limits,
    cuts_from_bbox,
    extract_svg,
    get_parula,
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
    annotate=False,
    swapaxes=False,
):
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

    if swapaxes:
        dslice = np.swapaxes(dslice, 0, 1)
        spacing = (spacing[1], spacing[0])

    ax.imshow(
        dslice,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        extent=[0, dslice.shape[1] * spacing[1],
                0, dslice.shape[0] * spacing[0]],
        interpolation="none",
        origin="lower",
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
            size=14,
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
            size=14,
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
            size=14,
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
    plot_sagittal=True,
    fig=None,
    maxrows=16,
    views=("axial", "sagittal", None),
):
    """Plot a mosaic of 2D cuts."""

    VIEW_AXES_ORDER = (2, 1, 0)

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
        zooms = nii.header.get_zooms()
    else:
        img_data = img
        zooms = [1.0, 1.0, 1.0]
        out_file = "mosaic.svg"

    shape = img_data.shape[:3]
    view_hratios = {
        "axial": 1.0,
        "coronal": (zooms[2] * shape[2]) / (zooms[1] * shape[1]),
        "sagittal": (zooms[2] * shape[2]) / (zooms[1] * shape[1]),
    }
    view_x = {"axial": 0, "coronal": 0, "sagittal": 1}
    view_y = {"axial": 1, "coronal": 2, "sagittal": 2}

    if plot_sagittal and views[1] is None and views[0] != "sagittal":
        views = (views[0], "sagittal", None)

    # Select the axis through which we cut the planes
    axes_order = [
        ["sagittal", "coronal", "axial"].index(views[0]),
    ]

    if views[1] is not None:
        axes_order.append(["sagittal", "coronal", "axial"].index(views[1]))

    # If 3D, complete last axis
    if img_data.ndim > 3:
        raise RuntimeError("Dataset has more than three dimensions")
    elif img_data.ndim == 3:
        axes_order += list(set(range(3)) - set(axes_order))

    # Remove extra dimensions
    img_data = np.moveaxis(
        np.squeeze(img_data),
        axes_order,
        VIEW_AXES_ORDER[:len(axes_order)],
    )

    # Load overlay if present
    if overlay_mask:
        overlay_data = np.moveaxis(
            nb.as_closest_canonical(nb.load(overlay_mask)).get_fdata(),
            axes_order,
            VIEW_AXES_ORDER[:len(axes_order)],
        )

    # Create mask for bounding box
    if bbox_mask_file is not None:
        bbox_data = np.moveaxis(
            nb.as_closest_canonical(nb.load(bbox_mask_file)).get_fdata(),
            axes_order,
            VIEW_AXES_ORDER[:len(axes_order)],
        )
        img_data = _bbox(img_data, bbox_data)
    elif img_data.shape[-1] > (ncols * maxrows):
        lowthres = np.percentile(img_data, 5)
        mask_file = np.ones_like(img_data)
        mask_file[img_data <= lowthres] = 0
        img_data = _bbox(img_data, mask_file)

    nrows = min((img_data.shape[-1] + 1) // ncols, maxrows)

    # Decimate if too many values
    z_vals = np.unique(np.linspace(
        0, img_data.shape[-1] - 1, num=(ncols * nrows), dtype=int, endpoint=True,
    ))
    n_gs = sum(bool(v) for v in views)

    main_mosaic_idx = np.full((nrows * ncols, ), -1, dtype=int)
    main_mosaic_idx[:len(z_vals)] = z_vals
    main_mosaic_idx = main_mosaic_idx.reshape(nrows, ncols)

    swapaxes = views in (
        ("axial", "coronal", None),
        ("axial", "coronal", "sagittal"),
        ("coronal", "axial", None),
        ("coronal", "axial", "sagittal"),
        ("sagittal", "axial", "coronal"),
        ("sagittal", "axial", None),

    )

    nrows = [nrows, 1, 1]

    # create figures

    if fig is None:
        fig = plt.figure(layout=None)

    fig_height = []
    panel_width = []
    for ii, vv in enumerate(views):
        if vv is None:
            break

        axis_x = view_x[vv]
        axis_y = view_y[vv]

        fig_height.append(zooms[axis_y] * shape[axis_y] * nrows[ii])
        panel_width.append(zooms[axis_x] * shape[axis_x])

    fig_ratio = sum(fig_height) / (panel_width[0] * ncols)
    fig.set_size_inches(20, 20 * fig_ratio)

    height_ratios = [
        view_hratios[r] * nrows[i]
        for i, r in enumerate(views) if r is not None
    ]
    subfigs = GridSpec(
        nrows=n_gs,
        ncols=1,
        top=0.96,
        bottom=0.01,
        hspace=0.08,
        height_ratios=height_ratios if len(height_ratios) > 1 else [1],
    )

    est_vmin, est_vmax = _get_limits(img_data, only_plot_noise=only_plot_noise)
    if not vmin:
        vmin = est_vmin
    if not vmax:
        vmax = est_vmax

    slice_spacing = [vs for i, vs in enumerate(zooms) if i != axes_order[0]]

    # Fill in the main mosaic panel
    panel_axs = subfigs[0].subgridspec(nrows[0], ncols, hspace=0.01, wspace=0.00001)
    for ii, row_slices in enumerate(main_mosaic_idx):
        for jj, z_val in enumerate(row_slices):
            if z_val < 0:
                break

            ax = fig.add_subplot(panel_axs[ii, jj])
            if overlay_mask:
                panel_axs[ii, jj].set_rasterized(True)

            plot_slice(
                img_data[:, :, z_val],
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                ax=ax,
                spacing=slice_spacing,
                label=f"{z_val:d}",
                swapaxes=swapaxes,
                annotate=annotate and views[0] in ("axial", "coronal"),
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
                    ax=panel_axs[ii, jj],
                    spacing=slice_spacing,
                    swapaxes=swapaxes,
                )

    if views[1] is not None:
        ncols_2 = math.floor((panel_width[0] * ncols) / panel_width[1]) - 1
        slice_spacing = [vs for i, vs in enumerate(zooms) if i != axes_order[1]]
        step = max(int(img_data.shape[1] / (ncols_2 + 1)), 1)
        start = step
        stop = img_data.shape[1] - step
        panel_axs = subfigs[1].subgridspec(1, ncols_2, wspace=0.0001)

        swapaxes = views in (
            ("axial", "coronal", None),
            ("axial", "coronal", "sagittal"),
            ("axial", "sagittal", "coronal"),
            ("axial", "sagittal", None),
            ("coronal", "axial", None),
            ("coronal", "axial", "sagittal"),
        )

        y_vals = np.linspace(start, stop, num=ncols_2, dtype=int, endpoint=True)
        for jj, slice_val in enumerate(y_vals):
            ax = fig.add_subplot(panel_axs[jj])
            plot_slice(
                img_data[:, slice_val, :],
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                ax=ax,
                label=f"{slice_val:d}",
                spacing=slice_spacing,
                swapaxes=swapaxes,
                annotate=annotate and views[1] in ("axial", "coronal"),
            )

    if views[1] is not None and views[2] is not None:
        ncols_3 = math.floor((panel_width[0] * ncols) / panel_width[2]) - 1
        slice_spacing = [vs for i, vs in enumerate(zooms) if i != axes_order[2]]
        step = max(int(img_data.shape[0] / (ncols_3 + 1)), 1)
        start = step
        stop = img_data.shape[0] - step
        panel_axs = subfigs[2].subgridspec(1, ncols_3, wspace=0.0001)

        swapaxes = views in (
            ("axial", "coronal", "sagittal"),
            ("axial", "sagittal", "coronal"),
            ("coronal", "sagittal", "axial"),
        )

        x_vals = np.linspace(start, stop, num=ncols_3, dtype=int, endpoint=True)
        for jj, slice_val in enumerate(x_vals):
            ax = fig.add_subplot(panel_axs[jj])
            plot_slice(
                img_data[slice_val, ...],
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                ax=ax,
                label=f"{slice_val:d}",
                spacing=slice_spacing,
                swapaxes=swapaxes,
                annotate=annotate and views[2] in ("axial", "coronal"),
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
    return out_file
