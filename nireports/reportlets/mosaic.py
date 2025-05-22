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

from __future__ import annotations

import math
import os
import typing as ty
import warnings
from collections.abc import Sequence
from os import path as op
from typing import Literal as L
from uuid import uuid4

import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nb
import nilearn
import numpy as np
import numpy.typing as npt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from nibabel.spatialimages import SpatialImage
from nilearn import image as nlimage
from nilearn.plotting import plot_anat
from packaging.version import Version

from nireports._vendored.svgutils.transform import SVGFigure, fromstring
from nireports.reportlets.utils import (
    _3d_in_file,
    _bbox,
    _get_limits,
    cuts_from_bbox,
    extract_svg,
    get_parula,
    robust_set_limits,
)
from nireports.tools.ndimage import load_api, rotate_affine, rotation2canonical


def _create_lscmap_with_alpha(
    cmap_name: str, cmap_name_alpha: ty.Union[str, None] = None, max_alpha: float = 0.75
) -> LinearSegmentedColormap:
    """Create a linear segmented colormap with custom alpha (transparency) values.

    Creates a new colormap with ``N+3`` elements by adjusting the alpha channel
    (transparency) of a given base colormap with ``N`` elements. The alpha
    values are distributed linearly between ``0`` and ``alpha_max`` across the
    values of the range of the new colormap.

    Parameters
    ----------
    cmap_name : str
        Name of the base colormap from Matplotlib.
    cmap_name_alpha : str, optional
        Name for the new colormap. If ``None``, append "Alpha" to the base colormap name.
    max_alpha : float, optional
        Maximum alpha value (transparency) to apply to the colormap.

    Returns
    -------
    obj:`matplotlib.colors.LinearSegmentedColormap`
        A linear segmented colormap instance based on the provided ``cmap_name``, with
        the adjusted alpha values.
    """

    _cmap_name_alpha = cmap_name_alpha if cmap_name_alpha is not None else cmap_name + "Alpha"

    reds_data_mpl = mpl._cm.datad[cmap_name]  # type: ignore[attr-defined]

    seq_pt_count = len(reds_data_mpl)

    pts = np.linspace(0, 1, seq_pt_count)
    red = np.vstack(
        [[pts[i], reds_data_mpl[i][0], reds_data_mpl[i][0]] for i in range(seq_pt_count)]
    )
    green = np.vstack(
        [[pts[i], reds_data_mpl[i][1], reds_data_mpl[i][1]] for i in range(seq_pt_count)]
    )
    blue = np.vstack(
        [[pts[i], reds_data_mpl[i][2], reds_data_mpl[i][2]] for i in range(seq_pt_count)]
    )

    n = mpl.colormaps[cmap_name].N
    _alpha = np.linspace(0, max_alpha, n + 3)
    _alpha_interp = np.interp(pts, np.linspace(0, 1, len(_alpha)), _alpha)

    alpha = np.vstack([[pts[i], _alpha_interp[i], _alpha_interp[i]] for i in range(seq_pt_count)])

    # Convert NumPy arrays to lists of tuples and add type hints to make type checking happy
    cdict: dict[L["red", "green", "blue", "alpha"], Sequence[tuple[float, ...]]] = {
        "red": [tuple(val) for val in red],
        "green": [tuple(val) for val in green],
        "blue": [tuple(val) for val in blue],
        "alpha": [tuple(val) for val in alpha],
    }

    lscmap = LinearSegmentedColormap(_cmap_name_alpha, cdict)

    lscmap._init()  # type: ignore[attr-defined]

    return lscmap


def plot_segs(
    image_nii: ty.Union[str, SpatialImage],
    seg_niis: list[ty.Union[str, SpatialImage]],
    bbox_nii: ty.Union[str, SpatialImage, None] = None,
    masked: bool = False,
    compress: ty.Union[bool, L["auto"]] = "auto",
    **plot_params,
) -> list[SVGFigure]:
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
    seg_imgs: list[SpatialImage] = [
        rotate_affine(_3d_in_file(f), rot=canonical_r) for f in seg_niis
    ]
    data = image_nii.get_fdata()

    plot_params = robust_set_limits(data, plot_params)

    bbox_nii = (
        image_nii if bbox_nii is None else rotate_affine(_3d_in_file(bbox_nii), rot=canonical_r)
    )

    if masked:
        if Version(nilearn.__version__) >= Version("0.11"):
            bbox_nii: SpatialImage = nlimage.threshold_img(bbox_nii, 1e-3, copy_header=True)  # type: ignore[no-redef]
        else:
            bbox_nii: SpatialImage = nlimage.threshold_img(bbox_nii, 1e-3)  # type: ignore[no-redef]

    cuts = cuts_from_bbox(bbox_nii, cuts=7)
    out_files = []
    for d in plot_params.pop("dimensions", ("z", "x", "y")):
        plot_params["display_mode"] = d
        plot_params["cut_coords"] = cuts[d]
        svg = _plot_anat_with_contours(image_nii, segs=seg_imgs, compress=compress, **plot_params)
        # Find and replace the figure_1 id.
        svg = svg.replace("figure_1", f"segmentation-{d}-{uuid4()}", 1)
        out_files.append(fromstring(svg))

    return out_files


def plot_registration(
    anat_nii: SpatialImage,
    div_id: str,
    plot_params: ty.Union[dict[str, ty.Any], None] = None,
    order: tuple[L["x", "y", "z"], L["x", "y", "z"], L["x", "y", "z"]] = ("z", "x", "y"),
    cuts: ty.Union[dict[str, list[float]], None] = None,
    estimate_brightness: bool = False,
    label: ty.Union[str, None] = None,
    contour: ty.Union[SpatialImage, None] = None,
    compress: ty.Union[bool, L["auto"]] = "auto",
    dismiss_affine: bool = False,
) -> list[SVGFigure]:
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

    ribbon = False
    if contour is not None:
        ribbon = np.array_equal(np.unique(contour.get_fdata()), [0, 2, 3, 41, 42])

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


def _plot_anat_with_contours(
    image: SpatialImage,
    segs: ty.Union[list[SpatialImage], None] = None,
    compress: ty.Union[bool, L["auto"]] = "auto",
    **plot_params,
) -> str:
    if segs is None:
        segs = []
    nsegs = len(segs)
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


def plot_segmentation(anat_file: str, segmentation: str, out_file: str, **kwargs) -> str:
    """Plot a segmentation (from MRIQC)."""
    from nilearn.plotting import plot_anat
    from nitransforms.io.afni import _dicom_real_to_card

    vmax = kwargs.get("vmax")
    vmin = kwargs.get("vmin")

    anat_ras = nb.as_closest_canonical(load_api(anat_file, SpatialImage))
    anat_ras_plumb = anat_ras.__class__(
        anat_ras.dataobj, _dicom_real_to_card(anat_ras.affine), anat_ras.header
    )

    seg_ras = nb.as_closest_canonical(load_api(segmentation, SpatialImage))
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
    dslice: npt.NDArray,
    spacing: ty.Union[tuple[float, float], None] = None,
    cmap: ty.Union[str, mpl.colors.Colormap] = "Greys_r",
    label: ty.Union[str, None] = None,
    ax: ty.Union[mpl.axes.Axes, None] = None,
    vmax: ty.Union[float, None] = None,
    vmin: ty.Union[float, None] = None,
    annotate: ty.Union[tuple[str, str], None] = None,
) -> mpl.axes.Axes:
    if isinstance(cmap, str):
        cmap = mpl.colormaps[cmap]

    est_vmin, est_vmax = _get_limits(dslice)
    if not vmin:
        vmin = est_vmin
    if not vmax:
        vmax = est_vmax

    if ax is None:
        ax = plt.gca()

    if spacing is None:
        spacing = (1.0, 1.0)

    # Always swap axes because imshow defines the image as (M, N)
    # where M are rows (i.e., Y axis) and N are columns (X axis)
    dslice = np.swapaxes(dslice, 0, 1)
    spacing = (spacing[1], spacing[0])

    ax.imshow(
        dslice,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        extent=(0, dslice.shape[1] * spacing[1], 0, dslice.shape[0] * spacing[0]),
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
    dslice: npt.NDArray,
    prev: ty.Union[npt.NDArray, None] = None,
    post: ty.Union[npt.NDArray, None] = None,
    spacing: ty.Union[tuple[float, float], None] = None,
    cmap: ty.Union[str, mpl.colors.Colormap] = "Greys_r",
    label: ty.Union[str, None] = None,
    ax: ty.Union[mpl.axes.Axes, None] = None,
    vmax: ty.Union[float, None] = None,
    vmin: ty.Union[float, None] = None,
) -> None:
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
        spacing = (1.0, 1.0)
    else:
        spacing = (spacing[1], spacing[0])

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
        extent=(0, phys_sp[1] * 3, 0, phys_sp[0]),
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
    in_file: str,
    in_fft: str,
    spikes_list: list[tuple[int, int]],
    cols: int = 3,
    labelfmt: str = "t={0:.3f}s (z={1:d})",
    out_file: ty.Union[str, os.PathLike[str], None] = None,
) -> ty.Union[str, os.PathLike[str]]:
    """Plot a mosaic enhancing EM spikes."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    nii = nb.as_closest_canonical(load_api(in_file, SpatialImage))
    fft = load_api(in_file, SpatialImage).get_fdata()

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
        out_file = op.abspath(f"{fname}.svg")

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
) -> str:
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
        warnings.warn(
            "Argument ``plot_sagittal`` for plot_mosaic() should not be used.",
            category=UserWarning,
            stacklevel=2,
        )
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
    axannotation = {"axial": ("R", "L"), "coronal": ("R", "L"), "sagittal": ("A", "P")}

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
    panel_width: list[float] = []
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
                ax.set_rasterized(True)

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
                msk_cmap = _create_lscmap_with_alpha("Reds")
                plot_slice(
                    view_overlay_data[:, :, z_val],
                    vmin=0,
                    vmax=1,
                    cmap=msk_cmap,
                    ax=ax,
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
