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
# niworkflows/viz/utils.py
"""Helper tools for visualization purposes."""

from __future__ import annotations

import base64
import os
import re
import shutil
import subprocess
import typing as ty
import warnings
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal as L
from typing import cast
from uuid import uuid4

import imageio.v3 as iio
import matplotlib as mpl
import nibabel as nb
import numpy as np
import numpy.typing as npt
import pandas as pd
from nibabel.spatialimages import SpatialImage
from nilearn import image as nlimage
from nilearn.masking import compute_epi_mask
from nilearn.plotting import plot_epi
from nilearn.plotting.find_cuts import find_xyz_cut_coords
from scipy import ndimage

import nireports._vendored.svgutils.transform as svgt
from nireports.reportlets import compression_missing_msg, have_compression
from nireports.tools.ndimage import load_api

SVGNS = "http://www.w3.org/2000/svg"

G = ty.TypeVar("G", bound=np.generic)
CropSlices = tuple[slice, slice, slice]
PadWidth3D = tuple[tuple[int, int], tuple[int, int], tuple[int, int]]


class DisplayObject(ty.Protocol):
    frame_axes: mpl.axes.Axes


def robust_set_limits(
    data: npt.NDArray,
    plot_params: dict[str, ty.Any],
    percentiles: tuple[float, float] = (15, 99.8),
) -> dict[str, ty.Any]:
    """Set (vmax, vmin) based on percentiles of the data."""
    plot_params["vmin"] = plot_params.get("vmin", np.percentile(data, percentiles[0]))
    plot_params["vmax"] = plot_params.get("vmax", np.percentile(data, percentiles[1]))
    return plot_params


def _get_limits(
    nifti_file: str | npt.NDArray,
    only_plot_noise: bool = False,
) -> tuple[float, float]:
    if isinstance(nifti_file, str):
        nii = nb.as_closest_canonical(load_api(nifti_file, nb.Nifti1Image))
        data: npt.NDArray = nii.get_fdata()
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


def svg_compress(image: str, compress: bool | L["auto"] = "auto") -> str:
    """Generate a blob SVG from a matplotlib figure, may perform compression."""
    # Check availability of svgo and cwebp
    if compress is True and not have_compression:
        raise RuntimeError(f"Compression is required, but {compression_missing_msg}")
    else:
        compress = (compress is True or compress == "auto") and have_compression

    # Compress the SVG file using SVGO
    if compress:
        cmd = "svgo -i - -o - -q -p 3 --pretty"
        try:
            pout = subprocess.run(
                cmd,
                input=image.encode("utf-8"),
                stdout=subprocess.PIPE,
                shell=True,
                check=True,
                close_fds=True,
            ).stdout
        except OSError as e:
            from errno import ENOENT

            if compress is True and e.errno == ENOENT:
                raise e
        else:
            image = pout.decode("utf-8")

    # Convert all of the rasters inside the SVG file with 80% compressed WEBP
    if compress:
        new_lines = []
        with StringIO(image) as fp:
            for line in fp:
                if "image/png" in line:
                    tmp_lines = [line]
                    while "/>" not in line:
                        line = fp.readline()
                        tmp_lines.append(line)
                    content = "".join(tmp_lines).replace("\n", "").replace(",  ", ",")

                    left = content.split("base64,")[0] + "base64,"
                    left = left.replace("image/png", "image/webp")
                    right = content.split("base64,")[1]
                    png_b64 = right.split('"')[0]
                    right = '"' + '"'.join(right.split('"')[1:])

                    cmd = "cwebp -quiet -noalpha -q 80 -o - -- -"
                    pout = subprocess.run(
                        cmd,
                        input=base64.b64decode(png_b64),
                        shell=True,
                        stdout=subprocess.PIPE,
                        check=True,
                        close_fds=True,
                    ).stdout
                    webpimg = base64.b64encode(pout).decode("utf-8")
                    new_lines.append(left + webpimg + right)
                else:
                    new_lines.append(line)
        lines = new_lines
    else:
        lines = image.splitlines()

    svg_start = 0
    for i, line in enumerate(lines):
        if "<svg " in line:
            svg_start = i
            continue

    image_svg = lines[svg_start:]  # strip out extra DOCTYPE, etc headers
    return "".join(image_svg)  # straight up giant string


def svg2str(display_object: DisplayObject, dpi: int = 300) -> str:
    """Serialize a nilearn display object to string."""
    from io import StringIO

    image_buf = StringIO()
    figure = display_object.frame_axes.figure
    assert isinstance(figure, mpl.figure.Figure)
    figure.savefig(image_buf, dpi=dpi, format="svg", facecolor="k", edgecolor="k")
    figure.clf()
    image_buf.seek(0)
    return image_buf.getvalue()


def combine_svg(svg_list: list[str], axis="vertical") -> svgt.SVGFigure:
    """
    Composes the input svgs into one standalone svg
    """
    import numpy as np

    # Read all svg files and get roots
    svgs = [svgt.fromstring(f.encode("utf-8")) for f in svg_list]
    roots = [f.getroot() for f in svgs]

    # Query the size of each
    sizes = [(int(f.width[:-2]), int(f.height[:-2])) for f in svgs]

    if axis == "vertical":
        # Calculate the scale to fit all widths
        scales = [1.0] * len(svgs)
        if not all(width[0] == sizes[0][0] for width in sizes[1:]):
            ref_size = sizes[0]
            for i, els in enumerate(sizes):
                scales[i] = ref_size[0] / els[0]

        newsizes = [tuple(size) for size in np.array(sizes) * np.array(scales)[..., np.newaxis]]
        totalsize = [newsizes[0][0], np.sum(newsizes, axis=0)[1]]

    elif axis == "horizontal":
        # Calculate the scale to fit all heights
        scales = [1.0] * len(svgs)
        if not all(height[0] == sizes[0][1] for height in sizes[1:]):
            ref_size = sizes[0]
            for i, els in enumerate(sizes):
                scales[i] = ref_size[1] / els[1]

        newsizes = [tuple(size) for size in np.array(sizes) * np.array(scales)[..., np.newaxis]]
        totalsize = [np.sum(newsizes, axis=0)[0], newsizes[0][1]]

    # Compose the views panel: total size is the width of
    # any element (used the first here) and the sum of heights
    fig = svgt.SVGFigure(totalsize[0], totalsize[1])

    if axis == "vertical":
        yoffset = 0
        for i, r in enumerate(roots):
            size = newsizes[i]
            r.moveto(0, yoffset, scale=scales[i])
            yoffset += size[1]
            fig.append(r)
    elif axis == "horizontal":
        xoffset = 0
        for i, r in enumerate(roots):
            size = newsizes[i]
            r.moveto(xoffset, 0, scale=scales[i])
            xoffset += size[0]
            fig.append(r)

    return fig


def extract_svg(
    display_object: DisplayObject,
    dpi: int = 300,
    compress: bool | L["auto"] = "auto",
) -> str:
    """Remove the preamble of the svg files generated with nilearn."""
    image_svg = svg2str(display_object, dpi)
    if compress is True or compress == "auto":
        image_svg = svg_compress(image_svg, compress)
    image_svg = re.sub(r' height="[0-9]+[a-z]*"', "", image_svg, count=1)
    image_svg = re.sub(r' width="[0-9]+[a-z]*"', "", image_svg, count=1)
    image_svg = re.sub(
        r" viewBox", ' preseveAspectRation="xMidYMid meet" viewBox', image_svg, count=1
    )
    start_tag = "<svg "
    start_idx = image_svg.find(start_tag)
    end_tag = "</svg>"
    end_idx = image_svg.rfind(end_tag)
    if start_idx == -1 or end_idx == -1:
        warnings.warn("svg tags not found in extract_svg", stacklevel=2)
    # rfind gives the start index of the substr. We want this substr
    # included in our return value so we add its length to the index.
    end_idx += len(end_tag)
    return image_svg[start_idx:end_idx]


def _bbox(img_data: npt.NDArray[G], bbox_data: npt.NDArray) -> npt.NDArray[G]:
    """Calculate the bounding box of a binary segmentation."""
    B = np.argwhere(bbox_data)
    (ystart, xstart, zstart), (ystop, xstop, zstop) = B.min(0), B.max(0) + 1
    return img_data[ystart:ystop, xstart:xstop, zstart:zstop]


def cuts_from_bbox(mask_nii: SpatialImage, cuts: int = 3) -> dict[str, list[float]]:
    """Find equi-spaced cuts for presenting images."""
    mask_data = np.asanyarray(mask_nii.dataobj) > 0.0

    # First, project the number of masked voxels on each axes
    ijk_counts = [
        mask_data.sum(2).sum(1),  # project sagittal planes to transverse (i) axis
        mask_data.sum(2).sum(0),  # project coronal planes to to longitudinal (j) axis
        mask_data.sum(1).sum(0),  # project axial planes to vertical (k) axis
    ]

    # If all voxels are masked in a slice (say that happens at k=10),
    # then the value for ijk_counts for the projection to k (ie. ijk_counts[2])
    # at that element of the orthogonal axes (ijk_counts[2][10]) is
    # the total number of voxels in that slice (ie. Ni x Nj).
    # Here we define some thresholds to consider the plane as "masked"
    # The thresholds vary because of the shape of the brain
    # I have manually found that for the axial view requiring 30%
    # of the slice elements to be masked drops almost empty boxes
    # in the mosaic of axial planes (and also addresses #281)
    ijk_th = np.ceil(
        [
            (mask_data.shape[1] * mask_data.shape[2]) * 0.2,  # sagittal
            (mask_data.shape[0] * mask_data.shape[2]) * 0.1,  # coronal
            (mask_data.shape[0] * mask_data.shape[1]) * 0.3,  # axial
        ]
    ).astype(int)

    vox_coords = np.zeros((4, cuts), dtype=np.float32)
    vox_coords[-1, :] = 1.0
    for ax, (c, th) in enumerate(zip(ijk_counts, ijk_th)):
        # Start with full plane if mask is seemingly empty
        smin: np.signedinteger | int = 0
        smax: np.signedinteger | int = mask_data.shape[ax] - 1

        B = np.argwhere(c > th)
        if B.size < cuts:  # Threshold too high
            B = np.argwhere(c > 0)
        if B.size:
            smin, smax = B.min(), B.max()

        vox_coords[ax, :] = np.linspace(smin, smax, num=cuts + 2)[1:-1]

    ras_coords = mask_nii.affine.dot(vox_coords)[:3, ...]
    return {k: list(v) for k, v in zip(["x", "y", "z"], np.around(ras_coords, 3))}


def _3d_in_file(
    in_file: SpatialImage | str | os.PathLike | list[str | os.PathLike],
) -> SpatialImage:
    """if self.inputs.in_file is 3d, return it.
    if 4d, pick an arbitrary volume and return that.

    if in_file is a list of files, return an arbitrary file from
    the list, and an arbitrary volume from that file
    """
    from nilearn import image as nlimage

    if isinstance(in_file, list):
        in_file = in_file[0]

    if not isinstance(in_file, SpatialImage):
        in_file = load_api(in_file, SpatialImage)

    if len(in_file.shape) == 3:
        return in_file

    return nlimage.index_img(in_file, 0)


def compose_view(
    bg_svgs: list[svgt.SVGFigure],
    fg_svgs: list[svgt.SVGFigure],
    ref: int = 0,
    out_file: str | os.PathLike[str] = "report.svg",
) -> str:
    """
    Compose svgs into one standalone svg with CSS flickering animation.

    Parameters
    ----------
    bg_svgs : :obj:`list`
        Full paths to input svgs for background.
    fg_svgs : :obj:`list`
        Full paths to input svgs for foreground.
    ref : :obj:`int`, optional
        Which panel to use as reference for sizing all panels. Default: 0
    out_file : :obj:`str`, optional
        Full path to the output file. Default: "report.svg".

    Returns
    -------
    out_file : same as input

    """
    out_file = Path(out_file).absolute()
    out_file.write_text("\n".join(_compose_view(bg_svgs, fg_svgs, ref=ref)))
    return str(out_file)


def _compose_view(
    bg_svgs: list[svgt.SVGFigure],
    fg_svgs: list[svgt.SVGFigure],
    ref: int = 0,
) -> list[str]:
    from nireports._vendored.svgutils.compose import Unit

    if fg_svgs is None:
        fg_svgs = []

    # Merge SVGs and get roots
    svgs = bg_svgs + fg_svgs
    roots = [f.getroot() for f in svgs]

    # Query the size of each
    sizes = np.array(
        [[int(float(val)) for val in f.root.get("viewBox").split(" ")[2:4]] for f in svgs]
    )
    nsvgs = len(bg_svgs)

    # Calculate the scale to fit all widths
    width = sizes[ref, 0]
    scales = width / sizes[:, 0]
    heights = sizes[:, 1] * scales

    # Compose the views panel: total size is the width of
    # any element (used the first here) and the sum of heights
    fig = svgt.SVGFigure(Unit(f"{width}px"), Unit(f"{heights[:nsvgs].sum()}px"))

    yoffset = 0
    for i, r in enumerate(roots):
        r.moveto(0, yoffset, scale_x=scales[i])
        if i == (nsvgs - 1):
            yoffset = 0
        else:
            yoffset += heights[i]

    # Group background and foreground panels in two groups
    if fg_svgs:
        newroots = [
            svgt.GroupElement(roots[:nsvgs], {"class": "background-svg"}),
            svgt.GroupElement(roots[nsvgs:], {"class": "foreground-svg"}),
        ]
    else:
        newroots = roots
    fig.append(newroots)
    fig.root.attrib.pop("width", None)
    fig.root.attrib.pop("height", None)
    fig.root.set("preserveAspectRatio", "xMidYMid meet")

    with TemporaryDirectory() as tmpdirname:
        out_file = Path(tmpdirname) / "tmp.svg"
        fig.save(str(out_file))
        # Post processing
        svg = out_file.read_text().splitlines()

    # Remove <?xml... line
    if svg[0].startswith("<?xml"):
        svg = svg[1:]

    # Add styles for the flicker animation
    if fg_svgs:
        svg.insert(
            2,
            """\
<style type="text/css">
@keyframes flickerAnimation%s { 0%% {opacity: 1;} 100%% { opacity: 0; }}
.foreground-svg { animation: 1s ease-in-out 0s alternate none infinite paused flickerAnimation%s;}
.foreground-svg:hover { animation-play-state: running;}
</style>"""  # noqa: UP031
            % tuple([uuid4()] * 2),
        )

    return svg


def transform_to_2d(data: npt.NDArray, max_axis: int) -> npt.NDArray:
    """
    Projects 3d data cube along one axis using maximum intensity with
    preservation of the signs. Adapted from nilearn.
    """
    import numpy as np

    # get the shape of the array we are projecting to
    new_shape = list(data.shape)
    del new_shape[max_axis]

    # generate a 3D indexing array that points to max abs value in the
    # current projection
    a1, a2 = np.indices(new_shape)
    inds = [a1, a2]
    inds.insert(max_axis, np.abs(data).argmax(axis=max_axis))

    # take the values where the absolute value of the projection
    # is the highest
    maximum_intensity_data = data[tuple(inds)]

    return np.rot90(maximum_intensity_data)


def get_parula() -> mpl.colors.LinearSegmentedColormap:
    """Generate a 'parula' colormap."""
    from matplotlib.colors import LinearSegmentedColormap

    cm_data = [
        [0.2081, 0.1663, 0.5292],
        [0.2116238095, 0.1897809524, 0.5776761905],
        [0.212252381, 0.2137714286, 0.6269714286],
        [0.2081, 0.2386, 0.6770857143],
        [0.1959047619, 0.2644571429, 0.7279],
        [0.1707285714, 0.2919380952, 0.779247619],
        [0.1252714286, 0.3242428571, 0.8302714286],
        [0.0591333333, 0.3598333333, 0.8683333333],
        [0.0116952381, 0.3875095238, 0.8819571429],
        [0.0059571429, 0.4086142857, 0.8828428571],
        [0.0165142857, 0.4266, 0.8786333333],
        [0.032852381, 0.4430428571, 0.8719571429],
        [0.0498142857, 0.4585714286, 0.8640571429],
        [0.0629333333, 0.4736904762, 0.8554380952],
        [0.0722666667, 0.4886666667, 0.8467],
        [0.0779428571, 0.5039857143, 0.8383714286],
        [0.079347619, 0.5200238095, 0.8311809524],
        [0.0749428571, 0.5375428571, 0.8262714286],
        [0.0640571429, 0.5569857143, 0.8239571429],
        [0.0487714286, 0.5772238095, 0.8228285714],
        [0.0343428571, 0.5965809524, 0.819852381],
        [0.0265, 0.6137, 0.8135],
        [0.0238904762, 0.6286619048, 0.8037619048],
        [0.0230904762, 0.6417857143, 0.7912666667],
        [0.0227714286, 0.6534857143, 0.7767571429],
        [0.0266619048, 0.6641952381, 0.7607190476],
        [0.0383714286, 0.6742714286, 0.743552381],
        [0.0589714286, 0.6837571429, 0.7253857143],
        [0.0843, 0.6928333333, 0.7061666667],
        [0.1132952381, 0.7015, 0.6858571429],
        [0.1452714286, 0.7097571429, 0.6646285714],
        [0.1801333333, 0.7176571429, 0.6424333333],
        [0.2178285714, 0.7250428571, 0.6192619048],
        [0.2586428571, 0.7317142857, 0.5954285714],
        [0.3021714286, 0.7376047619, 0.5711857143],
        [0.3481666667, 0.7424333333, 0.5472666667],
        [0.3952571429, 0.7459, 0.5244428571],
        [0.4420095238, 0.7480809524, 0.5033142857],
        [0.4871238095, 0.7490619048, 0.4839761905],
        [0.5300285714, 0.7491142857, 0.4661142857],
        [0.5708571429, 0.7485190476, 0.4493904762],
        [0.609852381, 0.7473142857, 0.4336857143],
        [0.6473, 0.7456, 0.4188],
        [0.6834190476, 0.7434761905, 0.4044333333],
        [0.7184095238, 0.7411333333, 0.3904761905],
        [0.7524857143, 0.7384, 0.3768142857],
        [0.7858428571, 0.7355666667, 0.3632714286],
        [0.8185047619, 0.7327333333, 0.3497904762],
        [0.8506571429, 0.7299, 0.3360285714],
        [0.8824333333, 0.7274333333, 0.3217],
        [0.9139333333, 0.7257857143, 0.3062761905],
        [0.9449571429, 0.7261142857, 0.2886428571],
        [0.9738952381, 0.7313952381, 0.266647619],
        [0.9937714286, 0.7454571429, 0.240347619],
        [0.9990428571, 0.7653142857, 0.2164142857],
        [0.9955333333, 0.7860571429, 0.196652381],
        [0.988, 0.8066, 0.1793666667],
        [0.9788571429, 0.8271428571, 0.1633142857],
        [0.9697, 0.8481380952, 0.147452381],
        [0.9625857143, 0.8705142857, 0.1309],
        [0.9588714286, 0.8949, 0.1132428571],
        [0.9598238095, 0.9218333333, 0.0948380952],
        [0.9661, 0.9514428571, 0.0755333333],
        [0.9763, 0.9831, 0.0538],
    ]

    return LinearSegmentedColormap.from_list("parula", cm_data)


def _latex_available() -> bool:
    """Return True when a LaTeX executable is available on PATH."""
    return shutil.which("latex") is not None


def _largest_connected_component(mask_data: np.ndarray) -> np.ndarray:
    """Return the largest connected component of a binary mask.

    Connected components are computed with :func:`scipy.ndimage.label`. If the
    mask contains zero or one component, the input is returned unchanged.

    Parameters
    ----------
    mask_data : :obj:`~numpy.ndarray`
        Boolean or binary mask array.

    Returns
    -------
    :obj:`~numpy.ndarray`
        Boolean array containing only the largest connected component.
    """

    labeled, num = ndimage.label(mask_data)
    if num <= 1:
        return mask_data
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    largest = counts.argmax()
    return labeled == largest


def _compute_crop_slices(img: nb.spatialimages.SpatialImage) -> tuple[slice, slice, slice] | None:
    """Compute tight 3D crop slices around foreground signal in an image.

    Foreground is estimated using :func:`nilearn.masking.compute_epi_mask` when
    available. If mask computation fails, a fallback threshold-based mask is
    computed from positive intensities (> 80th percentile). The largest connected
    component is then selected before deriving bounding-box slices.

    Parameters
    ----------
    img : :obj:`~nibabel.spatialimages.SpatialImage`
        Input 3D image used to estimate the crop region.

    Returns
    -------
    :obj:`tuple` of :obj:`slice` or ``None``
        Cropping slices in ``(x, y, z)`` order, or ``None`` when no foreground
        region can be identified.
    """

    try:
        mask_img = compute_epi_mask(img)
        mask_data = np.asanyarray(mask_img.dataobj) > 0
    except Exception:
        data = np.asanyarray(img.dataobj)
        positive = data[data > 0]
        if positive.size == 0:
            return None
        threshold = float(np.percentile(positive, 80))
        mask_data = data > threshold

    mask_data = _largest_connected_component(mask_data)

    if not mask_data.any():
        return None

    coords = np.array(np.where(mask_data))
    start = coords.min(axis=1)
    end = coords.max(axis=1) + 1
    crop_slices = tuple(slice(int(s), int(e)) for s, e in zip(start, end))
    return cast(tuple[slice, slice, slice], crop_slices)


def crop_img(
    img: nb.spatialimages.SpatialImage, crop_slices: tuple[slice, slice, slice] | None
) -> nb.spatialimages.SpatialImage:
    """Crop a spatial image and update its affine translation accordingly.

    Parameters
    ----------
    img : :obj:`~nibabel.spatialimages.SpatialImage`
        Input image to crop.
    crop_slices : :obj:`tuple` of :obj:`slice` or ``None``
        Cropping slices in ``(x, y, z)`` order. If ``None``, the input image is
        returned unchanged.

    Returns
    -------
    :obj:`~nibabel.spatialimages.SpatialImage`
        Cropped image with affine origin shifted to preserve world coordinates.
    """

    if crop_slices is None:
        return img

    data = np.asanyarray(img.dataobj)[crop_slices]
    affine = img.affine.copy()
    starts = np.array([slc.start or 0 for slc in crop_slices])
    affine[:3, 3] += affine[:3, :3] @ starts
    return img.__class__(data, affine, img.header)


def merge_crop_slices(
    first: tuple[slice, slice, slice] | None, second: tuple[slice, slice, slice] | None
) -> tuple[slice, slice, slice] | None:
    """Merge two 3D crop boxes into their union.

    Parameters
    ----------
    first : :obj:`tuple` of :obj:`slice` or ``None``
        First crop box in ``(x, y, z)`` order.
    second : :obj:`tuple` of :obj:`slice` or ``None``
        Second crop box in ``(x, y, z)`` order.

    Returns
    -------
    :obj:`tuple` of :obj:`slice` or ``None``
        Union of both crop boxes, or whichever one is not ``None``.
    """

    if first is None:
        return second
    if second is None:
        return first

    merged = []
    for first_slc, second_slc in zip(first, second):
        start = min(first_slc.start or 0, second_slc.start or 0)
        stop = max(first_slc.stop or 0, second_slc.stop or 0)
        merged.append(slice(start, stop))

    merged_tuple = tuple(merged)
    return cast(tuple[slice, slice, slice], merged_tuple)


def compute_display_params(
    img: nb.spatialimages.SpatialImage, crop_slices: tuple[slice, slice, slice] | None = None
):
    """Compute display parameters for plotting a (possibly 4D) neuroimaging file.

    A representative 3D volume is selected (the input itself if 3D, otherwise the
    temporal midpoint for 4D images), optionally cropped, and then used to compute
    robust display bounds and cut coordinates.

    Parameters
    ----------
    img : :obj:`~nibabel.spatialimages.SpatialImage`
        Input image.
    crop_slices : :obj:`tuple` of :obj:`slice`, optional
        Cropping slices to apply in ``(x, y, z)`` order. If ``None``, slices are
        computed from the selected representative volume.

    Returns
    -------
    cropped_mid : :obj:`~nibabel.spatialimages.SpatialImage`
        Cropped representative 3D image used for display parameter estimation.
    cut_coords : :obj:`tuple`
        Cut coordinates returned by :func:`nilearn.plotting.find_xyz_cut_coords`.
    vmin : :obj:`float`
        Lower display bound (80th percentile of cropped data).
    vmax : :obj:`float`
        Upper display bound (99.9th percentile of cropped data).
    crop_slices : :obj:`tuple` of :obj:`slice`
        Cropping slices actually used.
    """

    if img.ndim == 3:
        mid_img = img
    else:
        mid_img = nlimage.index_img(img, img.shape[-1] // 2)

    if crop_slices is None:
        crop_slices = _compute_crop_slices(mid_img)

    cropped_mid = crop_img(mid_img, crop_slices)
    data = cropped_mid.get_fdata().astype(float)
    vmax = float(np.percentile(data.flatten(), 99.9))
    vmin = float(np.percentile(data.flatten(), 80))
    cut_coords = find_xyz_cut_coords(cropped_mid)

    return cropped_mid, cut_coords, vmin, vmax, crop_slices


def compute_common_display_params(
    uncorr_img: nb.spatialimages.SpatialImage,
    corr_img: nb.spatialimages.SpatialImage,
) -> tuple[
    float,
    float,
    tuple[float, float, float],
    tuple[float, float, float],
    CropSlices | None,
]:
    """Compute shared crop and display params for uncorrected and corrected images.

    Parameters
    ----------
    uncorr_img : :obj:`~nibabel.spatialimages.SpatialImage`
        Uncorrected volume.
    corr_img : :obj:`~nibabel.spatialimages.SpatialImage`
        Motion-corrected volume.

    Returns
    -------
    :obj:`tuple`
        A 5-item tuple containing:

        1. ``vmin`` (:obj:`float`)
           Lower intensity bound used for plotting.
        2. ``vmax`` (:obj:`float`)
           Upper intensity bound used for plotting.
        3. ``cut_coords_orig`` (:obj:`tuple` of :obj:`float`)
           Orthogonal cut coordinates for the original image.
        4. ``cut_coords_corr`` (:obj:`tuple` of :obj:`float`)
           Orthogonal cut coordinates for the corrected image.
        5. ``crop_slices`` (:obj:`tuple` of :obj:`slice`)
           Common crop slices covering both images.
    """

    _, _, vmin, vmax, uncorr_crop_slices = compute_display_params(uncorr_img)
    _, _, _, _, corr_crop_slices = compute_display_params(corr_img)

    crop_slices = merge_crop_slices(uncorr_crop_slices, corr_crop_slices)

    _, cut_coords_uncorr, _, _, _ = compute_display_params(uncorr_img, crop_slices)
    _, cut_coords_corr, _, _, _ = compute_display_params(corr_img, crop_slices)

    return vmin, vmax, cut_coords_uncorr, cut_coords_corr, crop_slices


def render_comparison_frames(
    uncorr_img: nb.spatialimages.SpatialImage,
    corr_img: nb.spatialimages.SpatialImage,
    n_frames: int,
    vmin: float,
    vmax: float,
    cut_coords_uncorr: tuple[float, float, float],
    cut_coords_corr: tuple[float, float, float],
    crop_slices: CropSlices | None,
    cmap: mpl.colors.Colormap | str = "RdBu_r",
) -> list[np.ndarray]:
    """Render side-by-side uncorrected/corrected frames as RGB(A) arrays.

    Parameters
    ----------
    uncorr_img : :obj:`~nibabel.spatialimages.SpatialImage`
        Uncorrected volume.
    corr_img : :obj:`~nibabel.spatialimages.SpatialImage`
        Motion-corrected volume.
    n_frames : :obj:`int`
        Number of frames (timepoints) to render.
    vmin : :obj:`float`
        Lower intensity bound used for both plotted panels.
    vmax : :obj:`float`
        Upper intensity bound used for both plotted panels.
    cut_coords_uncorr : :obj:`tuple` of :obj:`float`
        Cut coordinates used when plotting uncorrected frames.
    cut_coords_corr : :obj:`tuple` of :obj:`float`
        Cut coordinates used when plotting corrected frames.
    crop_slices : :obj:`tuple` of :obj:`slice`
        Spatial crop slices applied to both images before plotting.
    cmap : :obj:`~matplotlib.colors.Colormap` or :obj:`str`, optional
        Colormap to use.

    Returns
    -------
    :obj:`list` of :obj:`~numpy.ndarray`
        Rendered side-by-side frame images, one array per timepoint, suitable
        for embedding into an SVG animation.
    """

    pad_mode: L["constant"] = "constant"
    pad_constant: int = 255

    display_mode = "ortho"
    colorbar = True

    frames: list[np.ndarray] = []

    with TemporaryDirectory() as tmpdir:
        for idx in range(n_frames):
            uncorr_png = Path(tmpdir) / f"uncorr_{idx:04d}.png"
            corr_png = Path(tmpdir) / f"corr_{idx:04d}.png"

            uncorr_frame = crop_img(nlimage.index_img(uncorr_img, idx), crop_slices)
            corr_frame = crop_img(nlimage.index_img(corr_img, idx), crop_slices)
            plot_epi(
                uncorr_frame,
                cut_coords=cut_coords_uncorr,
                output_file=str(uncorr_png),
                display_mode=display_mode,
                title=f"Before motion correction | Frame {idx + 1}",
                colorbar=colorbar,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            plot_epi(
                corr_frame,
                cut_coords=cut_coords_corr,
                output_file=str(corr_png),
                display_mode=display_mode,
                title=f"After motion correction | Frame {idx + 1}",
                colorbar=colorbar,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )

            uncorr_arr = np.asarray(iio.imread(uncorr_png))
            corr_arr = np.asarray(iio.imread(corr_png))

            max_height = max(uncorr_arr.shape[0], corr_arr.shape[0])
            if uncorr_arr.shape[0] < max_height:
                uncorr_pad_rows: int = int(max_height - uncorr_arr.shape[0])
                uncorr_pad_width: PadWidth3D = ((0, uncorr_pad_rows), (0, 0), (0, 0))
                uncorr_arr = np.pad(
                    uncorr_arr,
                    uncorr_pad_width,
                    mode=pad_mode,
                    constant_values=pad_constant,
                )
            if corr_arr.shape[0] < max_height:
                corr_pad_rows: int = int(max_height - corr_arr.shape[0])
                corr_pad_width: PadWidth3D = ((0, corr_pad_rows), (0, 0), (0, 0))
                corr_arr = np.pad(
                    corr_arr,
                    corr_pad_width,
                    mode=pad_mode,
                    constant_values=pad_constant,
                )

            combined = np.concatenate([uncorr_arr, corr_arr], axis=1)
            frames.append(combined.astype(uncorr_arr.dtype, copy=False))

    return frames


def load_framewise_displacement(fd_file: str, sep="\t") -> np.ndarray:
    """Load framewise displacement (FD) values from a delimiter-separated confounds file.

    The function expects either a ``framewise_displacement`` column (preferred)
    or an ``FD`` column. Missing values are replaced with ``0.0``.

    Parameters
    ----------
    fd_file : :obj:`str`
        Path to a tab-separated values (TSV) confounds file.
    sep : :obj:`str`, optional
        Separator character or pattern.

    Returns
    -------
    :obj:`~numpy.ndarray`
        One-dimensional array of framewise displacement values.

    Raises
    ------
    :exc:`ValueError`
        If neither a ``framewise_displacement`` nor an ``FD`` column is present in the file.
    """
    framewise_disp = pd.read_csv(fd_file, sep=sep)
    fd_values = framewise_disp.get("framewise_displacement", framewise_disp.get("FD"))
    if fd_values is None:
        available = ", ".join(framewise_disp.columns)
        raise ValueError(
            "Could not find a 'framewise_displacement' or 'FD' column in the "
            f"confounds file (available columns: {available})"
        )

    return np.asarray(fd_values.fillna(0.0), dtype=float)
