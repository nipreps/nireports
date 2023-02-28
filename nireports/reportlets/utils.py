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
from pathlib import Path
from shutil import which
from tempfile import TemporaryDirectory
import subprocess
import base64
import re
import warnings
from uuid import uuid4
from io import StringIO

import numpy as np
import nibabel as nb

from nipype.utils import filemanip


SVGNS = "http://www.w3.org/2000/svg"


def robust_set_limits(data, plot_params, percentiles=(15, 99.8)):
    """Set (vmax, vmin) based on percentiles of the data."""
    plot_params["vmin"] = plot_params.get("vmin", np.percentile(data, percentiles[0]))
    plot_params["vmax"] = plot_params.get("vmax", np.percentile(data, percentiles[1]))
    return plot_params


def svg_compress(image, compress="auto"):
    """Generate a blob SVG from a matplotlib figure, may perform compression."""
    # Check availability of svgo and cwebp
    has_compress = all((which("svgo"), which("cwebp")))
    if compress is True and not has_compress:
        raise RuntimeError(
            "Compression is required, but svgo or cwebp are not installed"
        )
    else:
        compress = (compress is True or compress == "auto") and has_compress

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


def svg2str(display_object, dpi=300):
    """Serialize a nilearn display object to string."""
    from io import StringIO

    image_buf = StringIO()
    display_object.frame_axes.figure.savefig(
        image_buf, dpi=dpi, format="svg", facecolor="k", edgecolor="k"
    )
    image_buf.seek(0)
    return image_buf.getvalue()


def combine_svg(svg_list, axis="vertical"):
    """
    Composes the input svgs into one standalone svg
    """
    import numpy as np
    import svgutils.transform as svgt

    # Read all svg files and get roots
    svgs = [svgt.fromstring(f.encode("utf-8")) for f in svg_list]
    roots = [f.getroot() for f in svgs]

    # Query the size of each
    sizes = [(int(f.width[:-2]), int(f.height[:-2])) for f in svgs]

    if axis == "vertical":
        # Calculate the scale to fit all widths
        scales = [1.0] * len(svgs)
        if not all([width[0] == sizes[0][0] for width in sizes[1:]]):
            ref_size = sizes[0]
            for i, els in enumerate(sizes):
                scales[i] = ref_size[0] / els[0]

        newsizes = [tuple(size) for size in np.array(sizes) * np.array(scales)[..., np.newaxis]]
        totalsize = [newsizes[0][0], np.sum(newsizes, axis=0)[1]]

    elif axis == "horizontal":
        # Calculate the scale to fit all heights
        scales = [1.0] * len(svgs)
        if not all([height[0] == sizes[0][1] for height in sizes[1:]]):
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


def extract_svg(display_object, dpi=300, compress="auto"):
    """Remove the preamble of the svg files generated with nilearn."""
    image_svg = svg2str(display_object, dpi)
    if compress is True or compress == "auto":
        image_svg = svg_compress(image_svg, compress)
    image_svg = re.sub(' height="[0-9]+[a-z]*"', "", image_svg, count=1)
    image_svg = re.sub(' width="[0-9]+[a-z]*"', "", image_svg, count=1)
    image_svg = re.sub(
        " viewBox", ' preseveAspectRation="xMidYMid meet" viewBox', image_svg, count=1
    )
    start_tag = "<svg "
    start_idx = image_svg.find(start_tag)
    end_tag = "</svg>"
    end_idx = image_svg.rfind(end_tag)
    if start_idx == -1 or end_idx == -1:
        warnings.warn("svg tags not found in extract_svg")
    # rfind gives the start index of the substr. We want this substr
    # included in our return value so we add its length to the index.
    end_idx += len(end_tag)
    return image_svg[start_idx:end_idx]


def cuts_from_bbox(mask_nii, cuts=3):
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
    ijk_th = np.ceil([
        (mask_data.shape[1] * mask_data.shape[2]) * 0.2,  # sagittal
        (mask_data.shape[0] * mask_data.shape[2]) * 0.1,  # coronal
        (mask_data.shape[0] * mask_data.shape[1]) * 0.3,  # axial
    ]).astype(int)

    vox_coords = np.zeros((4, cuts), dtype=np.float32)
    vox_coords[-1, :] = 1.0
    for ax, (c, th) in enumerate(zip(ijk_counts, ijk_th)):
        # Start with full plane if mask is seemingly empty
        smin, smax = (0, mask_data.shape[ax] - 1)

        B = np.argwhere(c > th)
        if B.size < cuts:  # Threshold too high
            B = np.argwhere(c > 0)
        if B.size:
            smin, smax = B.min(), B.max()

        vox_coords[ax, :] = np.linspace(smin, smax, num=cuts + 2)[1:-1]

    ras_coords = mask_nii.affine.dot(vox_coords)[:3, ...]
    return {k: list(v) for k, v in zip(["x", "y", "z"], np.around(ras_coords, 3))}


def _3d_in_file(in_file):
    """ if self.inputs.in_file is 3d, return it.
    if 4d, pick an arbitrary volume and return that.

    if in_file is a list of files, return an arbitrary file from
    the list, and an arbitrary volume from that file
    """
    from nilearn import image as nlimage

    in_file = filemanip.filename_to_list(in_file)[0]

    try:
        in_file = nb.load(in_file)
    except AttributeError:
        in_file = in_file

    if len(in_file.shape) == 3:
        return in_file

    return nlimage.index_img(in_file, 0)


def compose_view(bg_svgs, fg_svgs, ref=0, out_file="report.svg"):
    """Compose the input svgs into one standalone svg with CSS flickering animation."""
    out_file = Path(out_file).absolute()
    out_file.write_text("\n".join(_compose_view(bg_svgs, fg_svgs, ref=ref)))
    return str(out_file)


def _compose_view(bg_svgs, fg_svgs, ref=0):
    from svgutils.compose import Unit
    from svgutils.transform import SVGFigure, GroupElement

    if fg_svgs is None:
        fg_svgs = []

    # Merge SVGs and get roots
    svgs = bg_svgs + fg_svgs
    roots = [f.getroot() for f in svgs]

    # Query the size of each
    sizes = []
    for f in svgs:
        viewbox = [float(v) for v in f.root.get("viewBox").split(" ")]
        width = int(viewbox[2])
        height = int(viewbox[3])
        sizes.append((width, height))
    nsvgs = len(bg_svgs)

    sizes = np.array(sizes)

    # Calculate the scale to fit all widths
    width = sizes[ref, 0]
    scales = width / sizes[:, 0]
    heights = sizes[:, 1] * scales

    # Compose the views panel: total size is the width of
    # any element (used the first here) and the sum of heights
    fig = SVGFigure(Unit(f"{width}px"), Unit(f"{heights[:nsvgs].sum()}px"))

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
            GroupElement(roots[:nsvgs], {"class": "background-svg"}),
            GroupElement(roots[nsvgs:], {"class": "foreground-svg"}),
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
</style>"""
            % tuple([uuid4()] * 2),
        )

    return svg


def transform_to_2d(data, max_axis):
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
