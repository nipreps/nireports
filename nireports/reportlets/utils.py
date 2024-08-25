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

import base64
import re
import subprocess
import warnings
from io import StringIO
from pathlib import Path
from shutil import which
from tempfile import TemporaryDirectory
from uuid import uuid4

import nibabel as nb
import numpy as np
from nipype.utils import filemanip

SVGNS = "http://www.w3.org/2000/svg"


def robust_set_limits(data, plot_params, percentiles=(15, 99.8)):
    """Set (vmax, vmin) based on percentiles of the data."""
    plot_params["vmin"] = plot_params.get("vmin", np.percentile(data, percentiles[0]))
    plot_params["vmax"] = plot_params.get("vmax", np.percentile(data, percentiles[1]))
    return plot_params


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


def svg_compress(image, compress="auto"):
    """Generate a blob SVG from a matplotlib figure, may perform compression."""
    # Check availability of svgo and cwebp
    has_compress = all((which("svgo"), which("cwebp")))
    if compress is True and not has_compress:
        raise RuntimeError("Compression is required, but svgo or cwebp are not installed")
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
    display_object.frame_axes.figure.clf()
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
        warnings.warn("svg tags not found in extract_svg", stacklevel=2)
    # rfind gives the start index of the substr. We want this substr
    # included in our return value so we add its length to the index.
    end_idx += len(end_tag)
    return image_svg[start_idx:end_idx]


def _bbox(img_data, bbox_data):
    """Calculate the bounding box of a binary segmentation."""
    B = np.argwhere(bbox_data)
    (ystart, xstart, zstart), (ystop, xstop, zstop) = B.min(0), B.max(0) + 1
    return img_data[ystart:ystop, xstart:xstop, zstart:zstop]


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
    """if self.inputs.in_file is 3d, return it.
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


def _compose_view(bg_svgs, fg_svgs, ref=0):
    from svgutils.compose import Unit
    from svgutils.transform import GroupElement, SVGFigure

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


def get_parula():
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
