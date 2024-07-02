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
"""Visualizations for diffusion MRI data."""

import matplotlib as mpl
import nibabel as nb
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import art3d
from nilearn.plotting import plot_anat

from nireports.reportlets.nuisance import plot_raincloud


def plot_dwi(dataobj, affine, gradient=None, **kwargs):
    """
    Plot orthogonal (axial, coronal, sagittal) slices of a given DWI volume. The
    slices displayed are determined by a tuple contained in the ``cut_coords``
    keyword argument.

    Parameters
    ----------
    dataobj : :obj:`numpy.ndarray`
        DWI volume data: a single 3D volume from a given gradient direction.
    affine : :obj:`numpy.ndarray`
        Affine transformation matrix.
    gradient : :obj:`numpy.ndarray`
        Gradient values in RAS+b format at the chosen gradient direction.
    kwargs : :obj:`dict`
        Extra args given to :obj:`nilearn.plotting.plot_anat()`.

    Returns
    -------
    :class:`nilearn.plotting.displays.OrthoSlicer` or None
        An instance of the OrthoSlicer class. If ``output_file`` is defined,
        None is returned.

    """

    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"],
        }
    )

    affine = np.diag(nb.affines.voxel_sizes(affine).tolist() + [1])
    affine[:3, 3] = -1.0 * (affine[:3, :3] @ ((np.array(dataobj.shape) - 1) * 0.5))

    vmax = kwargs.pop("vmax", None) or np.percentile(dataobj, 98)
    cut_coords = kwargs.pop("cut_coords", None) or (0, 0, 0)

    return plot_anat(
        nb.Nifti1Image(dataobj, affine, None),
        vmax=vmax,
        cut_coords=cut_coords,
        title=(
            r"Reference $b$=0"
            if gradient is None
            else f"""\
$b$={gradient[3].astype(int)}, \
$\\vec{{b}}$ = ({', '.join(str(v) for v in gradient[:3])})"""
        ),
        **kwargs,
    )


def plot_heatmap(
    data,
    b_indices,
    bvals,
    mask,
    scalar,
    scalar_label="DWI-derived scalar (a.u.)",
    bins=(150, 11),
    imax=None,
    sub_size=100000,
    cmap="YlGn",
    sigma=None,
):
    """Create heatmap plot."""
    # Round scalar to a single-digit decimal
    scalar = np.clip(np.round(scalar + 0.005, 1), 0, 1)

    # Prepare data in shells
    shells = [np.rint(data[mask][..., idx].reshape(-1)) for idx in b_indices]

    # Maximum intensity level to be plotted
    if imax is None:
        # If not provided, set 75th percentile of lowest b-value.
        imax = np.percentile(shells[0], 75)

    fig, axs = plt.subplots(
        len(b_indices) + 1,
        sharex=True,
        figsize=(20, 1.6 * (len(b_indices) + 1)),
    )
    axs[-1].spines[:].set_visible(False)
    axs[-1].grid(which="minor", color="w", linestyle="-", linewidth=1)
    for i, shelldata in enumerate(shells):
        x = shelldata[shelldata < imax]
        y = np.array([scalar[mask]] * len(b_indices[i])).reshape(-1)[shelldata < imax]

        if sub_size is not None:
            choice = np.random.choice(range(x.size), size=sub_size)
            x = x[choice]
            y = y[choice]

        histdata, _, _ = np.histogram2d(x, y, bins=bins, range=((0, int(imax)), (0, 1)))
        axs[i].imshow(
            histdata.T,
            interpolation="nearest",
            origin="lower",
            aspect="auto",
            cmap=cmap,
        )

        # Show all ticks and label them with the respective list entries.
        axs[i].set_yticks(
            [0.5, 5, 9.5],
            labels=["0.0", "0.5", "1.0"],
        )

        # Turn spines off and create white grid.
        axs[i].spines[:].set_visible(False)

        # axs[i].set_xticks(np.arange(bins[0] + 1) - .5, minor=True)
        axs[i].set_yticks(np.arange(bins[1] + 1) - 0.5, minor=True)
        axs[i].grid(which="minor", color="w", linestyle="-", linewidth=1)
        axs[i].tick_params(which="minor", bottom=False, left=False)
        axs[i].set_ylabel(f"$b$ = {bvals[i]}\n($n$ = {len(b_indices[i])})", fontsize=15)

        marginal_H, edges = np.histogram(x, bins=bins[0], range=(0, int(imax)), density=True)
        axs[-1].bar(
            np.linspace(0, bins[0], num=bins[0], endpoint=False, dtype=int),
            marginal_H,
            alpha=0.4,
        )

    if sigma is not None:
        max_snr = imax / sigma
        labels_bins = [1.937, 5.0, 8.0, round(max_snr, 1)]
        labels_bins_position = bins[0] * np.array(labels_bins) / max_snr

    else:
        labels_bins_position = np.arange(bins[0], step=20) + 0.5
        labels_bins = (labels_bins_position - 0.5) * (imax / bins[0])

    axs[-1].set_xticks(
        labels_bins_position,
        labels=labels_bins_position
        if sigma is None
        else [f"{v}\n[{round(10 * np.log(v), 0):.0f} dB]" for v in labels_bins],
        fontsize=14,
    )
    axs[-1].legend([f"{b}" for b in bvals], ncol=len(bvals), title="$b$ value")
    axs[-1].set_yticks([], labels=[])
    axs[-1].set_xlabel(
        f"SNR [noise floor estimated at {sigma:0.2f}]" if sigma is not None else "DWI intensity",
        fontsize=20,
    )
    fig.supylabel(scalar_label, fontsize=20, y=0.65)
    fig.tight_layout(rect=[0.02, 0, 1, 1])

    return fig


def rotation_matrix(u, v):
    r"""
    Calculate the rotation matrix *R* such that :math:`R \cdot \mathbf{u} = \mathbf{v}`.

    Extracted from `Emmanuel Caruyer's code
    <https://github.com/ecaruyer/qspace/blob/master/qspace/visu/visu_points.py>`__,
    which is distributed under the revised BSD License:
    Copyright (c) 2013-2015, Emmanuel Caruyer
    All rights reserved.

    .. admonition :: List of changes

        Only minimal updates to leverage Numpy.

    Parameters
    ----------
    u : :obj:`numpy.ndarray`
        A vector.
    v : :obj:`numpy.ndarray`
        A vector.

    Returns
    -------
    R : :obj:`numpy.ndarray`
        The rotation matrix.

    """

    # the axis is given by the product u x v
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    w = np.asarray(
        [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ]
    )
    if (w**2).sum() < (np.finfo(w.dtype).eps * 10):
        # The vectors u and v are collinear
        return np.eye(3)

    # Compute sine and cosine
    c = u @ v
    s = np.linalg.norm(w)

    w = w / s
    P = np.outer(w, w)
    Q = np.asarray([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    R = P + c * (np.eye(3) - P) + s * Q
    return R


def draw_circles(positions, radius, n_samples=20):
    r"""
    Draw circular patches (lying on a sphere) at given positions.

    Adapted from `Emmanuel Caruyer's code
    <https://github.com/ecaruyer/qspace/blob/master/qspace/visu/visu_points.py>`__,
    which is distributed under the revised BSD License:
    Copyright (c) 2013-2015, Emmanuel Caruyer
    All rights reserved.

    .. admonition :: List of changes

        Modified to take the full list of normalized bvecs and corresponding circle
        radii instead of taking the list of bvecs and radii for a specific shell
        (*b*-value).

    Parameters
    ----------
    positions : :obj:`numpy.ndarray`
        An array :math:`N \times 3` of 3D cartesian positions.
    radius : :obj:`float`
        The reference radius (or, the radius in single-shell plots)
    n_samples : :obj:`int`
        The number of samples on the sphere.

    Returns
    -------
    circles : :obj:`numpy.ndarray`
        Circular patches.

    """

    # A circle centered at [1, 0, 0] with radius r
    t = np.linspace(0, 2 * np.pi, n_samples)

    nb_points = positions.shape[0]
    circles = np.zeros((nb_points, n_samples, 3))
    for i in range(positions.shape[0]):
        circle_x = np.zeros((n_samples, 3))
        dots_radius = np.sqrt(radius[i]) * 0.04
        circle_x[:, 1] = dots_radius * np.cos(t)
        circle_x[:, 2] = dots_radius * np.sin(t)
        norm = np.linalg.norm(positions[i])
        point = positions[i] / norm
        r1 = rotation_matrix(np.asarray([1, 0, 0]), point)
        circles[i] = positions[i] + np.dot(r1, circle_x.T).T
    return circles


def draw_points(gradients, ax, rad_min=0.3, rad_max=0.7, colormap="viridis"):
    """
    Draw the vectors on a shell.

    Adapted from `Emmanuel Caruyer's code
    <https://github.com/ecaruyer/qspace/blob/master/qspace/visu/visu_points.py>`__,
    which is distributed under the revised BSD License:
    Copyright (c) 2013-2015, Emmanuel Caruyer
    All rights reserved.

    .. admonition :: List of changes

        * The input is a single 2D numpy array of the gradient table in RAS+B format
        * The scaling of the circle radius for each bvec proportional to the inverse of
          the bvals. A minimum/maximal value for the radii can be specified.
        * Circles for each bvec are drawn at once instead of looping over the shells.
        * Some variables have been renamed (like vects to bvecs)

    Parameters
    ----------
    gradients : :obj:`numpy.ndarray`
        An (N, 4) shaped array of the gradient table in RAS+B format.
    ax : :obj:`matplotlib.axes.Axis`
        The matplotlib axes instance to plot in.
    rad_min : :obj:`float` between 0 and 1
        Minimum radius of the circle that renders a gradient direction.
    rad_max : :obj:`float` between 0 and 1
        Maximum radius of the circle that renders a gradient direction.
    colormap : :class:`str`
        matplotlib colormap name.

    """

    # Initialize 3D view
    elev = 90
    azim = 0
    ax.view_init(azim=azim, elev=elev)

    # Normalize to 1 the highest bvalue
    bvals = np.copy(gradients[:, 3])
    bvals = bvals / bvals.max()

    # Colormap depending on bvalue (for visualization)
    cmap = mpl.colormaps[colormap]
    colors = cmap(bvals)

    # Relative shell radii proportional to the inverse of bvalue (for visualization)
    rs = np.reciprocal(bvals)
    rs = rs / rs.max()

    # Readjust radius of the circle given the minimum and maximal allowed values.
    if rs.min() != rs.max():
        rs = rs - rs.min()
        rs = rs / (rs.max() - rs.min())
    rs = rs * (rad_max - rad_min) + rad_min

    bvecs = np.copy(
        gradients[:, :3],
    )
    bvecs[bvecs[:, 2] < 0] *= -1

    # Render all gradient direction of all b-values
    circles = draw_circles(bvecs, rs)
    ax.add_collection(art3d.Poly3DCollection(circles, facecolors=colors, linewidth=0))

    max_val = 0.6
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    ax.set_zlim(-max_val, max_val)
    ax.axis("off")


def plot_gradients(
    gradients,
    title=None,
    ax=None,
    spacing=0.05,
    **kwargs,
):
    """
    Draw the vectors on a unit sphere using a different color (as given by the
    ``colormap`` property in the extra keywowrd arguments) for each b-value.

    .. seealso:: :func:`draw_points`.

    Parameters
    ----------
    gradients : :obj:`numpy.ndarray`
        An (N, 4) shaped array of the gradient table in RAS+B format.
    title : :obj:`str`
        Plot title.
    ax : :obj:`matplotlib.axes.Axis`
        A figure's axis to plot on.
    spacing : :obj:`float`
        Plot spacing.
    kwargs : :obj:`dict`
        Extra args given to :func:`draw_points`.

    Returns
    -------
    ax : :obj:`matplotlib.axes.Axis`
        The figure's axis where the data is plotted.

    """

    # Initialize figure
    if ax is None:
        figsize = kwargs.pop("figsize", (9.0, 9.0))
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(bottom=spacing, top=1 - spacing, wspace=2 * spacing)

    # Draw points after re-projecting all shells to the unit sphere
    draw_points(gradients, ax, **kwargs)

    if title:
        plt.suptitle(title)

    return ax


def plot_tissue_values(data_file, group_name, feature, **kwargs):
    """Generate a raincloud plot with the data points corresponding to the
    ``feature`` value contained in the data file.

    Parameters
    ----------
    data_file : :obj:`str`
        File containing the data of interest.
    group_name : :obj:`str`
        The group name of interest to be plot.
    feature : :obj:`str`
        The feature of interest to be plot.
    kwargs : :obj:`dict`
        Extra args given to :func:~`nireports.reportlets.nuisance.plot_raincloud`.

    Returns
    -------
    axes and gridspec
        Plotting axes and gridspec. Returned only if ``output_file`` is ``None``.
    output_file : :obj:`str`
        The file where the figure is saved.
    """

    return plot_raincloud(data_file, group_name, feature, **kwargs)


def nii_to_carpetplot_data(
    nii,
    bvals=None,
    divide_by_b0=True,
    drop_b0=True,
    sort_by_bval=False,
    mask_nii=None,
    segment_labels=None,
):
    """
    Convert nii to data matrix for carpet plot

    Parameters
    ----------
    nii : Nifti1Image
        DW imaging data
    bvals : :obj:`numpy.ndarray`, optional
        Rounded bvals
    divide_by_b0 : :obj:`bool`, optional
        Divide data by mean b0
    drop_b0 : :obj:`bool`, optional
        Flag to drop b0 data
    sort_by_bval : :obj:`bool`, optional
        Flag to reorder time points by bvalue
    mask_nii : Nifti1Image, optional
        Boolean or segmentation mask of DW imaging data
    segment_labels : :obj:`dict`, optional
        Dictionary of segment labels
        e.g. {'Cerebral_White_Matter': [2, 41],
              'Cerebral_Cortex': [3, 42],
              'Ventricle': [4, 14, 15, 43, 72]}

    Returns
    ---------
    data : N x T :obj:`numpy.array`
        The functional data to be plotted
        (*N* sampling locations by *T* timepoints).
    segments: :obj:`dict`
        A mapping between segment labels (e.g., `"Left Cortex"`)
        and list of indexes in the data array.
    """

    nii_data = nii.get_fdata()

    if mask_nii is None:
        mask_data = np.ones(nii_data.shape[0:3])
    else:
        mask_data = np.asanyarray(mask_nii.dataobj, dtype=np.int16)

    if bvals is not None:
        if divide_by_b0:
            b0_data = nii_data[..., bvals == 0]
            bzero = np.mean(b0_data, -1)
            nii_data = nii_data / bzero[..., np.newaxis]

        if drop_b0:
            nii_data = nii_data[..., bvals > 0]
            bvals = bvals[bvals > 0]

        if sort_by_bval:
            sort_inds = np.argsort(bvals)
            nii_data = nii_data[..., sort_inds]

    # Reshape
    data = nii_data.reshape(-1, nii_data.shape[-1])
    mask_data = mask_data.reshape(-1)

    # Apply mask
    data = data[mask_data > 0, :]
    mask_data = mask_data[mask_data > 0]

    # Remove bad rows
    bad_row_ind = np.where(~np.isfinite(data))[0]
    good_row_ind = np.ones(data.shape[0], dtype=bool)
    good_row_ind[bad_row_ind] = False

    data = data[good_row_ind, :]
    mask_data = mask_data[good_row_ind]

    # Get segments dict
    if mask_nii is None or segment_labels is None:
        segments = None
    else:
        segments = get_segments(mask_data, segment_labels)

    return data, segments


def get_segments(segment_mask, segment_labels):
    """
    Return segments dict for plot_carpet function

    Parameters
    ----------
    segment_mask : :obj:`numpy.ndarray`
        Segmentation mask of DW imaging data
    segment_labels : :obj:`dict`
        Dictionary of segment labels
        e.g. {'Cerebral_White_Matter': [2, 41],
              'Cerebral_Cortex': [3, 42],
              'Ventricle': [4, 14, 15, 43, 72]}

    Returns
    ---------
    segments: :obj:`dict`
        A mapping between segment labels (e.g., `"Left Cortex"`)
        and list of indexes in the data array.
    """
    segments = {}

    for label, idx in segment_labels.items():
        indices = np.array([], dtype=int)
        for ii in idx:
            indices = np.concatenate([indices, np.nonzero(segment_mask == ii)[0]])
        segments[label] = indices

    return segments


def get_segment_labels(filepath, keywords, delimiter=" ", index_position=0, label_position=1):
    """
    Get segment labels from file by keyword for get_segments function

    Parameters
    ----------
    filepath : :obj:`string`
        Path to segment label text file, such as freesurfer label file
    keywords : list of :obj:`string`
        List of label keywords.
        All labels containing the keyword will be grouped together.
        e.g. ["Cerebral_White_Matter", "Cerebral_Cortex", "Ventricle"]
    delimiter : :obj:`string`, optional
        Delimiter between label index and label string in label file
        (' ' for freesurfer label file)
    index_position : :obj:`int`, optional
        Position of label index in label file
        (0 for freesurfer label file)
    label_position : :obj:`int`, optional
        Position of label string in label file
        (1 for freesurfer label file)
    Returns
    ---------
    dict
    e.g. {'Cerebral_White_Matter': [2, 41],
          'Cerebral_Cortex': [3, 42],
          'Ventricle': [4, 14, 15, 43, 72]}
    """
    segment_labels = {}

    with open(filepath, "r") as f:
        labels = f.read()

    labels_s = [label.split(delimiter) for label in labels.split("\n") if label != ""]

    for keyword in keywords:
        ind = [int(i[index_position]) for i in labels_s if keyword in i[label_position]]
        segment_labels[keyword] = ind

    return segment_labels
