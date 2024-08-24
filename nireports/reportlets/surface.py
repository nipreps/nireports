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
"""Plotting surface-supported data."""

import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize


def cifti_surfaces_plot(
    in_cifti,
    density="32k",
    surface_type="inflated",
    clip_range=(0, None),
    output_file=None,
    **kwargs,
):
    """
    Plots a CIFTI-2 dense timeseries onto left/right mesh surfaces.

    Parameters
    ----------
    in_cifti : str
        CIFTI-2 dense timeseries (.dtseries.nii)
    density : str
        Surface density
    surface_type : str
        Inflation level of mesh surfaces. Supported: midthickness, inflated, veryinflated
    clip_range : tuple or None
        Range to clip `in_cifti` data prior to plotting.
        If not None, two values must be provided as lower and upper bounds.
        If values are None, no clipping is performed for that bound.
    output_file: :obj:`str` or :obj:`None`
        Path where the output figure should be saved. If this is not defined,
        then the figure will be returned.
    kwargs : dict
        Keyword arguments for :obj:`nilearn.plotting.plot_surf`

    Outputs
    -------
    figure : matplotlib.pyplot.figure
        Surface plot figure. Returned only if ``output_file`` is ``None``.
    output_file: :obj:`str`
        The file where the figure is saved.

    """
    from nilearn.plotting import plot_surf

    def get_surface_meshes(density, surface_type):
        import templateflow.api as tf

        lh, rh = tf.get("fsLR", density=density, suffix=surface_type, extension=[".surf.gii"])
        return str(lh), str(rh)

    if density != "32k":
        raise NotImplementedError("Only 32k density is currently supported.")

    img = nb.cifti2.load(in_cifti)
    if img.nifti_header.get_intent()[0] != "ConnDenseSeries":
        raise TypeError(f"{in_cifti} is not a dense timeseries CIFTI file")

    geo = img.header.get_index_map(1)
    left_cortex, right_cortex = None, None
    for bm in geo.brain_models:
        if bm.brain_structure == "CIFTI_STRUCTURE_CORTEX_LEFT":
            left_cortex = bm
        elif bm.brain_structure == "CIFTI_STRUCTURE_CORTEX_RIGHT":
            right_cortex = bm

    if left_cortex is None or right_cortex is None:
        raise RuntimeError("CIFTI is missing cortex information")

    # calculate an average of the BOLD data, excluding the first 5 volumes
    # as potential nonsteady states
    data = img.dataobj[5:20].mean(axis=0)

    counts = (left_cortex.index_count, right_cortex.index_count)
    if density == "32k" and counts != (29696, 29716):
        raise ValueError("Cortex data is not in fsLR space")

    # medial wall needs to be added back in
    lh_data = np.full(left_cortex.surface_number_of_vertices, np.nan)
    rh_data = np.full(right_cortex.surface_number_of_vertices, np.nan)
    lh_data[left_cortex.vertex_indices] = _concat_brain_struct_data([left_cortex], data)
    rh_data[right_cortex.vertex_indices] = _concat_brain_struct_data([right_cortex], data)

    if clip_range:
        lh_data = np.clip(lh_data, clip_range[0], clip_range[1], out=lh_data)
        rh_data = np.clip(rh_data, clip_range[0], clip_range[1], out=rh_data)
        mn, mx = clip_range
    else:
        mn, mx = None, None

    if mn is None:
        mn = np.min(data)
    if mx is None:
        mx = np.max(data)

    cmap = kwargs.pop("cmap", "YlOrRd_r")
    cbar_map = cm.ScalarMappable(norm=Normalize(mn, mx), cmap=cmap)

    # Make background maps that rescale to a medium gray
    lh_bg = np.zeros(lh_data.shape, "int8")
    rh_bg = np.zeros(rh_data.shape, "int8")
    lh_bg[:2] = [3, -2]
    rh_bg[:2] = [3, -2]

    lh_mesh, rh_mesh = get_surface_meshes(density, surface_type)
    lh_kwargs = {"surf_mesh": lh_mesh, "surf_map": lh_data, "bg_map": lh_bg}
    rh_kwargs = {"surf_mesh": rh_mesh, "surf_map": rh_data, "bg_map": rh_bg}

    # Build the figure
    figure = plt.figure(figsize=plt.figaspect(0.25), constrained_layout=True)
    for i, view in enumerate(("lateral", "medial")):
        for j, hemi in enumerate(("left", "right")):
            title = f"{hemi.title()} - {view.title()}"
            ax = figure.add_subplot(1, 4, i * 2 + j + 1, projection="3d", rasterized=True)
            hemi_kwargs = (lh_kwargs, rh_kwargs)[j]
            plot_surf(
                hemi=hemi,
                view=view,
                title=title,
                cmap=cmap,
                vmin=mn,
                vmax=mx,
                axes=ax,
                **hemi_kwargs,
                **kwargs,
            )
            # plot_surf sets this to 8, which seems a little far out, but 6 starts clipping
            ax.dist = 7

    figure.colorbar(cbar_map, shrink=0.2, ax=figure.axes, location="bottom")

    if output_file is not None:
        figure.savefig(output_file, bbox_inches="tight", dpi=400)
        figure.clf()
        return output_file

    return figure


def _concat_brain_struct_data(structs, data):
    concat_data = np.array([], dtype=data.dtype)
    for struct in structs:
        struct_upper_bound = struct.index_offset + struct.index_count
        struct_data = data[struct.index_offset : struct_upper_bound]
        concat_data = np.concatenate((concat_data, struct_data))
    return concat_data
