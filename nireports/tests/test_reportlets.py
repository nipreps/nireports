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
"""Test reportlets module."""

import contextlib
import os
import warnings
from itertools import permutations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np
import pandas as pd
import pytest
from templateflow.api import get

import nireports._vendored.svgutils.transform as svgt
from nireports.reportlets import compression_missing_msg, have_compression
from nireports.reportlets.modality.func import fMRIPlot
from nireports.reportlets.mosaic import _create_lscmap_with_alpha, plot_mosaic, plot_segs
from nireports.reportlets.nuisance import plot_carpet, plot_dist, plot_fd, plot_raincloud
from nireports.reportlets.surface import cifti_surfaces_plot
from nireports.reportlets.utils import _3d_in_file
from nireports.reportlets.xca import compcor_variance_plot, plot_melodic_components
from nireports.tests.testing import _create_dtseries_cifti
from nireports.tests.utils import _generate_raincloud_random_data
from nireports.tools.timeseries import cifti_timeseries as _cifti_timeseries
from nireports.tools.timeseries import get_tr as _get_tr
from nireports.tools.timeseries import nifti_timeseries as _nifti_timeseries


@pytest.mark.parametrize("tr", (None, 0.7))
@pytest.mark.parametrize("sorting", (None, "ward", "linkage"))
def test_carpetplot(request, tr, sorting, outdir):
    """Write a carpetplot"""

    rng = request.node.rng
    plot_carpet(
        rng.normal(100, 20, size=(18000, 1900)),
        title="carpetplot with title",
        tr=tr,
        output_file=(
            outdir
            / (
                f"carpet_nosegs_{'index' if tr is None else 'time'}_"
                f"{'nosort' if sorting is None else sorting}.svg"
            )
        )
        if outdir is not None
        else None,
        sort_rows=sorting,
        drop_trs=15,
    )

    labels = ("Ctx GM", "Subctx GM", "WM+CSF", "Cereb.", "Edge")
    sizes = (200, 100, 50, 100, 50)
    total_size = np.sum(sizes)
    data = np.zeros((total_size, 300))

    indexes = np.arange(total_size)
    rng.shuffle(indexes)
    segments = {}
    start = 0
    for group, size in zip(labels, sizes):
        segments[group] = indexes[start : start + size]
        data[indexes[start : start + size]] = rng.normal(
            rng.standard_normal(1) * 100, rng.normal(20, 5, size=1), size=(size, 300)
        )
        start += size

    plot_carpet(
        data,
        segments,
        tr=tr,
        output_file=(
            outdir
            / (
                f"carpet_random_{'index' if tr is None else 'seg'}_"
                f"{'nosort' if sorting is None else sorting}.svg"
            )
        )
        if outdir is not None
        else None,
        sort_rows=sorting,
    )

    data = np.zeros((total_size, 300))
    indexes = np.arange(total_size)
    rng.shuffle(indexes)
    segments = {}
    start = 0
    for i, (group, size) in enumerate(zip(labels, sizes)):
        segments[group] = indexes[start : start + size]
        data[indexes[start : start + size]] = i
        start += size

    plot_carpet(
        data,
        segments,
        detrend=False,
        tr=tr,
        output_file=(
            outdir
            / (
                f"carpet_const_{'index' if tr is None else 'time'}_"
                f"{'nosort' if sorting is None else sorting}.svg"
            )
        )
        if outdir is not None
        else None,
        sort_rows=sorting,
    )


@pytest.mark.parametrize(
    "input_files",
    [
        ("sub-ds205s03_task-functionallocalizer_run-01_bold_volreg.nii.gz", None),
        ("sub-01_task-mixedgamblestask_run-02_space-fsLR_den-91k_bold.dtseries.nii", None),
        (
            "sub-ds205s03_task-functionallocalizer_run-01_bold_volreg.nii.gz",
            "sub-ds205s03_task-functionallocalizer_run-01_bold_parc.nii.gz",
        ),
    ],
)
def test_fmriplot(request, input_files, test_data_package, outdir):
    """Exercise the fMRIPlot class."""
    rng = request.node.rng

    in_file = os.path.join(test_data_package, input_files[0])
    seg_file = (
        os.path.join(test_data_package, input_files[1]) if input_files[1] is not None else None
    )

    dtype = "nifti" if input_files[0].endswith("volreg.nii.gz") else "cifti"
    has_seg = "_parc" if seg_file else ""

    timeseries, segments = (
        _nifti_timeseries(in_file, seg_file) if dtype == "nifti" else _cifti_timeseries(in_file)
    )

    fig = fMRIPlot(
        timeseries,
        segments,
        tr=_get_tr(nb.load(in_file)),
        confounds=pd.DataFrame(
            {
                "outliers": rng.normal(0.2, 0.2, timeseries.shape[-1] - 1),
                "DVARS": rng.normal(0.2, 0.2, timeseries.shape[-1] - 1),
                "FD": rng.normal(0.2, 0.2, timeseries.shape[-1] - 1),
            }
        ),
        units={"FD": "mm"},
        paired_carpet=dtype == "cifti",
    ).plot()
    if outdir is not None:
        fig.savefig(
            outdir / f"fmriplot_{dtype}{has_seg}.svg",
            bbox_inches="tight",
        )
        fig.clf()


@pytest.mark.skipif(not have_compression, reason=compression_missing_msg)
def test_plot_melodic_components(request, tmp_path, outdir):
    """Test plotting melodic components"""

    rng = request.node.rng

    if outdir is None:
        outdir = Path(str(tmp_path))

    all_noise = str(outdir / "melodic_all_noise.svg")
    no_noise = str(outdir / "melodic_no_noise.svg")
    no_classified = str(outdir / "melodic_no_classified.svg")

    # melodic directory
    melodic_dir = tmp_path / "melodic"
    melodic_dir.mkdir(exist_ok=True)
    # melodic_mix
    mel_mix = rng.integers(low=-5, high=5, size=[10, 2])
    mel_mix_file = str(melodic_dir / "melodic_mix")
    np.savetxt(mel_mix_file, mel_mix, fmt="%i")
    # melodic_FTmix
    mel_ftmix = rng.random((2, 5))
    mel_ftmix_file = str(melodic_dir / "melodic_FTmix")
    np.savetxt(mel_ftmix_file, mel_ftmix)
    # melodic_ICstats
    mel_icstats = rng.random((2, 2))
    mel_icstats_file = str(melodic_dir / "melodic_ICstats")
    np.savetxt(mel_icstats_file, mel_icstats)
    # melodic_IC
    mel_ic = rng.random((2, 2, 2, 2))
    mel_ic_file = str(melodic_dir / "melodic_IC.nii.gz")
    mel_ic_img = nb.Nifti2Image(mel_ic, np.eye(4))
    mel_ic_img.to_filename(mel_ic_file)
    # noise_components
    noise_comps = np.array([1, 2])
    noise_comps_file = str(tmp_path / "noise_ics.csv")
    np.savetxt(noise_comps_file, noise_comps, fmt="%i", delimiter=",")

    # create empty components file
    nocomps_file = str(tmp_path / "noise_none.csv")
    open(nocomps_file, "w").close()

    # in_file
    in_fname = str(tmp_path / "in_file.nii.gz")
    voxel_ts = rng.random((2, 2, 2, 10))
    in_file = nb.Nifti2Image(voxel_ts, np.eye(4))
    in_file.to_filename(in_fname)
    # report_mask
    report_fname = str(tmp_path / "report_mask.nii.gz")
    report_mask = nb.Nifti2Image(np.ones([2, 2, 2]), np.eye(4))
    report_mask.to_filename(report_fname)

    # run command with all noise components
    plot_melodic_components(
        str(melodic_dir),
        in_fname,
        tr=2.0,
        report_mask=report_fname,
        noise_components_file=noise_comps_file,
        out_file=all_noise,
    )
    # run command with no noise components
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="loadtxt: input contained no data",
            category=UserWarning,
        )
        plot_melodic_components(
            str(melodic_dir),
            in_fname,
            tr=2.0,
            report_mask=report_fname,
            noise_components_file=nocomps_file,
            out_file=no_noise,
        )

    # run command without noise components file
    plot_melodic_components(
        str(melodic_dir),
        in_fname,
        tr=2.0,
        report_mask=report_fname,
        out_file=no_classified,
    )


def test_compcor_variance_plot(tmp_path, test_data_package, outdir):
    """Test plotting CompCor variance"""

    if outdir is None:
        outdir = Path(str(tmp_path))

    out_file = str(outdir / "variance_plot_short.svg")
    metadata_file = os.path.join(test_data_package, "confounds_metadata_short_test.tsv")
    compcor_variance_plot([metadata_file], output_file=out_file)


@pytest.fixture
def create_surface_dtseries(request):
    """Create a dense timeseries CIFTI-2 file with only cortex structures"""

    rng = request.node.rng
    out_file = _create_dtseries_cifti(
        timepoints=10,
        models=[
            ("CIFTI_STRUCTURE_CORTEX_LEFT", rng.random((29696, 10))),
            ("CIFTI_STRUCTURE_CORTEX_RIGHT", rng.random((29716, 10))),
        ],
    )
    yield str(out_file)
    out_file.unlink()


def test_cifti_surfaces_plot(tmp_path, create_surface_dtseries, outdir):
    """Test plotting CIFTI-2 surfaces"""
    os.chdir(tmp_path)

    if outdir is None:
        outdir = Path(str(tmp_path))

    out_file = str(outdir / "cifti_surfaces_plot.svg")
    cifti_surfaces_plot(create_surface_dtseries, output_file=out_file)


def test_cifti_carpetplot(tmp_path, test_data_package, outdir):
    """Exercise extraction of timeseries from CIFTI2."""

    cifti_file = os.path.join(
        test_data_package,
        "sub-01_task-mixedgamblestask_run-02_space-fsLR_den-91k_bold.dtseries.nii",
    )
    data, segments = _cifti_timeseries(cifti_file)
    plot_carpet(
        data,
        segments,
        tr=_get_tr(nb.load(cifti_file)),
        output_file=outdir / "carpetplot_cifti.svg" if outdir is not None else None,
        drop_trs=0,
        cmap="paired",
    )


def test_nifti_carpetplot(tmp_path, test_data_package, outdir):
    """Exercise extraction of timeseries from CIFTI2."""

    nifti_file = os.path.join(
        test_data_package,
        "sub-ds205s03_task-functionallocalizer_run-01_bold_volreg.nii.gz",
    )
    seg_file = os.path.join(
        test_data_package,
        "sub-ds205s03_task-functionallocalizer_run-01_bold_parc.nii.gz",
    )
    data, segments = _nifti_timeseries(nifti_file, seg_file)
    plot_carpet(
        data,
        segments,
        tr=_get_tr(nb.load(nifti_file)),
        output_file=outdir / "carpetplot_nifti.svg" if outdir is not None else None,
        drop_trs=0,
    )


_views = list(permutations(("axial", "sagittal", "coronal", None), 3)) + [
    (v, None, None) for v in ("axial", "sagittal", "coronal")
]


@pytest.mark.parametrize("views", _views)
@pytest.mark.parametrize("plot_sagittal", (True, False))
def test_mriqc_plot_mosaic(tmp_path, test_data_package, outdir, views, plot_sagittal):
    """Exercise the generation of mosaics."""

    fname = f"mosaic_{'_'.join(v or 'none' for v in views)}_{plot_sagittal:d}.svg"

    if views[0] is None or ((views[1] is None) and (views[2] is not None)):
        context = pytest.raises(RuntimeError)
    elif plot_sagittal and views[1] is None and views[0] != "sagittal":
        context = pytest.warns(UserWarning, match=r".*plot_sagittal.*should not be used")
    else:
        context = contextlib.nullcontext()

    with context:
        plot_mosaic(
            get("MNI152NLin6Asym", resolution=2, desc="LR", suffix="T1w"),
            plot_sagittal=plot_sagittal,
            views=views,
            out_file=(outdir / fname) if outdir is not None else None,
            title=f"A mosaic plotting example: views={views}, plot_sagittal={plot_sagittal}",
            maxrows=5,
        )


@pytest.mark.parametrize("views", _views)
@pytest.mark.parametrize("plot_sagittal", (True, False))
def test_mriqc_plot_mosaic_1(tmp_path, test_data_package, outdir, views, plot_sagittal):
    """Exercise the generation of mosaics."""

    fname = f"mosaic_{'_'.join(v or 'none' for v in views)}_{plot_sagittal:d}.svg"

    if views[0] is None or ((views[1] is None) and (views[2] is not None)):
        context = pytest.raises(RuntimeError)
    elif plot_sagittal and views[1] is None and views[0] != "sagittal":
        context = pytest.warns(UserWarning, match=r".*plot_sagittal.*should not be used")
    else:
        context = contextlib.nullcontext()

    with context:
        plot_mosaic(
            get("MNI152NLin2009aAsym", resolution=1, suffix="T1w"),
            plot_sagittal=plot_sagittal,
            views=views,
            out_file=(outdir / fname) if outdir is not None else None,
            title=f"A mosaic plotting example: views={views}, plot_sagittal={plot_sagittal}",
            maxrows=5,
            overlay_mask=get("MNI152NLin2009aAsym", resolution=1, label="WM", suffix="probseg"),
        )


def test_mriqc_plot_mosaic_2(tmp_path, test_data_package, outdir):
    """Exercise the generation of mosaics."""
    plot_mosaic(
        get("Fischer344", desc=None, suffix="T2w"),
        plot_sagittal=False,
        ncols=6,
        views=("coronal", "axial", "sagittal"),
        out_file=(outdir / "rodent_mosaic.svg") if outdir is not None else None,
        maxrows=12,
        annotate=True,
    )


@pytest.mark.parametrize("orient", ["h", "v"])
@pytest.mark.parametrize("density", (True, False))
def test_plot_raincloud(orient, density, tmp_path):
    features_label = "value"
    group_label = "group"
    group_names = ["group1", "group2"]
    min_val_grp1 = 0.3
    max_val_grp1 = 1.0
    min_max_group1 = (min_val_grp1, max_val_grp1)
    min_val_grp2 = 0.0
    max_val_grp2 = 0.6
    min_max_group2 = (min_val_grp2, max_val_grp2)
    min_max = [min_max_group1, min_max_group2]
    n_grp_samples = 250
    data_file = tmp_path / "data.tsv"

    _generate_raincloud_random_data(
        min_max, n_grp_samples, features_label, group_label, group_names, data_file
    )

    palette = "Set2"
    mark_nans = False
    output_file = tmp_path / f"raincloud_reg_orient-{orient}_density-{density}.png"

    plot_raincloud(
        data_file,
        group_label,
        features_label,
        palette=palette,
        orient=orient,
        density=density,
        mark_nans=mark_nans,
        output_file=output_file,
    )

    group_nans = [50, 0]

    _generate_raincloud_random_data(
        min_max,
        n_grp_samples,
        features_label,
        group_label,
        group_names,
        data_file,
        group_nans=group_nans,
    )

    mark_nans = True
    nans_value = 2.0
    output_file = tmp_path / f"raincloud_nans_orient-{orient}_density-{density}.png"

    plot_raincloud(
        data_file,
        group_label,
        features_label,
        palette=palette,
        orient=orient,
        density=density,
        mark_nans=mark_nans,
        nans_value=nans_value,
        output_file=output_file,
    )

    min_val_grp1 = 0.3
    max_val_grp1 = 8.0
    min_max_group1 = (min_val_grp1, max_val_grp1)
    min_val_grp2 = -0.2
    max_val_grp2 = 0.6
    min_max_group2 = (min_val_grp2, max_val_grp2)
    min_max = [min_max_group1, min_max_group2]

    _generate_raincloud_random_data(
        min_max,
        n_grp_samples,
        features_label,
        group_label,
        group_names,
        data_file,
        group_nans=group_nans,
    )

    upper_limit_value = 1.0
    lower_limit_value = 0.0
    limit_offset = 0.5
    output_file = tmp_path / f"raincloud_nans_limits_orient-{orient}_density-{density}.png"

    plot_raincloud(
        data_file,
        group_label,
        features_label,
        palette=palette,
        orient=orient,
        density=density,
        upper_limit_value=upper_limit_value,
        lower_limit_value=lower_limit_value,
        limit_offset=limit_offset,
        mark_nans=mark_nans,
        nans_value=nans_value,
        output_file=output_file,
    )


def test_plot_segs(request, outdir):
    rng = request.node.rng

    image_data = rng.random((5, 5, 5))
    seg_data = rng.random((5, 5, 5))

    image_nii = _3d_in_file(nb.Nifti1Image(image_data, np.eye(4)))
    seg_niis = [_3d_in_file(nb.Nifti1Image(seg_data, np.eye(4)))]

    bbox_nii = None
    masked = False
    compress = "auto"

    result = plot_segs(image_nii, seg_niis, bbox_nii=bbox_nii, masked=masked, compress=compress)
    assert isinstance(result, list)
    assert len(result) == 3
    assert all(isinstance(svg, svgt.SVGFigure) for svg in result)

    if outdir is not None:
        [res.save(outdir / f"plot_seg_{idx:03d}.svg") for idx, res in enumerate(result)]


def test_plot_fd(request, tmp_path, outdir):
    rng = request.node.rng

    fd_file = os.path.join(tmp_path, "fd.txt")
    fd_radius = 50
    mean_fd_dist = rng.random(100)

    # Simulate FD file content
    with open(fd_file, "w") as f:
        for _ in range(100):
            f.write(" ".join(map(str, rng.random(6))) + "\n")

    fig = plot_fd(fd_file, fd_radius, mean_fd_dist=mean_fd_dist)
    assert fig is not None
    if outdir is not None:
        fig.savefig(outdir / "test_plot_fd.svg")
    plt.close(fig)


def test_plot_dist(request, tmp_path, outdir):
    rng = request.node.rng
    distribution = rng.random(100)
    xlabel = "Distribution"
    xlabel2 = "Another Distribution"

    fig = plot_dist(
        get("MNI152NLin2009aAsym", resolution=1, suffix="T1w"),
        get("MNI152NLin2009aAsym", resolution=1, desc="brain", suffix="mask"),
        xlabel,
        distribution=distribution,
        xlabel2=xlabel2,
    )
    assert fig is not None
    if outdir is not None:
        fig.savefig(outdir / "test_plot_dist.svg")
    plt.close(fig)


def test_create_cmap(outdir):
    cmap_name = "Reds"

    max_alpha = 0.75
    ls_cmap = _create_lscmap_with_alpha(cmap_name, max_alpha=max_alpha)

    orig_cmap = mpl.colormaps[cmap_name]
    orig_cmap._init()

    assert ls_cmap._lut.shape == orig_cmap._lut.shape
    assert np.allclose(ls_cmap._lut[0, -1], 0.0)
    assert np.allclose(ls_cmap._lut[ls_cmap.N - 1, -1], max_alpha)
    assert np.all(0.0 <= ls_cmap._lut[:, -1]) and np.all(ls_cmap._lut[:, -1] <= max_alpha)  # noqa: SIM300
    # Check that all values (excluding the bad/over/under) are monotonically increasing
    assert np.all(np.diff(ls_cmap._lut[: ls_cmap.N - 1, -1]) >= 0)
    assert [
        key1 == key2
        for key1, key2 in zip(orig_cmap._segmentdata.keys(), ls_cmap._segmentdata.keys())
    ]
    assert [
        np.allclose(val1, val2)
        for val1, val2 in zip(orig_cmap._segmentdata.values(), ls_cmap._segmentdata.values())
    ]

    fig, ax = plt.subplots(1, 1, figsize=(12, 10), constrained_layout=True)
    _ = plt.colorbar(plt.cm.ScalarMappable(cmap=ls_cmap), cax=ax, orientation="horizontal")
    if outdir:
        fig.savefig(outdir / f"{ls_cmap.name}.svg")
    plt.close(fig)
