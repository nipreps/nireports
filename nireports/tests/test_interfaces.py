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
"""Tests plotting interfaces."""

import os
from shutil import copy

import nibabel as nb
import numpy as np
import pytest

from nireports.interfaces.fmri import FMRISummary
from nireports.interfaces.nuisance import (
    CompCorVariancePlot,
    ConfoundsCorrelationPlot,
    RaincloudPlot,
)
from nireports.tests.utils import _generate_raincloud_random_data


def _smoke_test_report(report_interface, artifact_name):
    out_report = report_interface.run().outputs.out_file

    save_artifacts = os.getenv("SAVE_CIRCLE_ARTIFACTS", False)
    if save_artifacts:
        copy(out_report, os.path.join(save_artifacts, artifact_name))
    assert os.path.isfile(out_report), 'Report "%s" does not exist' % out_report


def test_CompCorVariancePlot(datadir):
    """CompCor variance report test"""
    metadata_file = os.path.join(datadir, "confounds_metadata_test.tsv")
    cc_rpt = CompCorVariancePlot(metadata_files=[metadata_file], metadata_sources=["aCompCor"])
    _smoke_test_report(cc_rpt, "compcor_variance.svg")


@pytest.mark.parametrize("ignore_initial_volumes", (0, 1))
def test_ConfoundsCorrelationPlot(datadir, ignore_initial_volumes):
    """confounds correlation report test"""
    confounds_file = os.path.join(datadir, "confounds_test.tsv")
    cc_rpt = ConfoundsCorrelationPlot(
        confounds_file=confounds_file,
        reference_column="a",
        ignore_initial_volumes=ignore_initial_volumes,
    )
    _smoke_test_report(cc_rpt, f"confounds_correlation_{ignore_initial_volumes}.svg")


@pytest.mark.parametrize("orient", ["h", "v"])
@pytest.mark.parametrize("density", (True, False))
def test_RaincloudPlot(orient, density, tmp_path):
    """Raincloud plot report test"""
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
    rc_rpt = RaincloudPlot(
        data_file=data_file,
        group_name=group_label,
        feature=features_label,
        palette=palette,
        orient=orient,
        density=density,
        mark_nans=mark_nans,
    )
    _smoke_test_report(rc_rpt, f"raincloud_orient-{orient}_density-{density}.svg")


def test_FMRISummary(testdata_path, tmp_path, outdir):
    """Exercise the FMRISummary interface."""
    rng = np.random.default_rng(2010)

    in_func = testdata_path / "sub-ds205s03_task-functionallocalizer_run-01_bold_volreg.nii.gz"
    ntimepoints = nb.load(in_func).shape[-1]

    np.savetxt(
        tmp_path / "fd.txt",
        rng.normal(0.2, 0.2, ntimepoints - 1).T,
    )

    np.savetxt(
        tmp_path / "outliers.txt",
        rng.normal(0.2, 0.2, ntimepoints - 1).T,
    )

    np.savetxt(
        tmp_path / "dvars.txt",
        rng.normal(0.2, 0.2, (ntimepoints - 1, 2)),
    )

    interface = FMRISummary(
        in_func=str(in_func),
        in_segm=str(
            testdata_path / "sub-ds205s03_task-functionallocalizer_run-01_bold_parc.nii.gz"
        ),
        fd=str(tmp_path / "fd.txt"),
        outliers=str(tmp_path / "outliers.txt"),
        dvars=str(tmp_path / "dvars.txt"),
    )

    result = interface.run()

    if outdir is not None:
        from shutil import copy

        copy(result.outputs.out_file, outdir / "fmriplot_nipype.svg")
