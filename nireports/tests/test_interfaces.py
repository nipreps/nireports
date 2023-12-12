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

import pytest

from nireports.interfaces.nuisance import CompCorVariancePlot, ConfoundsCorrelationPlot


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


@pytest.mark.parametrize('ignore_initial_volumes', (0, 1))
def test_ConfoundsCorrelationPlot(datadir, ignore_initial_volumes):
    """confounds correlation report test"""
    confounds_file = os.path.join(datadir, "confounds_test.tsv")
    cc_rpt = ConfoundsCorrelationPlot(
        confounds_file=confounds_file,
        reference_column="a",
        ignore_initial_volumes=ignore_initial_volumes,
    )
    _smoke_test_report(cc_rpt, f"confounds_correlation_{ignore_initial_volumes}.svg")
