# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2024 The NiPreps Developers <nipreps@gmail.com>
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

import matplotlib as mpl
import numpy as np
import pytest

from nireports.reportlets.modality.dwi import plot_dwi
from nireports.reportlets.utils import _latex_available


@pytest.mark.parametrize("expect_latex", [False, True], ids=["no-latex", "with-latex"])
def test_plot_dwi_uses_latex_typography(monkeypatch, expect_latex):
    """Check that plot_dwi enables the expected typography settings."""

    has_latex = _latex_available()
    if expect_latex != has_latex:
        pytest.skip(
            f"This parameter requires LaTeX  {'available' if expect_latex else 'unavailable'}."
        )

    observed = {}

    def fake_plot_anat(*args, **kwargs):
        observed["usetex"] = mpl.rcParams["text.usetex"]
        observed["family"] = mpl.rcParams["font.family"]
        observed["sans-serif"] = mpl.rcParams["font.sans-serif"]
        observed["title"] = kwargs["title"]

        return None

    monkeypatch.setattr(
        "nireports.reportlets.modality.dwi.plot_anat",
        fake_plot_anat,
    )

    data = np.random.default_rng(0).normal(size=(10, 10, 10))
    affine = np.eye(4)

    plot_dwi(data, affine)

    assert observed["usetex"] is expect_latex
    assert observed["family"] == ["sans-serif"]
    assert observed["sans-serif"] == ["Helvetica"]
    assert observed["title"]
