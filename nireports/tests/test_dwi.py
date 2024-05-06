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
"""Test DWI reportlets."""

import pytest

import numpy as np
from matplotlib import pyplot as plt

from nireports.reportlets.modality.dwi import plot_gradients


@pytest.mark.parametrize(
    'dwi_btable',
    ['ds000114_singleshell', 'hcph_multishell', 'ds004737_dsi'],
)
def test_plot_gradients(tmp_path, testdata_path, dwi_btable, outdir):
    """Check the plot of DWI gradients."""

    bvecs = np.loadtxt(testdata_path / f'{dwi_btable}.bvec').T
    bvals = np.loadtxt(testdata_path / f'{dwi_btable}.bval')

    b0s_mask = bvals < 50

    gradients = np.hstack([bvecs[~b0s_mask], bvals[~b0s_mask, None]])
    _ = plot_gradients(gradients)

    if outdir is not None:
        plt.savefig(outdir / f'{dwi_btable}.svg', bbox_inches='tight')