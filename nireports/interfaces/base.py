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
# STATEMENT OF CHANGES: This file was ported carrying over full git history from MRIQC,
# another NiPreps project licensed under the Apache-2.0 terms, and has been changed since.
# The original file this work derives from is found at:
# https://github.com/nipreps/mriqc/blob/1ffd4c8d1a20b44ebfea648a7b12bb32a425d4ec/
# mriqc/interfaces/viz.py
"""NiPype interface -- basic tooling."""

from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    File,
    traits,
)


class _PlotBaseInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc="File to be plotted")
    title = traits.Str(desc="a title string for the plot")
    annotate = traits.Bool(True, usedefault=True, desc="annotate left/right")
    figsize = traits.Tuple(
        (11.69, 8.27),
        traits.Float,
        traits.Float,
        usedefault=True,
        desc="Figure size",
    )
    dpi = traits.Int(300, usedefault=True, desc="Desired DPI of figure")
    out_file = File("mosaic.svg", usedefault=True, desc="output file name")
    cmap = traits.Str("Greys_r", usedefault=True)
