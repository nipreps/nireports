# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright The NiPreps Developers <nipreps@gmail.com>
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
"""Utilities for deprecating functions/classes."""

from nibabel.deprecator import Deprecator
from packaging.version import Version

from ._version import __version__


def version_comparator(target_version: str, pkg_version=__version__) -> int:
    """Compare ``target_version`` with ``pkg_version``.

    Returns 1 if target_version is greater than pkg_version, -1 if less, and 0 if equal.
    """
    version = Version(pkg_version)
    targ = Version(target_version)
    return (targ > version) - (targ < version)


deprecate_with_version = Deprecator(version_comparator=version_comparator)
