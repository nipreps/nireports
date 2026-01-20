# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2021 The NiPreps Developers <nipreps@gmail.com>
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
"""py.test configuration file"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
from templateflow.api import get as get_template

from nireports.tests.testing import data_env_canary


@pytest.fixture(scope="session")
def datadir():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "data") + os.path.sep)


@pytest.fixture
def reference():
    return str(get_template("MNI152Lin", resolution=2, desc=None, suffix="T1w"))


@pytest.fixture
def reference_mask():
    return str(get_template("MNI152Lin", resolution=2, desc="brain", suffix="mask"))


@pytest.fixture
def moving(test_data_home):
    data_env_canary()
    return str(test_data_home / "ds000003/sub-01/anat/sub-01_T1w.nii.gz")


@pytest.fixture
def nthreads():
    from os import cpu_count, getenv

    # Tests are linear, so don't worry about leaving space for a control thread
    return min(int(getenv("CIRCLE_NPROCS", "8")), cpu_count())


@pytest.fixture(autouse=True)
def close_mpl_figures():
    """Automatically close all Matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture(autouse=True)
def random_number_generator(request):
    """Automatically set a fixed-seed random number generator for all tests."""
    request.node.rng = np.random.default_rng(1234)


def pytest_addoption(parser):
    parser.addoption(
        "--warnings-as-errors",
        action="store_true",
        help="Consider all uncaught warnings as errors.",
    )


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    have_werrors = os.getenv("NIREPORTS_WERRORS", False)
    have_werrors = session.config.getoption("--warnings-as-errors", False) or have_werrors
    if have_werrors:
        # Check if there were any warnings during the test session
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter.stats.get("warnings", None):
            session.exitstatus = 2


@pytest.hookimpl
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    have_werrors = os.getenv("NIREPORTS_WERRORS", False)
    have_werrors = config.getoption("--warnings-as-errors", False) or have_werrors
    have_warnings = terminalreporter.stats.get("warnings", None)
    if have_warnings and have_werrors:
        terminalreporter.ensure_newline()
        terminalreporter.section("Werrors", sep="=", red=True, bold=True)
        terminalreporter.line(
            f"Warnings as errors: Activated.\n{len(have_warnings)} warnings were raised and treated as errors.\n"
        )
