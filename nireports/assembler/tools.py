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
# STATEMENT OF CHANGES: This file was ported carrying over full git history from niworkflows,
# another NiPreps project licensed under the Apache-2.0 terms, and has been changed since.
"""Utilities for the :mod:`~nireports.assembler` module."""

from pathlib import Path

from nireports.assembler.report import Report


def run_reports(
    out_dir,
    subject_label,
    run_uuid,
    bootstrap_file=None,
    reportlets_dir=None,
):
    """
    Run the reports.

    Examples
    --------
    .. testsetup::

       >>> from shutil import copytree
       >>> from nireports.assembler import data
       >>> test_data_path = data.load('tests', 'work')
       >>> testdir = Path(tmpdir)
       >>> data_dir = copytree(test_data_path, str(testdir / 'work'))
       >>> (testdir / 'nireports').mkdir(parents=True, exist_ok=True)

    >>> run_reports(testdir / 'out' / 'nireports', '01', 'madeoutuuid',
    ...             reportlets_dir=testdir / 'work' / 'reportlets' / 'nireports')
    0

    """
    return Report(
        out_dir,
        run_uuid,
        bootstrap_file=bootstrap_file,
        reportlets_dir=reportlets_dir,
        subject=subject_label,
    ).generate_report()


def generate_reports(
    subject_list,
    output_dir,
    run_uuid,
    bootstrap_file=None,
    work_dir=None,
):
    """Execute run_reports on a list of subjects."""
    reportlets_dir = None
    if work_dir is not None:
        reportlets_dir = Path(work_dir) / "reportlets"
    report_errors = [
        run_reports(
            output_dir,
            subject_label,
            run_uuid,
            bootstrap_file=bootstrap_file,
            reportlets_dir=reportlets_dir,
        )
        for subject_label in subject_list
    ]

    errno = sum(report_errors)
    if errno:
        import logging

        logger = logging.getLogger("cli")
        error_list = ", ".join(
            "%s (%d)" % (subid, err) for subid, err in zip(subject_list, report_errors) if err
        )
        logger.error(
            "Preprocessing did not finish successfully. Errors occurred while processing "
            "data from participants: %s. Check the HTML reports for details.",
            error_list,
        )
    return errno
