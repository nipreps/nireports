"""
Utilities for the :mod:`~nireports.assembler` module.
"""


from pathlib import Path

from nireports.assembler.report import Report


def run_reports(
    out_dir,
    subject_label,
    run_uuid,
    config=None,
    reportlets_dir=None,
    packagename=None,
):
    """
    Run the reports.

    .. testsetup::

    >>> cwd = os.getcwd()
    >>> os.chdir(tmpdir)

    >>> from pkg_resources import resource_filename
    >>> from shutil import copytree
    >>> test_data_path = resource_filename('nireports', 'assembler/data/tests/work')
    >>> testdir = Path(tmpdir)
    >>> data_dir = copytree(test_data_path, str(testdir / 'work'))
    >>> (testdir / 'fmriprep').mkdir(parents=True, exist_ok=True)

    .. doctest::

    >>> run_reports(testdir / 'out', '01', 'madeoutuuid', packagename='fmriprep',
    ...             reportlets_dir=testdir / 'work' / 'reportlets')
    0

    .. testcleanup::

    >>> os.chdir(cwd)

    """
    return Report(
        out_dir,
        run_uuid,
        config=config,
        subject_id=subject_label,
        packagename=packagename,
        reportlets_dir=reportlets_dir,
    ).generate_report()


def generate_reports(
    subject_list,
    output_dir,
    run_uuid,
    config=None,
    work_dir=None,
    packagename=None,
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
            config=config,
            packagename=packagename,
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
