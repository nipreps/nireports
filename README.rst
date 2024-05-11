
.. image:: https://img.shields.io/pypi/v/nireports.svg
  :target: https://pypi.python.org/pypi/nireports/
  :alt: Latest Version
.. image:: https://img.shields.io/badge/License-Apache_2.0-blue.svg
  :target: https://github.com/nipreps/eddymotion/blob/main/LICENSE
  :alt: License
.. image:: https://readthedocs.org/projects/nireports/badge/?version=latest
  :target: https://nireports.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status
.. image:: https://circleci.com/gh/nipreps/nireports/tree/main.svg?style=shield
  :target: https://circleci.com/gh/nipreps/nireports/tree/main
  :alt: Testing
.. image:: https://github.com/nipreps/nireports/actions/workflows/build_test_deploy.yml/badge.svg
  :target: https://github.com/nipreps/nireports/actions/workflows/build_test_deploy.yml
  :alt: Build-Test-Deploy
.. image:: https://codecov.io/gh/nipreps/nireports/branch/main/graph/badge.svg?token=OPH6D32GWN
  :target: https://codecov.io/gh/nipreps/nireports

*NiReports*: the *NiPreps*' reporting and visualization tools
=============================================================

*NiReports* contains the two main components of the *visual reporting system* of *NiPreps*:

* **Reportlets**: visualizations for assessing the quality of a particular processing step within the neuroimaging pipeline.
  Typically, reportlets show brain mosaics perhaps with contours and/or segmentations.
  They can be *dynamic* and flicker between two different *states* to help assess the accuracy of image registrations.
  However, the reportlets are not limited to brain mosaics, and can contain correlation plots, BOLD fMRI *carpetplots*, etc.
* **Assembler**: end-user *NiPreps* write out reportlets to a predetermined folder, which is then queried by the assembler using *PyBIDS*.
  The assembler follows a *report specification* in YAML format, which states the query to find specific reportlets and their corresponding metadata and text annotations.
  As a result, one HTML file with a concatenation of reportlets is produced.
