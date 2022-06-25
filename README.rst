
.. image:: https://readthedocs.org/projects/nireports/badge/?version=latest
  :target: https://nireports.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status
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
