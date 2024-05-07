23.2.1 (May 07, 2024)
=====================
Bug-fix release in the 23.2.x series.

CHANGES
-------

**Full Changelog**: https://github.com/nipreps/nireports/compare/23.2.0...23.2.1

* MNT: Fix matplotlib.cm.get_cmap deprecation (#98)

23.2.0 (December 13, 2023)
==========================

A new minor release with support for Python 3.12, matplotlib 3.8,
and dropping the implicit dependency on setuptools.

CHANGES
-------

**Full Changelog**: https://github.com/nipreps/nireports/compare/23.1.0...23.2.0

* FIX: Fix AttributeError Xtick has no attribute label (#84)
* FIX: Typos found by codespell (#79)
* ENH: Add session filtering to report generation (#82)
* ENH: Add `ignore_initial_volumes` param to `ConfoundsCorrelationPlot` (#83)
* RF: Purge pkg_resources, add data loader (#85)
* STY: Assorted pyupgrade suggestions (#80)

23.1.0 (June 13, 2023)
======================
A new minor release including several bugfixes and a new module for diffusion MRI data plotting tools.

CHANGES
-------
**Full Changelog**: https://github.com/nipreps/nireports/compare/23.0.1...23.1.0

* FIX: Calculation of aspect ratio of mosaics (#76)
* FIX: Bugs discovered generating DWI reports (#73)
* FIX: Improve handling of reportlet style (#68)
* FIX: Plugin inclusion via main bootstrap file did not work (#64)
* ENH: Better SNR levels for representation in DWI heatmaps (#77)
* ENH: Add a new DWI heatmap for quality control (#75)
* ENH: Port basic report-capable interfaces from *NiWorkflows* (#74)
* ENH: Add a ``bval-<label>`` entity (#72)
* ENH: Allow CSS styling of reportlets in bootstrap file (#67)
* ENH: Improve handling of auth token by rating-widget (#66)
* ENH: Advanced metadata interpolation (#65)
* ENH: BIDS filters and support *plugins* (incl. a rating widget as the example) (#62)
* ENH: Allow different types of reportlets, not only BIDS-based (#60)
* ENH: Upgrade bootstrap to 5.0.2 (#59)
* ENH: Allow plotting of image rotated to cardinal axes (#650)
* DOC: Adds a docstring to the ``compose_view`` function. (#63)
* DOC: Ensure copyright notice in all headers' comment (#635)
* MAINT: Replace distutils use, upgrade versioneer (#725)
* MAINT: Refactor structure of interfaces (#603)
* CI: Try older codecov orb (#70)
* CI: Purge codecov Python package (#69)

23.0.1 (March 10, 2023)
=======================
Hotfix release porting `nipreps/niworkflows#785 <https://github.com/nipreps/niworkflows/pull/785>`__.

23.0.0 (March 10, 2023)
=======================
The first OFFICIAL RELEASE of *NiReports* is out!
This first version of the package ports the visualization tools from *MRIQC* and *NiWorkflows* into a common API.
In addition, the plotting of mosaic views (*MRIQC*) is flexibilized so that rodent imaging can conveniently be also visualized.

CHANGES
-------
**Full Changelog**: https://github.com/nipreps/nireports/compare/0.2.0...23.0.0

* FIX: Bug in ``plot_mosaic`` introduced in #52 (666ac5b)
* ENH: Flexibilize views of ``plot_mosaic`` to render nonhuman imaging by @oesteban in https://github.com/nipreps/nireports/pull/52
* ENH: Set up CI on CircleCI for artifact visualization  by @esavary in https://github.com/nipreps/nireports/pull/50
* ENH: API refactor of *NiPype* interfaces by @oesteban in https://github.com/nipreps/nireports/pull/51
* MAINT: Updated ``MAINTAINERS.md`` by @esavary in https://github.com/nipreps/nireports/pull/49
* MAINT: Add Governance files (#48)


.. admonition:: Author list for papers based on *NiReports* 23.0 series

    As described in the `Contributor Guidelines
    <https://www.nipreps.org/community/CONTRIBUTING/#recognizing-contributions>`__,
    anyone listed as developer or contributor may write and submit manuscripts
    about *NiReports*.
    To do so, please move the author(s) name(s) to the front of the following list:

    Christopher J. Markiewicz \ :sup:`1`\ ; Zvi Baratz \ :sup:`2`\ ; Elodie Savary \ :sup:`3`\ ; Mathias Goncalves \ :sup:`1`\ ; Ross W. Blair \ :sup:`1`\ ; Eilidh MacNicol \ :sup:`4`\ ; Céline Provins \ :sup:`3`\ ; Dylan Nielson \ :sup:`5`\ ; Russell A. Poldrack \ :sup:`1`\ ; Oscar Esteban \ :sup:`6`\ .

    Affiliations:

      1. Department of Psychology, Stanford University, CA, USA
      2. Sagol School of Neuroscience, Tel Aviv University, Tel Aviv, Israel
      3. Department of Radiology, Lausanne University Hospital and University of Lausanne, Switzerland
      4. Department of Neuroimaging, Institute of Psychiatry, Psychology and Neuroscience, King's College London, London, UK
      5. Section on Clinical and Computational Psychiatry, National Institute of Mental Health, Bethesda, MD, USA
      6. Department of Radiology, Lausanne University Hospital and University of Lausanne

Pre 23.0.0
==========
A number of pre-releases were launched before 23.0.0 to test the deployment and the integration tests.
