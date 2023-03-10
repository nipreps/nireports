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
