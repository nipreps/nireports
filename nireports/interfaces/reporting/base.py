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
"""class mixin and utilities for enabling reports for nipype interfaces."""
from pathlib import Path
from nipype import logging
from nipype.utils.filemanip import fname_presuffix
from nipype.interfaces.base import File, traits, isdefined
from nipype.interfaces.mixins import reporting
from nireports.reportlets.utils import cuts_from_bbox, compose_view

_LOGGER = logging.getLogger("nipype.interface")


class _SVGReportCapableInputSpec(reporting.ReportCapableInputSpec):
    out_report = File("report.svg", usedefault=True, desc="filename for the visual report")
    compress_report = traits.Enum(
        "auto",
        True,
        False,
        usedefault=True,
        desc="Compress the reportlet using SVGO or"
        "WEBP. 'auto' - compress if relevant "
        "software is installed, True = force,"
        "False - don't attempt to compress",
    )


class _RegistrationRCInputSpecRPT(_SVGReportCapableInputSpec):
    fixed_params = traits.Dict(
        traits.Str, value={}, usedefault=True, desc="pass parameters to plotter",
    )
    moving_params = traits.Dict(
        traits.Str, value={}, usedefault=True, desc="pass parameters to plotter",
    )


class RegistrationRC(reporting.ReportCapableInterface):
    """An abstract mixin to registration nipype interfaces."""

    input_spec = _RegistrationRCInputSpecRPT

    _fixed_image = None
    _moving_image = None
    _fixed_image_mask = None
    _fixed_image_label = "fixed"
    _moving_image_label = "moving"
    _contour = None
    _dismiss_affine = False

    def _generate_report(self):
        """Generate the visual report."""
        from nilearn.image import threshold_img, load_img
        from nilearn.masking import apply_mask, unmask
        from nireports.reportlets.mosaic import plot_registration

        _LOGGER.info("Generating visual report")

        fixed_image_nii = load_img(self._fixed_image)
        moving_image_nii = load_img(self._moving_image)
        contour_nii = load_img(self._contour) if self._contour is not None else None

        if self._fixed_image_mask:
            fixed_image_nii = unmask(
                apply_mask(fixed_image_nii, self._fixed_image_mask),
                self._fixed_image_mask,
            )
            # since the moving image is already in the fixed image space we
            # should apply the same mask
            moving_image_nii = unmask(
                apply_mask(moving_image_nii, self._fixed_image_mask),
                self._fixed_image_mask,
            )
            mask_nii = load_img(self._fixed_image_mask)
        else:
            mask_nii = threshold_img(fixed_image_nii, 1e-3)

        n_cuts = 7
        if not self._fixed_image_mask and contour_nii:
            cuts = cuts_from_bbox(contour_nii, cuts=n_cuts)
        else:
            cuts = cuts_from_bbox(mask_nii, cuts=n_cuts)

        # Call composer
        compose_view(
            plot_registration(
                fixed_image_nii,
                "fixed-image",
                estimate_brightness=True,
                cuts=cuts,
                label=self._fixed_image_label,
                contour=contour_nii,
                compress=self.inputs.compress_report,
                dismiss_affine=self._dismiss_affine,
                plot_params=self.inputs.fixed_params,
            ),
            plot_registration(
                moving_image_nii,
                "moving-image",
                estimate_brightness=True,
                cuts=cuts,
                label=self._moving_image_label,
                contour=contour_nii,
                compress=self.inputs.compress_report,
                dismiss_affine=self._dismiss_affine,
                plot_params=self.inputs.moving_params,
            ),
            out_file=self._out_report,
        )


class SegmentationRC(reporting.ReportCapableInterface):
    """An abstract mixin to segmentation nipype interfaces."""

    def _generate_report(self):
        from nireports.reportlets.mosaic import plot_segs

        compose_view(
            plot_segs(
                image_nii=self._anat_file,
                seg_niis=self._seg_files,
                bbox_nii=self._mask_file,
                out_file=self.inputs.out_report,
                masked=self._masked,
                compress=self.inputs.compress_report,
            ),
            fg_svgs=None,
            out_file=self._out_report,
        )


class SurfaceSegmentationRC(reporting.ReportCapableInterface):
    """An abstract mixin to registration nipype interfaces."""

    _anat_file = None
    _mask_file = None
    _contour = None

    def _generate_report(self):
        """Generate the visual report."""
        from nilearn.image import threshold_img, load_img
        from nilearn.masking import apply_mask, unmask
        from nireports.reportlets.mosaic import plot_registration

        _LOGGER.info("Generating visual report")

        anat = load_img(self._anat_file)
        contour_nii = load_img(self._contour) if self._contour is not None else None

        if self._mask_file:
            anat = unmask(apply_mask(anat, self._mask_file), self._mask_file)
            mask_nii = load_img(self._mask_file)
        else:
            mask_nii = threshold_img(anat, 1e-3)

        n_cuts = 7
        if not self._mask_file and contour_nii:
            cuts = cuts_from_bbox(contour_nii, cuts=n_cuts)
        else:
            cuts = cuts_from_bbox(mask_nii, cuts=n_cuts)

        # Call composer
        compose_view(
            plot_registration(
                anat,
                "fixed-image",
                estimate_brightness=True,
                cuts=cuts,
                contour=contour_nii,
                compress=self.inputs.compress_report,
            ),
            [],
            out_file=self._out_report,
        )


class ReportingInterface(reporting.ReportCapableInterface):
    """
    Interface that always generates a report.

    A subclass must define an ``input_spec`` and override ``_generate_report``.

    """

    output_spec = reporting.ReportCapableOutputSpec

    def __init__(self, generate_report=True, **kwargs):
        super().__init__(generate_report=generate_report, **kwargs)

    def _run_interface(self, runtime):
        return runtime


class _SimpleBeforeAfterInputSpecRPT(_RegistrationRCInputSpecRPT):
    before = File(exists=True, mandatory=True, desc="file before")
    after = File(exists=True, mandatory=True, desc="file after")
    wm_seg = File(desc="reference white matter segmentation mask")
    before_label = traits.Str("before", usedefault=True)
    after_label = traits.Str("after", usedefault=True)
    dismiss_affine = traits.Bool(False, usedefault=True, desc="rotate image(s) to cardinal axes")


class SimpleBeforeAfterRPT(RegistrationRC, ReportingInterface):
    input_spec = _SimpleBeforeAfterInputSpecRPT

    def _post_run_hook(self, runtime):
        """there is not inner interface to run"""
        self._fixed_image_label = self.inputs.after_label
        self._moving_image_label = self.inputs.before_label
        self._fixed_image = self.inputs.after
        self._moving_image = self.inputs.before
        self._contour = self.inputs.wm_seg if isdefined(self.inputs.wm_seg) else None
        self._dismiss_affine = self.inputs.dismiss_affine
        _LOGGER.info(
            "Report - setting before (%s) and after (%s) images",
            self._fixed_image,
            self._moving_image,
        )

        return super()._post_run_hook(runtime)


class _ResampleBeforeAfterInputSpecRPT(_SimpleBeforeAfterInputSpecRPT):
    base = traits.Enum("before", "after", usedefault=True, mandatory=True)


class ResampleBeforeAfterRPT(SimpleBeforeAfterRPT):
    input_spec = _ResampleBeforeAfterInputSpecRPT

    def _post_run_hook(self, runtime):
        from nilearn import image as nli

        self._fixed_image = self.inputs.after
        self._moving_image = self.inputs.before
        if self.inputs.base == "before":
            resampled_after = nli.resample_to_img(self._fixed_image, self._moving_image)
            fname = fname_presuffix(self._fixed_image, suffix="_resampled", newpath=runtime.cwd)
            resampled_after.to_filename(fname)
            self._fixed_image = fname
        else:
            resampled_before = nli.resample_to_img(self._moving_image, self._fixed_image)
            fname = fname_presuffix(self._moving_image, suffix="_resampled", newpath=runtime.cwd)
            resampled_before.to_filename(fname)
            self._moving_image = fname
        self._contour = self.inputs.wm_seg if isdefined(self.inputs.wm_seg) else None
        _LOGGER.info(
            "Report - setting before (%s) and after (%s) images",
            self._fixed_image,
            self._moving_image,
        )

        runtime = super()._post_run_hook(runtime)
        _LOGGER.info("Successfully created report (%s)", self._out_report)
        Path(fname).unlink(missing_ok=True)

        return runtime
