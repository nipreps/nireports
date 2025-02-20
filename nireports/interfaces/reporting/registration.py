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
"""ReportCapableInterfaces for registration tools."""

import os

from looseversion import LooseVersion
from nipype import logging
from nipype.interfaces import freesurfer as fs
from nipype.interfaces import fsl
from nipype.interfaces.base import (
    File,
    TraitedSpec,
    isdefined,
    traits,
)
from nipype.interfaces.mixins import reporting
from nipype.utils.filemanip import fname_presuffix

from nireports.interfaces.reporting import base as nrb

_LOGGER = logging.getLogger("nipype.interface")


class _ApplyTOPUPInputSpecRPT(nrb._SVGReportCapableInputSpec, fsl.epi.ApplyTOPUPInputSpec):
    wm_seg = File(argstr="-wmseg %s", desc="reference white matter segmentation mask")


class _ApplyTOPUPOutputSpecRPT(reporting.ReportCapableOutputSpec, fsl.epi.ApplyTOPUPOutputSpec):
    pass


class ApplyTOPUPRPT(nrb.RegistrationRC, fsl.ApplyTOPUP):
    input_spec = _ApplyTOPUPInputSpecRPT
    output_spec = _ApplyTOPUPOutputSpecRPT

    def _post_run_hook(self, runtime):
        from nilearn.image import index_img

        self._fixed_image_label = "after"
        self._moving_image_label = "before"
        self._fixed_image = index_img(self.aggregate_outputs(runtime=runtime).out_corrected, 0)
        self._moving_image = index_img(self.inputs.in_files[0], 0)
        self._contour = self.inputs.wm_seg if isdefined(self.inputs.wm_seg) else None
        _LOGGER.info(
            "Report - setting corrected (%s) and warped (%s) images",
            self._fixed_image,
            self._moving_image,
        )

        return super()._post_run_hook(runtime)


class _FUGUEInputSpecRPT(nrb._SVGReportCapableInputSpec, fsl.preprocess.FUGUEInputSpec):
    wm_seg = File(argstr="-wmseg %s", desc="reference white matter segmentation mask")


class _FUGUEOutputSpecRPT(reporting.ReportCapableOutputSpec, fsl.preprocess.FUGUEOutputSpec):
    pass


class FUGUERPT(nrb.RegistrationRC, fsl.FUGUE):
    input_spec = _FUGUEInputSpecRPT
    output_spec = _FUGUEOutputSpecRPT

    def _post_run_hook(self, runtime):
        self._fixed_image_label = "after"
        self._moving_image_label = "before"
        self._fixed_image = self.aggregate_outputs(runtime=runtime).unwarped_file
        self._moving_image = self.inputs.in_file
        self._contour = self.inputs.wm_seg if isdefined(self.inputs.wm_seg) else None
        _LOGGER.info(
            "Report - setting corrected (%s) and warped (%s) images",
            self._fixed_image,
            self._moving_image,
        )

        return super()._post_run_hook(runtime)


class _FLIRTInputSpecRPT(nrb._SVGReportCapableInputSpec, fsl.preprocess.FLIRTInputSpec):
    pass


class _FLIRTOutputSpecRPT(reporting.ReportCapableOutputSpec, fsl.preprocess.FLIRTOutputSpec):
    pass


class FLIRTRPT(nrb.RegistrationRC, fsl.FLIRT):
    input_spec = _FLIRTInputSpecRPT
    output_spec = _FLIRTOutputSpecRPT

    def _post_run_hook(self, runtime):
        self._fixed_image = self.inputs.reference
        self._moving_image = self.aggregate_outputs(runtime=runtime).out_file
        self._contour = self.inputs.wm_seg if isdefined(self.inputs.wm_seg) else None
        _LOGGER.info(
            "Report - setting fixed (%s) and moving (%s) images",
            self._fixed_image,
            self._moving_image,
        )

        return super()._post_run_hook(runtime)


class _ApplyXFMInputSpecRPT(nrb._SVGReportCapableInputSpec, fsl.preprocess.ApplyXFMInputSpec):
    pass


class ApplyXFMRPT(FLIRTRPT, fsl.ApplyXFM):
    input_spec = _ApplyXFMInputSpecRPT
    output_spec = _FLIRTOutputSpecRPT


_BBRegisterInputSpec: type[TraitedSpec] = fs.preprocess.BBRegisterInputSpec6
if LooseVersion("0.0.0") < fs.Info.looseversion() < LooseVersion("6.0.0"):
    _BBRegisterInputSpec = fs.preprocess.BBRegisterInputSpec


class _BBRegisterInputSpecRPT(nrb._SVGReportCapableInputSpec, _BBRegisterInputSpec):
    # Adds default=True, usedefault=True
    out_lta_file = traits.Either(
        traits.Bool,
        File,
        default=True,
        usedefault=True,
        argstr="--lta %s",
        min_ver="5.2.0",
        desc="write the transformation matrix in LTA format",
    )


class _BBRegisterOutputSpecRPT(
    reporting.ReportCapableOutputSpec, fs.preprocess.BBRegisterOutputSpec
):
    pass


class BBRegisterRPT(nrb.RegistrationRC, fs.BBRegister):
    input_spec = _BBRegisterInputSpecRPT
    output_spec = _BBRegisterOutputSpecRPT

    def _post_run_hook(self, runtime):
        outputs = self.aggregate_outputs(runtime=runtime)
        mri_dir = os.path.join(self.inputs.subjects_dir, self.inputs.subject_id, "mri")
        target_file = os.path.join(mri_dir, "brainmask.mgz")

        # Apply transform for simplicity
        mri_vol2vol = fs.ApplyVolTransform(
            source_file=self.inputs.source_file,
            target_file=target_file,
            lta_file=outputs.out_lta_file,
            interp="nearest",
        )
        res = mri_vol2vol.run()

        self._fixed_image = target_file
        self._moving_image = res.outputs.transformed_file
        self._contour = os.path.join(mri_dir, "ribbon.mgz")
        _LOGGER.info(
            "Report - setting fixed (%s) and moving (%s) images",
            self._fixed_image,
            self._moving_image,
        )

        return super()._post_run_hook(runtime)


class _MRICoregInputSpecRPT(nrb._SVGReportCapableInputSpec, fs.registration.MRICoregInputSpec):
    pass


class _MRICoregOutputSpecRPT(
    reporting.ReportCapableOutputSpec, fs.registration.MRICoregOutputSpec
):
    pass


class MRICoregRPT(nrb.RegistrationRC, fs.MRICoreg):
    input_spec = _MRICoregInputSpecRPT
    output_spec = _MRICoregOutputSpecRPT

    def _post_run_hook(self, runtime):
        outputs = self.aggregate_outputs(runtime=runtime)
        mri_dir = None
        if isdefined(self.inputs.subject_id):
            mri_dir = os.path.join(self.inputs.subjects_dir, self.inputs.subject_id, "mri")

        if isdefined(self.inputs.reference_file):
            target_file = self.inputs.reference_file
        else:
            target_file = os.path.join(mri_dir, "brainmask.mgz")

        # Apply transform for simplicity
        mri_vol2vol = fs.ApplyVolTransform(
            source_file=self.inputs.source_file,
            target_file=target_file,
            lta_file=outputs.out_lta_file,
            interp="nearest",
        )
        res = mri_vol2vol.run()

        self._fixed_image = target_file
        self._moving_image = res.outputs.transformed_file
        if mri_dir is not None:
            self._contour = os.path.join(mri_dir, "ribbon.mgz")
        _LOGGER.info(
            "Report - setting fixed (%s) and moving (%s) images",
            self._fixed_image,
            self._moving_image,
        )

        return super()._post_run_hook(runtime)


class _SimpleBeforeAfterInputSpecRPT(nrb._SVGReportCapableInputSpec):
    before = File(exists=True, mandatory=True, desc="file before")
    after = File(exists=True, mandatory=True, desc="file after")
    wm_seg = File(desc="reference white matter segmentation mask")
    before_label = traits.Str("before", usedefault=True)
    after_label = traits.Str("after", usedefault=True)
    dismiss_affine = traits.Bool(False, usedefault=True, desc="rotate image(s) to cardinal axes")


class SimpleBeforeAfterRPT(nrb.RegistrationRC, nrb.ReportingInterface):
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
        os.unlink(fname)

        return runtime
