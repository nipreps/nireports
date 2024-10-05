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
"""Registration tests"""

import os
from shutil import copy
from tempfile import TemporaryDirectory

import pytest
from nipype.pipeline import engine as pe

from nireports.interfaces.reporting.registration import (
    FLIRTRPT,
    ApplyXFMRPT,
    BBRegisterRPT,
    MRICoregRPT,
    SimpleBeforeAfterRPT,
)
from nireports.tests.testing import _run_interface_mock, has_freesurfer, has_fsl


def _smoke_test_report(report_interface, artifact_name):
    with TemporaryDirectory() as tmpdir:
        res = pe.Node(report_interface, name="smoke_test", base_dir=tmpdir).run()
        out_report = res.outputs.out_report

        save_artifacts = os.getenv("SAVE_CIRCLE_ARTIFACTS", False)
        if save_artifacts:
            copy(out_report, os.path.join(save_artifacts, artifact_name))
        assert os.path.isfile(out_report), "Report does not exist"


@pytest.mark.skipif(not has_fsl, reason="No FSL")
def test_FLIRTRPT(reference, moving):
    """the FLIRT report capable test"""
    flirt_rpt = FLIRTRPT(generate_report=True, in_file=moving, reference=reference)
    _smoke_test_report(flirt_rpt, "testFLIRT.svg")


@pytest.mark.skipif(not has_freesurfer, reason="No FreeSurfer")
def test_MRICoregRPT(monkeypatch, test_data_package, reference, moving, nthreads):
    """the MRICoreg report capable test"""

    def _agg(objekt, runtime):
        outputs = objekt.output_spec()
        outputs.out_lta_file = str(
            (test_data_package / "testMRICoregRPT-out_lta_file.lta").resolve()
        )
        outputs.out_report = os.path.join(runtime.cwd, objekt.inputs.out_report)
        return outputs

    # Patch the _run_interface method
    monkeypatch.setattr(MRICoregRPT, "_run_interface", _run_interface_mock)
    monkeypatch.setattr(MRICoregRPT, "aggregate_outputs", _agg)

    mri_coreg_rpt = MRICoregRPT(
        generate_report=True,
        source_file=moving,
        reference_file=reference,
        num_threads=nthreads,
    )
    _smoke_test_report(mri_coreg_rpt, "testMRICoreg.svg")


@pytest.mark.skipif(not has_fsl, reason="No FSL")
def test_ApplyXFMRPT(reference, moving):
    """the ApplyXFM report capable test"""
    flirt_rpt = FLIRTRPT(generate_report=False, in_file=moving, reference=reference)

    applyxfm_rpt = ApplyXFMRPT(
        generate_report=True,
        in_file=moving,
        in_matrix_file=flirt_rpt.run().outputs.out_matrix_file,
        reference=reference,
        apply_xfm=True,
    )
    _smoke_test_report(applyxfm_rpt, "testApplyXFM.svg")


@pytest.mark.skipif(not has_fsl, reason="No FSL")
def test_SimpleBeforeAfterRPT(reference, moving):
    """the SimpleBeforeAfterRPT report capable test"""
    flirt_rpt = FLIRTRPT(generate_report=False, in_file=moving, reference=reference)

    ba_rpt = SimpleBeforeAfterRPT(
        generate_report=True, before=reference, after=flirt_rpt.run().outputs.out_file
    )
    _smoke_test_report(ba_rpt, "test_SimpleBeforeAfterRPT.svg")


@pytest.mark.skipif(not has_fsl, reason="No FSL")
def test_FLIRTRPT_w_BBR(reference, reference_mask, moving):
    """test FLIRTRPT with input `wm_seg` set.
    For the sake of testing ONLY, `wm_seg` is set to the filename of a brain mask"""
    flirt_rpt = FLIRTRPT(
        generate_report=True, in_file=moving, reference=reference, wm_seg=reference_mask
    )
    _smoke_test_report(flirt_rpt, "testFLIRTRPTBBR.svg")


@pytest.mark.skipif(not has_freesurfer, reason="No FreeSurfer")
def test_BBRegisterRPT(monkeypatch, moving, test_data_package):
    """the BBRegister report capable test"""

    def _agg(objekt, runtime):
        outputs = objekt.output_spec()
        outputs.out_lta_file = os.path.join(
            test_data_package, "testBBRegisterRPT-out_lta_file.lta"
        )
        outputs.out_report = os.path.join(runtime.cwd, objekt.inputs.out_report)
        return outputs

    # Patch the _run_interface method
    monkeypatch.setattr(BBRegisterRPT, "_run_interface", _run_interface_mock)
    monkeypatch.setattr(BBRegisterRPT, "aggregate_outputs", _agg)

    subject_id = "fsaverage"
    bbregister_rpt = BBRegisterRPT(
        generate_report=True,
        contrast_type="t1",
        init="fsl",
        source_file=moving,
        subject_id=subject_id,
        registered_file=True,
    )
    _smoke_test_report(bbregister_rpt, "testBBRegister.svg")
