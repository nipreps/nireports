import os
from datetime import datetime as dt
from functools import wraps
from pathlib import Path

import numpy as np
import pytest
from nipype.interfaces import afni, fsl
from nipype.interfaces import freesurfer as fs

from nireports.conftest import niprepsdev_path

has_fsl = fsl.Info.version() is not None
has_freesurfer = fs.Info.version() is not None
has_afni = afni.Info.version() is not None

test_output_dir = os.getenv("TEST_OUTPUT_DIR")
test_workdir = os.getenv("TEST_WORK_DIR")

data_dir = Path(niprepsdev_path) / "BIDS-examples-1-enh-ds054"


def create_canary(predicate, message):
    def canary():
        if predicate:
            pytest.skip(message)

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            canary()
            return f(*args, **kwargs)

        return wrapper

    return canary, decorator


data_env_canary, needs_data_env = create_canary(
    not Path(niprepsdev_path).is_dir(),
    "Test data must be made available in ~/.cache/stanford-crn or in a "
    "directory referenced by the TEST_DATA_HOME environment variable.",
)

data_dir_canary, needs_data_dir = create_canary(
    not Path(niprepsdev_path).is_dir(),
    "Test data must be made available in ~/.cache/stanford-crn or in a "
    "directory referenced by the TEST_DATA_HOME environment variable.",
)


def _run_interface_mock(objekt, runtime):
    runtime.returncode = 0
    runtime.endTime = dt.isoformat(dt.utcnow())

    objekt._out_report = str(Path(objekt.inputs.out_report).absolute())
    objekt._post_run_hook(runtime)
    objekt._generate_report()
    return runtime


def _create_dtseries_cifti(timepoints, models):
    """Create a dense timeseries CIFTI-2 file"""
    import nibabel.cifti2 as ci

    def create_series_map():
        return ci.Cifti2MatrixIndicesMap(
            (0,),
            "CIFTI_INDEX_TYPE_SERIES",
            number_of_series_points=timepoints,
            series_exponent=0,
            series_start=0,
            series_step=1,
            series_unit="SECOND",
        )

    def create_geometry_map():
        index_offset = 0
        brain_models = []
        timeseries = np.zeros((timepoints, 0))

        for name, data in models:
            if "CORTEX" in name:
                model_type = "CIFTI_MODEL_TYPE_SURFACE"
                attr = "vertex_indices"
                indices = ci.Cifti2VertexIndices(np.arange(len(data)))
            else:
                model_type = "CIFTI_MODEL_TYPE_VOXELS"
                attr = "voxel_indices_ijk"
                indices = ci.Cifti2VoxelIndicesIJK(np.arange(len(data)))
            bm = ci.Cifti2BrainModel(
                index_offset=index_offset,
                index_count=len(data),
                model_type=model_type,
                brain_structure=name,
            )
            setattr(bm, attr, indices)
            if model_type == "CIFTI_MODEL_TYPE_SURFACE":
                # define total vertices for surface models
                bm.surface_number_of_vertices = 32492
            index_offset += len(data)
            brain_models.append(bm)
            timeseries = np.column_stack((timeseries, data.T))

        brain_models.append(
            ci.Cifti2Volume(
                (4, 4, 4),
                ci.Cifti2TransformationMatrixVoxelIndicesIJKtoXYZ(-3, np.eye(4)),
            )
        )

        return (
            ci.Cifti2MatrixIndicesMap(
                (1,),
                "CIFTI_INDEX_TYPE_BRAIN_MODELS",
                maps=brain_models,
            ),
            timeseries,
        )

    matrix = ci.Cifti2Matrix()
    series_map = create_series_map()
    geometry_map, ts = create_geometry_map()
    matrix.append(series_map)
    matrix.append(geometry_map)
    hdr = ci.Cifti2Header(matrix)
    img = ci.Cifti2Image(dataobj=ts, header=hdr)
    img.nifti_header.set_intent("NIFTI_INTENT_CONNECTIVITY_DENSE_SERIES")

    out_file = Path("test.dtseries.nii").absolute()
    ci.save(img, out_file)
    return out_file
