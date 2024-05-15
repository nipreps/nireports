# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2023 The Axon Lab <theaxonlab@gmail.com>
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
"""Python module for loading and saving fMRI related data"""

import os
import re
import os.path as op
import pandas as pd
from collections import defaultdict
import logging
from typing import Optional, Union

from bids import BIDSLayout
import numpy as np

from pandas import read_csv
from nibabel import loadsave
from bids.layout import parse_file_entities
from bids.layout.writing import build_path
from nilearn.datasets import fetch_atlas_difumo
from nilearn.interfaces.fmriprep.load_confounds import _load_single_confounds_file

FC_PATTERN: list = [
    "sub-{subject}[/ses-{session}]/func/sub-{subject}"
    "[_ses-{session}][_task-{task}][_meas-{meas}]"
    "_{suffix}{extension}"
]
FC_FILLS: dict = {"suffix": "connectivity", "extension": ".tsv"}

TIMESERIES_PATTERN: list = [
    "sub-{subject}[/ses-{session}]/func/sub-{subject}"
    "[_ses-{session}][_task-{task}][_desc-{desc}]"
    "_{suffix}{extension}"
]
TIMESERIES_FILLS: dict = {"desc": "denoised", "extension": ".tsv"}

CONFOUND_PATTERN: list = [
    "sub-{subject}[_ses-{session}][_task-{task}][_part-{part}][_desc-{desc}]"
    "_{suffix}{extension}"
]
CONFOUND_FILLS: dict = {"desc": "confounds", "suffix": "timeseries", "extension": "tsv"}


def separate_by_similar_values(
    input_list: list, external_value: Optional[Union[list, np.ndarray]] = None
) -> dict:
    """This returns elements of `input_list` with similar values (optionally set by
    `external_value`) separated into sub-lists.

    Parameters
    ----------
    input_list : list
        List to be separated.
    external_value : Optional[list], optional
        Values corresponding to the elements of `input_list`, by default None

    Returns
    -------
    dict
        Dictionary where each entry is a list of elements that have similar values and
        the keys are the value for each list.
    """
    if external_value is None:
        external_value = input_list

    data_by_value = defaultdict(list)

    for val, data in zip(external_value, input_list):
        data_by_value[val].append(data)
    return data_by_value


def get_func_filenames_bids(
    paths_to_func_dir: str,
    task_filter: Optional[list] = None,
    ses_filter: Optional[list] = None,
    run_filter: Optional[list] = None,
) -> tuple[list[list[str]], list[float]]:
    """Return the BIDS functional imaging files matching the specified task and session
    filters as well as the first (if multiple) unique repetition time (TR).

    Parameters
    ----------
    paths_to_func_dir : str
        Path to the BIDS (usually derivatives) directory
    task_filter : list, optional
        List of task name(s) to consider, by default `None`
    ses_filter : list, optional
        List of session name(s) to consider, by default `None`
    run_filter : list, optional
        List of run(s) to consider, by default `None`

    Returns
    -------
    tuple[list[list[str]], list[float]]
        Returns two lists with: a list of sorted filenames and a list of TRs.
    """
    logging.debug("Using BIDS to find functional files...")

    layout = BIDSLayout(
        paths_to_func_dir,
        validate=False,
    )

    all_derivatives = layout.get(
        scope="all",
        return_type="file",
        extension=["nii.gz", "gz"],
        suffix="bold",
        task=task_filter or [],
        session=ses_filter or [],
        run=run_filter or [],
    )

    if not all_derivatives:
        raise ValueError(
            f"No functional derivatives were found under {paths_to_func_dir} with the following filters:"
            f"\nExtension: ['nii.gz', 'gz']"
            f"\nSuffix: bold"
            f"\nTask: {task_filter or []}"
            f"\nSession: {ses_filter or []}"
            f"\nRun: {run_filter or []}"
        )

    affines = []
    for file in all_derivatives:
        affines.append(loadsave.load(file).affine)

    similar_fov_dict = separate_by_similar_values(
        all_derivatives, np.array(affines)[:, 0, 0]
    )
    if len(similar_fov_dict) > 1:
        logging.warning(
            f"{len(similar_fov_dict)} different FoV found ! "
            "Files with similar FoV will be computed together. "
            "Computation time may increase."
        )

    separated_files = []
    separated_trs = []
    for file_group in similar_fov_dict.values():
        t_rs = []
        for file in file_group:
            t_rs.append(layout.get_metadata(file)["RepetitionTime"])

        similar_tr_dict = separate_by_similar_values(file_group, t_rs)
        separated_files += list(similar_tr_dict.values())
        separated_trs += list(similar_tr_dict.keys())

        if len(similar_tr_dict) > 1:
            logging.warning(
                "Multiple TR values found ! "
                "Files with similar TR will be computed together. "
                "Computation time may increase."
            )

    return separated_files, separated_trs


def get_bids_savename(filename: str, patterns: list, **kwargs) -> str:
    """Return the BIDS filename following the specified patterns and modifying the
    entities from the keywords arguments.

    Parameters
    ----------
    filename : str
        Name of the original BIDS file
    patterns : list, optional
        Patterns for the output file, by default FC_PATTERN

    Returns
    -------
    str
        BIDS output filename.
    """
    entity = parse_file_entities(filename)

    for key, value in kwargs.items():
        entity[key] = value

    bids_savename = build_path(entity, patterns)

    return str(bids_savename)


def get_atlas_data(atlas_name: str = "DiFuMo", **kwargs) -> dict:
    """Fetch the specifies atlas filename and data.

    Parameters
    ----------
    atlas_name : str, optional
        Name of the atlas to fetch, by default "DiFuMo"

    Returns
    -------
    dict
        Dictionary with keys "maps" (filename) and "labels" (ROI labels).
    """
    logging.info("Fetching the DiFuMo atlas ...")

    if kwargs["dimension"] not in [64, 128, 512]:
        logging.warning(
            "Dimension for DiFuMo atlas is different from 64, 128 or 512 ! Are you"
            "certain you want to deviate from those optimized modes? "
        )

    return fetch_atlas_difumo(legacy_format=False, **kwargs)


def find_atlas_dimension(path: str, atlas_name: str = "DiFuMo") -> int:
    """Fetch the atlas dimension from the path where the functional connectivity are saved.
    Parameters
    ----------
    path : str
        Path to the directory where functional connectivity are saved.
    atlas_name : str, optional
        Name of the atlas to fetch, by default "DiFuMo"

    Returns
    -------
    int
        Atlas dimension.
    """

    # Using regular expression to extract the number of dimensions
    dimension_match = re.search(rf"{atlas_name}(\d+)", path)

    if dimension_match:
        return int(dimension_match.group(1))
    else:
        raise ValueError(
            f"The output path {path} does not contain the expected pattern: {atlas_name} followed by digits."
        )


def find_derivative(path: str, derivatives_name: str = "derivatives") -> str:
    """Find the corresponding BIDS derivative folder (if existing, otherwise it will be
    created).

    Parameters
    ----------
    path : str
        Path to the BIDS (usually derivatives) dataset.
    derivatives_name : str, optional
        Name of the derivatives folder, by default "derivatives"

    Returns
    -------
    str
        Absolute path to the derivative folder.
    """
    splitted_path = path.split("/")
    try:
        while derivatives_name not in splitted_path[-1]:
            splitted_path.pop()
    except IndexError:
        logging.warning(
            f'"{derivatives_name}" could not be found on path - '
            f'creating at: {op.join(path, derivatives_name)}"'
        )
        return op.join(path, derivatives_name)

    return "/".join(splitted_path)


def find_mriqc(path: str) -> str:
    """Find the path to the MRIQC folder (if existing, otherwise it will be
    created).

    Parameters
    ----------
    path : str
        Path to the BIDS (usually derivatives) dataset.

    Returns
    -------
    str
        Absolute path to the mriqc folder.
    """
    logging.debug("Searching for MRIQC path...")
    derivative_path = find_derivative(path)

    folders = [
        f for f in os.listdir(derivative_path) if op.isdir(op.join(derivative_path, f))
    ]

    mriqc_path = [f for f in folders if "mriqc" in f]
    if len(mriqc_path) >= 2:
        logging.warning(
            f"More than one mriqc derivative folder was found: {mriqc_path}"
            f"The first instance {mriqc_path[0]} is used for the computation."
            "In case you want to use another mriqc derivative folder, use the --mriqc-path flag"
        )
    return op.join(derivative_path, mriqc_path[0])


def reorder_iqms(iqms_df: pd.DataFrame, fc_paths: list[str]):
    """Reorder the IQMs according to the list of filenames

    Parameters
    ----------
    iqms_df : pd.Dataframe
        Dataframe containing the IQMs value for each image
    fc_paths : list [str]
        List of paths to the functional connectivity matrices

    Returns
    -------
    panda.df
        Dataframe containing the IQMs dataframe with reordered rows.
    """
    iqms_df[["subject", "session", "task"]] = iqms_df["bids_name"].str.extract(
        r"sub-(\d+)_ses-(\d+)_task-(\w+)_"
    )
    entities_list = [parse_file_entities(filepath) for filepath in fc_paths]
    entities_df = pd.DataFrame(entities_list)

    return pd.merge(
        entities_df, iqms_df, on=["subject", "session", "task"], how="inner"
    )


def load_iqms(
    derivative_path: str,
    fc_paths: list[str],
    mriqc_path: str = None,
    mod="bold",
    iqms_name: list = ["fd_mean", "fd_num", "fd_perc"],
) -> str:
    """Load the IQMs and match their order with the corresponding functional matrix.

    Parameters
    ----------
    derivative_path : str
        Path to the BIDS dataset's derivatives.
    fc_paths : list [str]
        List of paths to the functional connectivity matrices
    mriqc_path : str, optional
        Name of the MRIQC derivative folder, by default None
    mod : str, optional
        Load the IQMs of that modality
    iqms_name : list, optional
        Name of the IQMs to find, by default ["fd_mean", "fd_num", "fd_perc"]

    Returns
    -------
    panda.df
        Dataframe containing the IQMs loaded from the derivatives folder.
    """
    # Find the MRIQC folder
    if mriqc_path is None:
        mriqc_path = find_mriqc(derivative_path)

    # Load the IQMs from the group tsv
    iqms_filename = op.join(mriqc_path, f"group_{mod}.tsv")
    iqms_df = read_csv(iqms_filename, sep="\t")
    # If multi-echo dataset and the IQMs of interest are motion-related, keep only the IQMs from the second echo
    if "echo" in iqms_df["bids_name"][0] and all("fd" in i for i in iqms_name):
        iqms_df = iqms_df[iqms_df["bids_name"].str.contains("echo-2")]
        logging.info(
            "In the case of a multi-echo dataset, the IQMs of the second echo are considered."
        )

    # Match the order of the rows in iqms_df with the corresponding FC
    iqms_df = reorder_iqms(iqms_df, fc_paths)

    # Keep only the IQMs of interest
    iqms_df = iqms_df[iqms_name]

    return iqms_df


def check_existing_output(
    output: str,
    func_filename: list[str],
    return_existing: bool = False,
    return_output: bool = False,
    **kwargs,
) -> tuple[list[str], list[str]]:
    """Check for existing output.

    Parameters
    ----------
    output : str
        Path to the output directory
    func_filename : list[str]
        Input files to be processed
    return_existing : bool, optional
        Condition to return the list of input corresponding to existing outputs, by default
        False
    return_output: bool, optional
        Condition to return the path of existing outputs, by default False

    Returns
    -------
    tuple[list[str], list[str]]
        List of missing data path (optionally, a second list of existing data path)
    """
    if return_output == True and return_existing == False:
        raise ValueError(
            "Setting return_output=True in check_existing_output requires return_existing=True."
        )

    missing_data_filter = [
        not op.exists(op.join(output, get_bids_savename(filename, **kwargs)))
        for filename in func_filename
    ]

    missing_data = np.array(func_filename)[missing_data_filter]
    logging.debug(
        f"\t{sum(missing_data_filter)} missing data found for files:"
        "\n\t" + "\n\t".join(missing_data)
    )

    if return_existing:
        if return_output:
            existing_output = [
                op.join(output, get_bids_savename(filename, **kwargs))
                for filename in func_filename
                if op.exists(op.join(output, get_bids_savename(filename, **kwargs)))
            ]
            return existing_output
        else:
            existing_data = np.array(func_filename)[
                [not fltr for fltr in missing_data_filter]
            ]
            return missing_data.tolist(), existing_data.tolist()

    return missing_data.tolist()
