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
# STATEMENT OF CHANGES: This file was ported carrying over full git history from niworkflows,
# another NiPreps project licensed under the Apache-2.0 terms, and has been changed since.
"""Miscellaneous utilities."""

from collections import defaultdict
from pathlib import Path

from bids.utils import listify
from nipype.utils.filemanip import loadcrash


def read_crashfile(path, root=None, root_replace="<workdir>"):
    """
    Prepare crashfiles for rendering into a report.

    Parameters
    ----------
    path : :obj:`str` or :obj:`~pathlib.Path`
        The path where the crash-file is located.
    root : :obj:`str` or :obj:`~pathlib.Path`
        The root folder. If provided, the path will be replaced with ``root_replace``.
    root_replace : :obj:`str`
        A replacement for the absolute root path.

    Examples
    --------
    .. testsetup::

       >>> new_path = Path(__file__).resolve().parent
       >>> test_data_path = new_path / 'data' / 'tests'

    >>> info = read_crashfile(test_data_path / 'crashfile.txt')
    >>> info['node']  # doctest: +ELLIPSIS
    '...func_preproc_task_machinegame_run_02_wf.carpetplot_wf.conf_plot'

    >>> info['traceback']  # doctest: +ELLIPSIS
    '...ValueError: zero-size array to reduction operation minimum which has no identity'

    >>> info['file']
    '...nireports/assembler/data/tests/crashfile.txt'

    >>> read_crashfile(
    ...     test_data_path / 'crashfile.txt',
    ...     root=test_data_path,
    ...     root_replace="<outdir>",
    ... )['file']
    '<outdir>/crashfile.txt'

    """
    errordata = _read_pkl(path) if str(path).endswith(".pklz") else _read_txt(path)
    if root:
        errordata["file"] = f"{root_replace}/{Path(errordata['file']).relative_to(root)}"
    return errordata


def _read_pkl(path):
    crash_data = loadcrash(path)
    data = {"file": path, "traceback": "".join(crash_data["traceback"])}
    if "node" in crash_data:
        data["node"] = crash_data["node"]
        if data["node"].base_dir:
            data["node_dir"] = data["node"].output_dir()
        else:
            data["node_dir"] = "Node crashed before execution"
        data["inputs"] = sorted(data["node"].inputs.trait_get().items())
    return data


def _read_txt(path):
    """
    Read a text crashfile.

    Examples
    --------
    >>> new_path = Path(__file__).resolve().parent
    >>> test_data_path = new_path / 'data' / 'tests'
    >>> info = _read_txt(test_data_path / 'crashfile.txt')
    >>> info['node']  # doctest: +ELLIPSIS
    '...func_preproc_task_machinegame_run_02_wf.carpetplot_wf.conf_plot'

    >>> info['traceback']  # doctest: +ELLIPSIS
    '...ValueError: zero-size array to reduction operation minimum which has no identity'

    """

    lines = Path(path).read_text().splitlines()
    data = {"file": str(path)}
    traceback_start = 0
    if lines[0].startswith("Node"):
        data["node"] = lines[0].split(": ", 1)[1].strip()
        data["node_dir"] = lines[1].split(": ", 1)[1].strip()
        inputs = []
        cur_key = ""
        cur_val = ""
        for i, line in enumerate(lines[5:]):
            if not line.strip():
                continue

            if line[0].isspace():
                cur_val += line
                continue

            if cur_val:
                inputs.append((cur_key, cur_val.strip()))

            if line.startswith("Traceback ("):
                traceback_start = i + 5
                break

            cur_key, cur_val = tuple(line.split(" = ", 1))

        data["inputs"] = sorted(inputs)
    else:
        data["node_dir"] = "Node crashed before execution"
    data["traceback"] = "\n".join(lines[traceback_start:]).strip()
    return data


def dict2html(indict, table_id):
    """Convert a dictionary into an HTML table."""
    rows = sorted(unfold_columns(indict))
    if not rows:
        return None

    width = max([len(row) for row in rows])

    result_str = '<table id="%s" class="table table-sm table-striped">\n' % table_id
    td = "<td{1}>{0}</td>".format
    for row in rows:
        result_str += "<tr>"
        ncols = len(row)
        for i, col in enumerate(row):
            colspan = 0
            colstring = ""
            if (width - ncols) > 0 and i == ncols - 2:
                colspan = (width - ncols) + 1
                colstring = " colspan=%d" % colspan
            result_str += td(col, colstring)
        result_str += "</tr>\n"
    result_str += "</table>\n"
    return result_str


def unfold_columns(indict, prefix=None, delimiter="_"):
    """
    Convert an input dict with flattened keys to an list of rows expanding columns.

    Parameters
    ----------
    indict : :obj:`dict`
        Input dictionary to be expanded as a list of lists.
    prefix : :obj:`str` or :obj:`list`
        A string or list of strings to expand columns on the left
        (that is, all *rows* will be added these prefixes to the
        left side).
    delimiter : :obj:`str`
        The delimiter string.

    Examples
    --------
    >>> unfold_columns({
    ...     "key1": "val1",
    ...     "nested_key1": "nested value",
    ...     "nested_key2": "another value",
    ... })
    [['key1', 'val1'],
    ['nested', 'key1', 'nested value'],
    ['nested', 'key2', 'another value']]

    If nested keys do not share prefixes, they should not be unfolded.

    >>> unfold_columns({
    ...     "key1": "val1",
    ...     "key1_split": "nonnested value",
    ...     "key2_singleton": "another value",
    ... })
    [['key1', 'val1'],
    ['key1_split', 'nonnested value'],
    ['key2_singleton', 'another value']]

    Nested/non-nested keys can be combined (see the behavior for key1):

    >>> unfold_columns({
    ...     "key1": "val1",
    ...     "key1_split1": "nested value",
    ...     "key1_split2": "nested value",
    ...     "key2_singleton": "another value",
    ... })
    [['key1', 'val1'],
    ['key1', 'split1', 'nested value'],
    ['key1', 'split2', 'nested value'],
    ['key2_singleton', 'another value']]

    >>> unfold_columns({
    ...     "key1": "val1",
    ...     "nested_key1": "nested value",
    ...     "nested_key2": "another value",
    ... }, prefix="prefix")
    [['prefix', 'key1', 'val1'],
    ['prefix', 'nested', 'key1', 'nested value'],
    ['prefix', 'nested', 'key2', 'another value']]

    >>> unfold_columns({
    ...     "key1": "val1",
    ...     "nested_key1": "nested value",
    ...     "nested_key2": "another value",
    ... }, prefix=["name", "lastname"])
    [['name', 'lastname', 'key1', 'val1'],
    ['name', 'lastname', 'nested', 'key1', 'nested value'],
    ['name', 'lastname', 'nested', 'key2', 'another value']]

    >>> unfold_columns({
    ...     "key1": "val1",
    ...     "nested_key1_sub1": "val2",
    ...     "nested_key1_sub2": "val3",
    ...     "nested_key2": "another value",
    ... })
    [['key1', 'val1'], ['nested', 'key2', 'another value'],
    ['nested', 'key1', 'sub1', 'val2'],
    ['nested', 'key1', 'sub2', 'val3']]

    """
    prefix = listify(prefix) if prefix is not None else []
    keys = sorted(set(indict.keys()))

    data = []
    subdict = defaultdict(dict, {})
    for key in keys:
        col = key.split(delimiter, 1)
        if len(col) == 1:
            data.append(prefix + [col[0], indict[col[0]]])
        else:
            subdict[col[0]][col[1]] = indict[key]

    if subdict:
        for skey in sorted(subdict.keys()):
            sskeys = list(subdict[skey].keys())

            # If there is only one subkey, merge back
            if len(sskeys) == 1:
                value = subdict[skey][sskeys[0]]
                newkey = delimiter.join([skey] + sskeys)
                data.append(prefix + [newkey, value])
            else:
                data += unfold_columns(subdict[skey], prefix=prefix + [skey])

    return data
