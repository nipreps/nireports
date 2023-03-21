"""Miscellaneous utilities."""
from pathlib import Path
from bids.utils import listify

from nipype.utils.filemanip import loadcrash


def read_crashfile(path, root=None):
    errordata = _read_pkl(path) if path.endswith(".pklz") else _read_txt(path)
    if root:
        errordata["file"] = f"&lt;workdir&gt;/{Path(errordata['file']).relative_to(root)}"
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
    """Read a txt crashfile

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
    """Converts a dictionary into an HTML table"""
    columns = sorted(unfold_columns(indict))
    if not columns:
        return None

    depth = max([len(col) for col in columns])

    result_str = '<table id="%s" class="table table-sm table-striped">\n' % table_id
    td = "<td{1}>{0}</td>".format
    for line in columns:
        result_str += "<tr>"
        ncols = len(line)
        for i, col in enumerate(line):
            colspan = 0
            colstring = ""
            if (depth - ncols) > 0 and i == ncols - 2:
                colspan = (depth - ncols) + 1
                colstring = " colspan=%d" % colspan
            result_str += td(col, colstring)
        result_str += "</tr>\n"
    result_str += "</table>\n"
    return result_str


def unfold_columns(indict, prefix=None, delimiter="_"):
    """
    Convert an input dict with flattened keys to an array of columns.

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
    [['key1', 'val1'], ['nested', 'key1', 'nested value'], ['nested', 'key2', 'another value']]

    If nested keys do not share prefixes, they should not be unfolded.
    >>> unfold_columns({
    ...     "key1": "val1",
    ...     "key1_split": "nonnested value",
    ...     "key2_singleton": "another value",
    ... })
    [['key1', 'val1'], ['key1_split', 'nonnested value'], ['key2_singleton', 'another value']]

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
    keys = sorted(set(list(indict.keys())))

    data = []
    subdict = {}
    for key in keys:
        col = key.split(delimiter, 1)
        if len(col) == 1:
            value = indict[col[0]]
            data.append(prefix + [col[0], value])
        else:
            if subdict.get(col[0]) is None:
                subdict[col[0]] = {}
            subdict[col[0]][col[1]] = indict[key]

    if subdict:
        for skey in sorted(list(subdict.keys())):
            sskeys = list(subdict[skey].keys())
            if len(sskeys) == 1:
                value = subdict[skey][sskeys[0]]
                newkey = delimiter.join([skey] + sskeys)
                data.append(prefix + [newkey, value])
            else:
                data += unfold_columns(subdict[skey], prefix=prefix + [skey])

    return data
