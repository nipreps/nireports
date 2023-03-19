"""Miscellaneous utilities."""
from pathlib import Path

from nipype.utils.filemanip import loadcrash


class Element:
    """Just a basic component of a report"""

    def __init__(self, name, title=None):
        self.name = name
        self.title = title


def read_crashfile(path):
    if path.endswith(".pklz"):
        return _read_pkl(path)
    elif path.endswith(".txt"):
        return _read_txt(path)
    raise RuntimeError("unknown crashfile format")


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


def unfold_columns(indict, prefix=None):
    """Converts an input dict with flattened keys to an array of columns"""
    if prefix is None:
        prefix = []
    keys = sorted(set(list(indict.keys())))

    data = []
    subdict = {}
    for key in keys:
        col = key.split("_", 1)
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
                newkey = "_".join([skey] + sskeys)
                data.append(prefix + [newkey, value])
            else:
                data += unfold_columns(subdict[skey], prefix=prefix + [skey])

    return data
