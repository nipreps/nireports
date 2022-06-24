"""Miscellaneous utilities."""
from pathlib import Path

from nipype.utils.filemanip import loadcrash

SVG_SNIPPET = [
    """\
<object class="svg-reportlet" type="image/svg+xml" data="./{0}">
Problem loading figure {0}. If the link below works, please try \
reloading the report in your browser.</object>
</div>
<div class="elem-filename">
    Get figure file: <a href="./{0}" target="_blank">{0}</a>
</div>
""",
    """\
<img class="svg-reportlet" src="./{0}" style="width: 100%" />
</div>
<div class="elem-filename">
    Get figure file: <a href="./{0}" target="_blank">{0}</a>
</div>
""",
]


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
