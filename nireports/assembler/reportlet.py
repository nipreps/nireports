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
"""The reporting visualization unit or *reportlet*."""

import re
from pathlib import Path
from uuid import uuid4

from nipype.utils.filemanip import copyfile

from nireports.assembler import data
from nireports.assembler.misc import dict2html, read_crashfile

IMG_SNIPPET = """\
<div class="reportlet">
<img class="{ext}-reportlet" src="./{name}" style="{style}" />
</div>
<small>Get figure file: <a href="./{name}" target="_blank">{name}</a></small>
"""

SVG_SNIPPET = """\
<div class="reportlet">
<object class="{ext}-reportlet" type="image/{ext}+xml" data="./{name}" style="{style}">
Problem loading figure {name}. If the link below works, please try \
reloading the report in your browser.</object>
</div>
<small>Get figure file: <a href="./{name}" target="_blank">{name}</a></small>
"""

METADATA_ACCORDION_BLOCK = """\
<div class="accordion accordion-flush" id="{metadata_id}">
"""


# aria-expanded="{metadata_folded}"
METADATA_ACCORDION_ITEM = """
  <div class="accordion-item">
    <h2 class="accordion-header" id="{metadata_id}-{metadata_index}">
      <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" \
data-bs-target="#{metadata_id}-{metadata_index}-collapse" \
aria-controls="{metadata_id}-{metadata_index}-collapse">
        {metadata_item_key}
      </button>
    </h2>
    <div id="{metadata_id}-{metadata_index}-collapse" class="accordion-collapse collapse" \
aria-labelledby="{metadata_id}-{metadata_index}-heading" \
data-bs-parent="#{metadata_id}-{metadata_index}">
      <div class="accordion-body metadata-table">
      {metadata_html}
      </div>
    </div>
  </div>
"""

ERROR_TEMPLATE = """
    <details>
        <summary>Node Name: {node}</summary><br>
        <div class="small">
            File: <code>{file}</code><br />
            Working Directory: <code>{node_dir}</code><br />
            Inputs: <br />
            <ul>
            {inputs}
            </ul>
            <pre>{traceback}</pre>
        </div>
    </details>
"""

BOILERPLATE_NAV_TEMPLATE = """
        <li class="nav-item" role="presentation">
        <button class="nav-link {active}" id="{anchor}-tab" data-bs-toggle="tab" \
data-bs-target="#{anchor}-tab-pane" type="button" role="tab" aria-controls="{anchor}-tab-pane"\
aria-selected="{selected}">{tab_title}</button>
        </li>
"""

BOILERPLATE_TXT_TEMPLATE = """
        <div class="tab-pane fade {active}" id="{anchor}-tab-pane" role="tabpanel" \
aria-labelledby="{anchor}-tab" tabindex="0">
            <div class="boiler-{anchor} p-3 m-4 bg-primary" \
style="--bs-bg-opacity: .04;{style}">{body}</div>
        </div>
"""

HTML_BOILER_STYLE = " font-family: 'Bitstream Charter', 'Georgia', Times;"


class Reportlet:
    """
    A visual report atom (*reportlet*).

    A reportlet has title, description and a list of components with either an
    HTML fragment or a path to an SVG file, and possibly a caption. This is a
    factory class to generate Reportlets reusing the layout from a ``Report``
    object.

    .. testsetup::

       >>> from shutil import copytree
       >>> from bids.layout import BIDSLayout, add_config_paths
       >>> from nireports.assembler import data
       >>> test_data_path = data.load('tests', 'work')
       >>> testdir = Path(tmpdir)
       >>> data_dir = copytree(test_data_path, str(testdir / 'work'))
       >>> out_figs = testdir / 'out' / 'fmriprep'
       >>> try:
       ...     add_config_paths(figures=data.load("nipreps.json"))
       ... except ValueError as e:
       ...     if "Configuration 'figures' already exists" != str(e):
       ...         raise
       >>> bl = BIDSLayout(str(testdir / 'work' / 'reportlets'),
       ...                 config='figures', validate=False)

    Examples
    --------
    >>> bl.get(subject='01', desc='reconall')[0]._path.as_posix() # doctest: +ELLIPSIS
    '.../nireports/sub-01/figures/sub-01_desc-reconall_T1w.svg'

    >>> len(bl.get(subject='01', space='.*', regex_search=True))
    2

    >>> r = Reportlet(bl, out_dir=out_figs, config={
    ...     'title': 'Some Title', 'bids': {'datatype': 'figures', 'desc': 'reconall'},
    ...     'description': 'Some description'})
    >>> r.name
    'datatype-figures_desc-reconall'

    >>> '<img ' in r.components[0][0]
    True

    >>> r = Reportlet(bl, out_dir=out_figs, config={
    ...     'title': 'Some Title', 'bids': {'datatype': 'figures', 'desc': 'reconall'},
    ...     'description': 'Some description', 'static': False})
    >>> r.name
    'datatype-figures_desc-reconall'

    >>> '<object ' in r.components[0][0]
    True

    >>> r = Reportlet(bl, out_dir=out_figs, config={
    ...     'title': 'Some Title', 'bids': {'datatype': 'figures', 'desc': 'summary'},
    ...     'description': 'Some description'})

    >>> '<h3 ' in r.components[0][0]
    True

    >>> r.components[0][1] is None
    True

    >>> r = Reportlet(bl, out_dir=out_figs, config={
    ...     'title': 'Some Title',
    ...     'bids': {'datatype': 'figures', 'space': '.*', 'regex_search': True},
    ...     'caption': 'Some description {space}'})
    >>> sorted(r.components)[0][1]
    'Some description MNI152NLin2009cAsym'

    >>> sorted(r.components)[1][1]
    'Some description MNI152NLin6Asym'

    >>> r = Reportlet(bl, out_dir=out_figs, config={
    ...     'title': 'Some Title',
    ...     'bids': {'datatype': 'fmap', 'space': '.*', 'regex_search': True},
    ...     'caption': 'Some description {space}'})
    >>> r.is_empty()
    True

    """

    __slots__ = {
        "components": "A list of visual elements for composite reportlets.",
        "description": "This reportlet's longer description.",
        "name": "A unique name for the reportlet (used to create HTML anchors).",
        "subtitle": "This reportlet's subtitle.",
        "title": "This reportlet's title.",
    }

    def __init__(self, layout, config=None, out_dir=None, bids_filters=None, metadata=None):
        if not config:
            raise RuntimeError("Reportlet must have a config object")

        if out_dir is None:
            raise RuntimeError("Reportlet must have an output directory")

        out_dir = Path(out_dir)
        self.title = config.get("title")
        self.subtitle = config.get("subtitle")
        self.description = config.get("description")
        self.components = []

        # Determine whether this is a "BIDS-type" reportlet (typically, an SVG file)
        if bidsquery := config.get("bids", {}):
            _bidsquery = (bids_filters or {}).copy()
            _bidsquery.update(bidsquery)
            bidsquery = _bidsquery
            self.name = config.get(
                "name",
                "_".join("%s-%s" % i for i in sorted(bidsquery.items())),
            )

            # Query the BIDS layout of reportlets
            files = layout.get(**bidsquery)

            for bidsfile in files:
                src = dst = Path(bidsfile.path)
                ext = "".join(src.suffixes)
                desc_text = config.get("caption")
                is_static = config.get("static", True)
                contents = None
                if ext == ".html":
                    contents = src.read_text().strip()
                elif ext == ".svg":
                    entities = dict(bidsfile.entities)
                    if desc_text:
                        desc_text = desc_text.format(**entities)

                    try:
                        html_anchor = src.relative_to(out_dir)
                    except ValueError:
                        html_anchor = src.relative_to(Path(layout.root))
                        dst = out_dir / html_anchor
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        copyfile(src, dst, copy=True, use_hardlink=True)

                    # Our current implementations of dynamic reportlets do this themselves,
                    # however I'll leave the code here since this is potentially something we
                    # will want to transfer from every figure generator to this location.
                    # The following code misses setting preserveAspecRatio="xMidYMid meet"
                    # if not is_static:
                    #     # set preserveAspectRatio

                    # Remove height and width attributes from initial <svg> tag
                    svglines = dst.read_text().splitlines()
                    expr = re.compile(r' (height|width)=["\'][0-9]+(\.[0-9]*)?[a-z]*["\']')
                    for ll, line in enumerate(svglines[:6]):
                        if line.strip().startswith("<svg"):
                            # It is critical that viewBox is correctly spelled out
                            fixedline = expr.sub("", line.replace("viewbox", "viewBox"))
                            dst.write_text("\n".join([fixedline] + svglines[ll + 1 :]))
                            break

                    style = {"width": "100%"} if is_static else {}
                    style.update(config.get("style", {}))

                    snippet = IMG_SNIPPET if is_static else SVG_SNIPPET
                    contents = snippet.format(
                        ext=ext[1:],
                        name=html_anchor,
                        style="; ".join(f"{k}: {v}" for k, v in style.items()),
                    )
                elif ext in (".png", ".jpg", ".jpeg"):
                    entities = dict(bidsfile.entities)
                    if desc_text:
                        desc_text = desc_text.format(**entities)

                    try:
                        html_anchor = src.relative_to(out_dir)
                    except ValueError:
                        html_anchor = src.relative_to(Path(layout.root))
                        dst = out_dir / html_anchor
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        copyfile(src, dst, copy=True, use_hardlink=True)

                    style = {"width": "100%"}
                    style.update(config.get("style", {}))

                    snippet = IMG_SNIPPET
                    contents = snippet.format(
                        ext=ext[1:],
                        name=html_anchor,
                        style="; ".join(f"{k}: {v}" for k, v in style.items()),
                    )
                else:
                    raise RuntimeError(f"Unsupported file extension: {ext}")

                if contents:
                    self.components.append((contents, desc_text))
        elif meta_reportlet := config.get("metadata", False):
            meta_settings = config.get("settings", {})
            meta_id = meta_settings.get("id", f"meta-{uuid4()}")
            self.name = f"meta-{meta_id}"

            if metadata is not None and not isinstance(meta_reportlet, dict):
                meta_reportlet = metadata.get(meta_id)

                if not meta_reportlet:
                    self.components.append(
                        (
                            '<p class="alert alert-success" role="alert">'
                            f'Could not find metadata for reportlet "{meta_id}"'
                            "</p>",
                            "",
                        )
                    )
                    return
            # meta_folded = meta_settings.get("folded", None)

            contents = [METADATA_ACCORDION_BLOCK.format(metadata_id=meta_id)]

            for ii, (group_name, values) in enumerate(meta_reportlet.items()):
                contents.append(
                    METADATA_ACCORDION_ITEM.format(
                        metadata_id=meta_id,
                        metadata_index=ii,
                        metadata_item_key=group_name,
                        metadata_html=dict2html(values, f"{meta_id}-table-{ii}"),
                    )
                )

            contents.append("</div>")
            self.components.append(("\n".join(contents), config.get("caption")))

        elif (custom := config.get("custom", None)) in ("boilerplate", "errors"):
            desc_text = config.get("caption")
            path = config.get("path", None)
            if custom == "errors":
                self.name = "errors"
                # Interpolate error log directory
                error_dir = Path(path)
                # Read in all crash files
                errors = [
                    read_crashfile(str(f), root=layout.root, root_replace="&lt;workdir&gt;")
                    for f in error_dir.glob("crash*.*")
                ]

                if not errors:
                    self.components.append(
                        (
                            '<p class="alert alert-success" role="alert">No errors to report!</p>',
                            desc_text,
                        )
                    )
                else:
                    contents = [
                        '<p class="alert alert-danger" role="alert">'
                        f"One or more execution steps failed ({len(errors)}). "
                        "Error details are attached below.</p>",
                    ]
                    for error in errors:
                        contents.append(
                            ERROR_TEMPLATE.format(
                                inputs="\n".join(
                                    [
                                        f"<li>{err_in[0]}: <code>{err_in[-1]}</code></li>"
                                        for err_in in error.pop("inputs", {})
                                    ]
                                ),
                                **error,
                            )
                        )
                    self.components.append(("\n".join(contents), desc_text))
            elif custom == "boilerplate":
                self.name = "boilerplate"
                logs_path = Path(path)
                bibfile = config.get("bibfile", ["nireports", "data/bibliography.bib"])

                boiler_tabs = ['<ul class="nav nav-tabs" id="myTab" role="tablist">']
                boiler_body = ['<div class="tab-content" id="myTabContent">']

                boiler_idx = 0
                for boiler_type in ("html", "md", "tex"):
                    if not (logs_path / f"CITATION.{boiler_type}").exists():
                        continue

                    text = ""
                    tab_title = ""
                    if boiler_type == "html":
                        text = (
                            re.compile("<body>(.*?)</body>", re.DOTALL | re.IGNORECASE)
                            .findall((logs_path / "CITATION.html").read_text())[0]
                            .strip()
                        )
                        tab_title = "HTML"
                    elif boiler_type == "md":
                        text = (logs_path / "CITATION.md").read_text()
                        text = f"<pre>{text}</pre>"
                        tab_title = "Markdown"
                    else:
                        text = (
                            re.compile(
                                r"\\begin{document}(.*?)\\end{document}",
                                re.DOTALL | re.IGNORECASE,
                            )
                            .findall((logs_path / "CITATION.tex").read_text())[0]
                            .strip()
                        )
                        text = f"""<pre>{text}</pre>
<h3>Bibliography</h3>
<pre>{data.Loader(bibfile[0]).readable(bibfile[1]).read_text()}</pre>
"""
                        tab_title = "LaTeX"

                    boiler_tabs.append(
                        BOILERPLATE_NAV_TEMPLATE.format(
                            active="active" if boiler_idx == 0 else "",
                            boiler_idx=boiler_idx,
                            selected="true" if boiler_idx == 0 else "false",
                            tab_title=tab_title,
                            anchor=boiler_type,
                        )
                    )

                    boiler_body.append(
                        BOILERPLATE_TXT_TEMPLATE.format(
                            active="show active" if boiler_idx == 0 else "",
                            boiler_idx=boiler_idx,
                            body=text,
                            anchor=boiler_type,
                            style="" if boiler_type != "html" else HTML_BOILER_STYLE,
                        )
                    )
                    boiler_idx += 1

                if boiler_idx == 0:
                    desc_text = None
                    self.components.append(
                        (
                            '<p class="alert alert-danger" role="alert">'
                            "Failed to generate the boilerplate</p>",
                            desc_text,
                        )
                    )
                else:
                    boiler_tabs.append("</ul>")
                    self.components.append(("\n".join(boiler_tabs + boiler_body), desc_text))

    def is_empty(self):
        """Determine whether the reportlet has no components."""
        return len(self.components) == 0
