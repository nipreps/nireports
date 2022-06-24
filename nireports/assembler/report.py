"""
Definition of the :class:`Report` class.
"""
import re
from collections import defaultdict
from itertools import compress
from pathlib import Path

import jinja2
from bids.layout import BIDSLayout, add_config_paths
from pkg_resources import resource_filename as pkgrf

from nireports.assembler.misc import Element, read_crashfile
from nireports.assembler.reportlet import Reportlet

# Add a new figures spec
try:
    add_config_paths(figures=pkgrf("nireports", "assembler/data/nipreps.json"))
except ValueError as e:
    if "Configuration 'figures' already exists" != str(e):
        raise

PLURAL_SUFFIX = defaultdict(str("s").format, [("echo", "es")])


class SubReport(Element):
    """SubReports are sections within a Report."""

    def __init__(self, name, isnested=False, reportlets=None, title=""):
        self.name = name
        self.title = title
        self.reportlets = reportlets or []
        self.isnested = isnested


class Report:
    """
    The full report object. This object maintains a BIDSLayout to index
    all reportlets.


    .. testsetup::

    >>> cwd = os.getcwd()
    >>> os.chdir(tmpdir)

    >>> from pkg_resources import resource_filename
    >>> from shutil import copytree
    >>> from bids.layout import BIDSLayout
    >>> test_data_path = resource_filename('nireports', 'assembler/data/tests/work')
    >>> testdir = Path(tmpdir)
    >>> data_dir = copytree(test_data_path, str(testdir / 'work'))
    >>> out_figs = testdir / 'out' / 'fmriprep'

    .. doctest::

    >>> robj = Report(testdir / 'out', 'madeoutuuid', subject_id='01', packagename='fmriprep',
    ...               reportlets_dir=testdir / 'work' / 'reportlets')
    >>> robj.layout.get(subject='01', desc='reconall')[0]._path.as_posix()  # doctest: +ELLIPSIS
    '.../figures/sub-01_desc-reconall_T1w.svg'

    >>> robj.generate_report()
    0
    >>> len((testdir / 'out' / 'fmriprep' / 'sub-01.html').read_text())
    36693

    .. testcleanup::

    >>> os.chdir(cwd)

    """

    def __init__(
        self,
        out_dir,
        run_uuid,
        config=None,
        out_filename="report.html",
        packagename=None,
        reportlets_dir=None,
        subject_id=None,
    ):
        self.root = Path(reportlets_dir or out_dir)

        # Initialize structuring elements
        self.sections = []
        self.errors = []
        self.out_dir = Path(out_dir)
        self.out_filename = out_filename
        self.run_uuid = run_uuid
        self.packagename = packagename
        self.subject_id = subject_id
        if subject_id is not None:
            self.subject_id = subject_id[4:] if subject_id.startswith("sub-") else subject_id
            self.out_filename = f"sub-{self.subject_id}.html"

        # Default template from nireports
        self.template_path = Path(pkgrf("nireports", "report.tpl"))
        self._load_config(Path(config or pkgrf("nireports", "default.yml")))
        assert self.template_path.exists()

    def _load_config(self, config):
        from yaml import safe_load as load

        settings = load(config.read_text())
        self.packagename = self.packagename or settings.get("package", None)

        if self.packagename is not None:
            self.root = self.root / self.packagename
            self.out_dir = self.out_dir / self.packagename

        if self.subject_id is not None:
            self.root = self.root / "sub-{}".format(self.subject_id)

        if "template_path" in settings:
            self.template_path = config.parent / settings["template_path"]

        self.index(settings["sections"])

    def init_layout(self):
        self.layout = BIDSLayout(self.root, config="figures", validate=False)

    def index(self, config):
        """
        Traverse the reports config definition and instantiate reportlets.

        This method also places figures in their final location.
        """
        # Initialize a BIDS layout
        self.init_layout()
        for subrep_cfg in config:
            # First determine whether we need to split by some ordering
            # (ie. sessions / tasks / runs), which are separated by commas.
            orderings = [s for s in subrep_cfg.get("ordering", "").strip().split(",") if s]
            entities, list_combos = self._process_orderings(orderings, self.layout)

            if not list_combos:  # E.g. this is an anatomical reportlet
                reportlets = [
                    Reportlet(self.layout, self.out_dir, config=cfg)
                    for cfg in subrep_cfg["reportlets"]
                ]
            else:
                # Do not use dictionary for queries, as we need to preserve ordering
                # of ordering columns.
                reportlets = []
                for c in list_combos:
                    # do not display entities with the value None.
                    c_filt = list(filter(None, c))
                    ent_filt = list(compress(entities, c))
                    # Set a common title for this particular combination c
                    title = "Reports for: %s." % ", ".join(
                        [
                            '%s <span class="bids-entity">%s</span>' % (ent_filt[i], c_filt[i])
                            for i in range(len(c_filt))
                        ]
                    )
                    for cfg in subrep_cfg["reportlets"]:
                        cfg["bids"].update({entities[i]: c[i] for i in range(len(c))})
                        rlet = Reportlet(self.layout, self.out_dir, config=cfg)
                        if not rlet.is_empty():
                            rlet.title = title
                            title = None
                            reportlets.append(rlet)

            # Filter out empty reportlets
            reportlets = [r for r in reportlets if not r.is_empty()]
            if reportlets:
                sub_report = SubReport(
                    subrep_cfg["name"],
                    isnested=bool(list_combos),
                    reportlets=reportlets,
                    title=subrep_cfg.get("title"),
                )
                self.sections.append(sub_report)

        # Populate errors section
        error_dir = self.out_dir / "sub-{}".format(self.subject_id) / "log" / self.run_uuid
        if error_dir.is_dir():
            self.errors = [read_crashfile(str(f)) for f in error_dir.glob("crash*.*")]

    def generate_report(self):
        """Once the Report has been indexed, the final HTML can be generated"""
        logs_path = self.out_dir / "logs"

        boilerplate = []
        boiler_idx = 0

        if (logs_path / "CITATION.html").exists():
            text = (
                re.compile("<body>(.*?)</body>", re.DOTALL | re.IGNORECASE)
                .findall((logs_path / "CITATION.html").read_text())[0]
                .strip()
            )
            boilerplate.append((boiler_idx, "HTML", f'<div class="boiler-html">{text}</div>'))
            boiler_idx += 1

        if (logs_path / "CITATION.md").exists():
            text = (logs_path / "CITATION.md").read_text()
            boilerplate.append((boiler_idx, "Markdown", f"<pre>{text}</pre>\n"))
            boiler_idx += 1

        if (logs_path / "CITATION.tex").exists():
            text = (
                re.compile(
                    r"\\begin{document}(.*?)\\end{document}",
                    re.DOTALL | re.IGNORECASE,
                )
                .findall((logs_path / "CITATION.tex").read_text())[0]
                .strip()
            )
            boilerplate.append(
                (
                    boiler_idx,
                    "LaTeX",
                    f"""<pre>{text}</pre>
<h3>Bibliography</h3>
<pre>{Path(pkgrf(self.packagename, 'data/boilerplate.bib')).read_text()}</pre>
""",
                )
            )
            boiler_idx += 1

        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(searchpath=str(self.template_path.parent)),
            trim_blocks=True,
            lstrip_blocks=True,
            autoescape=False,
        )
        report_tpl = env.get_template(self.template_path.name)
        report_render = report_tpl.render(
            sections=self.sections, errors=self.errors, boilerplate=boilerplate
        )

        # Write out report
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / self.out_filename).write_text(report_render, encoding="UTF-8")
        return len(self.errors)

    @staticmethod
    def _process_orderings(orderings, layout):
        """
        Generate relevant combinations of orderings with observed values.

        Arguments
        ---------
        orderings : :obj:`list` of :obj:`list` of :obj:`str`
            Sections prescribing an ordering to select across sessions, acquisitions, runs, etc.
        layout : :obj:`bids.layout.BIDSLayout`
            The BIDS layout

        Returns
        -------
        entities: :obj:`list` of :obj:`str`
            The relevant orderings that had unique values
        value_combos: :obj:`list` of :obj:`tuple`
            Unique value combinations for the entities

        """
        # get a set of all unique entity combinations
        all_value_combos = {
            tuple(bids_file.get_entities().get(k, None) for k in orderings)
            for bids_file in layout.get()
        }
        # remove the all None member if it exists
        none_member = tuple([None for k in orderings])
        if none_member in all_value_combos:
            all_value_combos.remove(tuple([None for k in orderings]))
        # see what values exist for each entity
        unique_values = [
            {value[idx] for value in all_value_combos} for idx in range(len(orderings))
        ]
        # if all values are None for an entity, we do not want to keep that entity
        keep_idx = [
            False if (len(val_set) == 1 and None in val_set) or not val_set else True
            for val_set in unique_values
        ]
        # the "kept" entities
        entities = list(compress(orderings, keep_idx))
        # the "kept" value combinations
        value_combos = [tuple(compress(value_combo, keep_idx)) for value_combo in all_value_combos]
        # sort the value combinations alphabetically from the first entity to the last entity
        value_combos.sort(
            key=lambda entry: tuple(str(value) if value is not None else "0" for value in entry)
        )

        return entities, value_combos
