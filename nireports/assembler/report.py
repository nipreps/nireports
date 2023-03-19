"""
Definition of the :class:`Report` class.
"""
import re
from collections import defaultdict
from itertools import compress
from pathlib import Path
from yaml import safe_load as load

import jinja2
from bids.layout import BIDSLayout, add_config_paths
from pkg_resources import resource_filename as pkgrf

from nireports.assembler.misc import Element
from nireports.assembler.reportlet import Reportlet

# Add a new figures spec
try:
    add_config_paths(figures=pkgrf("nireports.assembler", "data/nipreps.json"))
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

    >>> from pkg_resources import resource_filename
    >>> from shutil import copytree
    >>> from bids.layout import BIDSLayout
    >>> test_data_path = resource_filename('nireports', 'assembler/data/tests/work')
    >>> testdir = Path(tmpdir)
    >>> data_dir = copytree(test_data_path, str(testdir / 'work'))
    >>> out_figs = output_dir / 'nireports'

    .. doctest::

    >>> robj = Report(output_dir, 'madeoutuuid', subject_id='01', packagename='nireports',
    ...               reportlets_dir=testdir / 'work' / 'reportlets')

    >>> robj.generate_report()
    0
    >>> len((output_dir / 'nireports' / 'sub-01.html').read_text())
    39738

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
        out_dir = Path(out_dir)
        root = Path(reportlets_dir or out_dir)

        # TODO This probably breaks YODA and doesn't bring anything. Remove?
        if packagename is not None:
            root = root / packagename
            out_dir = out_dir / packagename

        if subject_id is not None:
            subject_id = subject_id[4:] if subject_id.startswith("sub-") else subject_id
            root = root / "sub-{}".format(subject_id)

        if subject_id is not None and out_filename == "report.html":
            out_filename = f"sub-{subject_id}.html"

        # Initialize structuring elements
        self.sections = []

        bootstrap_file = Path(
            config or pkgrf("nireports.assembler", "data/default.yml")
        )

        bootstrap_text = []
        expr = re.compile(r'{(subject_id|run_uuid|out_dir)}')
        for line in bootstrap_file.read_text().splitlines(keepends=False):
            if expr.search(line):
                line = line.format(
                    subject_id=subject_id if subject_id is not None else "null",
                    run_uuid=run_uuid if run_uuid is not None else "null",
                    out_dir=str(out_dir),
                )
            bootstrap_text.append(line)

        # Load report schema (settings YAML file)
        settings = load("\n".join(bootstrap_text))

        # Set the output path
        self.out_filename = Path(out_filename)
        if not self.out_filename.is_absolute():
            self.out_filename = Path(out_dir) / self.out_filename

        # Path to the Jinja2 template
        self.template_path = (
            Path(settings["template_path"])
            if "template_path" in settings
            else Path(pkgrf("nireports.assembler", "data/report.tpl")).absolute()
        )

        if not self.template_path.is_absolute():
            self.template_path = bootstrap_file / self.template_file

        assert self.template_path.exists()

        settings["root"] = root
        settings["out_dir"] = out_dir
        settings["run_uuid"] = run_uuid
        settings["bids_filters"] = {
            "subject": subject_id,
        }
        self.index(settings)

    def index(self, config):
        """
        Traverse the reports config definition and instantiate reportlets.

        This method also places figures in their final location.
        """
        # Initialize a BIDS layout
        layout = BIDSLayout(config["root"], config="figures", validate=False)

        out_dir = Path(config["out_dir"])
        for subrep_cfg in config["sections"]:
            # First determine whether we need to split by some ordering
            # (ie. sessions / tasks / runs), which are separated by commas.
            orderings = [s for s in subrep_cfg.get("ordering", "").strip().split(",") if s]
            entities, list_combos = self._process_orderings(orderings, layout)

            if not list_combos:  # E.g. this is an anatomical reportlet
                reportlets = [
                    Reportlet(layout, config=cfg, out_dir=out_dir)
                    for cfg in subrep_cfg["reportlets"]
                ]
                list_combos = subrep_cfg.get("nested", False)
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
                        rlet = Reportlet(layout, config=cfg, out_dir=out_dir)
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

    def generate_report(self):
        """Once the Report has been indexed, the final HTML can be generated"""
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(searchpath=str(self.template_path.parent)),
            trim_blocks=True,
            lstrip_blocks=True,
            autoescape=False,
        )
        report_tpl = env.get_template(self.template_path.name)
        report_render = report_tpl.render(sections=self.sections)

        # Write out report
        self.out_filename.parent.mkdir(parents=True, exist_ok=True)
        self.out_filename.write_text(report_render, encoding="UTF-8")
        return 0

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
