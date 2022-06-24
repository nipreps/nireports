import os
import tempfile
from pathlib import Path

import pytest

# disable ET
os.environ["NO_ET"] = "1"


@pytest.fixture(autouse=True)
def expand_namespace(doctest_namespace):
    doctest_namespace["os"] = os
    doctest_namespace["Path"] = Path

    tmpdir = tempfile.TemporaryDirectory()
    doctest_namespace["tmpdir"] = tmpdir.name

    cwd = os.getcwd()
    os.chdir(tmpdir.name)
    yield
    os.chdir(cwd)
    tmpdir.cleanup()
