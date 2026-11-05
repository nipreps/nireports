# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# Copyright 2026 The NiPreps Developers <nipreps@gmail.com>
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
"""Tests for reportlet utility functions."""

from io import StringIO
from subprocess import CompletedProcess
from unittest.mock import Mock

import pytest

from nireports.reportlets import utils


@pytest.fixture
def mock_compressors(monkeypatch):
    """Exercise SVG parsing without requiring the external compression tools."""

    def run(command, *, input, **kwargs):
        if command.startswith("svgo "):
            output = input
        else:
            assert command.startswith("cwebp ")
            output = b"webp"
        return CompletedProcess(command, 0, stdout=output)

    run_mock = Mock(side_effect=run)
    monkeypatch.setattr(utils, "have_compression", True)
    monkeypatch.setattr(utils.subprocess, "run", run_mock)
    return run_mock


@pytest.mark.parametrize("suffix", ("", '\n  cG5n"\n'), ids=("marker", "payload"))
def test_svg_compress_unterminated_png(monkeypatch, mock_compressors, suffix):
    """Reject a truncated PNG tag when its closing delimiter is missing."""

    class EOFGuard(StringIO):
        # Fail promptly if the parser regresses to repeatedly reading EOF.
        eof_reads = 0

        def readline(self, *args, **kwargs):
            line = super().readline(*args, **kwargs)
            if not line:
                self.eof_reads += 1
                assert self.eof_reads == 1, "Repeated read after SVG end of input"
            return line

    monkeypatch.setattr(utils, "StringIO", EOFGuard)
    image = '<svg xmlns="http://www.w3.org/2000/svg">\n<image href="data:image/png;base64,'

    with pytest.raises(ValueError, match=r"^Unterminated embedded PNG image in SVG$"):
        utils.svg_compress(image + suffix, compress=True)

    # Malformed input must stop before attempting raster conversion.
    assert mock_compressors.call_count == 1


def test_svg_compress_multiline_png(mock_compressors):
    """Convert a complete multiline PNG tag and retain surrounding SVG content."""
    image = (
        '<svg xmlns="http://www.w3.org/2000/svg">\n'
        '<image href="data:image/png;base64,\n'
        '  cG5n" width="1" height="1"\n'
        "/>\n"
        '<path d="M0 0"/>\n'
        "</svg>\n"
    )

    assert utils.svg_compress(image, compress=True) == (
        '<svg xmlns="http://www.w3.org/2000/svg">\n'
        '<image href="data:image/webp;base64,d2VicA==" width="1" height="1"/>'
        '<path d="M0 0"/>\n'
        "</svg>\n"
    )
    assert mock_compressors.call_count == 2
    assert mock_compressors.call_args_list[0].kwargs["input"] == image.encode("utf-8")
    assert mock_compressors.call_args_list[1].kwargs["input"] == b"png"
