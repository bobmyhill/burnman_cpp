# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Example output keeps informational diagnostics off the console."""

import burnman_cpp as bm
from conftest import diagnostic
from burnman_cpp.tools.phase_diagram_helpers import (
    report_phase_diagram,
    write_phase_diagram,
)


def test_informational_notes_are_saved_without_printing(tmp_path, capsys):
    result = dict(
        resolved=True,
        coordinate_diagnostics=[diagnostic("Interior location uncertain")],
        spacing_diagnostics=[diagnostic("Requested spacing not reached")],
    )
    report_phase_diagram(result, tmp_path)
    output = capsys.readouterr().out
    assert "Resolved: True" in output
    assert "native equilibrium solves" not in output
    assert "Interior location uncertain" not in output
    assert "Requested spacing not reached" not in output
    saved = (tmp_path / "diagnostics.txt").read_text()
    assert "Interior location uncertain" in saved
    assert "Requested spacing not reached" in saved
    assert "native equilibrium solves" in saved


def test_blocking_diagnostics_still_print_and_share_the_output_stem(tmp_path, capsys):
    phase = bm.minerals.HGP_2018_ds633.per()
    result = bm.phase_diagram(
        {"Mg": 1.0, "O": 1.0}, [phase], [1.0e9, 2.0e9], [800.0, 1000.0]
    ).to_dict()
    result.update(
        resolved=False,
        diagnostics=[diagnostic("An unfinished reaction remains")],
        coordinate_diagnostics=[diagnostic("Uncertain interior coordinates")],
    )
    write_phase_diagram(result, tmp_path / "example")
    output = capsys.readouterr().out
    assert "Resolved: False" in output
    assert "An unfinished reaction remains" in output
    assert "Uncertain interior coordinates" not in output
    saved = (tmp_path / "example_diagnostics.txt").read_text()
    assert "An unfinished reaction remains" in saved
    assert "Uncertain interior coordinates" in saved
    assert bm.load_phase_diagram(tmp_path / "example.json")[
        "coordinate_diagnostics"
    ] == [diagnostic("Uncertain interior coordinates")]
