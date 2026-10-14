# ------------------------------------------------------
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: Copyright (C) 2025-2026 by the BurnMan Team.
#
# This file is part of BurnMan.
#
# Detailed license information governing the source code
# and contributions can be found in the LICENSE document.
#
# ------------------------------------------------------

"""A completed equilibrium boundary covers a failed retrace near zero kelvin."""

from copy import deepcopy
import json
from pathlib import Path
import runpy

import pytest

import burnman_cpp as bm
from test_phase_diagram import curved_phases, settings


@pytest.fixture(scope="module")
def models():
    example = Path(__file__).parents[2] / "examples/example_pyrolite_phase_diagram.py"
    return runpy.run_path(str(example))["candidate_phases"]()


@pytest.mark.parametrize("reverse_order", [False, True])
@pytest.mark.parametrize("reverse_trace", [False, True])
def test_completed_boundary_survives_denser_unfinished_duplicate(
    models, reverse_order, reverse_trace
):
    path = Path(__file__).parent / "data/pyrolite_partial_duplicate.json"
    saved = json.loads(path.read_text())
    complete, partial = saved["boundaries"]
    expected = deepcopy(complete)
    assert len(partial["points"]) > len(complete["points"])
    before = bm.phase_diagram_field_polygons(saved)
    assert any(p.has_open_boundary for p in before.polygons)
    if reverse_trace:
        partial["points"].reverse()
        partial["start_node"], partial["end_node"] = (
            partial["end_node"],
            partial["start_node"],
        )
        partial["side_a"], partial["side_b"] = partial["side_b"], partial["side_a"]
        partial["termination"] = "; ".join(reversed(partial["termination"].split("; ")))
    if reverse_order:
        saved["boundaries"].reverse()
    original = deepcopy(saved)
    result = bm.refine_phase_diagram(saved["composition_start"], models, saved)
    assert saved == original
    assert result.resolved, result.diagnostics
    assert len(result.boundaries) == 1
    retained = result.to_dict()["boundaries"][0]
    assert retained["points"] == expected["points"]
    assert retained["start_node"] == expected["start_node"]
    assert retained["end_node"] == expected["end_node"]
    assert retained["termination"] == expected["termination"]
    assert retained["side_a"] == expected["side_a"]
    assert retained["side_b"] == expected["side_b"]
    assert retained["points"][0]["temperature"] == 0
    after = bm.phase_diagram_field_polygons(result)
    assert not after.diagnostics
    assert len(after.polygons) == len(before.polygons)
    assert all(not p.has_open_boundary and not p.is_provisional for p in after.polygons)


@pytest.mark.parametrize("short_tail", [False, True])
def test_uncovered_tail_is_not_discarded_as_a_partial_duplicate(short_tail):
    bulk = {"Mg": 1.0, "O": 1.0}
    phases = curved_phases()
    opts = settings()
    opts.max_recovery_passes = 0
    saved = bm.phase_diagram(
        bulk, phases, (0.0, 5.0e9), (600.0, 1400.0), opts
    ).to_dict()
    complete = saved["boundaries"][0]
    whole = deepcopy(complete)
    whole["points"] = whole["points"][:-1]
    # The purported completed cover ends halfway along the accepted branch.
    # An open record with the same reaction contains a genuine uncovered tail.
    halfway = len(whole["points"]) - 2 if short_tail else len(whole["points"]) // 2
    end_node = deepcopy(saved["nodes"][complete["end_node"]])
    end_node["id"] = len(saved["nodes"])
    for name in ("pressure", "temperature", "entropy", "volume"):
        end_node[name] = complete["points"][halfway][name]
    saved["nodes"].append(end_node)
    complete["points"] = complete["points"][: halfway + 1]
    complete["end_node"] = end_node["id"]
    whole.update(id=1, end_node=-1, termination="domain edge; corrector failed")
    saved["boundaries"].append(whole)
    saved["resolved"] = False
    result = bm.refine_phase_diagram(bulk, phases, saved, opts)
    assert len(result.boundaries) >= 2
    assert any(b.end_node < 0 or b.start_node < 0 for b in result.boundaries)
    assert not result.resolved
