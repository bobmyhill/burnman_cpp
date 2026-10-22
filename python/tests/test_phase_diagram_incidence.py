# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Certification derives node incidence from the retained boundary graph."""

from copy import deepcopy
import json

import pytest
import burnman_cpp as bm

from test_phase_diagram import crossing_phases, settings


@pytest.fixture(scope="module", params=["PT", "PS"])
def crossing_graph(request):
    opts = settings()
    opts.entropy_seeds = 3
    result = bm.phase_diagram(
        {"Mg": 1.0, "Fe": 1.0, "O": 2.0},
        crossing_phases(),
        pressure_range=(0.0, 2.0e9),
        temperature_range=(600.0, 1400.0),
        entropy_range=(210.0, 290.0),
        diagram=request.param,
        settings=opts,
    )
    assert result.resolved, result.diagnostics
    return json.loads(json.dumps(result.to_dict()))


def check_incidence(result):
    for node in result.nodes:
        expected = [
            line.id
            for line in result.boundaries
            if line.start_node == node.id or line.end_node == node.id
        ]
        assert sorted(node.incident_lines) == sorted(expected)


def test_stale_single_arm_does_not_reject_a_complete_graph(crossing_graph):
    saved = deepcopy(crossing_graph)
    junction = next(n for n in saved["nodes"] if n["kind"] == "junction")
    assert len(junction["incident_lines"]) > 1
    junction["incident_lines"] = junction["incident_lines"][:1]
    saved["diagnostics"] = [f"Unpaired internal junction at node {junction['id']}."]
    saved["resolved"] = False
    before = deepcopy(saved)

    result = bm.verify_phase_diagram_critical_endpoints(saved, crossing_phases())

    assert result.resolved, result.diagnostics
    check_incidence(result)
    assert result.to_dict()["boundaries"] == before["boundaries"]
    assert saved == before


def test_retired_junction_loses_its_stale_incident_line(crossing_graph):
    saved = deepcopy(crossing_graph)
    retired = deepcopy(next(n for n in saved["nodes"] if n["kind"] == "junction"))
    retired["id"] = len(saved["nodes"])
    retired["incident_lines"] = retired["incident_lines"][:1]
    saved["nodes"].append(retired)
    saved["diagnostics"] = [f"Unpaired internal junction at node {retired['id']}."]
    saved["resolved"] = False

    result = bm.verify_phase_diagram_critical_endpoints(saved, crossing_phases())

    assert result.resolved, result.diagnostics
    assert not result.nodes[retired["id"]].incident_lines
    check_incidence(result)


def test_stale_complete_incidence_cannot_hide_a_missing_branch(crossing_graph):
    saved = deepcopy(crossing_graph)
    junction = next(n for n in saved["nodes"] if n["kind"] == "junction")
    keep = junction["incident_lines"][0]
    saved["boundaries"] = [
        line
        for line in saved["boundaries"]
        if line["id"] == keep
        or junction["id"] not in (line["start_node"], line["end_node"])
    ]
    for i, line in enumerate(saved["boundaries"]):
        line["id"] = i
    # Retain the old four-arm certificate, with a genuinely incomplete graph.
    saved["resolved"] = False

    result = bm.verify_phase_diagram_critical_endpoints(saved, crossing_phases())

    assert not result.resolved
    assert f"Unpaired internal junction at node {junction['id']}." in result.diagnostics
    assert len(result.nodes[junction["id"]].incident_lines) == 1
    check_incidence(result)
