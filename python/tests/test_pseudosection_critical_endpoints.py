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

"""Physical critical replacements and genuine missing solvus arms."""

import json
from pathlib import Path
import runpy

import numpy as np
import pytest
import burnman_cpp as bm


@pytest.fixture
def critical_section():
    root = Path(__file__).parents[2]
    data = json.loads(
        (Path(__file__).parent / "data/pyrolite_critical_endpoints.json").read_text()
    )
    models = runpy.run_path(str(root / "examples/example_pyrolite_pseudosection.py"))
    return data, models["candidate_phases"]()


def test_univariant_critical_proof_preserves_geometry_and_roundtrips(critical_section):
    data, phases = critical_section
    data["diagnostics"].append("An unrelated diagnostic must survive.")
    before = bm.pseudosection_field_polygons(data)
    assert any(p.has_open_boundary for p in before.polygons)
    result = bm.verify_pseudosection_critical_endpoints(data, phases)
    checked = result.to_dict()
    certified = [
        data["original_nodes"][n.id]
        for n in result.nodes
        if n.is_verified_critical_endpoint
    ]
    assert certified == [384, 391]
    assert checked["boundaries"] == data["boundaries"]
    assert checked["samples"] == data["samples"]
    assert "An unrelated diagnostic must survive." in result.diagnostics
    for representation in [
        result,
        json.loads(json.dumps(checked)),
        bm.PseudosectionResult.from_dict(checked),
    ]:
        after = bm.pseudosection_field_polygons(representation)
        assert not any(p.has_open_boundary for p in after.polygons)
        assert len(after.polygons) == len(before.polygons)
        for a, b in zip(before.polygons, after.polygons):
            np.testing.assert_array_equal(a.vertices, b.vertices)
            assert len(a.holes) == len(b.holes)
            for ah, bh in zip(a.holes, b.holes):
                np.testing.assert_array_equal(ah, bh)


def test_false_critical_location_invalidates_saved_certificate(critical_section):
    data, phases = critical_section
    for node in data["nodes"]:
        if node["kind"] == "critical_point" and node["gibbs_variance"] == 1:
            node["is_verified_critical_endpoint"] = True
            node["temperature"] += 5.0
            for line in data["boundaries"]:
                if line["start_node"] == node["id"]:
                    line["points"][0]["temperature"] = node["temperature"]
                if line["end_node"] == node["id"]:
                    line["points"][-1]["temperature"] = node["temperature"]
    result = bm.verify_pseudosection_critical_endpoints(data, phases)
    assert not any(n.is_verified_critical_endpoint for n in result.nodes)
    assert any(
        p.has_open_boundary for p in bm.pseudosection_field_polygons(result).polygons
    )
    assert any("Unpaired solution critical endpoint" in d for d in result.diagnostics)


def test_missing_arm_of_finite_area_solvus_stays_unresolved(critical_section):
    data, phases = critical_section
    # The other garnet critical point bounds a real eight-phase finite field.
    # Its two distinct arms must not be treated as a univariant replacement.
    removed = data["original_boundaries"].index(691)
    data["boundaries"].pop(removed)
    for node in data["nodes"]:
        node["incident_lines"] = []
        node["is_verified_critical_endpoint"] = True
    for i, line in enumerate(data["boundaries"]):
        line["id"] = i
        for n in [line["start_node"], line["end_node"]]:
            if n >= 0:
                data["nodes"][n]["incident_lines"].append(i)
    result = bm.verify_pseudosection_critical_endpoints(data, phases)
    index = data["original_nodes"].index(385)
    assert not result.nodes[index].is_verified_critical_endpoint
    assert any(
        p.has_open_boundary for p in bm.pseudosection_field_polygons(result).polygons
    )
    assert any("Unpaired solution critical endpoint" in d for d in result.diagnostics)
