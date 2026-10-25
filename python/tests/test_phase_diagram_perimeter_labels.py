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

"""Recover side labels only from complete, reaction-compatible field loops."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import burnman_cpp as bm
from burnman_cpp.minerals import SLB_2024
from test_phase_diagram_plot import diagram, line, square


@pytest.fixture
def pyrolite_neighbours():
    data = json.loads(
        (Path(__file__).parent / "data/pyrolite_perimeter_neighbours.json").read_text()
    )
    factories = """olivine wadsleyite ringwoodite orthopyroxene clinopyroxene
        c2c_pyroxene garnet mg_fe_aluminous_spinel plagioclase ilmenite
        ferropericlase bridgmanite post_perovskite calcium_ferrite_structured_phase
        new_aluminous_phase capv neph ky qtz coes st apbo wo pwo fea feg fee""".split()
    phases = [getattr(SLB_2024, name)() for name in factories]
    for phase, name in zip(phases, data["phase_names"][::3]):
        phase.set_name(name)
    return data, phases


@pytest.mark.parametrize("unrelated", [False, True])
def test_closed_perimeters_recover_ambiguous_boundary_labels(
    pyrolite_neighbours, unrelated
):
    data, phases = pyrolite_neighbours
    if unrelated:
        data["diagnostics"].append("An unrelated diagnostic must survive.")
    before = bm.phase_diagram_field_polygons(data)
    full = [0, 12, 13, 21, 22, 24, 51, 54]
    absent = [0, 12, 13, 21, 22, 51, 54]
    assert before.boundary_sides[9] == [full, absent]
    original = deepcopy(data)
    result = bm.verify_phase_diagram_critical_endpoints(data, phases)
    assert data == original
    boundary = next(b for b in result.boundaries if b.id == 9)
    assert boundary.side_a == full and boundary.side_b == absent
    assert result.equilibrium_solves == data["equilibrium_solves"]
    # This extracted network has an unfinished exterior field. Recovering
    # boundary 9 must retain that diagnostic rather than certify the whole crop.
    expected = ["An unrelated diagnostic must survive."] if unrelated else []
    expected += list(before.diagnostics)
    assert list(result.diagnostics) == expected
    assert result.resolved == (not expected)
    checked = result.to_dict()
    assert checked["nodes"] == data["nodes"]
    assert checked["samples"] == data["samples"]
    for a, b in zip(checked["boundaries"], data["boundaries"]):
        assert {k: v for k, v in a.items() if k not in ("side_a", "side_b")} == {
            k: v for k, v in b.items() if k not in ("side_a", "side_b")
        }
    after = bm.phase_diagram_field_polygons(result)
    assert len(after.polygons) == len(before.polygons)
    for a, b in zip(after.polygons, before.polygons):
        np.testing.assert_array_equal(a.vertices, b.vertices)
        assert a.phases == b.phases


def test_open_perimeter_does_not_certify_boundary_sides(pyrolite_neighbours):
    data, phases = pyrolite_neighbours
    boundary = next(b for b in data["boundaries"] if b["id"] == 9)
    boundary["points"] = boundary["points"][:-2]
    boundary["end_node"] = -1
    boundary["termination"] = "junction; corrector failed"
    assert 9 not in bm.phase_diagram_field_polygons(data).boundary_sides
    result = bm.verify_phase_diagram_critical_endpoints(data, phases)
    assert not result.resolved
    assert data["diagnostics"][0] in result.diagnostics
    actual = next(b for b in result.boundaries if b.id == 9)
    assert actual.side_a == boundary["side_a"] and actual.side_b == boundary["side_b"]


def test_reaction_incompatible_perimeters_do_not_clear_warning(pyrolite_neighbours):
    data, phases = pyrolite_neighbours
    next(b for b in data["boundaries"] if b["id"] == 9)["zero_phase"] = 12
    assert 9 not in bm.phase_diagram_field_polygons(data).boundary_sides
    result = bm.verify_phase_diagram_critical_endpoints(data, phases)
    assert not result.resolved
    assert data["diagnostics"][0] in result.diagnostics


@pytest.mark.parametrize("reverse", [False, True])
def test_opposed_duplicate_loops_keep_each_records_side_orientation(reverse):
    points = square()
    data = diagram(
        [
            dict(
                id=3,
                zero_phase=0,
                assemblage=[0, 3],
                reaction_kind="phase_out",
                **line(points, [0, 3], [3]),
            ),
            dict(
                id=9,
                zero_phase=0,
                assemblage=[0, 3],
                reaction_kind="phase_out",
                **line(points[::-1], [3], [0, 3]),
            ),
        ]
    )
    if reverse:
        data["boundaries"].reverse()
    geometry = bm.phase_diagram_field_polygons(data)
    assert not geometry.diagnostics
    assert len(geometry.polygons) == 2
    assert geometry.boundary_sides[3] == [[3], [0, 3]]
    assert geometry.boundary_sides[9] == [[0, 3], [3]]


def test_equal_neighbours_do_not_certify_an_ordinary_phase_boundary():
    data = diagram(
        [
            dict(
                id=3,
                zero_phase=0,
                assemblage=[0, 3],
                **line([(0.5, 0), (0.5, 1)], [3], [3]),
            ),
        ]
    )
    assert not bm.phase_diagram_field_polygons(data).boundary_sides


def test_pyrolite_entropy_coexistence_field_survives_failed_side_orientation():
    data = json.loads(
        (
            Path(__file__).parent / "data/pyrolite_sp_reaction_classification.json"
        ).read_text()
    )
    models = data["workflow"]["models"]
    phases = [getattr(SLB_2024, name)() for name in models["phase_names"]]
    assert bm.phase_diagram_field_polygons(data).diagnostics
    result = bm.refine_phase_diagram(
        data["composition_start"],
        phases,
        data,
        bm.PhaseDiagramSettings.from_dict(data["settings"]),
        resolution=(2, 2),
    )
    # This crop omits the surrounding network. Its complete inner loop must
    # still retain the finite nine-phase field, including bg and ringwoodite.
    point = np.array([24.100611966328526e9, 192.03118480616763])
    geometry = bm.phase_diagram_field_polygons(result)
    field = next(
        p
        for p in geometry.polygons
        if p.contains_rectangle(point, np.array([1.0, 1.0e-8]))
    )
    expected = [6, 12, 15, 16, 24, 30, 33, 36, 72]
    assert field.phases == expected and not field.is_provisional
    state = bm.stable_equilibrium(
        data["composition_start"],
        phases,
        pressure=point[0],
        entropy=point[1],
        pressure_range=data["pressure_range"],
        temperature_range=data["temperature_range"],
    )
    assert state.success, state.message
    assert sorted(p.id for p in state.phases) == expected
    assert not any("conflicting perimeter" in d for d in result.diagnostics)


def test_trace_site_populations_preserve_a_finite_entropy_field_response():
    data = json.loads(
        (
            Path(__file__).parent / "data/pyrolite_sp_trace_site_response.json"
        ).read_text()
    )
    phases = [
        getattr(SLB_2024, name)() for name in data["workflow"]["models"]["phase_names"]
    ]
    result = bm.refine_phase_diagram(
        data["composition_start"],
        phases,
        data,
        resolution=(2, 2),
        region=(data["entropy_range"], data["pressure_range"]),
        pressure_unit="Pa",
    )
    # Trace site populations make this regular three-garnet coexistence
    # response stiff, but do not remove its finite extent in entropy.
    boundary = result.boundaries[0]
    full = (15, 18, 24, 33, 34, 35, 36, 72, 75)
    absent = (15, 18, 24, 33, 34, 36, 72, 75)
    assert {tuple(boundary.side_a), tuple(boundary.side_b)} == {full, absent}
    assert boundary.reaction_kind == "phase_out"
