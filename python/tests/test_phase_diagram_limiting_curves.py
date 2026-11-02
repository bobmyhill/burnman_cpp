# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Creation-time limiting curves contain one equilibrium per curve point."""

from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import burnman_cpp as bm

from conftest import oxide_params
from test_phase_diagram import pure, settings


@pytest.fixture(scope="module")
def original_diagram():
    first = pure("one", {"Mg": 1.0, "O": 1.0})
    second = bm.CombinedMineral([first], [1.0], [10000.0, 10.0, -1e-6], name="two")
    phases = [first, second]
    opts = settings()
    opts.max_phase_instances = 1
    diagram = bm.phase_diagram(
        {"Mg": 1.0, "O": 1.0}, phases, (0.5e9, 1.5e9), (850.0, 950.0), opts
    )
    assert diagram.resolved, diagram.diagnostics
    return diagram, phases


def test_original_creation_records_two_separate_curves(original_diagram):
    diagram, _ = original_diagram
    assert len(diagram.boundaries) == 1
    boundary = diagram.boundaries[0]
    for curve, assemblage in zip(
        boundary.limiting_curves, (boundary.side_a, boundary.side_b)
    ):
        assert isinstance(curve, bm.PhaseBoundaryCurve)
        assert curve.assemblage == assemblage
        assert len(curve.points) == len(boundary.points)
        assert all(p.success for p in curve.points), [
            p.message for p in curve.points if not p.success
        ]
        assert all(len(p.phases) == 1 for p in curve.points)
    for a, b in zip(*(curve.points for curve in boundary.limiting_curves)):
        assert a.pressure == pytest.approx(b.pressure, abs=1e-6)
        assert a.temperature == pytest.approx(b.temperature, abs=1e-9)
        assert abs(a.entropy - b.entropy) == pytest.approx(10.0, abs=1e-9)
        assert abs(a.volume - b.volume) == pytest.approx(1e-6, abs=1e-14)


def test_curve_json_roundtrip_and_older_diagrams(original_diagram, tmp_path):
    diagram, _ = original_diagram
    data = json.loads(json.dumps(diagram.to_dict(), allow_nan=False))
    assert data["schema_version"] == 5
    boundary = data["boundaries"][0]
    assert len(boundary["limiting_curves"]) == 2
    assert all("limiting_states" not in p for p in boundary["points"])
    assert bm.PhaseDiagramResult.from_dict(data).to_dict() == data
    path = tmp_path / "diagram.json"
    bm.save_phase_diagram(diagram, path)
    assert bm.load_phase_diagram(path) == data
    old = deepcopy(data)
    old["schema_version"] = 4
    for line in old["boundaries"]:
        del line["limiting_curves"]
    restored = bm.PhaseDiagramResult.from_dict(old)
    assert all(
        not c.points for line in restored.boundaries for c in line.limiting_curves
    )


def test_refinement_records_new_curve_points_and_reuses_saved_curves(original_diagram):
    diagram, phases = original_diagram
    refined = bm.refine_phase_diagram(
        diagram.composition_start,
        phases,
        diagram,
        resolution=(101, 101),
        temperature_unit="K",
    )
    boundary = refined.boundaries[0]
    assert len(boundary.points) > len(diagram.boundaries[0].points)
    assert all(len(c.points) == len(boundary.points) for c in boundary.limiting_curves)
    assert all(p.success for c in boundary.limiting_curves for p in c.points)
    saved = refined.to_dict()
    repeated = bm.refine_phase_diagram(diagram.composition_start, phases, refined)
    assert repeated.to_dict() == saved


def test_coarsened_trace_keeps_independently_sampled_curves(original_diagram):
    diagram, phases = original_diagram
    saved = diagram.to_dict()
    line = saved["boundaries"][0]
    line["points"] = [line["points"][0], line["points"][-1]]
    refined = bm.refine_phase_diagram(diagram.composition_start, phases, saved)
    assert refined.to_dict() == saved


@pytest.mark.parametrize("right", [True, False])
def test_equal_assemblages_select_the_geometric_side(right):
    from burnman_cpp.tools.phase_diagram_labels import _limiting_curve

    curves = [dict(assemblage=[0], points=[]), dict(assemblage=[0], points=[])]
    boundary = dict(
        limiting_curves=curves,
        points=[
            dict(pressure=0.0, temperature=0.5),
            dict(pressure=1.0, temperature=0.5),
        ],
    )
    polygon = SimpleNamespace(
        phases=[0],
        contains_rectangle=lambda position, size: (position[1] < 0.5) == right,
    )
    assert (
        _limiting_curve(
            polygon, boundary, ("pressure", "temperature"), ((0.0, 1.0), (0.0, 1.0))
        )
        is curves[0 if right else 1]
    )


def test_contours_read_saved_curves_without_equilibrating_or_rebalancing(
    original_diagram, monkeypatch
):
    from burnman_cpp.tools.composition_contours import _boundary_expression

    diagram, phases = original_diagram
    equilibria = bm.phase_diagram_grid(
        diagram, phases, "density", resolution=(11, 11), store_solutions=True
    )
    data = diagram.to_dict()

    def forbidden(*args, **kwargs):
        pytest.fail("Contouring must only read the saved limiting curves")

    for namespace, names in (
        (
            bm,
            (
                "phase_diagram",
                "stable_equilibrium",
                "equilibrate",
                "phase_diagram_grid",
                "phase_diagram_contours",
            ),
        ),
        (
            bm._core,
            (
                "phase_diagram",
                "stable_equilibrium",
                "_equilibrate",
                "_phase_diagram_grid",
                "phase_diagram_contours",
            ),
        ),
    ):
        for name in names:
            monkeypatch.setattr(namespace, name, forbidden)
    grid = bm.phase_diagram_property_grid(equilibria, phases, "density / 1000")
    sample = _boundary_expression(data, phases, next(iter(grid.values)))
    polygons = bm.phase_diagram_field_polygons(data).polygons
    with monkeypatch.context() as no_balance:
        no_balance.setattr(np.linalg, "lstsq", forbidden)
        for polygon in polygons:
            points, values = sample(polygon)
            assert len(points) == len(diagram.boundaries[0].points)
            assert np.isfinite(values).all()
        for curve in data["boundaries"][0]["limiting_curves"]:
            curve["points"][0]["success"] = False
        sample = _boundary_expression(data, phases, next(iter(grid.values)))
        for polygon in polygons:
            points, values = sample(polygon)
            assert len(points) == len(diagram.boundaries[0].points) - 1
            assert np.isfinite(values).all()
    reference = phases[0].copy()
    reference.set_state(1e9, 900.0)
    records = bm.phase_diagram_grid_contours(
        grid, [reference.density / 1000], diagram=data, phases=phases
    )
    assert records[0]["contours"]["lines"]
    assert records[0]["contours"]["equilibrium_solves"] == 0


@pytest.mark.parametrize("diagram_type", ["PS", "SV"])
def test_natural_coordinate_creation_retains_limiting_curves(diagram_type):
    first = bm.Mineral(oxide_params())
    second = bm.CombinedMineral([first], [1.0], [5000.0, 0.0, -1e-6], name="dense")
    first.set_state(5e9, 1000.0)
    entropy, volume = first.molar_entropy, first.molar_volume
    opts = settings()
    opts.max_phase_instances = 1
    opts.entropy_seeds = opts.volume_seeds = 3
    diagram = bm.phase_diagram(
        {"Mg": 1.0, "O": 1.0},
        [first, second],
        settings=opts,
        diagram=diagram_type,
        pressure_range=(4.5e9, 5.5e9),
        temperature_range=(800.0, 1200.0),
        entropy_range=(entropy - 1.0, entropy + 1.0),
        volume_range=(volume - 1.2e-6, volume + 0.2e-6),
    )
    assert diagram.resolved, diagram.diagnostics
    assert diagram.boundaries
    for boundary in diagram.boundaries:
        assert all(c.points for c in boundary.limiting_curves)
        for curve in boundary.limiting_curves:
            assert any(p.success for p in curve.points), [
                p.message for p in curve.points
            ]
            for point in curve.points:
                if point.success:
                    assert point.mass_balance_error <= opts.mass_balance_tolerance
                    assert point.equilibrium_error <= 0.1 * opts.affinity_tolerance
