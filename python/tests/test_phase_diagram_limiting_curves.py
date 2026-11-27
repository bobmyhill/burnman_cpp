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
    assert data["schema_version"] == 6
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


@pytest.mark.parametrize("missing", [False, True])
def test_explicit_refresh_rebuilds_legacy_limits_without_mutating_input(
    original_diagram, missing
):
    diagram, phases = original_diagram
    saved = diagram.to_dict()
    curves = saved["boundaries"][0]["limiting_curves"]
    curves[0]["points"] = deepcopy(curves[1]["points"])
    if missing:
        del saved["boundaries"][0]["limiting_curves"]
    before = deepcopy(saved)
    refreshed = bm.refresh_phase_diagram_limiting_curves(saved, phases).to_dict()
    assert saved == before
    assert refreshed == diagram.to_dict()


def test_explicit_refresh_keeps_invalid_limits_unsuccessful(original_diagram):
    diagram, phases = original_diagram
    saved = diagram.to_dict()
    for point in saved["boundaries"][0]["points"]:
        for phase in point["phases"]:
            phase["amount"] *= 2.0
    refreshed = bm.refresh_phase_diagram_limiting_curves(saved, phases)
    for curve in refreshed.boundaries[0].limiting_curves:
        assert all(not s.success for s in curve.points)
        assert all(s.message for s in curve.points)


def test_explicit_refresh_rejects_changed_candidate_order(original_diagram):
    diagram, phases = original_diagram
    with pytest.raises(ValueError, match="same candidate names"):
        bm.refresh_phase_diagram_limiting_curves(diagram, phases[::-1])


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("failed", [False, True])
def test_internal_limits_use_independent_curve_coordinates(reverse, failed):
    from burnman_cpp.tools._grid_contours import internal_boundary_samples

    def state(x, sign):
        return dict(pressure=x, temperature=0.5, success=True, value=sign * x)

    curves = [
        dict(assemblage=[0], points=[state(x, 1) for x in (0.0, 0.25, 1.0)]),
        dict(assemblage=[0], points=[state(x, -1) for x in (0.0, 0.5, 0.75, 1.0)]),
    ]
    if failed:
        curves[0]["points"][1]["success"] = False
    if reverse:
        for curve in curves:
            curve["points"].reverse()
    boundary = dict(points=[state(0.0, 0), state(1.0, 0)], limiting_curves=curves)
    before = deepcopy(boundary)
    polygon = SimpleNamespace(
        phases=[0],
        contains_rectangle=lambda *args: True,
        vertices=np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]),
        holes=[],
    )
    points, values, junctions = internal_boundary_samples(
        polygon,
        boundary,
        np.zeros(2),
        np.ones(2),
        lambda s, side, index: s["value"],
    )
    assert boundary == before
    right = points[:, 1] < 0.5
    left = points[:, 1] > 0.5
    if failed:
        assert np.isnan(values[right & (points[:, 0] < 1.0)]).all()
        np.testing.assert_array_equal(sorted(values[points[:, 0] == 1.0]), [-1.0, 1.0])
    else:
        np.testing.assert_allclose(values[right], points[right, 0])
    np.testing.assert_allclose(values[left], -points[left, 0])
    np.testing.assert_array_equal(junctions, [[1.0, 0.5]])


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
                "refresh_phase_diagram_limiting_curves",
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
                "refresh_phase_diagram_limiting_curves",
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
    refreshed = bm.refresh_phase_diagram_limiting_curves(diagram, [first, second])
    saved, updated = diagram.to_dict(), refreshed.to_dict()
    for key in ("equilibrium_solves", "minimization_calls"):
        assert updated.pop(key) >= saved.pop(key)
    assert updated == saved
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
