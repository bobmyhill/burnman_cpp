# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Expression evaluation and interpolation use only retained compositions."""

from copy import deepcopy
import json

import numpy as np
import pytest
import burnman_cpp as bm

from conftest import make_model, oxide_params


@pytest.fixture(autouse=True)
def forbid_equilibrium_calculations(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Composition processing must never solve equilibria")

    for name in (
        "phase_diagram",
        "stable_equilibrium",
        "_equilibrate",
        "_phase_diagram_grid",
        "phase_diagram_contours",
    ):
        monkeypatch.setattr(bm._core, name, forbidden)
    for name in (
        "phase_diagram",
        "stable_equilibrium",
        "equilibrate",
        "phase_diagram_grid",
        "phase_diagram_contours",
    ):
        monkeypatch.setattr(bm, name, forbidden)


def stored_grid(phases, *, diagram=None, x=None, y=None, phase_states=None):
    x = np.linspace(1.0e9, 2.0e9, 5) if x is None else np.asarray(x)
    y = np.linspace(800.0, 1200.0, 5) if y is None else np.asarray(y)
    context = dict(
        diagram
        or dict(
            diagram_type="PT",
            pressure_range=[x[0], x[-1]],
            temperature_range=[y[0], y[-1]],
            resolved=True,
            phase_names=[
                p.name + (f" #{i+1}" if i else "") for p in phases for i in range(3)
            ],
        ),
        boundaries=[],
        samples=[],
        nodes=[],
        fields=[],
    )
    context = bm.PhaseDiagramResult.from_dict(context).to_dict()
    solutions = []
    for t in y:
        row = []
        for p in x:
            value = (p - x[0]) / (x[-1] - x[0])
            states = (
                phase_states(p, t)
                if phase_states
                else [
                    dict(
                        id=0,
                        candidate_index=0,
                        name=phases[0].name,
                        amount=1.0,
                        composition=[value, 1.0 - value],
                    )
                ]
            )
            row.append(
                dict(
                    pressure=float(p),
                    temperature=float(t),
                    entropy=float(t),
                    volume=1.0,
                    composition_coordinate=0.0,
                    mass_balance_error=0.0,
                    minimum_affinity=0.0,
                    residual=0.0,
                    n_moles=sum(s["amount"] for s in states),
                    molar_fractions=[1.0 / len(states)] * len(states),
                    phases=states,
                    phase_bases=[
                        np.eye(len(s["composition"])).tolist() for s in states
                    ],
                    phase_compositions=[s["composition"] for s in states],
                )
            )
        solutions.append(row)
    shape = (len(y), len(x))
    return bm.PhaseDiagramPropertyGrid.from_dict(
        dict(
            format="burnman_cpp.property_grid",
            schema_version=1,
            context=context,
            x=x,
            y=y,
            relaxed=False,
            resolved=True,
            values={"temperature": np.broadcast_to(y[:, None], shape)},
            field_numbers=np.ones(shape, int),
            success=np.ones(shape, bool),
            equilibrium_success=np.ones(shape, bool),
            diagnostics=[],
            equilibrium_diagnostics=[],
            equilibrium_solves=123,
            solutions=solutions,
        )
    )


@pytest.fixture
def sample():
    phases = [
        bm.Solution(make_model(), [0.5, 0.5], name="oxide"),
        bm.Mineral(oxide_params()),
    ]
    return stored_grid(phases), phases


def values(grid):
    return next(iter(grid.values.values()))


def reference_volumes(states, phases, pressure, temperature):
    """Independent native Assemblage normalization for the stored amounts."""
    materials = []
    amounts = np.array([p["amount"] for p in states])
    for state in states:
        material = phases[state["candidate_index"]].copy()
        if isinstance(material, bm.Solution):
            material.set_composition(state["composition"])
        materials.append(material)
    assemblage = bm.Assemblage(materials, amounts / amounts.sum())
    assemblage.set_state(pressure, temperature)
    return amounts.sum() * assemblage.molar_volume, assemblage.volume_fractions


@pytest.fixture
def volume_sample():
    parameters = oxide_params()
    parameters.update(
        name="other", V_0=parameters["V_0"] * 3.0, K_0=parameters["K_0"] / 3.0
    )
    phases = [
        bm.Solution(make_model(), [0.3, 0.7], name="oxide"),
        bm.Mineral(parameters),
    ]

    def states(p, t):
        x = (p - 1.0e9) / 1.0e9
        return [
            dict(
                id=0,
                candidate_index=0,
                name="oxide",
                amount=1.0 + x,
                composition=[0.25 + 0.5 * x, 0.75 - 0.5 * x],
            ),
            dict(id=3, candidate_index=1, name="other", amount=2.0, composition=[1.0]),
        ]

    grid = stored_grid(phases, phase_states=states).to_dict()
    expected = {
        "volume_fraction": np.empty((5, 5)),
        "standard_volume_fraction": np.empty((5, 5)),
    }
    for j, row in enumerate(grid["solutions"]):
        for i, state in enumerate(row):
            total, fractions = reference_volumes(
                state["phases"], phases, state["pressure"], state["temperature"]
            )
            state["volume"] = total
            state["molar_fractions"] = [
                p["amount"] / state["n_moles"] for p in state["phases"]
            ]
            expected["volume_fraction"][j, i] = fractions[0]
            _, fractions = reference_volumes(state["phases"], phases, 1.0e5, 298.15)
            expected["standard_volume_fraction"][j, i] = fractions[0]
    return bm.PhaseDiagramPropertyGrid.from_dict(grid), phases, expected


@pytest.mark.parametrize("expression", ["volume_fraction", "standard_volume_fraction"])
def test_volume_fractions_use_molar_volumes_and_freeze_the_saved_assemblage(
    volume_sample, expression
):
    source, phases, expected = volume_sample
    snapshot = source.to_dict()
    phases[0].set_state(3.0e9, 1500.0)
    phases[0].set_composition([0.4, 0.6])
    result = bm.phase_composition_grid(source, phases, "oxide", expression)
    np.testing.assert_allclose(values(result), expected[expression], rtol=1.0e-12)
    other = bm.phase_composition_grid(source, phases, "other", expression)
    np.testing.assert_allclose(values(result) + values(other), 1.0, rtol=1.0e-12)
    assert result.equilibrium_solves == 0 and source.to_dict() == snapshot
    np.testing.assert_array_equal(phases[0].molar_fractions, [0.4, 0.6])
    assert (phases[0].pressure, phases[0].temperature) == (3.0e9, 1500.0)
    # Molar proportions alone would miss the unequal phase molar volumes.
    molar = np.array(
        [[s["molar_fractions"][0] for s in row] for row in snapshot["solutions"]]
    )
    assert np.max(np.abs(values(result) - molar)) > 0.1


def test_standard_volume_fraction_changes_every_phase_volume(volume_sample):
    source, phases, expected = volume_sample
    result = bm.phase_composition_grid(
        source, phases, "oxide", "100*standard_volume_fraction"
    )
    np.testing.assert_allclose(
        values(result), 100.0 * expected["standard_volume_fraction"], rtol=1.0e-12
    )
    assert (
        np.max(
            np.abs(expected["volume_fraction"] - expected["standard_volume_fraction"])
        )
        > 1.0e-4
    )
    # All rows share the same frozen amounts and compositions at each pressure.
    np.testing.assert_allclose(
        values(result), np.broadcast_to(values(result)[0], (5, 5))
    )


def test_volume_fraction_sums_copies_and_is_zero_for_an_absent_phase(volume_sample):
    source, phases, _ = volume_sample
    data = source.to_dict()
    for row in data["solutions"]:
        for state in row:
            copy = deepcopy(state["phases"][0])
            copy.update(id=1, name="oxide #2", amount=0.5, composition=[0.8, 0.2])
            state["phases"].append(copy)
            state["phase_bases"].append(np.eye(2).tolist())
            state["phase_compositions"].append(copy["composition"])
            state["volume"] = reference_volumes(
                state["phases"], phases, state["pressure"], state["temperature"]
            )[0]
    total = bm.phase_composition_grid(data, phases, "oxide", "volume_fraction")
    first = bm.phase_composition_grid(
        data, phases, "oxide", "volume_fraction", instance=1
    )
    second = bm.phase_composition_grid(
        data, phases, "oxide", "volume_fraction", instance=2
    )
    np.testing.assert_allclose(
        values(total), values(first) + values(second), rtol=1.0e-12
    )
    state = data["solutions"][0][0]
    state["phases"] = [state["phases"][1]]
    state["phase_bases"] = [[[1.0]]]
    state["phase_compositions"] = [[1.0]]
    for expression in ("volume_fraction", "standard_volume_fraction"):
        result = bm.phase_composition_grid(data, phases, "oxide", expression)
        assert values(result)[0, 0] == 0.0 and result.success[0, 0]


def test_volume_boundary_values_use_preexisting_amounts_and_preserve_zero_modes(
    volume_sample,
):
    from burnman_cpp.tools.composition_contours import _boundary_expression

    source, phases, _ = volume_sample
    diagram = split_diagram(source, solved_boundary=True)
    polygon = max(
        bm.phase_diagram_field_polygons(diagram).polygons,
        key=lambda p: p.label_position[0],
    )
    for expression in ("volume_fraction", "standard_volume_fraction"):
        points, boundary = _boundary_expression(
            diagram, phases, f"other: {expression}"
        )(polygon)
        assert len(points) == 5
        np.testing.assert_array_equal(boundary, np.zeros(5))


@pytest.mark.parametrize("expression", ["volume_fraction", "standard_volume_fraction"])
def test_excluded_denominator_phases_are_renormalized_and_saved(
    volume_sample, tmp_path, expression
):
    source, phases, _ = volume_sample
    result = bm.phase_composition_grid(
        source, phases, "oxide", expression, exclude_phases=("other",)
    )
    np.testing.assert_allclose(values(result), 1.0)
    path = bm.save_property_grid(result, tmp_path / "volume.json.gz")
    loaded = bm.load_property_grid(path)
    assert next(iter(loaded.values)) == f'oxide: {expression}; exclude_phases=["other"]'
    np.testing.assert_array_equal(values(loaded), values(result))
    records = bm.phase_composition_contours(
        loaded, [0.5], diagram=source.context, phases=phases
    )
    assert not records[0]["contours"]["lines"]


def test_exclusions_are_reused_when_evaluating_preexisting_boundary_states(
    volume_sample,
):
    from burnman_cpp.tools.composition_contours import _boundary_expression

    source, phases, _ = volume_sample
    diagram = split_diagram(source, solved_boundary=True)
    for point in diagram["boundaries"][0]["points"]:
        point["phases"][1]["amount"] = 1.0
    polygon = max(
        bm.phase_diagram_field_polygons(diagram).polygons,
        key=lambda p: p.label_position[0],
    )
    quantity = 'oxide: standard_volume_fraction; exclude_phases=["other"]'
    _, boundary = _boundary_expression(diagram, phases, quantity)(polygon)
    np.testing.assert_allclose(boundary, 1.0)


def test_excluding_water_allows_frozen_standard_state_without_evaluating_its_eos(
    sample,
):
    source, phases = sample
    water = bm.minerals.model_sets.metapelite().phases[21]
    phases = [phases[0], water]

    def states(p, t):
        return [
            dict(
                id=0,
                candidate_index=0,
                name="oxide",
                amount=1.0,
                composition=[0.5, 0.5],
            ),
            dict(id=3, candidate_index=1, name="H2O", amount=0.2, composition=[1.0]),
        ]

    source = stored_grid(phases, phase_states=states)
    full = bm.phase_composition_grid(
        source, phases, "oxide", "standard_volume_fraction"
    )
    assert not full.success.any()
    assert any("PitzerSterner domain" in message for message in full.diagnostics)
    dry = bm.phase_composition_grid(
        source, phases, "oxide", "standard_volume_fraction", exclude_phases=("H2O",)
    )
    assert dry.success.all()
    np.testing.assert_allclose(values(dry), 1.0)


@pytest.mark.parametrize(
    "expression,excluded",
    [
        ("volume_fraction", ("unknown",)),
        ("volume_fraction", ("oxide",)),
        ("Mg_A", ("other",)),
    ],
)
def test_invalid_volume_exclusions_are_rejected(volume_sample, expression, excluded):
    source, phases, _ = volume_sample
    with pytest.raises(ValueError):
        bm.phase_composition_grid(
            source, phases, "oxide", expression, exclude_phases=excluded
        )


def split_diagram(grid, *, solved_boundary=False):
    data = grid.context.to_dict()
    samples = [
        dict(
            pressure=p,
            temperature=900.0,
            success=True,
            phases=[
                dict(
                    id=i,
                    candidate_index=i // 3,
                    name=data["phase_names"][i],
                    amount=1.0,
                    composition=[0.25, 0.75] if i == 0 else [1.0],
                )
                for i in ids
            ],
        )
        for p, ids in [(1.1e9, [0]), (1.8e9, [0, 3])]
    ]
    points = []
    for t in np.linspace(800.0, 1200.0, 5):
        point = dict(pressure=1.5e9, temperature=float(t))
        if solved_boundary:
            x = 0.2 + 0.8 * (t - 800.0) / 400.0
            point["phases"] = [
                dict(
                    id=0,
                    candidate_index=0,
                    name="oxide",
                    amount=1.0,
                    composition=[x, 1 - x],
                ),
                dict(
                    id=3,
                    candidate_index=1,
                    name=data["phase_names"][3],
                    amount=0.0,
                    composition=[1.0],
                ),
            ]
        points.append(point)
    data.update(
        samples=samples,
        fields=[
            dict(id=i, phases=ids, sample_indices=[i])
            for i, ids in enumerate([[0], [0, 3]])
        ],
        boundaries=[
            dict(
                id=0,
                zero_phase=3,
                assemblage=[0, 3],
                side_b=[0],
                side_a=[0, 3],
                start_node=-1,
                end_node=-1,
                reaction_kind="phase_out",
                points=points,
            )
        ],
    )
    return data


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("Mg_A/(Mg_A+Fe_A)", lambda x: x),
        ("-Mg_A + +Fe_A*2 - 1/4", lambda x: 1.75 - 3 * x),
        ("Mg/(Mg+Fe)", lambda x: x),
        ("X_MgO + X_FeO", lambda x: np.ones_like(x)),
        ("0.5+2*(Mg_A-Fe_A)/4", lambda x: x),
        ("3/2", lambda x: np.full_like(x, 1.5)),
    ],
)
def test_arithmetic_site_element_and_endmember_symbols(sample, expression, expected):
    source, phases = sample
    snapshot = source.to_dict()

    result = bm.phase_composition_grid(source, phases, "oxide", expression)
    expected_values = np.broadcast_to(expected(np.linspace(0, 1, 5)), (5, 5))
    np.testing.assert_allclose(values(result), expected_values)
    assert result.success.all() and result.equilibrium_solves == 0
    assert not result.has_solutions
    assert source.to_dict() == snapshot
    records = bm.phase_composition_contours(
        result, [0.25, 0.75], diagram=source.context
    )
    assert all(r["contours"]["equilibrium_solves"] == 0 for r in records)


@pytest.mark.parametrize(
    "expression",
    [
        '__import__("os")',
        "Mg_A.real",
        "Mg_A[0]",
        "True",
        '"Mg_A"',
        "Mg_A**2",
        "Mg_A//2",
        "Mg_A%2",
        "Mg_A > 0",
        "1e999",
        "missing",
        "Mg_A +",
    ],
)
def test_invalid_expressions_are_rejected(sample, expression):
    source, phases = sample
    with pytest.raises(ValueError):
        bm.phase_composition_grid(source, phases, "oxide", expression)


def test_failed_absent_and_undefined_nodes_are_masked_independently_of_properties(
    sample,
):
    source, phases = sample
    data = source.to_dict()
    data["solutions"][0][0] = None
    data["equilibrium_success"][0][0] = False
    data["success"][0][0] = False
    state = data["solutions"][1][1]
    state["phases"] = [
        dict(
            id=3, candidate_index=1, name=phases[1].name, amount=1.0, composition=[1.0]
        )
    ]
    state["phase_bases"] = [[[1.0]]]
    state["phase_compositions"] = [[1.0]]
    data["success"][3][3] = False  # An unrelated property failed; equilibrium is valid.
    result = bm.phase_composition_grid(data, phases, "oxide", "1/(2*Mg_A-1)")
    assert np.isnan(values(result)[0, 0]) and np.isnan(values(result)[1, 1])
    assert np.isnan(values(result)[:, 2]).all()
    assert result.success[3, 3]
    assert not result.equilibrium_success[0, 0] and result.equilibrium_success[1, 1]
    assert any("5 composition values" in d for d in result.diagnostics)


def test_reduced_active_face_uses_original_composition(sample):
    source, phases = sample
    data = source.to_dict()
    for row in data["solutions"]:
        state = row[0]
        state["phase_bases"] = [[[0.0, 1.0]]]
        state["phase_compositions"] = [[1.0]]
    result = bm.phase_composition_grid(data, phases, "oxide", "Fe_A")
    np.testing.assert_allclose(values(result)[:, 0], 1.0)


def test_coexisting_copies_require_an_explicit_selection(sample):
    source, phases = sample
    data = source.to_dict()
    for row in data["solutions"]:
        for state in row:
            second = deepcopy(state["phases"][0])
            second.update(id=1, name="oxide #2", composition=[0.8, 0.2])
            state["phases"].append(second)
            state["phase_bases"].append(np.eye(2).tolist())
            state["phase_compositions"].append([0.8, 0.2])
    with pytest.raises(ValueError, match="coexisting copies"):
        bm.phase_composition_grid(data, phases, "oxide", "Mg_A")
    for kwargs in [dict(phase="oxide", instance=2), dict(phase="oxide #2")]:
        result = bm.phase_composition_grid(data, phases, expression="Mg_A", **kwargs)
        np.testing.assert_allclose(values(result), 0.8)
    with pytest.raises(ValueError, match="instance"):
        bm.phase_composition_grid(data, phases, "oxide #2", "Mg_A", instance=1)


def test_persistence_and_chunked_equilibria(sample, tmp_path):
    source, phases = sample
    expected = bm.phase_composition_grid(source, phases, "oxide", "Mg_A")
    path = bm.save_property_grid(source, tmp_path / "source.json.gz")
    np.testing.assert_array_equal(
        values(bm.phase_composition_grid(path, phases, "oxide", "Mg_A")),
        values(expected),
    )
    scalar = bm.save_property_grid(expected, tmp_path / "composition.json.gz")
    assert bm.load_property_grid(scalar).equilibrium_solves == 0
    assert bm.phase_composition_contours(scalar, [0.5], diagram=source.context)[0][
        "contours"
    ]["lines"]
    chunks = []
    for index, (start, end) in enumerate([(0, 2), (2, 5)]):
        data = source.to_dict()
        data["y"] = data["y"][start:end]
        for name in ("field_numbers", "success", "equilibrium_success", "solutions"):
            data[name] = data[name][start:end]
        data["values"] = {k: v[start:end] for k, v in data["values"].items()}
        filename = f"chunk{index}.json.gz"
        bm.save_property_grid(
            bm.PhaseDiagramPropertyGrid.from_dict(data), tmp_path / filename
        )
        chunks.append(dict(file=filename, rows=[start, end], columns=[0, 5]))
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            dict(
                format="burnman_cpp.property_grid_chunks",
                schema_version=1,
                complete=True,
                x=source.x.tolist(),
                y=source.y.tolist(),
                chunks=chunks,
            )
        )
    )
    np.testing.assert_array_equal(
        values(bm.phase_composition_grid(tmp_path, phases, "oxide", "Mg_A")),
        values(expected),
    )
    property_grid = bm.phase_diagram_property_grid(path, phases, "density")
    np.testing.assert_array_equal(
        values(bm.phase_diagram_property_grid(tmp_path, phases, "density")),
        values(property_grid),
    )
    scalar = bm.save_property_grid(property_grid, tmp_path / "density.json.gz")
    assert bm.phase_diagram_grid_contours(
        scalar,
        [float(np.median(values(property_grid)))],
        diagram=source.context,
        phases=phases,
    )[0]["contours"]["lines"]


def test_contours_interpolate_known_values_and_do_not_cross_fields_or_holes(sample):
    source, phases = sample
    data = source.to_dict()
    for j, row in enumerate(data["solutions"]):
        for state in row:
            state["phases"][0]["composition"] = [j / 4.0, 1 - j / 4.0]
    data["field_numbers"] = [[1, 1, 2, 2, 2] for _ in range(5)]
    grid = bm.phase_composition_grid(data, phases, "oxide", "Mg_A")
    diagram = split_diagram(source)
    record = bm.phase_composition_contours(grid, [0.5], diagram=diagram)[0]
    lines = record["contours"]["lines"]
    assert {line["field_number"] for line in lines} == {1, 2}
    for line in lines:
        np.testing.assert_allclose(
            [p["temperature"] for p in line["points"]], 1000.0, atol=1.0e-9
        )
        pressures = [p["pressure"] for p in line["points"]]
        assert (
            max(pressures) if line["field_number"] == 1 else min(pressures)
        ) == pytest.approx(1.5e9)
    hole = grid.to_dict()
    hole["success"][2][3] = False
    lines = bm.phase_composition_contours(
        bm.PhaseDiagramPropertyGrid.from_dict(hole), [0.5], diagram=diagram
    )[0]["contours"]["lines"]
    assert all(line["field_number"] == 1 for line in lines)


def test_preexisting_boundary_equilibria_anchor_the_contour_endpoints(sample):
    source, phases = sample
    data = source.to_dict()
    for j, row in enumerate(data["solutions"]):
        for state in row:
            state["phases"][0]["composition"] = [j / 4.0, 1.0 - j / 4.0]
    data["field_numbers"] = [[1, 1, 2, 2, 2] for _ in range(5)]
    grid = bm.phase_composition_grid(data, phases, "oxide", "Mg_A")
    diagram = split_diagram(source, solved_boundary=True)
    record = bm.phase_composition_contours(grid, [0.5], diagram=diagram, phases=phases)[
        0
    ]
    assert {line["field_number"] for line in record["contours"]["lines"]} == {1, 2}
    for line in record["contours"]["lines"]:
        assert line["preexisting_boundary_samples"] == 5
        endpoints = [line["points"][0], line["points"][-1]]
        boundary = next(p for p in endpoints if p["pressure"] == pytest.approx(1.5e9))
        assert boundary["temperature"] == pytest.approx(950.0, abs=1.0e-6)


@pytest.mark.parametrize("hole", [False, True])
def test_interpolated_lines_reach_polygon_and_hole_edges_without_leaving_them(hole):
    from types import SimpleNamespace
    from shapely.geometry import LineString, Polygon
    from burnman_cpp.tools._grid_contours import field_contours

    outer = np.array([[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8], [0.2, 0.2]])
    inner = np.array(
        [[0.45, 0.45], [0.55, 0.45], [0.55, 0.55], [0.45, 0.55], [0.45, 0.45]]
    )
    polygon = SimpleNamespace(vertices=outer, holes=[inner] if hole else [])
    points = np.stack(np.meshgrid([0.3, 0.5, 0.7], [0.3, 0.5, 0.7]), axis=-1).reshape(
        -1, 2
    )
    if hole:
        points = points[np.any(points != 0.5, axis=1)]
    lines = list(field_contours(points, points[:, 1], polygon, np.empty((0, 2)), [0.5]))
    region = Polygon(outer, polygon.holes)
    assert len(lines) == (2 if hole else 1)
    endpoints = sorted(p[0] for _, segment in lines for p in [segment[0], segment[-1]])
    np.testing.assert_allclose(
        endpoints, [0.2, 0.45, 0.55, 0.8] if hole else [0.2, 0.8], atol=1.0e-12
    )
    for _, segment in lines:
        assert region.buffer(1.0e-12).covers(LineString(segment))


def test_invalid_ring_spikes_do_not_extend_the_contour_outside_the_field():
    from types import SimpleNamespace
    from burnman_cpp.tools._grid_contours import field_contours

    ring = np.array(
        [
            [0.2, 0.2],
            [0.8, 0.2],
            [0.8, 0.8],
            [0.9, 0.9],
            [0.8, 0.8],
            [0.2, 0.8],
            [0.2, 0.2],
        ]
    )
    polygon = SimpleNamespace(vertices=ring, holes=[])
    points = np.stack(np.meshgrid([0.3, 0.5, 0.7], [0.3, 0.5, 0.7]), axis=-1).reshape(
        -1, 2
    )
    lines = list(
        field_contours(
            points, points[:, 1] - points[:, 0], polygon, np.empty((0, 2)), [0.0]
        )
    )
    assert lines
    for _, segment in lines:
        assert segment.min() >= 0.2 - 1.0e-12 and segment.max() <= 0.8 + 1.0e-12


def test_concave_field_splits_contours_at_its_actual_edges():
    from types import SimpleNamespace
    from shapely.geometry import Polygon, Point
    from burnman_cpp.tools._grid_contours import field_contours

    ring = np.array(
        [
            [0.1, 0.1],
            [0.9, 0.1],
            [0.9, 0.9],
            [0.7, 0.9],
            [0.7, 0.3],
            [0.3, 0.3],
            [0.3, 0.9],
            [0.1, 0.9],
            [0.1, 0.1],
        ]
    )
    polygon = SimpleNamespace(vertices=ring, holes=[])
    region = Polygon(ring)
    points = np.stack(
        np.meshgrid([0.2, 0.4, 0.6, 0.8], [0.2, 0.4, 0.6, 0.8]), axis=-1
    ).reshape(-1, 2)
    points = points[[region.covers(Point(p)) for p in points]]
    lines = list(field_contours(points, points[:, 1], polygon, np.empty((0, 2)), [0.5]))
    endpoints = sorted(p[0] for _, segment in lines for p in [segment[0], segment[-1]])
    np.testing.assert_allclose(endpoints, [0.1, 0.3, 0.7, 0.9], atol=1.0e-12)


def test_non_pt_contours_keep_native_coordinate_names(sample):
    source, phases = sample
    data = source.to_dict()
    data["context"].update(diagram_type="PS", entropy_range=[800.0, 1200.0])
    source = bm.PhaseDiagramPropertyGrid.from_dict(data)
    grid = bm.phase_composition_grid(source, phases, "oxide", "Mg_A")
    records = bm.phase_composition_contours(grid, [0.5], diagram=source.context)
    points = [p for line in records[0]["contours"]["lines"] for p in line["points"]]
    assert points and all(set(p) == {"pressure", "entropy"} for p in points)
    np.testing.assert_allclose([p["pressure"] for p in points], 1.5e9)


def test_secondary_labels_follow_compositions_when_grid_copy_ids_swap():
    from test_phase_diagram_labels import moving_solvus

    diagram = moving_solvus()
    phases = [
        bm.Solution(make_model(), [0.5, 0.5], name="hb"),
        bm.Mineral(oxide_params()),
    ]
    phases[1].set_name("q")
    x = np.array([0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9])
    y = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

    def states(p, t):
        roots = [0] if p < 0.3 else ([1] if p > 0.7 else [0, 1])
        records = [
            dict(
                id=i,
                candidate_index=0,
                name="hb",
                amount=1.0,
                composition=[
                    0.05 + 0.15 * root + 0.6 * t,
                    0.95 - 0.15 * root - 0.6 * t,
                ],
            )
            for i, root in enumerate(roots)
        ]
        if len(records) == 2 and t > 0.5:
            records[0]["id"], records[1]["id"] = records[1]["id"], records[0]["id"]
        records.append(
            dict(id=3, candidate_index=1, name="q", amount=1.0, composition=[1.0])
        )
        return records

    source = stored_grid(phases, diagram=diagram, x=x, y=y, phase_states=states)
    polygons = sorted(
        bm.phase_diagram_field_polygons(diagram).polygons,
        key=lambda p: p.label_position[0],
    )
    data = source.to_dict()
    data["field_numbers"] = [
        [polygons[0 if p < 0.3 else (2 if p > 0.7 else 1)].field_number for p in x]
        for _ in y
    ]
    for label, root, columns in [("hb_1", 0, slice(0, 5)), ("hb_2", 1, slice(2, None))]:
        result = bm.phase_composition_grid(data, phases, label, "Mg_A", diagram=diagram)
        np.testing.assert_allclose(
            values(result)[:, columns],
            np.broadcast_to(
                (0.05 + 0.15 * root + 0.6 * y)[:, None],
                values(result)[:, columns].shape,
            ),
        )
        assert result.equilibrium_solves == 0
        property_grid = bm.phase_diagram_property_grid(
            data, phases, "density", phase=label, diagram=diagram
        )
        expected_density = np.full((len(y), len(x)), np.nan)
        reference = phases[0].copy()
        for j, t in enumerate(y):
            composition = 0.05 + 0.15 * root + 0.6 * t
            reference.set_composition([composition, 1.0 - composition])
            for i in range(len(x))[columns]:
                reference.set_state(x[i], t)
                expected_density[j, i] = reference.density
        np.testing.assert_allclose(values(property_grid), expected_density)
        from burnman_cpp.tools.composition_contours import _boundary_expression

        sample_boundary = _boundary_expression(
            diagram, phases, next(iter(property_grid.values))
        )
        for polygon in polygons:
            points, boundary_values = sample_boundary(polygon)
            if polygon is polygons[2 if root == 0 else 0]:
                assert not np.isfinite(boundary_values).any()
                continue
            for (p, t), value in zip(points, boundary_values):
                composition = 0.05 + 0.15 * root + 0.6 * t
                reference.set_composition([composition, 1.0 - composition])
                reference.set_state(p, t)
                assert value == pytest.approx(reference.density)


@pytest.mark.parametrize(
    "attribute", ["density", "molar_entropy", "n_phases", "n_moles"]
)
def test_attribute_grids_match_direct_assemblage_getters(volume_sample, attribute):
    source, phases, _ = volume_sample
    snapshot = source.to_dict()
    phases[0].set_state(3e9, 1500.0)
    phases[0].set_composition([0.4, 0.6])
    expected = np.empty(source.success.shape)
    for j, row in enumerate(snapshot["solutions"]):
        for i, state in enumerate(row):
            materials = []
            amounts = np.array([p["amount"] for p in state["phases"]])
            for p in state["phases"]:
                material = phases[p["candidate_index"]].copy()
                if isinstance(material, bm.Solution):
                    material.set_composition(p["composition"])
                materials.append(material)
            assemblage = bm.Assemblage(materials, amounts / amounts.sum())
            assemblage.n_moles = float(amounts.sum())
            assemblage.set_state(state["pressure"], state["temperature"])
            expected[j, i] = getattr(assemblage, attribute)
    result = bm.phase_diagram_property_grid(source, phases, attribute)
    np.testing.assert_allclose(values(result), expected, rtol=1e-12)
    assert result.equilibrium_solves == 0 and not result.has_solutions
    assert source.to_dict() == snapshot
    np.testing.assert_array_equal(phases[0].molar_fractions, [0.4, 0.6])
    assert (phases[0].pressure, phases[0].temperature) == (3e9, 1500.0)


def test_phase_attributes_select_copies_and_mask_absent_or_failed_nodes(sample):
    source, phases = sample
    data = source.to_dict()
    for row in data["solutions"]:
        for state in row:
            state["phases"].append(
                dict(
                    id=1,
                    candidate_index=0,
                    name="oxide #2",
                    amount=0.5,
                    composition=[0.8, 0.2],
                )
            )
            state["phase_bases"].append(np.eye(2).tolist())
            state["phase_compositions"].append([0.8, 0.2])
    with pytest.raises(TypeError, match="coexisting copies"):
        bm.phase_diagram_property_grid(data, phases, "density", phase="oxide")
    for options in [dict(phase="oxide #2"), dict(phase="oxide", instance=2)]:
        result = bm.phase_diagram_property_grid(data, phases, "density", **options)
        reference = phases[0].copy()
        reference.set_composition([0.8, 0.2])
        for j, t in enumerate(source.y):
            for i, p in enumerate(source.x):
                reference.set_state(p, t)
                assert values(result)[j, i] == pytest.approx(reference.density)
    data["success"][0][0] = data["equilibrium_success"][0][0] = False
    data["solutions"][0][0] = None
    data["solutions"][1][1]["phases"].pop()
    data["solutions"][1][1]["phase_bases"].pop()
    data["solutions"][1][1]["phase_compositions"].pop()
    result = bm.phase_diagram_property_grid(data, phases, "density", phase="oxide #2")
    assert np.isnan(values(result)[0, 0]) and np.isnan(values(result)[1, 1])
    assert not result.equilibrium_success[0, 0] and result.equilibrium_success[1, 1]


def test_property_boundary_values_replay_saved_states_and_zero_amount_phases(
    volume_sample,
):
    from burnman_cpp.tools.composition_contours import _boundary_expression

    source, phases, _ = volume_sample
    diagram = split_diagram(source, solved_boundary=True)
    polygon = next(
        p for p in bm.phase_diagram_field_polygons(diagram).polygons if 3 in p.phases
    )
    for phase in (None, "other"):
        grid = bm.phase_diagram_property_grid(source, phases, "density", phase=phase)
        points, boundary_values = _boundary_expression(
            diagram, phases, next(iter(grid.values))
        )(polygon)
        for (p, t), value in zip(points, boundary_values):
            if phase is None:
                x = 0.2 + 0.8 * (t - 800.0) / 400.0
                reference = phases[0].copy()
                reference.set_composition([x, 1.0 - x])
            else:
                reference = phases[1].copy()
            reference.set_state(p, t)
            assert value == pytest.approx(reference.density)
        assert len(points) == 5
    # A temperature attribute has straight contours ending at the shared field edge.
    data = source.to_dict()
    polygons = sorted(
        bm.phase_diagram_field_polygons(diagram).polygons,
        key=lambda p: p.label_position[0],
    )
    data["field_numbers"] = [
        [polygons[int(p > 1.5e9)].field_number for p in source.x] for _ in source.y
    ]
    grid = bm.phase_diagram_property_grid(data, phases, "temperature")
    records = bm.phase_diagram_grid_contours(
        grid, [950.0], diagram=diagram, phases=phases
    )
    assert len(records[0]["contours"]["lines"]) == 2
    for line in records[0]["contours"]["lines"]:
        assert line["preexisting_boundary_samples"] == 5
        assert any(p["pressure"] == pytest.approx(1.5e9) for p in line["points"])
        assert all(p["temperature"] == pytest.approx(950.0) for p in line["points"])


def test_attribute_grid_restores_reduced_bases_for_relaxed_properties(sample):
    source, phases = sample
    data = source.to_dict()
    for row in data["solutions"]:
        for state in row:
            state["phases"][0]["composition"] = [0.0, 1.0]
            state["phase_bases"] = [[[0.0, 1.0]]]
            state["phase_compositions"] = [[1.0]]
    result = bm.phase_diagram_property_grid(
        data, phases, "molar_heat_capacity_p", relaxed=True
    )
    reference = bm.Mineral(oxide_params("Fe"))
    for j, t in enumerate(source.y):
        for i, p in enumerate(source.x):
            reference.set_state(p, t)
            assert values(result)[j, i] == pytest.approx(
                reference.molar_heat_capacity_p
            )
    assert result.success.all(), result.diagnostics


def test_nonfinite_attributes_are_masked_with_diagnostics(sample, monkeypatch):
    source, phases = sample
    monkeypatch.setattr(bm.Assemblage, "density", property(lambda self: np.nan))
    result = bm.phase_diagram_property_grid(source, phases, "density")
    assert result.equilibrium_success.all() and not result.success.any()
    assert np.isnan(values(result)).all() and not result.resolved
    assert any("nonfinite" in d for d in result.diagnostics)


@pytest.mark.parametrize(
    "attribute,options,error",
    [
        ("__dict__", {}, ValueError),
        ("set_state", {}, ValueError),
        ("density.real", {}, ValueError),
        ("unknown", {}, ValueError),
        ("formula", {}, TypeError),
        ("molar_fractions", {}, TypeError),
        ("density", dict(instance=1), ValueError),
        ("density", dict(phase="oxide", instance=4), ValueError),
        ("density", dict(phase="oxide", instance=True), ValueError),
        ("density", dict(phase="oxide #2", instance=1), ValueError),
        ("density", dict(phase="unknown"), ValueError),
        ("density", dict(relaxed="yes"), TypeError),
        ("density", dict(averaging_scheme="VRH"), TypeError),
        ("n_phases", dict(relaxed=True), ValueError),
    ],
)
def test_attribute_grid_rejects_invalid_or_nonscalar_properties(
    sample, attribute, options, error
):
    source, phases = sample
    with pytest.raises(error):
        bm.phase_diagram_property_grid(source, phases, attribute, **options)


def test_missing_retained_states_and_wrong_candidates_are_rejected(sample):
    source, phases = sample
    with pytest.raises(ValueError, match="original order"):
        bm.phase_composition_grid(source, phases[::-1], "oxide", "Mg_A")
    data = source.to_dict()
    data["solutions"] = None
    with pytest.raises(ValueError, match="retain equilibria"):
        bm.phase_composition_grid(data, phases, "oxide", "Mg_A")


def test_contour_plotting_uses_existing_display_transform(sample):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source, phases = sample
    result = bm.phase_composition_grid(source, phases, "oxide", "Mg_A")
    record = bm.phase_composition_contours(result, [0.5], diagram=source.context)[0]
    figure, axes = plt.subplots()
    bm.plot_phase_diagram_contours(
        source.context,
        record["contours"],
        axes,
        pressure_unit="kbar",
        temperature_unit="C",
        value=0.5,
        inline_labels=False,
    )
    vertices = axes.collections[0].get_paths()[0].vertices
    np.testing.assert_allclose(vertices[:, 1], 15.0)
    np.testing.assert_allclose(
        [vertices[:, 0].min(), vertices[:, 0].max()], [526.85, 926.85]
    )
    plt.close(figure)
