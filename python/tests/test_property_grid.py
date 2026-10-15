# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Analytic reaction responses checked against independent equilibrium solves."""

from pathlib import Path
import runpy

import numpy as np
import pytest

import burnman_cpp as bm
from conftest import oxide_params


def exchange_phases():
    phases = []
    for k, shifts in enumerate(((0.0, 12000.0), (12000.0, 0.0))):
        endmembers = []
        for element, shift in zip(("Mg", "Fe"), shifts):
            params = oxide_params(element, shift)
            params["V_0"] *= 1.0 if k == 0 else 0.94
            params["Debye_0"] *= 1.0 if k == 0 else 1.1
            endmembers.append((bm.Mineral(params), f"[{element}]O"))
        phases.append(
            bm.Solution(
                bm.IdealSolution(endmembers),
                [0.7, 0.3] if k == 0 else [0.3, 0.7],
                name=f"phase{k}",
            )
        )
    return phases


def equilibrated(p=4.0e9, t=1000.0, scale=1.0, doubled=False):
    phases = exchange_phases()
    if doubled:
        model = bm.IdealSolution(
            [
                (
                    bm.CombinedMineral(
                        [bm.Mineral(oxide_params(e, shift))], [2.0], name=e + "2"
                    ),
                    f"[{e}]O",
                )
                for e, shift in (("Mg", 12000.0), ("Fe", 0.0))
            ]
        )
        phases[1] = bm.Solution(model, [0.3, 0.7], name="double")
    a = bm.Assemblage(phases, [0.5, 0.5])
    a.n_moles = scale
    a.set_state(p, t)
    result = bm.equilibrate(
        {"Mg": 0.5 * scale, "Fe": 0.5 * scale, "O": scale},
        a,
        [bm.PressureConstraint(p), bm.TemperatureConstraint(t)],
        tol=1.0e-10,
    )
    assert result.sol_array.flat[0].success, result.sol_array.flat[0].message
    return a


def test_relaxed_response_matches_fixed_bulk_finite_differences():
    a = equilibrated()
    names = bm.thermodynamic_property_names()
    relaxed = bm.assemblage_properties(a, names, relaxed=True)
    frozen = bm.assemblage_properties(a, names, relaxed=False)
    p, t, dt, dp = a.pressure, a.temperature, 0.02, 2.0e4
    hot, cold = equilibrated(p, t + dt), equilibrated(p, t - dt)
    high, low = equilibrated(p + dp, t), equilibrated(p - dp, t)
    volume = a.n_moles * a.molar_volume
    dvdt = (hot.n_moles * hot.molar_volume - cold.n_moles * cold.molar_volume) / (
        2 * dt
    )
    dsdt = (hot.n_moles * hot.molar_entropy - cold.n_moles * cold.molar_entropy) / (
        2 * dt
    )
    dvdp = (high.n_moles * high.molar_volume - low.n_moles * low.molar_volume) / (
        2 * dp
    )
    assert relaxed["thermal_expansivity"] == pytest.approx(dvdt / volume, rel=2e-6)
    assert relaxed["molar_heat_capacity_p"] == pytest.approx(
        t * dsdt / a.n_moles, rel=2e-6
    )
    assert relaxed["isothermal_compressibility_reuss"] == pytest.approx(
        -dvdp / volume, rel=2e-6
    )
    assert relaxed["molar_heat_capacity_p"] > frozen["molar_heat_capacity_p"]
    assert (
        relaxed["isothermal_compressibility_reuss"]
        > frozen["isothermal_compressibility_reuss"]
    )
    scaled = bm.assemblage_properties(equilibrated(scale=7.0), names, relaxed=True)
    np.testing.assert_allclose(list(scaled.values()), list(relaxed.values()), rtol=1e-8)
    alpha, cp, kt, v = [
        relaxed[n]
        for n in (
            "thermal_expansivity",
            "molar_heat_capacity_p",
            "isothermal_bulk_modulus_reuss",
            "molar_volume",
        )
    ]
    cv = cp - t * v * alpha**2 * kt
    assert relaxed["molar_heat_capacity_v"] == pytest.approx(cv)
    assert relaxed["isentropic_bulk_modulus_reuss"] == pytest.approx(kt * cp / cv)
    for name in ("density", "molar_gibbs", "molar_entropy", "molar_enthalpy"):
        assert relaxed[name] == frozen[name]


def test_response_uses_extensive_derivatives_when_reactions_change_total_moles():
    a = equilibrated(doubled=True)
    dt = 0.02
    hot = equilibrated(t=1000.0 + dt, doubled=True)
    cold = equilibrated(t=1000.0 - dt, doubled=True)
    assert abs(hot.n_moles - cold.n_moles) > 1e-8
    expected = (
        a.temperature
        * (hot.n_moles * hot.molar_entropy - cold.n_moles * cold.molar_entropy)
        / (2 * dt * a.n_moles)
    )
    cp = bm.assemblage_properties(a, ["molar_heat_capacity_p"])["molar_heat_capacity_p"]
    assert cp == pytest.approx(expected, rel=2e-6)


@pytest.fixture
def single_phase():
    phase = bm.Mineral(oxide_params())
    settings = bm.PhaseDiagramSettings()
    settings.pressure_seeds = settings.temperature_seeds = 2
    settings.max_phase_instances = 1
    result = bm.phase_diagram(
        {"Mg": 1.0, "O": 1.0}, [phase], (3e9, 5e9), (800.0, 1200.0), settings
    )
    return result, phase


@pytest.mark.parametrize("relaxed", [False, True])
def test_grid_all_scalar_properties_and_saved_inputs(single_phase, relaxed, tmp_path):
    source, phase = single_phase
    before = source.to_dict()
    phase.set_state(2e9, 700.0)
    path = tmp_path / "diagram.json"
    bm.save_phase_diagram(source, path)
    names = bm.thermodynamic_property_names()
    grid = bm.phase_diagram_grid(
        path, [phase], names, resolution=(3, 4), relaxed=relaxed
    )
    assert grid.resolved, grid.diagnostics
    assert grid.success.shape == (4, 3)
    assert grid.success.dtype == np.bool_
    assert grid.success.all()
    assert (grid.field_numbers == 1).all()
    assert grid.equilibrium_solves >= 12
    assert source.to_dict() == before
    assert bm.load_phase_diagram(path) == before
    assert (phase.pressure, phase.temperature) == (2e9, 700.0)
    assert set(grid.values) == set(names)
    for j, t in enumerate(grid.y):
        for i, p in enumerate(grid.x):
            phase.set_state(p, t)
            for name in names:
                assert grid.values[name][j, i] == pytest.approx(
                    getattr(phase, name), rel=1e-9
                )


def test_grid_matches_direct_assemblage_response():
    phases = exchange_phases()
    options = bm.PhaseDiagramSettings()
    options.pressure_seeds = options.temperature_seeds = 2
    options.max_phase_instances = 1
    source = bm.phase_diagram(
        {"Mg": 0.5, "Fe": 0.5, "O": 1.0},
        phases,
        (3.9e9, 4.1e9),
        (999.0, 1001.0),
        options,
    )
    names = [
        "thermal_expansivity",
        "molar_heat_capacity_p",
        "isothermal_bulk_modulus_reuss",
    ]
    for relaxed in (False, True):
        grid = bm.phase_diagram_grid(
            source, phases, names, x=[4e9], y=[1000.0], relaxed=relaxed
        )
        assert grid.resolved, grid.diagnostics
        expected = bm.assemblage_properties(equilibrated(), names, relaxed=relaxed)
        for name in names:
            assert grid.values[name][0, 0] == pytest.approx(expected[name], rel=1e-8)


def test_excluded_holes_remain_nan(single_phase):
    source, phase = single_phase
    saved = source.to_dict()
    saved["excluded_regions"] = [
        [
            [3.8e9, 950.0],
            [4.2e9, 950.0],
            [4.2e9, 1050.0],
            [3.8e9, 1050.0],
            [3.8e9, 950.0],
        ]
    ]
    grid = bm.phase_diagram_grid(saved, [phase], "density", resolution=(5, 5))
    assert np.isnan(grid.values["density"][2, 2])
    assert grid.field_numbers[2, 2] == 0
    assert grid.success[0, 0]


def test_singular_relaxation_preserves_state_properties():
    a = bm.Mineral(oxide_params())
    b = bm.CombinedMineral([a], [1.0], [5000.0, 0.0, -1e-6], name="dense")
    composite = bm.Assemblage([a, b], [0.5, 0.5])
    composite.set_state(5e9, 1000.0)
    assert np.isfinite(bm.assemblage_properties(composite, ["density"])["density"])
    with pytest.raises(RuntimeError, match="Singular reaction Hessian"):
        bm.assemblage_properties(composite, ["thermal_expansivity"])
    assert np.isfinite(
        bm.assemblage_properties(composite, ["thermal_expansivity"], relaxed=False)[
            "thermal_expansivity"
        ]
    )


def test_grid_singular_coexistence_and_negative_curvature():
    first = bm.Mineral(oxide_params())
    dense = bm.CombinedMineral([first], [1.0], [5000.0, 0.0, -1e-6], name="dense")
    first.set_state(5e9, 1000.0)
    volume = first.molar_volume - 0.5e-6
    settings = bm.PhaseDiagramSettings()
    settings.temperature_seeds = settings.volume_seeds = 2
    settings.max_phase_instances = 1
    source = bm.phase_diagram(
        {"Mg": 1.0, "O": 1.0},
        [first, dense],
        (4e9, 6e9),
        (999.0, 1001.0),
        settings,
        diagram="TV",
        volume_range=(volume - 1e-7, volume + 1e-7),
    )
    for relaxed in (False, True):
        grid = bm.phase_diagram_grid(
            source,
            [first, dense],
            ["density", "thermal_expansivity"],
            x=[1000.0],
            y=[volume],
            relaxed=relaxed,
            store_solutions=True,
        )
        assert np.isfinite(grid.values["density"][0, 0])
        if relaxed:
            assert np.isnan(grid.values["thermal_expansivity"][0, 0])
            assert any(
                "Singular reaction Hessian" in message for message in grid.diagnostics
            )
            assert grid.equilibrium_success.all()
            recovered = bm.evaluate_property_grid(grid, [first, dense], "density")
            assert recovered.success.all(), recovered.diagnostics
            assert recovered.resolved
            assert recovered.equilibrium_solves == 0
            frozen = bm.evaluate_property_grid(
                grid, [first, dense], "thermal_expansivity", relaxed=False
            )
            assert frozen.success.all(), frozen.diagnostics
        else:
            assert grid.success.all(), grid.diagnostics

    # A stationary solution with negative ordering curvature must not return
    # a spurious finite equilibrium response from an indefinite inverse.
    model = bm.SymmetricRegularSolution(
        [(bm.Mineral(oxide_params()), "[Mg]O"), (bm.Mineral(oxide_params()), "[Fe]O")],
        energy_interaction=[[30000.0]],
    )
    solution = bm.Solution(model, [0.5, 0.5])
    composite = bm.Assemblage([solution], [1.0])
    composite.set_state(4e9, 500.0)
    with pytest.raises(RuntimeError, match="negative Gibbs curvature"):
        bm.assemblage_properties(composite, ["thermal_expansivity"])


def test_grid_input_validation(single_phase):
    source, phase = single_phase
    for kwargs, message in [
        (dict(resolution=(1, 2)), "resolution"),
        (dict(x=[4e9]), "both x and y"),
        (dict(x=[4e9, 3e9], y=[1000.0]), "increasing"),
        (dict(x=[6e9], y=[1000.0]), "inside the diagram"),
        (dict(x=[np.nan], y=[1000.0]), "finite"),
    ]:
        with pytest.raises(ValueError, match=message):
            bm.phase_diagram_grid(source, [phase], "density", **kwargs)
    with pytest.raises(ValueError, match="Unknown scalar"):
        bm.phase_diagram_grid(source, [phase], "not_a_property")
    with pytest.raises(ValueError, match="candidate names/order"):
        bm.phase_diagram_grid(source, [], "density")


@pytest.mark.parametrize(
    "diagram", ["PT", "PS", "PV", "TS", "TV", "SV", "PX", "TX", "SX", "VX"]
)
def test_grid_uses_all_native_section_coordinates(diagram):
    phase = bm.Mineral(oxide_params())
    phase.set_state(4e9, 1000.0)
    ranges = dict(
        P=(3.9e9, 4.1e9),
        T=(999.0, 1001.0),
        S=(phase.molar_entropy - 0.05, phase.molar_entropy + 0.05),
        V=(phase.molar_volume - 1e-9, phase.molar_volume + 1e-9),
        X=(0.0, 1.0),
    )
    names = dict(
        P="pressure", T="temperature", S="entropy", V="volume", X="composition"
    )
    options = {names[c] + "_range": ranges[c] for c in diagram}
    if "X" in diagram:
        options["composition_end"] = {"Mg": 1.001, "O": 1.001}
        options["temperature" if diagram[0] != "T" else "pressure"] = (
            1000.0 if diagram[0] != "T" else 4e9
        )
    if "P" not in diagram and "pressure" not in options:
        options["pressure_range"] = (0.0, 8e9)
    if "T" not in diagram and "temperature" not in options:
        options["temperature_range"] = (500.0, 1500.0)
    settings = bm.PhaseDiagramSettings()
    settings.pressure_seeds = settings.temperature_seeds = settings.entropy_seeds = (
        settings.volume_seeds
    ) = settings.composition_seeds = 2
    settings.max_phase_instances = 1
    source = bm.phase_diagram(
        {"Mg": 1.0, "O": 1.0}, [phase], settings=settings, diagram=diagram, **options
    )
    grid = bm.phase_diagram_grid(
        source,
        [phase],
        ["pressure", "temperature", "molar_entropy", "molar_volume"],
        resolution=(3, 3),
        store_solutions=True,
    )
    assert grid.resolved, grid.diagnostics
    assert grid.success.all()
    restored = bm.PhaseDiagramPropertyGrid.from_dict(grid.to_dict())
    replay = bm.evaluate_property_grid(restored, [phase], list(grid.values))
    assert replay.equilibrium_solves == 0
    assert replay.success.all(), replay.diagnostics
    for name in grid.values:
        np.testing.assert_allclose(replay.values[name], grid.values[name], rtol=1e-10)
    for j, y in enumerate(grid.y):
        for i, x in enumerate(grid.x):
            phase.set_state(
                grid.values["pressure"][j, i], grid.values["temperature"][j, i]
            )
            for c, target in zip(diagram, (x, y)):
                amount = 1 + 0.001 * y if "X" in diagram else 1.0
                actual = dict(
                    P=phase.pressure,
                    T=phase.temperature,
                    S=amount * phase.molar_entropy,
                    V=amount * phase.molar_volume,
                    X=y,
                )[c]
                assert actual == pytest.approx(target, rel=1e-8)


def test_figure4_example_plot_and_grid(single_phase, monkeypatch, tmp_path):
    import matplotlib.pyplot as plt

    source, phase = single_phase
    example = runpy.run_path(
        str(
            Path(__file__).resolve().parents[2]
            / "examples/example_pyrolite_expansivity.py"
        )
    )
    monkeypatch.setattr(example["pyrolite"], "candidate_phases", lambda: [phase])
    grids = example["calculate_grids"](source, resolution=(4, 3))
    fig, axes = example["plot"](source, grids)
    assert axes[0].get_xlim() == (3.0, 5.0)
    assert axes[0].get_ylim() == (800.0, 1200.0)
    assert grids["relaxed"].values["thermal_expansivity"].shape == (3, 4)
    fig.savefig(tmp_path / "figure4.png")
    assert (tmp_path / "figure4.png").stat().st_size > 1000
    plt.close(fig)


def test_figure4_isentropes_use_saved_total_entropy_without_solves(
    single_phase, monkeypatch, tmp_path
):
    _, phase = single_phase
    settings = bm.PhaseDiagramSettings()
    settings.pressure_seeds = settings.temperature_seeds = 2
    settings.max_phase_instances = 1
    diagram = bm.phase_diagram(
        {"Mg": 2.0, "O": 2.0}, [phase], (0.0, 2e9), (800.0, 1600.0), settings
    )
    grid = bm.phase_diagram_grid(
        diagram, [phase], "density", resolution=(9, 17), store_solutions=True
    )
    assert grid.equilibrium_success.all(), grid.diagnostics
    grid = bm.load_property_grid(bm.save_property_grid(grid, tmp_path / "grid.json.gz"))
    example = runpy.run_path(
        str(
            Path(__file__).resolve().parents[2]
            / "examples/example_pyrolite_expansivity.py"
        )
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("Isentropes must use saved grid entropy only.")

    monkeypatch.setattr(bm, "stable_equilibrium", unexpected)
    monkeypatch.setattr(bm, "phase_diagram_contours", unexpected)
    monkeypatch.setattr(bm, "phase_diagram_grid", unexpected)
    monkeypatch.setattr(bm, "evaluate_property_grid", unexpected)
    records = example["calculate_isentropes"](grid, (1000.0, 1025.0, 700.0))
    assert records[0]["entropy"] == pytest.approx(grid.solution_at(4, 0).state.entropy)
    expected = (
        grid.solution_at(4, 0).state.entropy + grid.solution_at(5, 0).state.entropy
    ) / 2
    assert records[1]["entropy"] == pytest.approx(expected)
    for record in records[:2]:
        assert record["lines"] and not record["diagnostics"]
        starts = [
            point
            for line in record["lines"]
            for point in line["points"]
            if point["pressure"] == 0.0
        ]
        assert any(
            p["temperature"] == pytest.approx(record["potential_temperature"])
            for p in starts
        )
        for line in record["lines"]:
            for point in line["points"]:
                phase.set_state(point["pressure"], point["temperature"])
                assert 2 * phase.molar_entropy == pytest.approx(
                    record["entropy"], rel=3e-4
                )
    assert not records[2]["lines"]
    assert "outside the saved grid" in records[2]["diagnostics"][0]


def test_figure4_isentropes_mask_missing_saved_entropy():
    from types import SimpleNamespace

    example = runpy.run_path(
        str(
            Path(__file__).resolve().parents[2]
            / "examples/example_pyrolite_expansivity.py"
        )
    )
    # A gap separates two valid contour segments. Property failures must not
    # hide accepted equilibria, while missing equilibria must break the line.
    x, y = np.arange(7.0), np.array([800.0, 1000.0, 1200.0])
    valid = np.ones((len(y), len(x)), dtype=bool)
    valid[:, 3] = False

    def solution_at(row, col):
        assert valid[row, col]
        return SimpleNamespace(state=SimpleNamespace(entropy=y[row] - 10.0 * x[col]))

    grid = SimpleNamespace(
        diagram_type="PT",
        has_solutions=True,
        x=x,
        y=y,
        equilibrium_success=valid,
        success=np.zeros_like(valid),
        solution_at=solution_at,
    )
    record = example["calculate_isentropes"](grid, (900.0,))[0]
    assert len(record["lines"]) == 2
    assert not record["diagnostics"]
    for line in record["lines"]:
        p = np.array([point["pressure"] for point in line["points"]])
        assert (p <= 2).all() or (p >= 4).all()
    valid[1, 0] = False
    failed = example["calculate_isentropes"](grid, (900.0,))[0]
    assert not failed["lines"]
    assert "entropy is unavailable" in failed["diagnostics"][0]


@pytest.mark.parametrize("suffix", [".json", ".json.gz"])
def test_saved_solutions_round_trip_and_add_properties(single_phase, tmp_path, suffix):
    import json
    import gzip

    source, phase = single_phase
    grid = bm.phase_diagram_grid(
        source, [phase], "density", resolution=(3, 4), store_solutions=True
    )
    assert grid.has_solutions
    path = bm.save_property_grid(grid, tmp_path / ("grid" + suffix))
    loaded = bm.load_property_grid(path)
    assert loaded.to_dict() == grid.to_dict()
    state = loaded.solution_at(1, 2).state
    assert state.pressure == pytest.approx(grid.x[2])
    assert state.temperature == pytest.approx(grid.y[1])
    assert state.phases[0].amount == pytest.approx(1.0)
    names = [
        "thermal_expansivity",
        "molar_heat_capacity_p",
        "isothermal_bulk_modulus_reuss",
    ]
    for relaxed in (True, False):
        replay = bm.evaluate_property_grid(path, [phase], names, relaxed=relaxed)
        direct = bm.phase_diagram_grid(
            source, [phase], names, resolution=(3, 4), relaxed=relaxed
        )
        assert replay.equilibrium_solves == 0
        assert replay.success.all(), replay.diagnostics
        for name in names:
            np.testing.assert_allclose(
                replay.values[name], direct.values[name], rtol=1e-11
            )
    assert set(grid.values) == {"density"}
    opener = gzip.open if suffix.endswith("gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        data = json.load(stream)
    assert data["schema_version"] == 1
    assert data["context"]["samples"] == []


def test_saved_solutions_preserve_excluded_points(single_phase, tmp_path):
    source, phase = single_phase
    saved = source.to_dict()
    saved["excluded_regions"] = [
        [
            [3.8e9, 950.0],
            [4.2e9, 950.0],
            [4.2e9, 1050.0],
            [3.8e9, 1050.0],
            [3.8e9, 950.0],
        ]
    ]
    grid = bm.phase_diagram_grid(
        saved, [phase], "density", resolution=(5, 5), store_solutions=True
    )
    path = bm.save_property_grid(grid, tmp_path / "masked.json.gz")
    loaded = bm.load_property_grid(path)
    assert loaded.solution_at(2, 2) is None
    assert not loaded.equilibrium_success[2, 2]
    replay = bm.evaluate_property_grid(loaded, [phase], "molar_heat_capacity_p")
    assert np.isnan(replay.values["molar_heat_capacity_p"][2, 2])
    assert replay.equilibrium_solves == 0
    np.testing.assert_array_equal(replay.field_numbers, grid.field_numbers)
    np.testing.assert_array_equal(replay.equilibrium_success, grid.equilibrium_success)


def test_saved_solution_active_faces_and_multiple_instances(tmp_path):
    from conftest import make_model

    settings = bm.PhaseDiagramSettings()
    settings.pressure_seeds = settings.temperature_seeds = 2
    phase = bm.Solution(make_model(), [1.0, 0.0], name="oxide")
    source = bm.phase_diagram(
        {"Mg": 1.0, "O": 1.0}, [phase], (3e9, 5e9), (800.0, 1200.0), settings
    )
    grid = bm.phase_diagram_grid(
        source,
        [phase],
        ["density", "molar_entropy", "thermal_expansivity"],
        resolution=(3, 3),
        store_solutions=True,
    )
    solution = grid.solution_at(1, 1)
    assert solution.phase_bases[0].shape == (1, 2)
    np.testing.assert_array_equal(solution.phase_compositions[0], [1.0])
    loaded = bm.load_property_grid(
        bm.save_property_grid(grid, tmp_path / "face.json.gz")
    )
    replay = bm.evaluate_property_grid(loaded, [phase], list(grid.values))
    assert replay.success.all(), replay.diagnostics
    for name in grid.values:
        np.testing.assert_allclose(replay.values[name], grid.values[name], rtol=1e-11)

    model = make_model(bm.SymmetricRegularSolution, energy_interaction=[[30000.0]])
    phase = bm.Solution(model, [0.5, 0.5], name="oxide")
    source = bm.phase_diagram(
        {"Mg": 0.5, "Fe": 0.5, "O": 1.0}, [phase], (3e9, 5e9), (500.0, 600.0), settings
    )
    grid = bm.phase_diagram_grid(
        source,
        [phase],
        ["density", "thermal_expansivity", "molar_heat_capacity_p"],
        resolution=(3, 3),
        store_solutions=True,
    )
    assert grid.success.all(), grid.diagnostics
    assert len(grid.solution_at(1, 1).state.phases) == 2
    restored = bm.PhaseDiagramPropertyGrid.from_dict(grid.to_dict())
    replay = bm.evaluate_property_grid(restored, [phase], list(grid.values))
    assert replay.success.all(), replay.diagnostics
    for name in grid.values:
        np.testing.assert_allclose(replay.values[name], grid.values[name], rtol=1e-10)


def test_saved_grid_validation(single_phase):
    import copy

    source, phase = single_phase
    plain = bm.phase_diagram_grid(source, [phase], "density", resolution=(2, 2))
    assert not plain.has_solutions
    with pytest.raises(ValueError, match="store_solutions=True"):
        bm.evaluate_property_grid(plain, [phase], "density")
    grid = bm.phase_diagram_grid(
        source, [phase], "density", resolution=(2, 2), store_solutions=True
    )
    with pytest.raises(ValueError, match="candidate names/order"):
        bm.evaluate_property_grid(grid, [], "density")
    with pytest.raises(IndexError):
        grid.solution_at(2, 0)
    data = grid.to_dict()
    for key, value, message in [
        ("schema_version", 999, "schema"),
        ("success", [[True]], "shape"),
        ("solutions", [[None]], "shape"),
    ]:
        bad = copy.deepcopy(data)
        bad[key] = value
        with pytest.raises(ValueError, match=message):
            bm.PhaseDiagramPropertyGrid.from_dict(bad)
    bad = copy.deepcopy(data)
    bad["solutions"][0][0] = None
    with pytest.raises(ValueError, match="missing its solution"):
        bm.PhaseDiagramPropertyGrid.from_dict(bad)
