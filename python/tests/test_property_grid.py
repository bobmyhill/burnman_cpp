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


def test_prepared_batches_own_models_and_count_only_their_solves(single_phase):
    source, phase = single_phase
    before = source.to_dict()
    x, y = np.array([3e9, 5e9]), np.array([1000.0])
    names = ["density", "molar_enthalpy"]
    original = bm._core._PropertyGridder(source, [phase], x, y)
    # A second prepared calculation must own its changed model. Neither may
    # borrow thermodynamic state or field guesses from the other's batches.
    shift = dict(delta_E=1200.0, delta_S=0.0, delta_V=1.0e-7)
    phase.set_property_modifiers([("linear", shift)])
    phase.set_state(2e9, 700.0)
    modified = bm._core._PropertyGridder(source, [phase], x, y)
    reference = bm.Mineral(oxide_params())
    for p in x:
        plain = original.evaluate(names, [p], y, True, True)
        shifted = modified.evaluate(names, [p], y, True, True)
        fresh = bm.phase_diagram_grid(
            source, [reference], names, x=[p], y=y, store_solutions=True
        )
        assert plain.resolved and shifted.resolved
        for name in names:
            np.testing.assert_allclose(
                plain.values[name], fresh.values[name], rtol=1e-12
            )
        reference.set_state(p, y[0])
        assert shifted.values["density"][0, 0] == pytest.approx(
            reference.molar_mass / (reference.molar_volume + shift["delta_V"])
        )
        assert shifted.values["molar_enthalpy"][0, 0] == pytest.approx(
            plain.values["molar_enthalpy"][0, 0]
            + shift["delta_E"]
            + p * shift["delta_V"]
        )
        assert plain.equilibrium_solves == fresh.equilibrium_solves
        solves = plain.equilibrium_solves
        original.retry(plain)
        assert plain.equilibrium_solves == solves
    assert (phase.pressure, phase.temperature) == (2e9, 700.0)
    assert source.to_dict() == before


def test_resumed_diagram_audits_saved_states_without_new_solves(single_phase):
    source, phase = single_phase
    data = source.to_dict()
    assert data.pop("thermodynamic_validation")["checked_pairs"] > 0
    legacy = bm.PhaseDiagramResult.from_dict(data)
    assert legacy.thermodynamic_validation["scope"] == "not_checked"
    assert legacy.to_dict() == data
    # Keep a previously resolved diagram but give one accepted root bad energy.
    data["samples"][0]["gibbs"] += 1.0e6
    resumed = bm.refine_phase_diagram(data["composition_start"], [phase], data)
    assert not resumed.resolved
    assert resumed.thermodynamic_validation["scope"] == "aligned_samples"
    assert resumed.thermodynamic_validation["violations"] > 0
    saved = resumed.to_dict()
    for key in ("samples", "boundaries", "nodes", "equilibrium_solves"):
        assert saved[key] == data[key]
    restored = bm.PhaseDiagramResult.from_dict(saved)
    assert restored.thermodynamic_validation == resumed.thermodynamic_validation


def test_final_grid_audit_replaces_stale_messages_without_equilibrating(single_phase):
    source, phase = single_phase
    original = bm.phase_diagram_grid(
        source, [phase], "density", resolution=(2, 2), store_solutions=True
    ).to_dict()
    data = dict(original)
    data.pop("thermodynamic_validation")
    # Legacy snapshots can reconstruct energy from their phase compositions.
    for row in data["solutions"]:
        for state in row:
            state.pop("gibbs")
    stale = (
        "grid [0, 0]: Thermodynamic inconsistency with a neighbouring "
        "equilibrium remains after recovery."
    )
    data.update(resolved=False, diagnostics=[stale], equilibrium_diagnostics=[stale])
    grid = bm.PhaseDiagramPropertyGrid.from_dict(data)
    prepared = bm._core._PropertyGridder(source, [phase], grid.x, grid.y)
    prepared.audit(grid, reset=True)
    assert grid.resolved and not grid.equilibrium_diagnostics
    assert grid.thermodynamic_validation == dict(
        scope="grid_and_joins", checked_pairs=4, skipped_pairs=0, violations=0
    )
    assert grid.equilibrium_solves == original["equilibrium_solves"]
    np.testing.assert_array_equal(grid.values["density"], original["values"]["density"])
    for j in range(2):
        for i in range(2):
            assert np.isfinite(grid.solution_at(j, i).gibbs)


@pytest.mark.parametrize("chunk_size", [1, 2, 4])
@pytest.mark.parametrize("bad_energy", [False, True])
def test_chunk_final_audit_waits_for_repairs_and_covers_all_joins(
    single_phase, tmp_path, monkeypatch, chunk_size, bad_energy
):
    import json

    source, phase = single_phase
    records = []
    originals = {}
    for n, (rows, cols, grid) in enumerate(
        bm.phase_diagram_grid_chunks(
            source, [phase], "density", resolution=(3, 3), chunk_size=chunk_size
        )
    ):
        data = grid.to_dict()
        data.pop("thermodynamic_validation")  # An older saved batch.
        if bad_energy and rows.start <= 1 < rows.stop and cols.start <= 1 < cols.stop:
            data["solutions"][1 - rows.start][1 - cols.start]["gibbs"] += 1.0e6
        filename = f"chunk_{n}.json"
        bm.save_property_grid(
            bm.PhaseDiagramPropertyGrid.from_dict(data), tmp_path / filename
        )
        originals[filename] = data
        records.append(
            dict(
                file=filename,
                rows=[rows.start, rows.stop],
                columns=[cols.start, cols.stop],
            )
        )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            dict(
                format="burnman_cpp.property_grid_chunks",
                schema_version=1,
                x=[3e9, 4e9, 5e9],
                y=[800.0, 1000.0, 1200.0],
                chunks=records,
            )
        )
    )
    native = bm._core._PropertyGridder
    repaired = []

    class CappedRecovery:
        """Leave states as they are; exercise the real final audit after retry."""

        def __init__(self, *args):
            self.prepared = native(*args)

        def retry(self, grid):
            repaired.append((grid.x[0], grid.y[0]))

        def audit(self, *args, **kwargs):
            return self.prepared.audit(*args, **kwargs)

    monkeypatch.setattr(bm._core, "_PropertyGridder", CappedRecovery)
    checked = violations = 0
    for _, _, grid in bm.retry_property_grid_chunks(source, [phase], tmp_path):
        assert len(repaired) == len(records)  # No provisional result was yielded.
        coverage = grid.thermodynamic_validation
        assert coverage["scope"] == "grid_and_joins"
        checked += coverage["checked_pairs"]
        violations += coverage["violations"]
        assert grid.resolved == (coverage["violations"] == 0)
    assert checked == 12  # All horizontal/vertical edges of a 3x3 grid, once.
    assert violations == (4 if bad_energy else 0)
    for record in records:
        saved = bm.load_property_grid(tmp_path / record["file"]).to_dict()
        original = originals[record["file"]]
        for key in ("solutions", "values", "success", "equilibrium_solves"):
            assert saved[key] == original[key]
        assert saved["thermodynamic_validation"]["scope"] == "grid_and_joins"


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


@pytest.fixture
def inexact_binary_field():
    from test_composition_phase_diagram import (
        START,
        END,
        binary_phases,
        section_settings,
    )

    phases = binary_phases()
    source = bm.phase_diagram(
        START,
        phases,
        diagram="PX",
        composition_end=END,
        pressure_range=(0.0, 2e9),
        temperature=1000.0,
        settings=section_settings(),
    )
    saved = source.to_dict()
    # An inaccurate chord assigns X=.95 to the coexistence field, where its
    # equilibrium would require a negative amount. The nearby B field is valid.
    upper = max(
        saved["boundaries"],
        key=lambda line: np.mean([p["composition_coordinate"] for p in line["points"]]),
    )
    for point in upper["points"]:
        point["composition_coordinate"] = 0.99
    for node in saved["nodes"]:
        if node["id"] in (upper["start_node"], upper["end_node"]):
            node["composition_coordinate"] = 0.99
    return source, phases, saved


@pytest.mark.parametrize("provisional", [False, True])
def test_grid_tries_nearby_assemblages_when_drawn_fields_are_inexact(
    inexact_binary_field, provisional
):
    from copy import deepcopy

    source, phases, saved = inexact_binary_field
    if provisional:
        for line in saved["boundaries"]:
            line["side_a"] = line["side_b"] = []
        assert any(
            p.is_provisional for p in bm.phase_diagram_field_polygons(saved).polygons
        )
    original = deepcopy(saved)
    expected = bm.phase_diagram_grid(
        source,
        phases,
        "density",
        x=[1e9],
        y=[0.05, 0.95],
        store_solutions=True,
    )
    grid = bm.phase_diagram_grid(
        saved,
        phases,
        "density",
        x=[1e9],
        y=[0.05, 0.95],
        store_solutions=True,
    )
    assert grid.success.all(), grid.diagnostics
    np.testing.assert_allclose(
        grid.values["density"], expected.values["density"], rtol=1e-9
    )
    for row, phase_id in enumerate((0, 1)):
        state = grid.solution_at(row, 0).state
        assert [p.id for p in state.phases] == [phase_id]
        assert state.mass_balance_error <= source.settings.mass_balance_tolerance
        assert grid.field_numbers[row, 0] > 0
    assert saved == original


@pytest.mark.parametrize("chunk_size", [None, 1, 4])
@pytest.mark.parametrize(
    "coordinates, recovered",
    [
        ([0.90, 0.95], True),
        ([0.94, 0.95], False),
        ([0.60, 0.94, 0.95], False),
        ([0.90, 0.94, 0.95, 0.999], True),
        ([0.95], False),
    ],
)
def test_grid_fallback_only_crosses_boundaries_within_one_local_spacing(
    inexact_binary_field, coordinates, recovered, chunk_size
):
    _, phases, saved = inexact_binary_field
    target = coordinates.index(0.95)
    # X=.95 lies in the drawn A+B field but only B can equilibrate. B's
    # boundary at .99 must meet the one-node window, even in singleton batches.
    arguments = dict(x=[1e9], y=coordinates, store_solutions=True)
    if chunk_size is None:
        grid = bm.phase_diagram_grid(saved, phases, "density", **arguments)
        row = target
    else:
        for rows, _, batch in bm.phase_diagram_grid_chunks(
            saved, phases, "density", chunk_size=chunk_size, **arguments
        ):
            if rows.start <= target < rows.stop:
                grid, row = batch, target - rows.start
                break
    assert bool(grid.equilibrium_success[row, 0]) == recovered, grid.diagnostics
    if recovered:
        assert [p.id for p in grid.solution_at(row, 0).state.phases] == [1]
    else:
        assert grid.solution_at(row, 0) is None
        assert np.isnan(grid.values["density"][row, 0])


def test_grid_fallback_does_not_cross_two_boundaries(inexact_binary_field):
    from copy import deepcopy

    _, phases, saved = inexact_binary_field
    upper = max(
        saved["boundaries"],
        key=lambda line: np.mean([p["composition_coordinate"] for p in line["points"]]),
    )
    # Insert an incorrectly drawn A strip between A+B and B. At X=.95 both
    # separating boundaries are within one grid spacing. Only A is adjacent;
    # its root is metastable and must fail the alternative's affinity check.
    extra = deepcopy(upper)
    extra.update(
        id=max(line["id"] for line in saved["boundaries"]) + 1,
        start_node=-1,
        end_node=-1,
        zero_phase=1,
        side_a=[0, 1],
        side_b=[0],
    )
    for point in extra["points"]:
        point["composition_coordinate"] = 0.98
    saved["boundaries"].append(extra)
    upper.update(side_a=[0], side_b=[1], is_solution_replacement=True)
    grid = bm.phase_diagram_grid(
        saved, phases, "density", x=[1e9], y=[0.90, 0.95], store_solutions=True
    )
    assert not grid.equilibrium_success[1, 0]
    assert grid.solution_at(1, 0) is None


@pytest.mark.parametrize("chunk_size", [1, 4, 11])
def test_neighbour_warm_starts_cross_rows_and_chunks_in_inverse_sections(chunk_size):
    phases = exchange_phases()
    state = equilibrated()
    options = bm.PhaseDiagramSettings()
    options.pressure_seeds = options.entropy_seeds = 2
    options.max_phase_instances = 1
    source = bm.phase_diagram(
        {"Mg": 0.5, "Fe": 0.5, "O": 1.0},
        phases,
        pressure_range=(3.9e9, 4.1e9),
        temperature_range=(800.0, 1200.0),
        entropy_range=(
            state.n_moles * state.molar_entropy - 0.05,
            state.n_moles * state.molar_entropy + 0.05,
        ),
        diagram="PS",
        settings=options,
    )
    names = ["density", "thermal_expansivity", "temperature"]
    full = bm.phase_diagram_grid(source, phases, names, resolution=(5, 3))
    assert full.success.all(), full.diagnostics
    # One-point chunks exercise left/above warm starts even without retaining
    # solutions in the returned batches; larger chunks cross both row seams.
    for rows, cols, batch in bm.phase_diagram_grid_chunks(
        source,
        phases,
        names,
        resolution=(5, 3),
        chunk_size=chunk_size,
        store_solutions=False,
    ):
        assert batch.success.all(), batch.diagnostics
        assert not batch.has_solutions
        for name in names:
            np.testing.assert_allclose(
                batch.values[name],
                full.values[name][rows, cols],
                rtol=1e-8,
            )


@pytest.fixture
def cold_entropy_seed():
    """An inverse section whose left saved P/T guess is at zero heat capacity."""
    phase = bm.Mineral(oxide_params())
    phase.set_state(4e9, 1000.0)
    entropy = phase.molar_entropy
    settings = bm.PhaseDiagramSettings()
    settings.pressure_seeds = settings.entropy_seeds = 2
    settings.max_phase_instances = 1
    source = bm.phase_diagram(
        {"Mg": 1.0, "O": 1.0},
        [phase],
        (3.9e9, 4.1e9),
        (0.0, 1200.0),
        settings,
        diagram="PS",
        entropy_range=(entropy - 1.0, entropy + 1.0),
    ).to_dict()
    # The equilibria themselves are known; deliberately supply one poor saved
    # starting temperature. A later forward point has a useful starting state.
    source["fields"] = []
    first = dict(
        source["samples"][0], pressure=3.9e9, entropy=entropy - 1.0, temperature=0.0
    )
    last = dict(first, pressure=4.1e9, temperature=1000.0)
    source["samples"] = [first, last]
    return (
        source,
        phase,
        np.array([3.9e9, 4.1e9]),
        np.linspace(entropy - 1.0, entropy + 1.0, 3),
    )


@pytest.mark.parametrize("store_solutions", [False, True])
def test_reverse_pass_recovers_from_right_without_resolving_successes(
    cold_entropy_seed, store_solutions
):
    source, phase, x, y = cold_entropy_seed
    prepared = bm._core._PropertyGridder(source, [phase], x, y)
    forward = prepared.evaluate(["temperature", "density"], x, y, True, True)
    assert not forward.equilibrium_success[0, 0]
    assert forward.equilibrium_success[0, 1]
    original = forward.to_dict()
    prepared.retry(forward)
    assert forward.equilibrium_success.all(), forward.diagnostics
    assert not forward.equilibrium_diagnostics
    assert forward.resolved
    for j, i in np.argwhere(np.array(original["equilibrium_success"])):
        assert forward.to_dict()["solutions"][j][i] == original["solutions"][j][i]
        for name in forward.values:
            assert forward.values[name][j, i] == original["values"][name][j][i]
    solves = forward.equilibrium_solves
    prepared.retry(forward)
    assert forward.equilibrium_solves == solves
    complete = bm.phase_diagram_grid(
        source,
        [phase],
        list(forward.values),
        x=x,
        y=y,
        store_solutions=store_solutions,
    )
    assert complete.equilibrium_success.all(), complete.diagnostics
    assert complete.has_solutions == store_solutions


@pytest.mark.parametrize("chunk_size", [1, 2, 4])
def test_reverse_pass_recovers_across_saved_batch_rows(
    cold_entropy_seed, tmp_path, chunk_size
):
    import json

    source, phase, x, y = cold_entropy_seed
    records = []
    originals = {}
    for n, (rows, cols, grid) in enumerate(
        bm.phase_diagram_grid_chunks(
            source, [phase], "density", x=x, y=y, chunk_size=chunk_size
        )
    ):
        data = grid.to_dict()
        # A failed middle row must borrow equilibria from the next saved row.
        for j in range(len(grid.y)):
            for i in range(len(grid.x)):
                if rows.start + j == 1:
                    data["solutions"][j][i] = None
                    data["equilibrium_success"][j][i] = False
                    data["success"][j][i] = False
                    data["values"]["density"][j][i] = None
                    message = f"grid [{j}, {i}]: forced failed equilibrium"
                    data["diagnostics"].append(message)
                    data["equilibrium_diagnostics"].append(message)
                    data["resolved"] = False
        grid = bm.PhaseDiagramPropertyGrid.from_dict(data)
        name = f"chunk_{n}.json.gz"
        bm.save_property_grid(grid, tmp_path / name)
        records.append(
            dict(
                file=name, rows=[rows.start, rows.stop], columns=[cols.start, cols.stop]
            )
        )
        originals[(rows.start, cols.start)] = grid.to_dict()
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            dict(
                format="burnman_cpp.property_grid_chunks",
                schema_version=1,
                x=x.tolist(),
                y=y.tolist(),
                chunks=records,
            )
        )
    )
    order = []
    for rows, cols, grid in bm.retry_property_grid_chunks(source, [phase], tmp_path):
        order.append((rows.start, cols.start))
        assert grid.equilibrium_success.all(), grid.diagnostics
        assert grid.resolved and not grid.diagnostics
        original = originals[(rows.start, cols.start)]
        for j, i in np.argwhere(np.array(original["equilibrium_success"])):
            assert grid.to_dict()["solutions"][j][i] == original["solutions"][j][i]
    assert order == sorted(order, reverse=True)
    for _, _, grid in bm.load_property_grid_chunks(tmp_path):
        assert grid.equilibrium_success.all() and not grid.diagnostics


def test_reverse_pass_preserves_property_failures_at_accepted_equilibria(single_phase):
    source, phase = single_phase
    grid = bm.phase_diagram_grid(
        source, [phase], "density", resolution=(3, 3), store_solutions=True
    )
    data = grid.to_dict()
    data["success"][1][1] = False
    data["values"]["density"][1][1] = None
    data["diagnostics"] = ["Field 1: grid [1, 1]: density: unavailable response"]
    data["resolved"] = False
    grid = bm.PhaseDiagramPropertyGrid.from_dict(data)
    prepared = bm._core._PropertyGridder(source, [phase], grid.x, grid.y)
    prepared.retry(grid)
    assert grid.to_dict() == data


def test_grid_keeps_successful_assigned_assemblage_even_when_provisional(
    single_phase,
):
    source, phase = single_phase
    params = oxide_params(energy_shift=1e5)
    params["name"] = "unstable"
    unstable = bm.Mineral(params)
    saved = source.to_dict()
    saved["fields"] = []
    saved["phase_names"].append("unstable")
    extra = dict(saved["samples"][0], pressure=4e9, temperature=1000.0)
    extra["phases"] = [
        dict(extra["phases"][0], id=1, candidate_index=1, name="unstable")
    ]
    saved["samples"] = [saved["samples"][0], extra]
    field = bm.phase_diagram_field_polygons(saved).polygons[0]
    assert field.is_provisional and field.phases == [1]
    # The assigned phase satisfies P/T, mass balance and local curvature.
    # A lower-energy alternative must not trigger global phase reselection.
    grid = bm.phase_diagram_grid(
        saved,
        [phase, unstable],
        "density",
        resolution=(3, 3),
        store_solutions=True,
    )
    assert grid.success.all(), grid.diagnostics
    assert grid.equilibrium_solves == 9
    for j in range(3):
        for i in range(3):
            assert [p.candidate_index for p in grid.solution_at(j, i).state.phases] == [
                1
            ]


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
    for rows, cols, batch in bm.phase_diagram_grid_chunks(
        saved, [phase], "density", resolution=(5, 5), chunk_size=3
    ):
        np.testing.assert_array_equal(
            batch.field_numbers, grid.field_numbers[rows, cols]
        )
        np.testing.assert_allclose(
            batch.values["density"], grid.values["density"][rows, cols], equal_nan=True
        )
        for j in range(len(batch.y)):
            for i in range(len(batch.x)):
                assert (batch.solution_at(j, i) is None) == (
                    batch.field_numbers[j, i] == 0
                )


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


def test_saved_grid_can_replay_an_equilibrium_without_a_closed_field(single_phase):
    source, phase = single_phase
    saved = bm.phase_diagram_grid(
        source,
        [phase],
        "density",
        resolution=(2, 2),
        store_solutions=True,
    ).to_dict()
    # A sampled fallback can solve a gap for which no closed loop supplies a
    # display number. The saved equilibrium remains usable independently.
    saved["field_numbers"][0][0] = 0
    restored = bm.PhaseDiagramPropertyGrid.from_dict(saved)
    replay = bm.evaluate_property_grid(restored, [phase], "density")
    assert replay.equilibrium_success[0, 0] and replay.success[0, 0]
    assert replay.field_numbers[0, 0] == 0


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


@pytest.mark.parametrize("chunk_size", [1, 4, 11, 100])
def test_chunked_grid_matches_full_grid_and_replays_saved_states(
    single_phase, tmp_path, chunk_size
):
    import json

    source, phase = single_phase
    source_data = source.to_dict()
    full = bm.phase_diagram_grid(
        source,
        [phase],
        ["density", "thermal_expansivity"],
        resolution=(7, 5),
        store_solutions=True,
    )
    values = np.full(full.success.shape, np.nan)
    coverage = np.zeros(full.success.shape, dtype=int)
    records = []
    for number, (rows, cols, grid) in enumerate(
        bm.phase_diagram_grid_chunks(
            source, [phase], list(full.values), resolution=(7, 5), chunk_size=chunk_size
        )
    ):
        assert len(grid.x) * len(grid.y) <= chunk_size
        np.testing.assert_array_equal(
            grid.field_numbers, full.field_numbers[rows, cols]
        )
        np.testing.assert_allclose(
            grid.values["density"], full.values["density"][rows, cols]
        )
        coverage[rows, cols] += 1
        file = f"chunk_{number}.json.gz"
        bm.save_property_grid(grid, tmp_path / file)
        records.append(
            dict(
                file=file, rows=[rows.start, rows.stop], columns=[cols.start, cols.stop]
            )
        )
    assert (coverage == 1).all()
    assert source.to_dict() == source_data
    (tmp_path / "chunks.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            dict(
                format="burnman_cpp.property_grid_chunks",
                schema_version=1,
                chunks_file="chunks.jsonl",
                complete=True,
            )
        )
    )
    for rows, cols, grid in bm.load_property_grid_chunks(tmp_path):
        replay = bm.evaluate_property_grid(
            grid, [phase], "thermal_expansivity", relaxed=False
        )
        assert replay.equilibrium_solves == 0
        assert replay.success.all(), replay.diagnostics
        values[rows, cols] = replay.values["thermal_expansivity"]
    np.testing.assert_allclose(values, full.values["thermal_expansivity"], rtol=1e-10)


@pytest.mark.parametrize("chunk_size", [0, -1, 1.5, True])
def test_chunked_grid_rejects_invalid_batch_size(single_phase, chunk_size):
    source, phase = single_phase
    with pytest.raises(ValueError, match="chunk_size"):
        next(
            bm.phase_diagram_grid_chunks(
                source, [phase], "density", chunk_size=chunk_size
            )
        )


def test_sp_isotherms_keep_gaps_and_all_strip_cells():
    from types import SimpleNamespace

    example = runpy.run_path(
        str(
            Path(__file__).resolve().parents[2]
            / "examples/example_plot_pyrolite_sp_relaxed_properties.py"
        )
    )
    x, y = np.arange(7.0), np.arange(9.0)
    temperatures = y[:, None] - x[None, :]
    temperatures[:, 3] = np.nan
    grid = SimpleNamespace(
        diagram_type="PS",
        has_solutions=True,
        x=x,
        y=y,
        values={"temperature": temperatures},
    )
    records = example["calculate_isotherms"](grid, (1.25,), chunk_rows=3)
    intervals = []
    for line in records[0]["lines"]:
        points = line["points"]
        assert all(p["entropy"] - p["pressure"] == pytest.approx(1.25) for p in points)
        low, high = min(p["pressure"] for p in points), max(
            p["pressure"] for p in points
        )
        assert high <= 2 or low >= 4
        intervals.append(high - low)
    # The missing column masks its two adjoining cells: lengths 2 + 2.
    assert sum(intervals) == pytest.approx(4.0)


@pytest.mark.parametrize("entropy_jump", [-1.0, 1.0])
def test_coexistence_expansivity_keeps_signed_infinity_and_saved_states(
    tmp_path, entropy_jump
):
    first = bm.Mineral(oxide_params())
    dense = bm.CombinedMineral(
        [first],
        [1.0],
        [5000.0 + 1000.0 * entropy_jump, entropy_jump, -1e-6],
        name="dense",
    )
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
    grid = bm.phase_diagram_grid(
        source,
        [first, dense],
        ["thermal_expansivity", "temperature"],
        x=[1000.0],
        y=[volume],
        store_solutions=True,
    )
    expected = -np.copysign(np.inf, entropy_jump)
    assert grid.success.all(), grid.diagnostics
    assert grid.values["thermal_expansivity"][0, 0] == expected
    loaded = bm.load_property_grid(
        bm.save_property_grid(grid, tmp_path / "infinite.json.gz")
    )
    assert loaded.values["thermal_expansivity"][0, 0] == expected
    replay = bm.evaluate_property_grid(loaded, [first, dense], "thermal_expansivity")
    assert replay.equilibrium_solves == 0
    assert replay.success.all(), replay.diagnostics
    assert replay.values["thermal_expansivity"][0, 0] == expected
    assert np.isfinite(
        bm.evaluate_property_grid(
            loaded, [first, dense], "thermal_expansivity", relaxed=False
        ).values["thermal_expansivity"]
    ).all()


def test_soft_pyrolite_curvature_remains_finite():
    example = runpy.run_path(
        str(
            Path(__file__).resolve().parents[2]
            / "examples/example_pyrolite_phase_diagram.py"
        )
    )
    grid = bm.load_property_grid(
        Path(__file__).with_name("data") / "pyrolite_sp_soft_grid_point.json"
    )
    replay = bm.evaluate_property_grid(
        grid,
        example["candidate_phases"](),
        ["thermal_expansivity", "molar_heat_capacity_p"],
    )
    assert replay.equilibrium_solves == 0
    assert replay.success.all(), replay.diagnostics
    assert np.isfinite(replay.values["thermal_expansivity"]).all()
    assert np.isfinite(replay.values["molar_heat_capacity_p"]).all()


def test_disk_grid_reverse_pass_updates_manifest_and_saved_states(
    cold_entropy_seed, tmp_path, capsys
):
    import json
    from burnman_cpp.tools.phase_diagram_workflow import _grid_to_disk

    source, phase, _, _ = cold_entropy_seed
    manifest_path, diagnostics = _grid_to_disk(
        bm.PhaseDiagramResult.from_dict(source),
        [phase],
        (2, 3),
        tmp_path / "grid_solutions",
        1,
        True,
    )
    report = json.loads(manifest_path.read_text())
    assert report["complete"] and report["equilibrium_successes"] == 6
    assert not diagnostics and not report["diagnostics"]
    for _, _, grid in bm.load_property_grid_chunks(manifest_path):
        assert grid.equilibrium_success.all() and not grid.diagnostics
        assert np.isfinite(grid.values["temperature"]).all()
    output = capsys.readouterr().out
    assert "Retrying failed grid nodes backwards" in output
    assert "grid [" not in output


@pytest.mark.parametrize("chunked", [False, True])
def test_saved_sp_response_panels_need_no_equilibrium_solves(
    cold_entropy_seed, tmp_path, monkeypatch, chunked
):
    import json
    import matplotlib.pyplot as plt

    source, phase, x, y = cold_entropy_seed
    example = runpy.run_path(
        str(
            Path(__file__).resolve().parents[2]
            / "examples/example_plot_pyrolite_sp_relaxed_properties.py"
        )
    )
    names = [name for name, _, _ in example["PROPERTIES"]]
    original = bm.phase_diagram_grid(
        source, [phase], names + ["temperature"], x=x, y=y, store_solutions=True
    )
    assert original.equilibrium_success.all(), original.diagnostics
    expected = {
        mode: bm.evaluate_property_grid(original, [phase], names, relaxed=relaxed)
        for mode, relaxed in (("relaxed", True), ("unrelaxed", False))
    }
    saved = bm.save_property_grid(original, tmp_path / "saved.json.gz")
    if chunked:
        manifest = dict(
            format="burnman_cpp.property_grid_chunks",
            schema_version=1,
            x=x.tolist(),
            y=y.tolist(),
            chunks=[dict(file=saved.name, rows=[0, len(y)], columns=[0, len(x)])],
        )
        saved = tmp_path / "manifest.json"
        saved.write_text(json.dumps(manifest))
    before = saved.read_bytes()

    def unexpected(*args, **kwargs):
        raise AssertionError("Plotting saved equilibria must not solve them again.")

    monkeypatch.setattr(bm, "phase_diagram_grid", unexpected)
    monkeypatch.setattr(bm, "phase_diagram_grid_chunks", unexpected)
    monkeypatch.setattr(bm, "retry_property_grid_chunks", unexpected)
    gx, gy, arrays = example["evaluate_saved"](saved, [phase], tmp_path / "output")
    np.testing.assert_array_equal(gx, x)
    np.testing.assert_array_equal(gy, y)
    for mode in expected:
        assert expected[mode].equilibrium_solves == 0
        for name in names:
            np.testing.assert_allclose(
                arrays[f"{mode}_{name}"], expected[mode].values[name]
            )
    assert saved.read_bytes() == before
    with np.load(tmp_path / "output/properties.npz") as archive:
        for key, array in arrays.items():
            np.testing.assert_array_equal(archive[key], array)

    # A failure of one property must not mask the others at the same point.
    arrays["relaxed_molar_heat_capacity_p"][0, 0] = np.nan
    isotherms = example["calculate_isotherms"](
        type(
            "Temperatures",
            (),
            dict(
                x=x,
                y=y,
                diagram_type="PS",
                has_solutions=True,
                values={"temperature": arrays["temperature"]},
            ),
        )(),
        (1000.0,),
    )
    fig, axes = example["plot"](source, x, y, arrays, isotherms=isotherms)
    assert axes.shape == (3, 2)
    assert np.ma.is_masked(axes[2, 0].images[0].get_array()[0, 0])
    assert not np.ma.is_masked(axes[0, 0].images[0].get_array()[0, 0])
    for row in axes:
        assert row[0].images[0].norm.vmin == row[1].images[0].norm.vmin
        assert row[0].images[0].norm.vmax == row[1].images[0].norm.vmax
        assert all(any(t.get_text() == "1000 K" for t in ax.texts) for ax in row)
    plt.close(fig)
