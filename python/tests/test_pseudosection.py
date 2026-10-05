import sys
import json
from pathlib import Path
import runpy

import numpy as np
import pytest
import burnman_cpp as bm
from burnman_cpp.minerals import (
    HP_2011_ds62 as HP,
    mb50NCKFMASHTO as MB,
    HGP_2018_ds633 as HGP,
)


def pure(name, formula):
    return bm.Mineral(
        dict(
            name=name,
            formula=formula,
            equation_of_state="hp_tmt",
            H_0=-100000.0,
            S_0=50.0,
            V_0=2e-5,
            K_0=1e12,
            Kprime_0=4.0,
            Kdprime_0=-4e-12,
            Cp=[60.0, 0.0, 0.0, 0.0],
            a_0=0.0,
            n=2,
            molar_mass=0.05,
            G_0=0.0,
            Gprime_0=0.0,
        )
    )


def crossing_phases():
    a = pure("A low", {"Mg": 1.0, "O": 1.0})
    b = pure("B low", {"Fe": 1.0, "O": 1.0})
    return [
        a,
        bm.CombinedMineral([a], [1.0], [1000.0, 0.0, -1e-6], name="A high"),
        b,
        bm.CombinedMineral([b], [1.0], [10000.0, 10.0, 0.0], name="B high"),
    ]


def settings():
    s = bm.PseudosectionSettings()
    s.pressure_seeds = s.temperature_seeds = 3
    s.step = 0.05
    return s


def test_four_branches_at_a_phase_swap_node():
    phases = crossing_phases()
    result = bm.pseudosection(
        {"Mg": 1.0, "Fe": 1.0, "O": 2.0},
        phases,
        (0.0, 2e9),
        (600.0, 1400.0),
        settings(),
    )
    assert result.resolved, result.diagnostics
    assert len(result.fields) == len(result.boundaries) == 4
    junctions = [n for n in result.nodes if n.kind == "junction"]
    assert len(junctions) == 1
    node = junctions[0]
    np.testing.assert_allclose(
        [node.pressure, node.temperature], [1e9, 1000.0], rtol=1e-10
    )
    assert len(node.incident_lines) == 4
    assert node.gibbs_variance == node.pt_nullity == 0
    for line in result.boundaries:
        assert line.side_a != line.side_b
        assert len(line.points) > 5
        for p in line.points:
            # Independently known coexistence equations of the test model.
            assert abs(p.pressure - 1e9) < 1.0 or abs(p.temperature - 1000.0) < 1e-6
            assert p.mass_balance_error < 1e-9
            assert p.minimum_affinity > -0.2
            if len(p.phases) == len(line.assemblage):
                zero = next(ph for ph in p.phases if ph.id == line.zero_phase)
                assert abs(zero.amount) < 1e-9
    # Input objects were copied, including the solution/model state.
    assert all(not phase.has_state() for phase in phases)


def test_resume_uses_accepted_endpoints_and_closes_truncated_fields():
    s = settings()
    s.max_trace_steps = 2
    s.max_recovery_passes = 0
    bulk = {"Mg": 1.0, "Fe": 1.0, "O": 2.0}
    result = bm.pseudosection(bulk, crossing_phases(), (0.0, 2e9), (600.0, 1400.0), s)
    assert not result.resolved
    for line in result.boundaries:
        back, forward = line.termination.split("; ")
        if back == "trace step limit":
            assert line.start_node == -1
        if forward == "trace step limit":
            assert line.end_node == -1
    s.max_trace_steps = 500
    s.max_recovery_passes = 1
    resumed = bm.refine_pseudosection(bulk, crossing_phases(), result, s)
    assert resumed.resolved, resumed.diagnostics
    assert len(resumed.boundaries) == 4
    assert (
        len(next(n for n in resumed.nodes if n.kind == "junction").incident_lines) == 4
    )
    assert all(
        line.start_node >= 0 and line.end_node >= 0 for line in resumed.boundaries
    )
    polygons = bm.pseudosection_field_polygons(resumed)
    assert not polygons.diagnostics
    assert len(polygons.polygons) == 4
    assert all(p.n_phases == 2 for p in polygons.polygons)
    assert resumed.equilibrium_solves > result.equilibrium_solves
    assert not result.resolved  # The previous result was not mutated.
    with pytest.raises(ValueError, match="candidate names"):
        bm.refine_pseudosection(bulk, crossing_phases()[:-1], result, s)


@pytest.mark.parametrize("saved_as_dict", [False, True])
def test_standard_result_roundtrip_and_resume_preserve_settings(saved_as_dict):
    bulk = {"Mg": 1.0, "Fe": 1.0, "O": 2.0}
    s = settings()
    # A nondefault ID stride and strict EOS policy must survive saving and an
    # omitted settings argument on resume, for both native and JSON results.
    s.max_phase_instances = 2
    s.exclude_invalid_eos = False
    s.node_tolerance = 1.0e-4
    r = bm.pseudosection(bulk, crossing_phases(), (0.0, 2e9), (600.0, 1400.0), s)
    assert r.resolved, r.diagnostics
    data = json.loads(json.dumps(r.to_dict(), allow_nan=False))
    restored = bm.PseudosectionResult.from_dict(data)
    assert restored.to_dict() == data
    assert restored.settings.to_dict() == s.to_dict()
    assert restored.resolved and restored.diagnostics == r.diagnostics
    copied_settings = restored.settings
    copied_settings.max_phase_instances = 7
    assert restored.settings.max_phase_instances == 2
    previous = data if saved_as_dict else restored
    resumed = bm.refine_pseudosection(bulk, crossing_phases(), previous)
    assert resumed.settings.to_dict() == s.to_dict()
    assert resumed.resolved, resumed.diagnostics
    polygons = bm.pseudosection_field_polygons(resumed)
    assert len(polygons.polygons) == 4
    assert all(p.n_phases == 2 and not p.has_open_boundary for p in polygons.polygons)
    assert sum(p.area for p in polygons.polygons) == pytest.approx(1.0)


def test_settings_dictionary_rejects_unknown_options():
    with pytest.raises(ValueError, match="Unknown pseudosection setting"):
        bm.PseudosectionSettings.from_dict({"minimum_step": 1.0e-7})


def test_polymorph_reduced_variance_has_three_lines():
    phases = [HP.andalusite(), HP.ky(), HP.sill()]
    r = bm.pseudosection(
        {"Al": 2.0, "Si": 1.0, "O": 5.0},
        phases,
        (1e5, 1.0e9),
        (500.0, 1200.0),
        settings(),
    )
    assert r.resolved, r.diagnostics
    assert len(r.fields) == 3
    junctions = [n for n in r.nodes if n.kind == "junction"]
    assert len(junctions) == 1
    assert len(junctions[0].incident_lines) == 3
    assert junctions[0].gibbs_variance == 0
    assert len(r.boundaries) == 3


@pytest.mark.parametrize("node_tolerance", [2.0e-4, 2.0e-3])
def test_short_branch_is_labelled_between_its_two_junctions(node_tolerance):
    phases = crossing_phases()
    phases.insert(
        2,
        bm.CombinedMineral([phases[0]], [1.0], [2002.0, 0.0, -2e-6], name="A highest"),
    )
    opts = settings()
    opts.node_tolerance = node_tolerance
    r = bm.pseudosection(
        {"Mg": 1.0, "Fe": 1.0, "O": 2.0}, phases, (0.0, 2e9), (600.0, 1400.0), opts
    )
    assert r.resolved, r.diagnostics
    assert len(r.fields) == 6
    junctions = [n for n in r.nodes if n.kind == "junction"]
    assert len(junctions) == 2
    assert all(len(n.incident_lines) == 4 for n in junctions)
    short = [
        line
        for line in r.boundaries
        if {line.start_node, line.end_node} == {n.id for n in junctions}
    ]
    assert len(short) == 1
    assert short[0].side_a != short[0].side_b
    assert all(3 in sides for sides in [short[0].side_a, short[0].side_b])


def test_stability_selects_lower_energy_and_bulk_scaling():
    phases = crossing_phases()
    for pressure, temperature, expected in [
        (5e8, 800.0, ["A low", "B low"]),
        (1.5e9, 1200.0, ["A high", "B high"]),
    ]:
        state = bm.stable_equilibrium(
            {"Mg": 2.0, "Fe": 2.0, "O": 4.0}, phases, pressure, temperature
        )
        assert state.success, state.message
        assert [p.name for p in state.phases] == expected
        np.testing.assert_allclose(
            [p.amount for p in state.phases], [2.0, 2.0], atol=1e-9
        )
        assert state.mass_balance_error < 1e-9


def test_water_eos_thermodynamic_derivatives_and_gas_limit():
    fluid = bm.water_fluid()
    for pressure, temperature in [
        (1e5, 573.15),
        (1e8, 573.15),
        (1e9, 873.15),
        (2e9, 1173.15),
    ]:
        fluid.set_state(pressure, temperature)
        volume, entropy = fluid.molar_volume, fluid.molar_entropy
        dp, dt = max(10.0, pressure * 1e-5), 0.02
        fluid.set_state(pressure + dp, temperature)
        high = fluid.molar_gibbs
        fluid.set_state(pressure - dp, temperature)
        low = fluid.molar_gibbs
        assert (high - low) / (2 * dp) == pytest.approx(volume, rel=1e-6)
        fluid.set_state(pressure, temperature + dt)
        high = fluid.molar_gibbs
        fluid.set_state(pressure, temperature - dt)
        low = fluid.molar_gibbs
        assert -(high - low) / (2 * dt) == pytest.approx(entropy, rel=1e-7)
    fluid.set_state(100.0, 873.15)
    assert fluid.molar_volume == pytest.approx(8.314510 * 873.15 / 100.0, rel=1e-4)
    with pytest.raises(ValueError, match="Water fluid"):
        fluid.set_state(0.0, 873.15)
        _ = fluid.molar_volume


def test_invalid_bulk_ranges_and_settings():
    phases = crossing_phases()
    with pytest.raises(ValueError, match="chemical span"):
        bm.stable_equilibrium({"Ca": 1.0}, phases, 1e9, 1000.0)
    with pytest.raises(ValueError, match="increasing"):
        bm.pseudosection(
            {"Mg": 1.0, "Fe": 1.0, "O": 2.0}, phases, (1e9, 0.0), (600.0, 1400.0)
        )
    s = settings()
    s.min_step = -1
    with pytest.raises(ValueError, match="settings"):
        bm.pseudosection(
            {"Mg": 1.0, "Fe": 1.0, "O": 2.0}, phases, (0.0, 2e9), (600.0, 1400.0), s
        )
    with pytest.raises(ValueError, match="finite"):
        bm.Composition({"SiO2": float("nan")}, unit_type="mass")


def test_trace_limit_is_reported_as_unresolved():
    s = settings()
    s.max_trace_steps = 2
    result = bm.pseudosection(
        {"Mg": 1.0, "Fe": 1.0, "O": 2.0},
        crossing_phases(),
        (0.0, 2e9),
        (600.0, 1400.0),
        s,
    )
    assert not result.resolved
    assert any("trace step limit" in d for d in result.diagnostics)


def test_stability_of_bulk_on_a_chemical_face():
    state = bm.stable_equilibrium({"Mg": 1.0, "O": 1.0}, crossing_phases(), 5e8, 800.0)
    assert state.success, state.message
    assert [p.name for p in state.phases] == ["A low"]


def test_stability_finds_two_solution_instances_and_merges_above_solvus():
    mg = pure("Mg oxide", {"Mg": 1.0, "O": 1.0})
    fe = pure("Fe oxide", {"Fe": 1.0, "O": 1.0})
    phase = bm.Solution(
        bm.SymmetricRegularSolution(
            [(mg, "[Mg]O"), (fe, "[Fe]O")], energy_interaction=[[13000.0]]
        ),
        [0.5, 0.5],
        name="oxide",
    )
    bulk = {"Mg": 0.5, "Fe": 0.5, "O": 1.0}
    state = bm.stable_equilibrium(bulk, [phase], 1e9, 600.0)
    assert state.success, state.message
    assert [p.id for p in state.phases] == [0, 1]
    np.testing.assert_allclose([p.amount for p in state.phases], [0.5, 0.5], atol=1e-9)
    x = state.phases[0].composition[0]
    # Independent common-tangent condition for a symmetric regular solution.
    assert 8.31446261815324 * 600.0 * np.log(x / (1.0 - x)) + 13000.0 * (
        1.0 - 2.0 * x
    ) == pytest.approx(0.0, abs=1e-5)
    np.testing.assert_allclose(
        state.phases[1].composition, state.phases[0].composition[::-1], atol=1e-9
    )
    state = bm.stable_equilibrium(bulk, [phase], 1e9, 900.0)
    assert state.success, state.message
    assert len(state.phases) == 1 and state.phases[0].id == 0
    np.testing.assert_allclose(state.phases[0].composition, [0.5, 0.5], atol=1e-9)
    assert state.mass_balance_error < 1e-9


def test_regular_solution_solvus_boundary_keeps_distinct_compositions():
    mg = pure("Mg oxide", {"Mg": 1.0, "O": 1.0})
    fe = pure("Fe oxide", {"Fe": 1.0, "O": 1.0})
    phase = bm.Solution(
        bm.SymmetricRegularSolution(
            [(mg, "[Mg]O"), (fe, "[Fe]O")], energy_interaction=[[13000.0]]
        ),
        [0.3, 0.7],
        name="oxide",
    )
    r = bm.pseudosection(
        {"Mg": 0.3, "Fe": 0.7, "O": 1.0},
        [phase],
        (0.0, 2.0e9),
        (600.0, 900.0),
        settings(),
    )
    expected = 13000.0 * (0.7 - 0.3) / (8.31446261815324 * np.log(0.7 / 0.3))
    assert r.resolved, r.diagnostics
    assert len(r.boundaries) == 1
    for point in r.boundaries[0].points:
        assert point.temperature == pytest.approx(expected, abs=1e-5)
        compositions = [p.composition for p in point.phases]
        np.testing.assert_allclose(
            sorted(c[0] for c in compositions), [0.3, 0.7], atol=1e-8
        )
    polygons = bm.pseudosection_field_polygons(r)
    assert not polygons.diagnostics
    assert sorted(p.n_phases for p in polygons.polygons) == [1, 2]
