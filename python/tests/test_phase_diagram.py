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


import numpy as np
import pytest
import burnman_cpp as bm
from conftest import make_model
from burnman_cpp.minerals import HP_2011_ds62


def pure(name, formula, **parameters):
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
        | parameters
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
    s = bm.PhaseDiagramSettings()
    s.pressure_seeds = s.temperature_seeds = 3
    s.step = 0.05
    return s


def test_pt_seed_grid_reuses_a_neighbour_without_repeating_phase_discovery(capfd):
    phase = pure("MgO", {"Mg": 1.0, "O": 1.0})
    opts = settings()
    opts.verbose = True
    result = bm.phase_diagram(
        {"Mg": 1.0, "O": 1.0}, [phase], (0.9e9, 1.1e9), (995.0, 1005.0), opts
    )
    assert result.resolved, result.diagnostics
    assert len(result.samples) == result.equilibrium_solves == 9
    # The initial seed discovers its phases; the other eight reuse that result.
    assert capfd.readouterr().err.count("LP 0 ") == 1
    for sample in result.samples:
        assert sample.success, sample.message
        phase.set_state(sample.pressure, sample.temperature)
        assert sample.gibbs == pytest.approx(phase.molar_gibbs, abs=1e-8)
        assert sample.mass_balance_error <= opts.mass_balance_tolerance


def test_pt_warm_seeds_discover_a_new_stable_polymorph(capfd):
    first = pure("low pressure", {"Mg": 1.0, "O": 1.0})
    dense = bm.CombinedMineral(
        [first], [1.0], [1000.0, 0.0, -1e-6], name="high pressure"
    )
    phases = [first, dense]
    opts = settings()
    opts.verbose = True
    result = bm.phase_diagram(
        {"Mg": 1.0, "O": 1.0}, phases, (0.9e9, 1.1e9), (995.0, 1005.0), opts
    )
    assert result.resolved, result.diagnostics
    capfd.readouterr()
    seeds = result.samples[:9]
    for sample in seeds:
        assert sample.success, sample.message
        for phase in phases:
            phase.set_state(sample.pressure, sample.temperature)
        assert sample.gibbs == pytest.approx(
            min(p.molar_gibbs for p in phases), abs=1e-8
        )
        if sample.pressure != 1e9:
            assert [p.candidate_index for p in sample.phases] == [
                int(sample.pressure > 1e9)
            ]
    assert len(result.boundaries) == 1
    assert all(
        p.pressure == pytest.approx(1e9, abs=0.2) for p in result.boundaries[0].points
    )


def test_trace_phase_uses_configured_amount_tolerance_in_saved_samples():
    phases = [
        pure("MgO", {"Mg": 1.0, "O": 1.0}),
        pure("FeO", {"Fe": 1.0, "O": 1.0}),
    ]
    opts = settings()
    opts.pressure_seeds = opts.temperature_seeds = 2
    opts.amount_tolerance = 1.0e-9
    result = bm.phase_diagram(
        {"Mg": 1.0, "Fe": 5.0e-8, "O": 1.00000005},
        phases,
        (1.0e9, 2.0e9),
        (600.0, 800.0),
        opts,
    )
    assert result.resolved, result.diagnostics
    assert all(len(s.phases) == 2 for s in result.samples if s.success)
    assert all(not s.is_field_verification for s in result.samples)
    geometry = bm.phase_diagram_field_polygons(result)
    assert not geometry.diagnostics
    assert len(geometry.polygons) == 1
    assert geometry.polygons[0].n_phases == 2
    assert geometry.polygons[0].phases == result.fields[0].phases


def test_two_phase_entries_between_seeds_retain_the_intermediate_field():
    """Two closely spaced precipitations must not appear as one phase line."""
    elements = ("Mg", "Fe", "Ca")
    endmembers = [pure(element, {element: 1.0, "O": 1.0}) for element in elements]
    solution = bm.Solution(
        bm.IdealSolution(
            [
                (mineral, f"[{element}]O")
                for mineral, element in zip(endmembers, elements)
            ]
        ),
        [0.3, 0.3, 0.4],
        name="solution",
    )
    phases = [solution] + [
        bm.CombinedMineral(
            [endmembers[i]], [1.0], [1000.0 + 10.0 * i, 0.0, -1.0e-5], name=element
        )
        for i, element in enumerate(elements[:2])
    ]
    opts = settings()
    opts.pressure_seeds = opts.temperature_seeds = 2
    result = bm.phase_diagram(
        {"Mg": 0.3, "Fe": 0.3, "Ca": 0.4, "O": 1.0},
        phases,
        (0.0, 2.0e9),
        (600.0, 1400.0),
        opts,
    )
    assert result.resolved, result.diagnostics
    geometry = bm.phase_diagram_field_polygons(result)
    assert not geometry.diagnostics
    assert sorted(field.n_phases for field in geometry.polygons) == [1, 2, 3]
    assert sum(field.area for field in geometry.polygons) == pytest.approx(1.0)
    assert len(result.boundaries) == 2
    # At Mg entry the solution still has its bulk composition; equality of
    # its partial Gibbs energy and the shifted pure Mg phase gives this line.
    line = next(
        line
        for line in result.boundaries
        if result.phase_names[line.zero_phase] == "Mg"
    )
    for point in line.points:
        pressure = (
            1000.0 - 8.31446261815324 * point.temperature * np.log(0.3)
        ) / 1.0e-5
        assert point.pressure == pytest.approx(pressure, abs=0.1)


def test_boundary_probes_keep_a_narrow_field_between_equal_phase_counts():
    endmembers = [pure(e, {e: 1.0, "O": 1.0}) for e in ("Mg", "Fe", "Ca")]
    solution = bm.Solution(
        bm.IdealSolution(
            [(phase, f"[{e}]O") for phase, e in zip(endmembers, ("Mg", "Fe", "Ca"))]
        ),
        [1.0 / 3.0] * 3,
        name="solution",
    )
    phases = [solution] + [
        bm.CombinedMineral(
            [endmembers[i]],
            [1.0],
            [sign * 10000.0 - 1.0, bm.constants.gas_constant * np.log(3), -sign * 1e-5],
            name=name,
        )
        for i, sign, name in [(0, 1.0, "Mg high"), (1, -1.0, "Fe low")]
    ]
    bulk = {"Mg": 1.0, "Fe": 1.0, "Ca": 1.0, "O": 3.0}
    result = bm.phase_diagram(bulk, phases, (0.0, 2.0e9), (600.0, 1400.0), settings())

    def check_neighbours(result):
        assert result.resolved, result.diagnostics
        assert len(result.boundaries) == 2
        for line in result.boundaries:
            assert line.reaction_kind == "phase_out"
            assert line.assemblage == [0, 3, 6]
            missing = [p for p in line.assemblage if p != line.zero_phase]
            assert {tuple(line.side_a), tuple(line.side_b)} == {
                (0, 3, 6),
                tuple(missing),
            }

    # The initial probe crosses the entire three-phase field, whose width is
    # about 0.6 MPa inside a 2 GPa section. Both distant neighbours have two
    # phases; they are not the immediate neighbours of either boundary.
    check_neighbours(result)
    saved = result.to_dict()
    for line in saved["boundaries"]:
        line.update(side_a=[0, 6], side_b=[0, 3])
    saved["resolved"] = False  # Deliberately stale labels remain unchanged.
    # The reaction graph can recover all three finite assemblages from the
    # verified phase-out conditions, independently of those stale probes.
    for representation in (saved, bm.PhaseDiagramResult.from_dict(saved)):
        geometry = bm.phase_diagram_field_polygons(representation)
        assert not geometry.diagnostics
        assert {tuple(p.phases) for p in geometry.polygons} == {
            (0, 3, 6),
            (0, 3),
            (0, 6),
        }


def test_four_branches_at_a_phase_swap_node():
    phases = crossing_phases()
    result = bm.phase_diagram(
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
        assert len(line.points) >= 2
        assert node.id in (line.start_node, line.end_node)
        first, last = line.points[0], line.points[-1]
        # Each known coexistence arm must reach the frame from the invariant;
        # the number of continuation steps is not a physical requirement.
        assert abs(last.pressure - first.pressure) == pytest.approx(1e9) or abs(
            last.temperature - first.temperature
        ) == pytest.approx(400.0)
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


def test_two_phases_disappear_together_at_a_compound_replacement():
    """A + B = AB has a one-phase field across a three-phase boundary."""
    a = pure("A", {"Mg": 1.0, "O": 1.0})
    b = pure("B", {"Fe": 1.0, "O": 1.0})
    compound = bm.CombinedMineral([a, b], [1.0, 1.0], [1000.0, 0.0, -1.0e-6], name="AB")
    opts = settings()
    opts.max_phase_instances = 1
    result = bm.phase_diagram(
        {"Mg": 1.0, "Fe": 1.0, "O": 2.0},
        [a, b, compound],
        (0.0, 2.0e9),
        (600.0, 1400.0),
        opts,
    )
    assert result.resolved, result.diagnostics
    assert len(result.boundaries) == 1
    line = result.boundaries[0]
    assert len(line.assemblage) == 3
    assert {tuple(line.side_a), tuple(line.side_b)} == {(0, 1), (2,)}
    assert not line.is_solution_replacement
    assert line.start_node >= 0 and line.end_node >= 0
    for point in line.points:
        assert point.pressure == pytest.approx(1.0e9, abs=0.1)
    geometry = bm.phase_diagram_field_polygons(result)
    assert not geometry.diagnostics
    assert sorted(p.n_phases for p in geometry.polygons) == [1, 2]
    assert sum(p.area for p in geometry.polygons) == pytest.approx(1.0)


def curved_phases():
    first = pure("large volume", {"Mg": 1.0, "O": 1.0})
    second = pure(
        "small volume",
        {"Mg": 1.0, "O": 1.0},
        H_0=-50000.0,
        S_0=70.0,
        V_0=1.0e-5,
        Cp=[70.0, 0.0, 0.0, 0.0],
    )
    return [first, second]


def test_resume_refines_curved_chords_only_when_resolution_is_requested():
    phases = curved_phases()
    opts = settings()
    bulk = {"Mg": 1.0, "O": 1.0}
    result = bm.phase_diagram(bulk, phases, (0.0, 5.0e9), (600.0, 1400.0), opts)
    assert result.resolved, result.diagnostics
    saved = result.to_dict()
    line = saved["boundaries"][0]
    line["points"] = [line["points"][0], line["points"][-1]]
    assert not bm.phase_diagram_field_polygons(saved).diagnostics
    unchanged = bm.refine_phase_diagram(bulk, phases, saved, opts)
    assert len(unchanged.boundaries[0].points) == 2
    assert unchanged.equilibrium_solves == saved["equilibrium_solves"]
    repaired = bm.refine_phase_diagram(bulk, phases, saved, opts, resolution=(31, 31))
    assert repaired.resolved, repaired.diagnostics
    assert len(repaired.boundaries) == 1
    assert len(repaired.boundaries[0].points) > 2
    assert not bm.phase_diagram_field_polygons(repaired).diagnostics
    for p in repaired.boundaries[0].points:
        assert p.mass_balance_error < 1e-9
        assert p.minimum_affinity >= -opts.affinity_tolerance
        assert p.residual < 0.01


@pytest.mark.parametrize(
    "resolution, saved_resolved", [(None, False), ((17, 21), False), ((17, 21), True)]
)
def test_refinement_preserves_truncated_boundaries_and_diagnostics(
    resolution, saved_resolved
):
    s = settings()
    s.max_trace_steps = 2
    s.max_recovery_passes = 0
    bulk = {"Mg": 1.0, "Fe": 1.0, "O": 2.0}
    result = bm.phase_diagram(bulk, crossing_phases(), (0.0, 2e9), (600.0, 1400.0), s)
    assert not result.resolved
    for line in result.boundaries:
        back, forward = line.termination.split("; ")
        if back == "trace step limit":
            assert line.start_node == -1
        if forward == "trace step limit":
            assert line.end_node == -1
    s.max_trace_steps = 500
    s.max_recovery_passes = 1
    previous = result.to_dict()
    previous["resolved"] = saved_resolved
    resumed = bm.refine_phase_diagram(
        bulk, crossing_phases(), previous, s, resolution=resolution
    )
    assert not resumed.resolved
    after = resumed.to_dict()
    assert after["diagnostics"] == previous["diagnostics"]
    for key in ("nodes", "fields", "samples", "excluded_regions"):
        assert after[key] == previous[key]
    assert len(after["boundaries"]) == len(previous["boundaries"])
    for old, new in zip(previous["boundaries"], after["boundaries"]):
        assert {
            k: v for k, v in new.items() if k not in ("points", "limiting_curves")
        } == {k: v for k, v in old.items() if k not in ("points", "limiting_curves")}
        assert new["points"][0] == old["points"][0]
        assert new["points"][-1] == old["points"][-1]
    if resolution is None:
        assert after["boundaries"] == previous["boundaries"]
        assert resumed.equilibrium_solves == previous["equilibrium_solves"]
    assert not result.resolved  # The previous result was not mutated.
    with pytest.raises(ValueError, match="candidate names"):
        bm.refine_phase_diagram(bulk, crossing_phases()[:-1], result, s)


@pytest.mark.parametrize("saved_as_dict", [False, True])
def test_standard_result_roundtrip_and_resume_preserve_settings(
    saved_as_dict, tmp_path
):
    bulk = {"Mg": 1.0, "Fe": 1.0, "O": 2.0}
    s = settings()
    # A nondefault ID stride and strict EOS policy must survive saving and an
    # omitted settings argument on resume, for both native and JSON results.
    s.max_phase_instances = 2
    s.exclude_invalid_eos = False
    s.node_tolerance = 1.0e-4
    r = bm.phase_diagram(bulk, crossing_phases(), (0.0, 2e9), (600.0, 1400.0), s)
    assert r.resolved, r.diagnostics
    path = tmp_path / "phase_diagram.json"
    bm.save_phase_diagram(r, str(path))
    data = bm.load_phase_diagram(str(path))
    restored = bm.PhaseDiagramResult.from_dict(data)
    assert restored.to_dict() == data
    assert restored.settings.to_dict() == s.to_dict()
    assert restored.resolved and restored.diagnostics == r.diagnostics
    copied_settings = restored.settings
    copied_settings.max_phase_instances = 7
    assert restored.settings.max_phase_instances == 2
    previous = data if saved_as_dict else restored
    resumed = bm.refine_phase_diagram(bulk, crossing_phases(), previous)
    assert resumed.settings.to_dict() == s.to_dict()
    assert resumed.resolved, resumed.diagnostics
    polygons = bm.phase_diagram_field_polygons(resumed)
    assert len(polygons.polygons) == 4
    assert all(p.n_phases == 2 and not p.has_open_boundary for p in polygons.polygons)
    assert sum(p.area for p in polygons.polygons) == pytest.approx(1.0)


def test_save_refined_diagram_preserves_metadata_and_replaces_old_phase_lines(tmp_path):
    from burnman_cpp.tools import load_phase_diagram, save_phase_diagram

    bulk = {"Mg": 1.0, "Fe": 1.0, "O": 2.0}
    result = bm.phase_diagram(
        bulk, crossing_phases(), (0.0, 2.0e9), (600.0, 1400.0), settings()
    )
    metadata = dict(model_set={"name": "Test set"}, description="Closed bulk at 600 °C")
    path = tmp_path / "new_directory" / "section.json"
    assert save_phase_diagram(result, path, metadata=metadata) == path
    saved = load_phase_diagram(path)
    before = path.read_text(encoding="utf-8")
    refined = bm.refine_phase_diagram(
        bulk, crossing_phases(), path, resolution=(101, 201)
    )
    assert refined.resolved, refined.diagnostics
    assert refined.to_dict()["boundaries"] != saved["boundaries"]
    destination = tmp_path / "refined.json"
    save_phase_diagram(refined, destination, metadata=saved)
    loaded = load_phase_diagram(destination)
    assert all(loaded[key] == value for key, value in metadata.items())
    assert loaded == metadata | refined.to_dict()
    assert path.read_text(encoding="utf-8") == before


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_invalid_metadata_does_not_overwrite_an_existing_diagram(
    tmp_path, invalid_value
):
    saved = {"diagram_type": "PT", "phase_names": ["A", "B"], "boundaries": []}
    path = bm.save_phase_diagram(saved, tmp_path / "section.json")
    original = path.read_bytes()
    with pytest.raises(ValueError, match="JSON"):
        bm.save_phase_diagram(saved, path, metadata={"bulk_mass": invalid_value})
    assert path.read_bytes() == original
    assert bm.load_phase_diagram(path) == saved


def test_settings_dictionary_rejects_unknown_options():
    with pytest.raises(ValueError, match="Unknown phase diagram setting"):
        bm.PhaseDiagramSettings.from_dict({"minimum_step": 1.0e-7})


def test_polymorph_reduced_variance_has_three_lines():
    phases = [HP_2011_ds62.andalusite(), HP_2011_ds62.ky(), HP_2011_ds62.sill()]
    r = bm.phase_diagram(
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
    r = bm.phase_diagram(
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
    with pytest.raises(ValueError, match="Pressure outside the PitzerSterner domain"):
        fluid.set_state(0.0, 873.15)
        _ = fluid.molar_volume


def test_invalid_bulk_ranges_and_settings():
    phases = crossing_phases()
    with pytest.raises(ValueError, match="chemical span"):
        bm.stable_equilibrium({"Ca": 1.0}, phases, 1e9, 1000.0)
    with pytest.raises(ValueError, match="increasing"):
        bm.phase_diagram(
            {"Mg": 1.0, "Fe": 1.0, "O": 2.0}, phases, (1e9, 0.0), (600.0, 1400.0)
        )
    s = settings()
    s.min_step = -1
    with pytest.raises(ValueError, match="settings"):
        bm.phase_diagram(
            {"Mg": 1.0, "Fe": 1.0, "O": 2.0}, phases, (0.0, 2e9), (600.0, 1400.0), s
        )
    with pytest.raises(ValueError, match="finite"):
        bm.Composition({"SiO2": float("nan")}, unit_type="mass")


def test_trace_limit_is_reported_as_unresolved():
    s = settings()
    s.max_trace_steps = 2
    result = bm.phase_diagram(
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


@pytest.mark.parametrize("temperature", [0.0, 0.05, 0.1])
def test_cold_ideal_solution_has_one_mass_balanced_composition(temperature):
    phase = bm.Solution(make_model(), [0.4, 0.6], name="oxide")
    state = bm.stable_equilibrium(
        {"Mg": 0.4, "Fe": 0.6, "O": 1.0}, [phase], 1.0e9, temperature
    )
    assert state.success, state.message
    assert len(state.phases) == 1
    assert state.phases[0].amount == pytest.approx(1.0, abs=1.0e-9)
    np.testing.assert_allclose(state.phases[0].composition, [0.4, 0.6], atol=1.0e-9)
    assert state.mass_balance_error < 1.0e-9


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
    r = bm.phase_diagram(
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
    polygons = bm.phase_diagram_field_polygons(r)
    assert not polygons.diagnostics
    assert sorted(p.n_phases for p in polygons.polygons) == [1, 2]
