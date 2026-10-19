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

import sys
import json
from pathlib import Path
import runpy

import numpy as np
import pytest
import burnman_cpp as bm
from conftest import make_model
from burnman_cpp.minerals import (
    HP11,
    MB16,
    HGP18,
)


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


def test_pyrolite_pt_warm_grid_matches_independent_phase_discovery(capfd):
    from burnman_cpp.tools.phase_diagram_helpers import load_example

    example = load_example(
        Path(__file__).resolve().parents[2]
        / "examples/example_pyrolite_phase_diagram.py"
    )
    bulk = example.PYROLITE_COMPOSITION.atomic_composition
    opts = example.settings(3)
    opts.verbose = True
    result = bm.phase_diagram(
        bulk, example.candidate_phases(), (20e9, 20.2e9), (1400.0, 1420.0), opts
    )
    assert result.resolved, result.diagnostics
    assert len(result.samples) == 9
    assert capfd.readouterr().err.count("LP 0 ") == 1
    opts.verbose = False
    for sample in result.samples:
        reference = bm.stable_equilibrium(
            bulk, example.candidate_phases(), sample.pressure, sample.temperature, opts
        )
        assert reference.success, reference.message
        assert sample.gibbs == pytest.approx(reference.gibbs, abs=1e-5)
        assert sample.mass_balance_error <= opts.mass_balance_tolerance
        assert sample.minimum_affinity >= -opts.affinity_tolerance
        assert [p.candidate_index for p in sample.phases] == [
            p.candidate_index for p in reference.phases
        ]
        for actual, expected in zip(sample.phases, reference.phases):
            assert actual.amount == pytest.approx(expected.amount, abs=1e-8)
            np.testing.assert_allclose(
                actual.composition, expected.composition, atol=1e-7, rtol=0.0
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
    # A saved diagram with that line omitted still has closed, coloured faces.
    # Its perimeter labels still agree, but a sampled assemblage has no
    # corresponding region. Report that gap without adding a new line.
    incomplete = result.to_dict()
    incomplete["resolved"] = False  # Removing a boundary invalidates completion.
    incomplete["boundaries"] = [
        boundary
        for boundary in incomplete["boundaries"]
        if boundary["zero_phase"] != line.zero_phase
    ]
    geometry = bm.phase_diagram_field_polygons(incomplete)
    assert not geometry.diagnostics
    for passes in (0, 2):
        opts.max_recovery_passes = passes
        unresolved = bm.refine_phase_diagram(
            {"Mg": 0.3, "Fe": 0.3, "Ca": 0.4, "O": 1.0}, phases, incomplete, opts
        )
        assert not unresolved.resolved
        assert any("No closed region" in d for d in unresolved.diagnostics)
        assert len(unresolved.boundaries) == 1
        assert all(
            p.phases for p in bm.phase_diagram_field_polygons(unresolved).polygons
        )


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
    saved["resolved"] = False  # Deliberately stale labels require recovery.
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
    check_neighbours(bm.refine_phase_diagram(bulk, phases, saved))


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


def test_resume_removes_a_verified_retraced_part_of_a_curved_boundary():
    phases = curved_phases()
    opts = settings()
    opts.max_recovery_passes = 0
    bulk = {"Mg": 1.0, "O": 1.0}
    result = bm.phase_diagram(bulk, phases, (0.0, 5.0e9), (600.0, 1400.0), opts)
    assert result.resolved, result.diagnostics
    assert len(result.boundaries) == 1
    saved = result.to_dict()
    line = saved["boundaries"][0]
    original = line["points"]
    assert len(original) > 10
    # The retained edge and the duplicate have real equilibrium states, but
    # use different chords. A closed out-and-back trace has zero area; it
    # must not create a field or make an otherwise complete loop open.
    line["points"] = original[::6] + [original[-1]]
    duplicate = dict(
        line,
        id=1,
        end_node=line["start_node"],
        points=[original[0], original[3], original[0]],
        termination="junction; junction",
    )
    saved["boundaries"].append(duplicate)
    saved["resolved"] = False  # Request repair of the injected retrace.
    assert not any(
        p.has_open_boundary for p in bm.phase_diagram_field_polygons(saved).polygons
    )
    repaired = bm.refine_phase_diagram(bulk, phases, saved, opts)
    assert repaired.resolved, repaired.diagnostics
    assert len(repaired.boundaries) == 1
    geometry = bm.phase_diagram_field_polygons(repaired)
    assert not geometry.diagnostics
    assert len(geometry.polygons) == 2
    assert sum(p.area for p in geometry.polygons) == pytest.approx(1.0)


def test_resume_does_not_relocate_or_reverify_samples_from_another_domain():
    mg = pure("Mg", {"Mg": 1.0, "O": 1.0})
    fe = pure("Fe", {"Fe": 1.0, "O": 1.0})
    phases = [
        bm.Solution(
            bm.IdealSolution([(mg, "[Mg]O"), (fe, "[Fe]O")]),
            [0.3, 0.7],
            name="solution",
        ),
        bm.CombinedMineral([mg], [1.0], [1000.0, 0.0, -1.0e-5], name="Mg high"),
    ]
    bulk = {"Mg": 0.3, "Fe": 0.7, "O": 1.0}
    opts = settings()
    opts.pressure_seeds = opts.temperature_seeds = 2
    opts.max_recovery_passes = 0
    result = bm.phase_diagram(bulk, phases, (1.5e9, 1.6e9), (700.0, 800.0), opts)
    assert result.resolved, result.diagnostics
    assert all(len(s.phases) == 2 for s in result.samples if s.success)
    saved = result.to_dict()
    # Reuse a distant warm start in a small subdomain where the second phase
    # is unstable. Dropping it must preserve the requested field coordinates.
    saved["boundaries"] = []
    saved["nodes"] = []
    saved["pressure_range"] = [1e5, 1.01e5]
    saved["temperature_range"] = [700.0, 700.1]
    saved["samples"] = saved["samples"][:1]
    repaired = bm.refine_phase_diagram(bulk, phases, saved, opts)
    assert repaired.to_dict()["samples"] == saved["samples"]
    assert repaired.equilibrium_solves == saved["equilibrium_solves"]
    geometry = bm.phase_diagram_field_polygons(repaired)
    assert geometry.polygons[0].is_provisional
    assert any("label is provisional" in d for d in geometry.diagnostics)


def test_solvus_recovery_matches_compositions_of_renumbered_solution_copies():
    example = runpy.run_path(
        str(Path(__file__).parents[2] / "examples/example_pyrolite_phase_diagram.py")
    )
    # Two native warm starts around the pyrolite calcium-ferrite solvus.
    # The sole CF composition corresponds to copy #2 of the two-phase state.
    previous = json.loads(
        (Path(__file__).parent / "data" / "pyrolite_solvus_seeds.json").read_text()
    )
    opts = settings()
    opts.max_lines = 1  # Isolate this branch from subsequent junction searches.
    result = bm.refine_phase_diagram(
        example["PYROLITE_COMPOSITION"].atomic_composition,
        example["candidate_phases"](),
        previous,
        opts,
    )
    assert len(result.boundaries) == 1, result.diagnostics
    line = result.boundaries[0]
    assert line.zero_phase == 39
    assert sorted([len(line.side_a), len(line.side_b)]) == [7, 8]
    for point in line.points:
        assert point.mass_balance_error < 1e-8
        assert point.minimum_affinity >= -opts.affinity_tolerance
        assert point.residual < 0.02
        zero = next(p for p in point.phases if p.id == line.zero_phase)
        assert abs(zero.amount) < 1e-9


@pytest.mark.parametrize("reverse_order", [False, True])
def test_equivalent_retrace_inherits_verified_labels_with_correct_orientation(
    reverse_order,
):
    example = runpy.run_path(
        str(Path(__file__).parents[2] / "examples/example_pyrolite_phase_diagram.py")
    )
    # Opposite traces of the same native spinel-out line: one has reliable
    # adjacent-field solves, while the other has unresolved side labels.
    previous = json.loads(
        (Path(__file__).parent / "data" / "pyrolite_duplicate_curve.json").read_text()
    )
    expected = previous["boundaries"][0]
    if reverse_order:
        previous["boundaries"].reverse()
    opts = settings()
    opts.max_lines = 2
    opts.max_recovery_passes = 0
    result = bm.refine_phase_diagram(
        example["PYROLITE_COMPOSITION"].atomic_composition,
        example["candidate_phases"](),
        previous,
        opts,
    )
    assert len(result.boundaries) == 1, result.diagnostics
    line = result.boundaries[0]
    assert line.side_a == expected["side_a"]
    assert line.side_b == expected["side_b"]
    assert not any("neighbouring fields" in d for d in result.diagnostics)
    for point in line.points:
        assert point.mass_balance_error < 1e-8
        assert point.minimum_affinity >= -opts.affinity_tolerance
        assert point.residual < 0.02


@pytest.mark.parametrize("ambiguous_edges", [1, 2])
def test_label_recovery_still_merges_duplicate_two_point_boundaries(ambiguous_edges):
    bulk = {"Mg": 1.0, "Fe": 1.0, "O": 2.0}
    opts = settings()
    opts.max_recovery_passes = 0
    saved = bm.phase_diagram(
        bulk, crossing_phases(), (0.0, 2e9), (600.0, 1400.0), opts
    ).to_dict()
    line = saved["boundaries"][0]
    expected = dict(line)
    line["points"] = [line["points"][0], line["points"][-1]]
    duplicate = dict(
        line,
        id=len(saved["boundaries"]),
        start_node=line["end_node"],
        end_node=line["start_node"],
        points=line["points"][::-1],
        side_a=[],
        side_b=[],
    )
    if ambiguous_edges == 2:
        line["side_a"] = line["side_b"] = []
    saved["boundaries"].append(duplicate)
    saved["resolved"] = False  # Request repair of the injected duplicate.
    result = bm.refine_phase_diagram(bulk, crossing_phases(), saved, opts)
    assert result.resolved, result.diagnostics
    assert len(result.boundaries) == 4
    retained = next(b for b in result.boundaries if b.assemblage == line["assemblage"])
    aligned = retained.start_node == expected["start_node"]
    assert retained.side_a == expected["side_a" if aligned else "side_b"]
    assert retained.side_b == expected["side_b" if aligned else "side_a"]
    geometry = bm.phase_diagram_field_polygons(result)
    assert not geometry.diagnostics
    assert len(geometry.polygons) == 4
    assert sum(p.area for p in geometry.polygons) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "resolution, saved_resolved", [(None, False), ((17, 21), False), ((17, 21), True)]
)
def test_resume_uses_accepted_endpoints_and_closes_truncated_fields(
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
    assert resumed.resolved, resumed.diagnostics
    assert len(resumed.boundaries) == 4
    assert (
        len(next(n for n in resumed.nodes if n.kind == "junction").incident_lines) == 4
    )
    assert all(
        line.start_node >= 0 and line.end_node >= 0 for line in resumed.boundaries
    )
    polygons = bm.phase_diagram_field_polygons(resumed)
    assert not polygons.diagnostics
    assert len(polygons.polygons) == 4
    assert all(p.n_phases == 2 for p in polygons.polygons)
    assert resumed.equilibrium_solves > result.equilibrium_solves
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


def test_conflicting_loops_recheck_stale_edges_without_interior_samples():
    bulk = {"Mg": 1.0, "Fe": 1.0, "O": 2.0}
    previous = bm.phase_diagram(
        bulk, crossing_phases(), (0.0, 2e9), (600.0, 1400.0), settings()
    ).to_dict()
    expected = {tuple(f["phases"]) for f in previous["fields"]}
    labels = [f["phases"] for f in previous["fields"][:2]]
    for line in previous["boundaries"]:
        line["side_a"], line["side_b"] = labels
    previous["samples"] = previous["fields"] = []
    previous["resolved"] = False  # Recheck the deliberately corrupted labels.
    opts = settings()
    opts.max_lines = 8
    opts.max_recovery_passes = 0
    result = bm.refine_phase_diagram(bulk, crossing_phases(), previous, opts)
    assert not any("No field edge" in d for d in result.diagnostics)
    polygons = bm.phase_diagram_field_polygons(result)
    assert len(polygons.polygons) == 4
    assert all(p.n_phases == 2 and p.phases for p in polygons.polygons)
    assert {tuple(p.phases) for p in polygons.polygons} == expected
    assert not any(p.is_provisional for p in polygons.polygons)
    assert result.resolved, result.diagnostics
    assert result.equilibrium_solves > previous["equilibrium_solves"]
    assert not any(s.is_field_verification for s in result.samples)


@pytest.mark.parametrize("case_index", range(3))
def test_metasediment_phase_entry_recovers_verified_junction(case_index):
    """Matched metapelite: a solvus, a polymorph swap and a solution entering."""
    example = runpy.run_path(
        str(
            Path(__file__).parents[2]
            / "examples"
            / "example_metasediment_phase_diagram.py"
        )
    )
    fixtures = json.loads(
        (Path(__file__).parent / "data" / "metasediment_endpoints.json").read_text()
    )
    case = fixtures["cases"][case_index]
    s = settings()
    s.max_lines = 1  # Isolate the endpoint from the subsequent branch search.
    previous = case["previous"]
    phases = example["candidate_phases"]()
    result = bm.refine_phase_diagram(
        example["METASEDIMENT_COMPOSITION"].atomic_composition,
        phases,
        previous,
        s,
    )
    line = result.boundaries[0]
    end = case["end"]
    assert getattr(line, end + "_node") >= 0, (case["name"], result.diagnostics)
    assert line.termination.split("; ")[0 if end == "start" else 1] == "junction"
    point = line.points[0 if end == "start" else -1]
    node = result.nodes[getattr(line, end + "_node")]
    assert node.kind == "junction"
    assert sorted(node.zero_phases) == sorted(case["expected_zero_phases"])
    assert point.mass_balance_error < 1.0e-8
    assert point.minimum_affinity >= -0.2
    assert point.residual < 0.02
    for zero in node.zero_phases:
        assert abs(next(p.amount for p in point.phases if p.id == zero)) < 1.0e-9
    # The original states initialise the solve. Points before the event remain
    # intact; a tolerance-sized overshoot is trimmed from the displayed curve.
    original = previous["boundaries"][0]["points"]
    retained = np.array([(p.pressure, p.temperature) for p in line.points])
    for p in original[1:] if end == "start" else original[:-1]:
        assert np.any(
            np.all(
                np.isclose(
                    retained,
                    [p["pressure"], p["temperature"]],
                    rtol=1.0e-12,
                    atol=1.0e-8,
                ),
                axis=1,
            )
        )
    approach = line.points[:3][::-1] if end == "start" else line.points[-3:]
    u = np.array([(p.pressure, p.temperature) for p in approach]) / [
        2.0e9 - 1.0e5,
        600.0,
    ]
    assert (u[2] - u[1]) @ (u[1] - u[0]) >= -1.0e-18


@pytest.mark.parametrize(
    "pressure,temperature,absent",
    [
        (15.715867e8, 802.309284, "ru"),
        (5.1996211e8, 588.4336727, "melt"),
    ],
)
def test_basalt_fixed_pt_drops_blocked_zero_amount_phase(pressure, temperature, absent):
    example = runpy.run_path(
        str(Path(__file__).parents[2] / "examples" / "example_basalt_phase_diagram.py")
    )
    state = bm.stable_equilibrium(
        example["BASALT_COMPOSITION"].atomic_composition,
        example["candidate_phases"](),
        pressure,
        temperature,
    )
    assert state.success, state.message
    assert state.mass_balance_error < 1.0e-8
    assert state.equilibrium_error <= 0.02
    assert state.minimum_affinity >= -0.2
    assert absent not in [p.name for p in state.phases]
    total = sum(p.amount for p in state.phases)
    assert all(p.amount > 1.0e-7 * total for p in state.phases)


def test_polymorph_reduced_variance_has_three_lines():
    phases = [HP11.andalusite(), HP11.ky(), HP11.sill()]
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


@pytest.mark.parametrize("clear_labels", [False, True])
@pytest.mark.parametrize("coarse", [False, True])
def test_pyrolite_replacement_probe_does_not_skip_the_narrow_ak_field(
    clear_labels, coarse
):
    saved = json.loads(
        (
            Path(__file__).parent / "data" / "pyrolite_narrow_solution_replacement.json"
        ).read_text()
    )["previous"]
    saved["resolved"] = False  # This fixture has an incorrect branch label.
    if not clear_labels:
        saved["boundaries"][0]["side_a"] = [6, 27, 39, 40, 45, 60, 78]
        saved["boundaries"][0]["side_b"] = [6, 18, 27, 39, 40, 45, 60, 78]
    if coarse:
        points = saved["boundaries"][0]["points"]
        saved["boundaries"][0]["points"] = [points[0], points[-1]]
    example = runpy.run_path(
        str(
            Path(__file__).parents[2] / "examples" / "example_pyrolite_phase_diagram.py"
        )
    )
    result = bm.refine_phase_diagram(
        saved["composition_start"],
        example["candidate_phases"](),
        saved,
        resolution=(2, 2),
    )
    assert len(result.boundaries) == 1
    boundary = result.boundaries[0]
    assert boundary.is_solution_replacement
    assert boundary.side_a == boundary.side_b == [6, 18, 27, 39, 40, 45, 60, 78]
    # Both sides contain garnet and one ak, but on different solution branches.
    nearby = [s for s in result.samples if s.success and len(s.phases) == 8]
    ak = [next(p for p in s.phases if p.candidate_index == 9) for s in nearby]
    assert any(p.composition[2] > 0.9 for p in ak)  # Corundum-rich.
    assert any(p.composition[0] > 0.9 for p in ak)  # Mg-akimotoite-rich.


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


@pytest.mark.parametrize("crowded_neighbours", [False, True])
def test_resume_retains_conflicting_seed_assemblages_as_provisional_evidence(
    crowded_neighbours,
):
    lower = pure("lower", {"Mg": 1.0, "O": 1.0})
    higher = bm.CombinedMineral([lower], [1.0], [0.1, 0.0, 0.0], name="higher")
    opts = settings()
    result = bm.phase_diagram(
        {"Mg": 1.0, "O": 1.0},
        [higher, lower],
        (1.0e9, 2.0e9),
        (600.0, 800.0),
        opts,
    )
    saved = result.to_dict()
    saved["samples"] = saved["samples"][:2]
    saved["resolved"] = False  # The altered sample evidence needs rechecking.
    first, second = saved["samples"]
    first.update(
        pressure=1.5e9,
        temperature=700.0,
        is_field_verification=True,
        minimum_affinity=-0.1,
    )
    first["phases"][0].update(id=0, candidate_index=0, name="higher")
    second.update(pressure=1.2e9, temperature=640.0)
    if crowded_neighbours:
        # The lower-energy state is far along the region. Nearby higher-energy
        # roots must not crowd that known conflicting assemblage out of recovery.
        saved["samples"].extend(
            dict(first, pressure=1.5e9 + (i + 1) * 100.0) for i in range(36)
        )
        saved["settings"]["max_recovery_passes"] = 5
    recovered = bm.refine_phase_diagram(
        saved["composition_start"], [higher, lower], saved
    )
    if crowded_neighbours:
        recovered = bm.refine_phase_diagram(
            saved["composition_start"], [higher, lower], recovered
        )
    assert not recovered.resolved
    assert not recovered.boundaries
    assert {tuple(f.phases) for f in recovered.fields} == {(0,), (3,)}
    assert recovered.to_dict()["samples"] == saved["samples"]
    assert recovered.equilibrium_solves == saved["equilibrium_solves"]
    geometry = bm.phase_diagram_field_polygons(recovered)
    assert len(geometry.polygons) == 1
    assert geometry.polygons[0].is_provisional
    assert geometry.polygons[0].phases == [0]


def test_pyrolite_small_lp_seed_grows_during_equilibrium_refinement():
    example = runpy.run_path(
        str(Path(__file__).parents[2] / "examples/example_pyrolite_phase_diagram.py")
    )
    opts = bm.PhaseDiagramSettings()
    state = bm.stable_equilibrium(
        example["PYROLITE_COMPOSITION"].atomic_composition,
        example["candidate_phases"](),
        4.92563045e9,
        143.78190438,
        opts,
    )
    assert state.success, state.message
    spinel = next(phase for phase in state.phases if phase.name == "sp")
    assert spinel.amount > opts.amount_tolerance * sum(p.amount for p in state.phases)
    assert state.minimum_affinity >= -opts.affinity_tolerance
    assert state.mass_balance_error < opts.mass_balance_tolerance


@pytest.mark.parametrize("case_index", [0, 1])
def test_pyrolite_short_branches_keep_both_verified_neighbours(case_index):
    example = runpy.run_path(
        str(Path(__file__).parents[2] / "examples/example_pyrolite_phase_diagram.py")
    )
    cases = json.loads(
        (Path(__file__).parent / "data/pyrolite_short_branch_labels.json").read_text()
    )
    saved = cases[case_index]
    result = bm.refine_phase_diagram(
        saved["composition_start"], example["candidate_phases"](), saved
    )
    assert len(result.boundaries) == 1
    line = result.boundaries[0]
    assert line.side_a and line.side_b
    if case_index == 0:
        # Nine phases can coexist on this PT line, but the eight-component
        # bulk cannot support a nine-phase, two-dimensional neighbouring field.
        assert max(len(line.side_a), len(line.side_b)) <= 8
        assert line.is_solution_replacement
        assert line.side_a == line.side_b == [6, 12, 18, 19, 27, 39, 60, 78]
        neighbours = [s for s in result.samples if s.success and len(s.phases) == 8]
        ak = [next(p for p in s.phases if p.candidate_index == 9) for s in neighbours]
        assert any(p.composition[2] > 0.9 for p in ak)  # Corundum-rich.
        assert any(p.composition[4] > 0.9 for p in ak)  # Hematite-rich.
        for state in neighbours:
            assert state.minimum_affinity >= -result.settings.affinity_tolerance
            assert state.equilibrium_error <= result.settings.affinity_tolerance * 0.1
            assert state.mass_balance_error <= result.settings.mass_balance_tolerance
    else:
        assert line.side_a != line.side_b
    assert not any("neighbouring fields" in d for d in result.diagnostics)


def test_required_eos_limit_keeps_the_accepted_pyrolite_boundary():
    example = runpy.run_path(
        str(Path(__file__).parents[2] / "examples/example_pyrolite_phase_diagram.py")
    )
    opts = bm.PhaseDiagramSettings()
    opts.pressure_seeds = opts.temperature_seeds = 2
    opts.required_eos_phases = ["opx"]
    result = bm.phase_diagram(
        example["PYROLITE_COMPOSITION"].atomic_composition,
        example["candidate_phases"](),
        (7.0e9, 10.0e9),
        (3800.0, 4000.0),
        opts,
    )
    assert result.resolved, result.diagnostics
    boundary = next(
        line
        for line in result.boundaries
        if line.zero_phase == 0 and "model domain limit" in line.termination
    )
    assert len(boundary.points) >= 2
    assert all(p.minimum_affinity >= -opts.affinity_tolerance for p in boundary.points)
    assert all(
        p.mass_balance_error < opts.mass_balance_tolerance for p in boundary.points
    )
    assert any(node.kind == "model_domain_limit" for node in result.nodes)


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


@pytest.mark.parametrize(
    "pressure,temperature",
    [(1e9, 873.15), (2e9, 573.15), (1.3333666666666665e9, 573.15)],
)
def test_basalt_setup_exact_water_and_native_stable_state(pressure, temperature):
    reference_loaded = "burnman" in sys.modules
    example = runpy.run_path(
        str(
            Path(__file__).resolve().parents[2]
            / "examples/example_basalt_phase_diagram.py"
        )
    )
    oxides = example["BASALT_OXIDES"]
    assert sum(oxides.values()) == 100.0
    assert oxides["H2O"] == 2.0
    composition = example["BASALT_COMPOSITION"]
    assert isinstance(composition, bm.Composition)
    assert sum(composition.mass_composition.values()) == pytest.approx(0.1)
    assert composition.mass_composition["H2O"] == pytest.approx(0.002)
    bulk = composition.atomic_composition
    state = bm.stable_equilibrium(
        bulk, example["candidate_phases"](), pressure, temperature
    )
    assert state.success, state.message
    assert state.mass_balance_error < 1e-8
    assert state.minimum_affinity >= -0.2
    assert state.equilibrium_error <= 0.02
    assert ("burnman" in sys.modules) == reference_loaded


@pytest.mark.parametrize(
    "pressure,temperature", [(1.0e9, 873.15), (2.0e9, 573.15), (5.0e8, 1073.15)]
)
def test_metasediment_setup_exact_water_and_native_stable_state(pressure, temperature):
    reference_loaded = "burnman" in sys.modules
    example = runpy.run_path(
        str(
            Path(__file__).resolve().parents[2]
            / "examples/example_metasediment_phase_diagram.py"
        )
    )
    oxides = example["METASEDIMENT_OXIDES"]
    assert sum(oxides.values()) == 100.0
    assert oxides["H2O"] == 2.0
    phases = example["candidate_phases"]()
    names = [p.name for p in phases]
    assert len(names) == len(set(names))
    assert {
        "ms",
        "bi",
        "chl",
        "ctd",
        "st",
        "cd",
        "g",
        "fsp",
        "and",
        "ky",
        "sill",
        "melt",
        "H2O",
    } <= set(names)
    composition = example["METASEDIMENT_COMPOSITION"]
    assert isinstance(composition, bm.Composition)
    assert sum(composition.mass_composition.values()) == pytest.approx(0.1)
    assert composition.mass_composition["H2O"] == pytest.approx(0.002)
    state = bm.stable_equilibrium(
        composition.atomic_composition, phases, pressure, temperature
    )
    assert state.success, state.message
    assert state.mass_balance_error < 1.0e-8
    assert state.minimum_affinity >= -0.2
    assert state.equilibrium_error <= 0.02
    assert ("burnman" in sys.modules) == reference_loaded


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


@pytest.mark.parametrize("alphas", [(1.0, 1.0), (1.5, 1.0)])
@pytest.mark.parametrize("step", [0.025, 0.05, 0.1])
def test_two_solvus_arms_close_at_composition_critical_point(alphas, step, tmp_path):
    R, W = 8.31446261815324, 13000.0
    a, b = alphas
    x = b / (a + np.sqrt(a * a - (a - b) * b))
    S = a * x + b * (1.0 - x)
    B = 2.0 * W * a * b / (a + b)
    critical_temperature = 2.0 * B * a * b * x * (1.0 - x) / (R * S**3)
    chemical_difference = (
        R * critical_temperature * np.log(x / (1.0 - x))
        + B * ((1.0 - 2.0 * x) * S - x * (1.0 - x) * (a - b)) / S**2
    )
    mg = pure("Mg oxide", {"Mg": 1.0, "O": 1.0})
    fe = pure("Fe oxide", {"Fe": 1.0, "O": 1.0})
    main = bm.Solution(
        bm.AsymmetricRegularSolution(
            [(mg, "[Mg]O"), (fe, "[Fe]O")],
            alphas=list(alphas),
            energy_interaction=[[W]],
        ),
        [0.5, 0.5],
        name="oxide",
    )
    mg2 = pure("Mg dioxide", {"Mg": 1.0, "O": 2.0})
    fe2 = bm.CombinedMineral(
        [pure("Fe dioxide", {"Fe": 1.0, "O": 2.0})],
        [1.0],
        [1000.0 - chemical_difference, 0.0, -1.0e-6],
        name="Fe dioxide shifted",
    )
    auxiliary = bm.Solution(
        bm.IdealSolution([(mg2, "[Mg]O2"), (fe2, "[Fe]O2")]), [0.5, 0.5], name="dioxide"
    )
    bulk = {"Mg": 0.6 * x + 0.2, "Fe": 0.8 - 0.6 * x, "O": 1.4}
    s = settings()
    s.step = step
    s.pressure_seeds = s.temperature_seeds = 5
    r = bm.phase_diagram(bulk, [main, auxiliary], (0.0, 2.0e9), (600.0, 900.0), s)
    assert r.resolved, r.diagnostics
    # Oxygen balance fixes main/auxiliary amounts at .6/.4. At the critical
    # auxiliary composition .5 gives the critical pressure 1 GPa. For the
    # asymmetric regular solution, G''=G'''=0 give x and T analytically above;
    # the symmetric limit is x=.5 and T=W/(2R).
    critical = [n for n in r.nodes if n.kind == "critical_point" and n.incident_lines]
    assert len(critical) == 1, r.diagnostics
    node = critical[0]
    assert abs(node.pressure - 1.0e9) / (2.0e9) < s.node_tolerance * 2.0
    assert abs(node.temperature - critical_temperature) / 300.0 < s.node_tolerance * 2.0
    assert len(node.incident_lines) == 2
    assert np.linalg.norm(node.critical_mode) == pytest.approx(1.0)
    for index in node.incident_lines:
        edge = r.boundaries[index]
        other = r.nodes[
            edge.end_node if edge.start_node == node.id else edge.start_node
        ]
        # In this model each arm approaches its critical pressure from one
        # side. Nearly identical-copy roots must not overshoot and double back.
        lower, upper = sorted([node.pressure, other.pressure])
        pressures = [p.pressure for p in edge.points]
        assert min(pressures) >= lower - 0.3
        assert max(pressures) <= upper + 0.3
        point = edge.points[0 if edge.start_node == node.id else -1]
        copies = [p.composition for p in point.phases if p.candidate_index == 0]
        assert len(copies) == 2
        np.testing.assert_allclose(copies, [[x, 1.0 - x], [x, 1.0 - x]], atol=1e-8)
    geometry = bm.phase_diagram_field_polygons(r)
    assert all(p.n_phases > 0 for p in geometry.polygons), geometry.diagnostics
    assert {p.n_phases for p in geometry.polygons} == {2, 3}
    assert sum(p.area for p in geometry.polygons) == pytest.approx(1.0)
    # Verified critical endpoints intentionally have equal compositions.
    # Saving/resuming must preserve them rather than rewinding them as false
    # junctions; the recorded mode can seed an incomplete adjoining arm.
    import json

    example = runpy.run_path(
        str(
            Path(__file__).resolve().parents[2]
            / "examples/example_basalt_phase_diagram.py"
        )
    )
    path = tmp_path / "critical.json"
    example["save_json"](r, path)
    resumed = bm.refine_phase_diagram(
        bulk, [main, auxiliary], json.loads(path.read_text()), s
    )
    assert resumed.resolved, resumed.diagnostics
    assert len(resumed.boundaries) == 2
    assert all(
        p.n_phases > 0 for p in bm.phase_diagram_field_polygons(resumed).polygons
    )
