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

"""Thermodynamic sections use constrained totals, including latent volume."""

import json
import re
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

import burnman_cpp as bm
from conftest import oxide_params

AXES = dict(
    P="pressure", T="temperature", S="entropy", V="volume", X="composition_coordinate"
)


def test_local_pyrolite_akimotoite_branches_bound_equilibratable_fields():
    from burnman_cpp.tools.phase_diagram_helpers import load_example

    saved = json.loads(
        (
            Path(__file__).with_name("data") / "pyrolite_sp_ak_branch_switch.json"
        ).read_text()
    )
    example = load_example(
        Path(__file__).resolve().parents[2]
        / "examples/example_pyrolite_phase_diagram.py"
    )
    patch = bm.calculate_phase_diagram_patch(
        saved,
        example.candidate_phases(),
        ((138.0, 155.5), (21.2, 22.5)),
        pressure_unit="GPa",
        entropy_unit="J/K",
    )
    assert patch.resolved, patch.diagnostics
    double_zero = [node for node in patch.nodes if node.zero_phases == [27, 28]]
    assert len(double_zero) == 1
    assert double_zero[0].pressure == pytest.approx(21.429396678594486e9, abs=1.0)
    assert double_zero[0].entropy == pytest.approx(139.09580456113144, abs=1.0e-6)
    grid = bm.phase_diagram_grid(
        patch,
        example.candidate_phases(),
        ["temperature", "thermal_expansivity"],
        **saved["grid"],
        store_solutions=True,
    )
    assert grid.success.all(), grid.diagnostics
    # The reported point was assigned akimotoite but cannot have positive ak.
    row = np.flatnonzero(np.isclose(grid.y, 145.1533803755081))[0]
    column = np.flatnonzero(np.isclose(grid.x, 22.0e9))[0]
    state = grid.solution_at(int(row), int(column)).state
    assert {p.candidate_index for p in state.phases} == {2, 6, 13, 20, 26}
    assert state.temperature == pytest.approx(787.5233434907014, abs=1.0e-4)


def test_original_scale_curve_does_not_attach_to_another_akimotoite_branch():
    from burnman_cpp.tools.phase_diagram_helpers import load_example

    saved = json.loads(
        (
            Path(__file__).with_name("data") / "pyrolite_sp_wrong_branch_junction.json"
        ).read_text()
    )
    example = load_example(
        Path(__file__).resolve().parents[2]
        / "examples/example_pyrolite_phase_diagram.py"
    )
    result = bm.refine_phase_diagram(
        saved["composition_start"],
        example.candidate_phases(),
        saved,
        resolution=(2, 2),
        region=(saved["entropy_range"], saved["pressure_range"]),
        pressure_unit="Pa",
        entropy_unit="J/K",
    )
    # Both curves share the present assemblage, but the capv/ak junction lies
    # on only one akimotoite composition branch. Coarse chords of the other
    # curve pass nearby in the original full-diagram frame.
    a, b, c = (saved["fixture_original_nodes"].index(i) for i in [214, 242, 243])
    assert any({line.start_node, line.end_node} == {a, c} for line in result.boundaries)
    assert len(result.nodes[b].incident_lines) == 1
    assert len(result.boundaries) == 2


@pytest.mark.parametrize("diagram", ["PS", "TV", "SV"])
def test_natural_coordinate_grid_solves_without_pt_equilibrium_search(diagram, capfd):
    phase = bm.Mineral(oxide_params())
    phase.set_state(1e9, 1000.0)
    entropy, volume = phase.molar_entropy, phase.molar_volume
    opts = bm.PhaseDiagramSettings()
    opts.pressure_seeds = opts.temperature_seeds = 3
    opts.entropy_seeds = opts.volume_seeds = 3
    opts.max_phase_instances = 1
    opts.verbose = True
    result = bm.phase_diagram(
        {"Mg": 1.0, "O": 1.0},
        [phase],
        diagram=diagram,
        pressure_range=(0.9e9, 1.1e9),
        temperature_range=(995.0, 1005.0),
        entropy_range=(entropy - 0.2, entropy + 0.2),
        volume_range=(volume * 0.999, volume * 1.001),
        settings=opts,
    )
    assert result.resolved, result.diagnostics
    assert len(result.samples) == 9
    # One solve per requested equilibrium; no preliminary P-T equilibrium.
    assert result.equilibrium_solves == 9
    assert "LP " not in capfd.readouterr().err
    expected = {
        "P": np.linspace(0.9e9, 1.1e9, 3),
        "T": np.linspace(995.0, 1005.0, 3),
        "S": np.linspace(entropy - 0.2, entropy + 0.2, 3),
        "V": np.linspace(volume * 0.999, volume * 1.001, 3),
    }
    for i, state in enumerate(result.samples):
        assert state.success, state.message
        for axis, index in zip(diagram, divmod(i, 3)):
            assert getattr(state, AXES[axis]) == pytest.approx(
                expected[axis][index], rel=1e-9
            )
        phase.set_state(state.pressure, state.temperature)
        assert state.entropy == pytest.approx(phase.molar_entropy, rel=1e-9)
        assert state.volume == pytest.approx(phase.molar_volume, rel=1e-9)


@pytest.mark.parametrize("reverse", [False, True])
def test_saved_entropy_junction_recovers_all_four_branches(reverse):
    from burnman_cpp.tools.phase_diagram_helpers import load_example

    saved = json.loads(
        (Path(__file__).with_name("data") / "pyrolite_sp_junction.json").read_text()
    )
    if reverse:
        for line in saved["boundaries"]:
            line["points"].reverse()
            line["start_node"], line["end_node"] = (
                line["end_node"],
                line["start_node"],
            )
            line["side_a"], line["side_b"] = line["side_b"], line["side_a"]
            line["termination"] = "; ".join(reversed(line["termination"].split("; ")))
    original = deepcopy(saved)
    example = load_example(
        Path(__file__).resolve().parents[2]
        / "examples/example_pyrolite_phase_diagram.py"
    )
    result = bm.refine_phase_diagram(
        saved["composition_start"], example.candidate_phases(), saved
    )
    # These branches start from the saved 1952 K junction. Replacing its T by
    # the physical seed's 2273 K changes its entropy and rejects both trials.
    junction = result.nodes[saved["fixture_original_nodes"].index(89)]
    assert len(junction.incident_lines) == 4
    assert junction.pressure == pytest.approx(12.400943648e9, abs=1.0)
    assert junction.temperature == pytest.approx(1952.634895, abs=1e-6)
    assert junction.entropy == pytest.approx(266.543204, abs=1e-6)
    added = [line for line in result.boundaries if line.id >= 2]
    assert {line.zero_phase for line in added} == {9, 75}  # opx and Fe-fcc.
    for line in added:
        assert junction.id in (line.start_node, line.end_node)
        assert line.assemblage == [0, 9, 12, 15, 18, 75]
        absent = [i for i in line.assemblage if i != line.zero_phase]
        assert {tuple(line.side_a), tuple(line.side_b)} == {
            tuple(line.assemblage),
            tuple(absent),
        }
        for point in line.points:
            assert point.minimum_affinity >= -result.settings.affinity_tolerance
            assert point.mass_balance_error <= result.settings.mass_balance_tolerance
            assert point.residual < 0.01
    assert saved == original


def test_intensive_invariant_is_a_latent_entropy_edge():
    first = bm.Mineral(oxide_params())
    dense = bm.CombinedMineral(
        [first], [2.0], [10000.0, 0.0, -2.0e-6], name="double dense"
    )
    hot = bm.CombinedMineral(
        [first], [1.0], [7500.0, 5.0, -0.5e-6], name="high entropy"
    )
    first.set_state(5e9, 1000.0)
    entropy = first.molar_entropy
    opts = bm.PhaseDiagramSettings()
    opts.max_phase_instances = 1
    opts.pressure_seeds = opts.entropy_seeds = 3
    result = bm.phase_diagram(
        {"Mg": 1.0, "O": 1.0},
        [first, dense, hot],
        diagram="PS",
        pressure_range=(4.5e9, 5.5e9),
        temperature_range=(800.0, 1200.0),
        entropy_range=(entropy - 2.0, entropy + 7.0),
        settings=opts,
    )
    assert result.resolved, result.diagnostics
    latent = [b for b in result.boundaries if b.assemblage == [0, 1, 2]]
    assert len(latent) == 1
    line = latent[0]
    assert {tuple(line.side_a), tuple(line.side_b)} == {(0, 2), (1, 2)}
    assert all(p.pressure == pytest.approx(5e9, abs=0.2) for p in line.points)
    assert sorted([line.points[0].entropy, line.points[-1].entropy]) == pytest.approx(
        [entropy, entropy + 5.0], abs=1e-7
    )


def test_saved_garnet_saddle_junction_reopens_to_the_true_critical_point():
    from burnman_cpp.tools.phase_diagram_helpers import load_example

    saved = json.loads(
        (
            Path(__file__).with_name("data") / "pyrolite_sp_false_junction.json"
        ).read_text()
    )
    example = load_example(
        Path(__file__).resolve().parents[2]
        / "examples/example_pyrolite_phase_diagram.py"
    )
    result = bm.refine_phase_diagram(
        saved["composition_start"], example.candidate_phases(), saved
    )
    assert not result.nodes[saved["fixture_original_nodes"].index(192)].incident_lines
    critical = result.nodes[saved["fixture_original_nodes"].index(348)]
    assert critical.kind == "critical_point"
    assert len(critical.incident_lines) == 2
    assert critical.pressure == pytest.approx(19.78944e9, abs=1e4)
    assert critical.temperature == pytest.approx(667.4895, abs=1e-3)
    assert all("solution critical point" in b.termination for b in result.boundaries)
    # The two-line fixture can discover an additional junction at its outer
    # cut, where max_lines prevents tracing its other arms. Check the repaired
    # saddle and critical endpoint; that separate search limit is outside this
    # critical-point regression.
    repaired = [saved["fixture_original_nodes"].index(i) for i in (192, 348)]
    assert not any(
        "Unpaired" in diagnostic and re.search(rf"\bat node {node}\b", diagnostic)
        for diagnostic in result.diagnostics
        for node in repaired
    )


def test_nonisolated_cpx_third_copy_recovers_the_shared_critical_endpoint():
    from burnman_cpp.tools.phase_diagram_helpers import load_example

    saved = json.loads(
        (
            Path(__file__).with_name("data") / "pyrolite_sp_false_cpx_junction.json"
        ).read_text()
    )
    example = load_example(
        Path(__file__).resolve().parents[2]
        / "examples/example_pyrolite_phase_diagram.py"
    )
    result = bm.refine_phase_diagram(
        saved["composition_start"], example.candidate_phases(), saved
    )
    old = saved["fixture_original_nodes"]
    assert not result.nodes[old.index(377)].incident_lines
    critical = result.nodes[old.index(376)]
    assert critical.kind == "critical_point"
    assert len(critical.incident_lines) == 2
    assert critical.pressure == pytest.approx(13.137396517e9, abs=1.0)
    assert critical.temperature == pytest.approx(730.5591003, abs=1e-6)
    for line in result.boundaries:
        assert critical.id in (line.start_node, line.end_node)
        assert "solution critical point" in line.termination
        endpoint = line.points[-1]
        copies = [p for p in endpoint.phases if p.candidate_index == 4]
        assert len(copies) == 2
        assert np.linalg.norm(copies[0].composition - copies[1].composition) < 1e-8
        assert endpoint.mass_balance_error <= result.settings.mass_balance_tolerance
    assert not any("Unpaired" in diagnostic for diagnostic in result.diagnostics)


def test_parallel_arc_cleanup_retains_the_recovered_akimotoite_boundary():
    from burnman_cpp.tools.phase_diagram_helpers import load_example

    saved = json.loads(
        (
            Path(__file__).with_name("data") / "pyrolite_sp_missing_ak_boundary.json"
        ).read_text()
    )
    example = load_example(
        Path(__file__).resolve().parents[2]
        / "examples/example_pyrolite_phase_diagram.py"
    )
    result = bm.refine_phase_diagram(
        saved["composition_start"], example.candidate_phases(), saved
    )
    junction = result.nodes[saved["fixture_original_nodes"].index(321)]
    assert len(junction.incident_lines) == 4
    ak_out = [
        b for b in result.boundaries if b.zero_phase == 27 and 33 not in b.assemblage
    ]
    assert len(ak_out) == 1
    absent = (30, 39, 42, 45, 60, 78)
    assert {tuple(ak_out[0].side_a), tuple(ak_out[0].side_b)} == {
        absent,
        (27,) + absent,
    }
    for point in ak_out[0].points:
        assert point.minimum_affinity >= -result.settings.affinity_tolerance
        assert point.mass_balance_error <= result.settings.mass_balance_tolerance
    polygons = bm.phase_diagram_field_polygons(result)
    assert any(
        tuple(p.phases) == absent and not p.has_open_boundary for p in polygons.polygons
    )


def test_saved_trace_sites_recover_missing_spinel_and_cf_reaction_arms():
    from burnman_cpp.tools.phase_diagram_helpers import load_example

    saved = json.loads(
        (
            Path(__file__).with_name("data") / "pyrolite_sp_trace_site_junctions.json"
        ).read_text()
    )
    example = load_example(
        Path(__file__).resolve().parents[2]
        / "examples/example_pyrolite_phase_diagram.py"
    )
    result = bm.refine_phase_diagram(
        saved["composition_start"], example.candidate_phases(), saved
    )
    original = saved["fixture_original_nodes"]
    for node in (41, 43, 46, 48):
        assert len(result.nodes[original.index(node)].incident_lines) == 4
    full = [0, 9, 12, 18, 21, 22, 39, 51]
    recovered = [
        b
        for b in result.boundaries
        if b.assemblage == full and b.zero_phase in (22, 39)
    ]
    assert sorted(b.zero_phase for b in recovered) == [22, 39]
    for line in recovered:
        assert line.start_node < len(original) and line.end_node < len(original)
        for point in line.points:
            assert point.mass_balance_error <= result.settings.mass_balance_tolerance
            assert point.minimum_affinity >= -result.settings.affinity_tolerance
            assert point.residual < 0.01
    geometry = bm.phase_diagram_field_polygons(result)
    for phases in (
        [0, 9, 12, 13, 18, 21, 22, 39, 51],
        [0, 9, 12, 18, 21, 39, 51],
        [0, 9, 12, 18, 21, 22, 51],
    ):
        assert any(
            p.phases == phases and not p.is_provisional and not p.has_open_boundary
            for p in geometry.polygons
        )


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("fixture", ["unconnected", "cf_copy"])
def test_duplicate_entropy_junctions_preserve_coexisting_solution_copies(
    fixture, reverse
):
    from burnman_cpp.tools.phase_diagram_helpers import load_example

    saved = json.loads(
        (
            Path(__file__).with_name("data") / f"pyrolite_sp_{fixture}_junctions.json"
        ).read_text()
    )
    if reverse:
        for line in saved["boundaries"]:
            line["points"].reverse()
            line["start_node"], line["end_node"] = (
                line["end_node"],
                line["start_node"],
            )
            line["side_a"], line["side_b"] = line["side_b"], line["side_a"]
            line["termination"] = "; ".join(reversed(line["termination"].split("; ")))
    first, last = saved["fixture_duplicate_nodes"][0]
    bridges = sum(
        {line["start_node"], line["end_node"]} == {first, last}
        for line in saved["boundaries"]
    )
    assert bridges == (fixture == "cf_copy")
    original = deepcopy(saved)
    example = load_example(
        Path(__file__).resolve().parents[2]
        / "examples/example_pyrolite_phase_diagram.py"
    )
    result = bm.refine_phase_diagram(
        saved["composition_start"],
        example.candidate_phases(),
        saved,
        resolution=(2, 2),
        region=(
            ((110.0, 114.0), (5.42e9, 5.45e9))
            if fixture == "unconnected"
            else ((184.0, 187.0), (24.3e9, 24.7e9))
        ),
        pressure_unit="Pa",
        entropy_unit="J/K",
    )
    assert saved == original
    assert len(result.nodes[first].incident_lines) == (
        4 if fixture == "unconnected" else 5
    )
    assert not result.nodes[last].incident_lines
    assert len(result.boundaries) == len(saved["boundaries"]) - bridges


def test_fresh_entropy_diagram_preserves_reduced_bases_at_latent_junctions():
    from burnman_cpp.tools.phase_diagram_helpers import load_example

    example = load_example(
        Path(__file__).resolve().parents[2]
        / "examples/example_pyrolite_sp_phase_diagram.py"
    )
    opts = example.settings(3)
    opts.pressure_seeds = opts.entropy_seeds = 3
    opts.required_eos_phases = []
    opts.max_trace_steps = 200
    result = bm.phase_diagram(
        example.pyrolite.PYROLITE_COMPOSITION.atomic_composition,
        example.candidate_phases(),
        diagram="PS",
        pressure_range=(21.387e9, 21.389e9),
        entropy_range=(77.7, 78.15),
        temperature_range=(350.0, 550.0),
        settings=opts,
    )
    assert result.resolved, result.diagnostics
    junctions = sorted(
        (n for n in result.nodes if n.kind == "junction"), key=lambda n: n.entropy
    )
    assert len(junctions) == 2
    assert {tuple(sorted(n.zero_phases)) for n in junctions} == {(18, 42), (27, 42)}
    assert all(len(n.incident_lines) == 4 for n in junctions)
    # These are distinct latent-entropy vertices, not duplicate node records.
    assert junctions[1].entropy - junctions[0].entropy > 0.1
    saved = json.loads(json.dumps(result.to_dict()))
    assert any(
        "composition_basis" in phase
        for line in saved["boundaries"]
        for point in line["points"]
        for phase in point["phases"]
    )
    assert bm.PhaseDiagramResult.from_dict(saved).to_dict() == saved
    saved["resolved"] = False  # Exercise native reconstruction after JSON reload.
    saved["settings"]["max_recovery_passes"] = 0
    resumed = bm.refine_phase_diagram(
        saved["composition_start"],
        example.candidate_phases(),
        saved,
        resolution=(2, 2),
        region=((77.7, 78.15), (21.387e9, 21.389e9)),
        pressure_unit="Pa",
        entropy_unit="J/K",
    )
    assert resumed.resolved, resumed.diagnostics
    assert len(resumed.boundaries) == len(result.boundaries)
    assert all(
        len(n.incident_lines) == 4 for n in resumed.nodes if n.kind == "junction"
    )


def test_fresh_entropy_diagram_keeps_weakly_coupled_nine_phase_fields():
    from burnman_cpp.tools.phase_diagram_helpers import load_example

    example = load_example(
        Path(__file__).resolve().parents[2]
        / "examples/example_pyrolite_sp_phase_diagram.py"
    )
    opts = example.settings(5)
    opts.pressure_seeds = opts.entropy_seeds = 5
    opts.required_eos_phases = []
    opts.max_trace_steps = 200
    result = bm.phase_diagram(
        example.pyrolite.PYROLITE_COMPOSITION.atomic_composition,
        example.candidate_phases(),
        diagram="PS",
        pressure_range=(21.0e9, 21.8e9),
        entropy_range=(74.0, 85.0),
        temperature_range=(350.0, 550.0),
        settings=opts,
    )
    assert result.resolved, result.diagnostics
    assert all(len(n.incident_lines) == 4 for n in result.nodes if n.kind == "junction")
    polygons = bm.phase_diagram_field_polygons(result).polygons
    assert all(p.phases and not p.is_provisional for p in polygons)
    # The small amount/entropy coupling is nonzero: this nine-phase field
    # separates its two eight-phase limits over a finite S-P region.
    full = [6, 27, 28, 39, 40, 42, 45, 60, 78]
    assert any(p.phases == full and not p.has_open_boundary for p in polygons)
    for boundary in result.boundaries:
        for point in boundary.points:
            assert point.mass_balance_error <= result.settings.mass_balance_tolerance
            assert point.minimum_affinity >= -result.settings.affinity_tolerance
            # This is the norm over several reaction equations, each checked
            # against the configured reaction/affinity tolerance by the solver.
            assert point.residual <= result.settings.affinity_tolerance


@pytest.mark.parametrize("entropy", [153.46945231266858, 153.46945231266855])
def test_failed_entropy_bracket_recovers_at_its_requested_coordinates(entropy):
    from burnman_cpp.tools.phase_diagram_helpers import load_example

    saved = json.loads(
        (
            Path(__file__).with_name("data") / "pyrolite_sp_failed_bracket.json"
        ).read_text()
    )
    saved["samples"][0]["entropy"] = entropy
    original = deepcopy(saved)
    failed = saved["samples"][0]
    assert not failed["success"]
    example = load_example(
        Path(__file__).resolve().parents[2]
        / "examples/example_pyrolite_sp_phase_diagram.py"
    )
    result = bm.refine_phase_diagram(
        saved["composition_start"], example.candidate_phases(), saved
    )
    assert saved == original
    recovered = result.samples[0]
    assert recovered.success, recovered.message
    assert recovered.pressure == pytest.approx(failed["pressure"], abs=1.0)
    assert recovered.entropy == pytest.approx(failed["entropy"], rel=1e-9)
    assert recovered.mass_balance_error <= result.settings.mass_balance_tolerance
    assert recovered.minimum_affinity >= -result.settings.affinity_tolerance
    assert recovered.equilibrium_error <= result.settings.affinity_tolerance
    assert not any("Unresolved seed" in message for message in result.diagnostics)


def test_nearby_latent_polymorph_vertices_keep_distinct_entropy_and_labels():
    first = bm.Mineral(oxide_params())
    dense = bm.CombinedMineral([first], [1.0], [5001.0, 0.001, -1.0e-6], name="dense")
    hot = bm.CombinedMineral(
        [first], [1.0], [7500.0, 5.0, -0.5e-6], name="high entropy"
    )
    first.set_state(5e9, 1000.0)
    entropy = first.molar_entropy
    opts = bm.PhaseDiagramSettings()
    opts.max_phase_instances = 1
    opts.pressure_seeds = opts.entropy_seeds = 3
    result = bm.phase_diagram(
        {"Mg": 1.0, "O": 1.0},
        [first, dense, hot],
        diagram="PS",
        pressure_range=(4.5e9, 5.5e9),
        temperature_range=(800.0, 1200.0),
        entropy_range=(entropy - 2.0, entropy + 7.0),
        settings=opts,
    )
    assert result.resolved, result.diagnostics
    vertices = sorted(n.entropy - entropy for n in result.nodes if n.kind == "junction")
    assert vertices == pytest.approx([0.0, 0.001, 5.0], abs=1e-7)
    edges = sorted(
        [b for b in result.boundaries if b.assemblage == [0, 1, 2]],
        key=lambda b: min(b.points[0].entropy, b.points[-1].entropy),
    )
    assert len(edges) == 2
    assert {tuple(edges[0].side_a), tuple(edges[0].side_b)} == {(0, 1), (0, 2)}
    assert {tuple(edges[1].side_a), tuple(edges[1].side_b)} == {(1, 2), (0, 2)}


def test_entropy_diagram_closes_at_the_required_pyrolite_eos_limit():
    from burnman_cpp.tools.phase_diagram_helpers import load_example

    example = load_example(
        Path(__file__).resolve().parents[2]
        / "examples/example_pyrolite_phase_diagram.py"
    )
    opts = example.settings(2)
    opts.entropy_seeds = 2
    result = bm.phase_diagram(
        example.PYROLITE_COMPOSITION.atomic_composition,
        example.candidate_phases(),
        diagram="PS",
        pressure_range=(0.0, 1.2e9),
        entropy_range=(300.0, 313.393912642678),
        temperature_range=(2300.0, 2550.0),
        settings=opts,
    )
    assert result.resolved, result.diagnostics
    assert len(result.excluded_regions) == 1
    assert len(result.boundaries) == 2
    assert all("model domain limit" in b.termination for b in result.boundaries)
    assert all(b.start_node >= 0 and b.end_node >= 0 for b in result.boundaries)
    assert any(n.kind == "model_domain_limit" for n in result.nodes)
    for b in result.boundaries:
        endpoint = b.points[-1]
        check = bm.stable_equilibrium(
            example.PYROLITE_COMPOSITION.atomic_composition,
            example.candidate_phases(),
            endpoint.pressure,
            endpoint.temperature,
            opts,
        )
        assert check.success, check.message
        assert check.entropy == pytest.approx(endpoint.entropy, rel=1e-8)


@pytest.mark.parametrize("scale", [0.1, 3.0])
@pytest.mark.parametrize("diagram", ["PS", "TV", "SV"])
def test_constrained_potential_minima_recover_latent_mixtures(diagram, scale):
    # Independently known coexistence conditions. B contains twice the formula
    # amount of A, so minimizing molar energy would give the wrong objective.
    first = bm.Mineral(oxide_params())
    dense = bm.CombinedMineral(
        [first], [2.0], [10000.0, 0.0, -2.0e-6], name="double dense"
    )
    hot = bm.CombinedMineral(
        [first], [1.0], [7500.0, 5.0, -0.5e-6], name="high entropy"
    )
    first.set_state(5e9, 1000.0)
    s, v = first.molar_entropy, first.molar_volume
    opts = bm.PhaseDiagramSettings()
    opts.max_phase_instances = 1
    options = dict(
        settings=opts, pressure_range=(0.0, 20e9), temperature_range=(400.0, 2400.0)
    )
    if diagram == "PS":
        # A and C coexist at T=1000 K, P=5 GPa, with half of each formula.
        candidates = [first, hot]
        options.update(pressure=5e9, entropy=scale * (s + 2.5))
        expected = {"MgO": 0.5 * scale, "high entropy": 0.5 * scale}
    elif diagram == "TV":
        candidates = [first, dense]
        options.update(temperature=1000.0, volume=scale * (v - 0.4e-6))
        expected = {"MgO": 0.6 * scale, "double dense": 0.2 * scale}
    else:
        candidates = [first, dense, hot]
        options.update(entropy=scale * (s + 2.5), volume=scale * (v - 0.4e-6))
        expected = {
            "MgO": 0.35 * scale,
            "double dense": 0.075 * scale,
            "high entropy": 0.5 * scale,
        }
    result = bm.stable_equilibrium({"Mg": scale, "O": scale}, candidates, **options)
    assert result.success, result.message
    assert result.pressure == pytest.approx(5e9, abs=1.0)
    assert result.temperature == pytest.approx(1000.0, abs=1e-6)
    assert result.mass_balance_error <= opts.mass_balance_tolerance
    assert result.minimum_affinity >= -opts.affinity_tolerance
    actual = {phase.name: phase.amount for phase in result.phases}
    assert actual == pytest.approx(expected, rel=1e-7, abs=1e-9 * scale)
    if "entropy" in options:
        assert result.entropy == pytest.approx(options["entropy"], rel=1e-9)
    if "volume" in options:
        assert result.volume == pytest.approx(options["volume"], rel=1e-9)


@pytest.mark.parametrize("fix_volume", [False, True])
def test_entropy_minimum_can_start_with_an_invalid_hot_eos_endpoint(fix_volume):
    phase = bm.Mineral(oxide_params())
    phase.set_state(1e9, 1000.0)
    entropy = phase.molar_entropy
    options = {"volume": phase.molar_volume} if fix_volume else {"pressure": 1e9}
    result = bm.stable_equilibrium(
        {"Mg": 1.0, "O": 1.0},
        [phase],
        entropy=entropy,
        temperature_range=(900.0, 10000.0),
        pressure_range=(0.0, 2e9),
        **options,
    )
    assert result.success, result.message
    assert result.temperature == pytest.approx(1000.0, abs=1e-6)
    assert result.entropy == pytest.approx(entropy, rel=1e-9)


@pytest.mark.parametrize(
    "pressure, entropy",
    [
        (1.5e9, 214.24074723807217),  # Seed 293 of the 81-by-81 S-P diagram.
        (1e9, 166.26340913906927),
        (1.5e9, 185.4543448798405),
        (1e9, 57.51477611466293),
        (1e9, 172.66038755226973),
        (3e9, 306.99693422947763),
        (3e9, 313.393912642678),
        (12e9, 57.51477611466293),  # Seed 493: repeated add/drop cycles.
    ],
)
def test_pyrolite_entropy_seeds_change_the_metastable_assemblage(
    pressure, entropy, capfd
):
    from burnman_cpp.tools.phase_diagram_helpers import load_example

    example = load_example(
        Path(__file__).resolve().parents[2]
        / "examples/example_pyrolite_phase_diagram.py"
    )
    opts = example.settings(41)
    opts.verbose = pressure == 12e9
    result = bm.stable_equilibrium(
        example.PYROLITE_COMPOSITION.atomic_composition,
        example.candidate_phases(),
        pressure=pressure,
        entropy=entropy,
        settings=opts,
        pressure_range=(0.0, 40e9),
        temperature_range=(273.15, 4273.15),
    )
    assert result.success, result.message
    assert result.entropy == pytest.approx(entropy, rel=1e-9)
    assert result.pressure == pytest.approx(pressure, abs=1.0)
    assert result.mass_balance_error <= opts.mass_balance_tolerance
    assert result.minimum_affinity >= -opts.affinity_tolerance
    if opts.verbose:
        output = capfd.readouterr().err
        assert "minimisation stopped:" in output
        assert "minimisation 32:" not in output
    # An independent fixed-P,T discovery must have the same total Gibbs energy.
    check = bm.stable_equilibrium(
        example.PYROLITE_COMPOSITION.atomic_composition,
        example.candidate_phases(),
        pressure,
        result.temperature,
        opts,
    )
    assert check.success, check.message
    assert result.gibbs == pytest.approx(check.gibbs, rel=1e-10, abs=0.01)


@pytest.mark.parametrize("entropy", [306.99693422947763, 313.393912642678])
def test_pyrolite_entropy_search_retains_required_eos_domain_failures(entropy):
    from burnman_cpp.tools.phase_diagram_helpers import load_example

    example = load_example(
        Path(__file__).resolve().parents[2]
        / "examples/example_pyrolite_phase_diagram.py"
    )
    result = bm.stable_equilibrium(
        example.PYROLITE_COMPOSITION.atomic_composition,
        example.candidate_phases(),
        pressure=0.0,
        entropy=entropy,
        settings=example.settings(41),
        pressure_range=(0.0, 40e9),
        temperature_range=(273.15, 4273.15),
    )
    assert not result.success
    assert result.pressure == 0.0
    assert result.entropy == entropy
    assert result.outside_model_domain
    assert any(reason.startswith("opx:") for reason in result.excluded_phases)


@pytest.mark.parametrize(
    "diagram", ["PT", "PS", "PV", "TS", "TV", "SV", "PX", "TX", "SX", "VX"]
)
def test_all_coordinate_pairs_solve_and_plot_extensive_totals(diagram):
    mineral = bm.Mineral(oxide_params())
    mineral.set_state(4.0e9, 1000.0)
    ranges = dict(
        P=(3.9e9, 4.1e9),
        T=(999.0, 1001.0),
        S=(mineral.molar_entropy - 0.05, mineral.molar_entropy + 0.05),
        V=(mineral.molar_volume - 1.0e-9, mineral.molar_volume + 1.0e-9),
        X=(0.0, 1.0),
    )
    settings = bm.PhaseDiagramSettings()
    settings.pressure_seeds = settings.temperature_seeds = settings.entropy_seeds = (
        settings.volume_seeds
    ) = settings.composition_seeds = 2
    settings.max_phase_instances = 1
    options = {
        AXES[c].replace("composition_coordinate", "composition") + "_range": ranges[c]
        for c in diagram
    }
    if "X" in diagram:
        # Unequal supplied amounts exercise the extensive-coordinate scaling.
        options["composition_end"] = {"Mg": 1.001, "O": 1.001}
        options["temperature" if diagram[0] != "T" else "pressure"] = (
            1000.0 if diagram[0] != "T" else 4.0e9
        )
    if "P" not in diagram and "pressure" not in options:
        options["pressure_range"] = (0.0, 8.0e9)
    if "T" not in diagram and "temperature" not in options:
        options["temperature_range"] = (500.0, 1500.0)
    result = bm.phase_diagram(
        {"Mg": 1.0, "O": 1.0}, [mineral], settings=settings, diagram=diagram, **options
    )
    assert result.resolved, result.diagnostics
    assert all(s.success for s in result.samples), [s.message for s in result.samples]
    for s in result.samples:
        mineral.set_state(s.pressure, s.temperature)
        amount = 1.0 + 0.001 * s.composition_coordinate
        assert s.volume == pytest.approx(amount * mineral.molar_volume, rel=1.0e-9)
        assert s.entropy == pytest.approx(amount * mineral.molar_entropy, rel=1.0e-9)
        for axis in diagram:
            if s.is_field_verification:
                continue
            assert (
                min(abs(getattr(s, AXES[axis]) - q) for q in ranges[axis])
                <= (ranges[axis][1] - ranges[axis][0]) * 1.0e-6
            )
    saved = json.loads(json.dumps(result.to_dict()))
    restored = bm.PhaseDiagramResult.from_dict(saved)
    assert restored.coordinate_ranges == result.coordinate_ranges
    assert restored.samples[0].entropy == result.samples[0].entropy
    assert restored.samples[0].volume == result.samples[0].volume
    geometry = bm.phase_diagram_field_polygons(saved)
    assert not geometry.diagnostics
    assert len(geometry.polygons) == 1
    assert geometry.polygons[0].n_phases == 1
    assert geometry.polygons[0].area == pytest.approx(1.0)
    import matplotlib.pyplot as plt

    fig, ax = bm.plot_phase_diagram(
        saved,
        temperature_unit="K",
        pressure_unit="GPa",
        volume_unit="cm3",
        entropy_unit="kJ/K",
        label_key_path=None,
    )
    assert ax.get_ylabel().startswith(AXES[diagram[0]].capitalize())
    plt.close(fig)


def test_volume_section_keeps_first_order_coexistence():
    # Both phases have the same chemistry; equality of Gibbs energies is
    # exactly P=5 GPa. Their volume jump must become a finite two-phase field.
    first = bm.Mineral(oxide_params())
    second = bm.CombinedMineral([first], [1.0], [5000.0, 0.0, -1.0e-6], name="dense")
    first.set_state(5.0e9, 1000.0)
    v = first.molar_volume
    settings = bm.PhaseDiagramSettings()
    settings.volume_seeds = settings.composition_seeds = 5
    settings.max_phase_instances = 1
    settings.step = 0.1
    result = bm.phase_diagram(
        {"Mg": 1.0, "O": 1.0},
        [first, second],
        diagram="VX",
        volume_range=(v - 1.2e-6, v + 0.2e-6),
        temperature=1000.0,
        composition_end={"Mg": 1.01, "O": 1.01},
        settings=settings,
    )
    assert result.resolved, result.diagnostics
    assert all(s.success for s in result.samples), [
        s.message for s in result.samples if not s.success
    ]
    assert len(result.boundaries) == 2
    for line in result.boundaries:
        for p in line.points:
            assert p.pressure == pytest.approx(5.0e9, abs=0.3)
            assert p.temperature == pytest.approx(1000.0, abs=1.0e-6)
            amount = 1.0 + 0.01 * p.composition_coordinate
            fractions = {s.candidate_index: s.amount for s in p.phases}
            assert sum(fractions.values()) == pytest.approx(amount, abs=1.0e-8)
            assert p.volume == pytest.approx(
                amount * v - fractions.get(1, 0.0) * 1.0e-6, abs=1.0e-13
            )
    polygons = bm.phase_diagram_field_polygons(result)
    assert not polygons.diagnostics
    assert sorted(p.n_phases for p in polygons.polygons) == [1, 1, 2]
    assert sum(p.area for p in polygons.polygons) == pytest.approx(1.0, abs=1.0e-8)


@pytest.mark.parametrize("step", [0.025, 0.1])
def test_entropy_volume_invariant_has_a_finite_three_phase_field(step):
    # A, B and C meet at P=5 GPa, T=1000 K. Their S/V values form
    # a triangle: barycentric phase amounts vary inside at fixed P,T.
    first = bm.Mineral(oxide_params())
    second = bm.CombinedMineral([first], [1.0], [5000.0, 0.0, -1.0e-6], name="B")
    third = bm.CombinedMineral([first], [1.0], [7500.0, 5.0, -0.5e-6], name="C")
    first.set_state(5.0e9, 1000.0)
    s, v = first.molar_entropy, first.molar_volume
    settings = bm.PhaseDiagramSettings()
    settings.entropy_seeds = settings.volume_seeds = 5
    settings.max_phase_instances = 1
    settings.step = step
    result = bm.phase_diagram(
        {"Mg": 1.0, "O": 1.0},
        [first, second, third],
        diagram="SV",
        entropy_range=(s - 1.0, s + 6.0),
        volume_range=(v - 1.2e-6, v + 0.2e-6),
        pressure_range=(0.0, 10.0e9),
        temperature_range=(800.0, 1200.0),
        settings=settings,
    )
    assert result.resolved, result.diagnostics
    assert all(p.success for p in result.samples), [
        p.message for p in result.samples if not p.success
    ]
    polygons = bm.phase_diagram_field_polygons(result)
    assert not polygons.diagnostics
    assert sorted(p.n_phases for p in polygons.polygons) == [1, 1, 1, 2, 2, 2, 3]
    triple = next(p for p in polygons.polygons if p.n_phases == 3)
    assert triple.area == pytest.approx(0.5 * 5.0 * 1.0e-6 / (7.0 * 1.4e-6), rel=1.0e-6)
    for sample in result.samples:
        if len(sample.phases) == 3 and min(p.amount for p in sample.phases) > 1.0e-6:
            assert sample.pressure == pytest.approx(5.0e9, abs=0.3)
            assert sample.temperature == pytest.approx(1000.0, abs=1.0e-6)
            amounts = {p.candidate_index: p.amount for p in sample.phases}
            assert sample.entropy == pytest.approx(s + 5.0 * amounts[2], abs=1.0e-7)
            assert sample.volume == pytest.approx(
                v - amounts[1] * 1.0e-6 - amounts[2] * 0.5e-6, abs=1.0e-13
            )


@pytest.mark.parametrize(
    "diagram,fixed",
    [
        ("PX", "S"),
        ("PX", "V"),
        ("TX", "S"),
        ("TX", "V"),
        ("SX", "P"),
        ("SX", "V"),
        ("VX", "P"),
        ("VX", "S"),
    ],
)
def test_x_sections_accept_each_remaining_fixed_coordinate(diagram, fixed):
    mineral = bm.Mineral(oxide_params())
    mineral.set_state(4.0e9, 1000.0)
    values = dict(P=4.0e9, T=1000.0, S=mineral.molar_entropy, V=mineral.molar_volume)
    widths = dict(P=1.0e8, T=1.0, S=0.05, V=1.0e-9)
    axis = diagram[0]
    options = {
        AXES[axis]
        + "_range": (values[axis] - widths[axis], values[axis] + widths[axis]),
        AXES[fixed]: values[fixed],
    }
    for physical, bounds in [("P", (0.0, 8.0e9)), ("T", (500.0, 1500.0))]:
        if physical not in (axis, fixed):
            options[AXES[physical] + "_range"] = bounds
    settings = bm.PhaseDiagramSettings()
    settings.pressure_seeds = settings.temperature_seeds = settings.entropy_seeds = (
        settings.volume_seeds
    ) = settings.composition_seeds = 2
    settings.max_phase_instances = 1
    result = bm.phase_diagram(
        {"Mg": 1.0, "O": 1.0},
        [mineral],
        diagram=diagram,
        composition_end={"Mg": 1.001, "O": 1.001},
        settings=settings,
        **options,
    )
    assert result.resolved, result.diagnostics
    assert result.fixed_coordinate == fixed
    for sample in result.samples:
        assert sample.success, sample.message
        assert getattr(sample, AXES[fixed]) == pytest.approx(
            values[fixed], rel=1.0e-8, abs=1.0e-12
        )
    restored = bm.PhaseDiagramResult.from_dict(json.loads(json.dumps(result.to_dict())))
    resumed = bm.refine_phase_diagram({"Mg": 1.0, "O": 1.0}, [mineral], restored)
    assert resumed.resolved, resumed.diagnostics
    assert resumed.fixed_coordinate == fixed
    assert resumed.fixed_value == values[fixed]


def test_entropy_volume_json_requires_its_geometry_coordinates():
    data = dict(
        diagram_type="SV",
        pressure_range=(0.0, 10.0e9),
        temperature_range=(500.0, 1500.0),
        entropy_range=(0.0, 1.0),
        volume_range=(1.0e-5, 2.0e-5),
        boundaries=[dict(points=[dict(pressure=1.0e9, temperature=1000.0)])],
    )
    with pytest.raises(ValueError, match="require entropy"):
        bm.phase_diagram_field_polygons(data)
    data["boundaries"][0]["points"][0]["entropy"] = 0.5
    with pytest.raises(ValueError, match="require volume"):
        bm.phase_diagram_field_polygons(data)
