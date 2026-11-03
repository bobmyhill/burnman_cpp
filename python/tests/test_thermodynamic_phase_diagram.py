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

import numpy as np
import pytest

import burnman_cpp as bm
from conftest import oxide_params

AXES = dict(
    P="pressure", T="temperature", S="entropy", V="volume", X="composition_coordinate"
)


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
