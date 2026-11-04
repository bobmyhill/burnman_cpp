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

"""Composition sections against independently known binary coexistence curves."""

import json

import numpy as np
import pytest

import burnman_cpp as bm
from conftest import oxide_params


def binary_phases():
    magnesium = bm.Mineral(oxide_params("Mg"))
    iron = bm.Mineral(oxide_params("Fe"))
    shifted_magnesium = bm.CombinedMineral(
        [magnesium], [1.0], [5000.0, 0.0, -1.0e-6], name="Mg B"
    )
    shifted_iron = bm.CombinedMineral([iron], [1.0], [-5000.0, 0.0, 0.0], name="Fe B")
    return [
        bm.Solution(
            bm.IdealSolution([(mg, "[Mg]O"), (fe, "[Fe]O")]),
            [0.5, 0.5],
            name=name,
        )
        for name, mg, fe in [
            ("A", magnesium, iron),
            ("B", shifted_magnesium, shifted_iron),
        ]
    ]


def section_settings():
    settings = bm.PhaseDiagramSettings()
    settings.pressure_seeds = 3
    settings.temperature_seeds = 3
    settings.composition_seeds = 5
    settings.max_phase_instances = 1
    settings.step = 0.1
    settings.affinity_tolerance = 0.002
    return settings


START = {"Mg": 1.0, "O": 1.0}
END = {"Fe": 1.0, "O": 1.0}


def calculate(diagram, settings=None, end=END):
    domains = (
        dict(pressure_range=(0.0, 2.0e9), temperature=1000.0)
        if diagram == "PX"
        else dict(temperature_range=(600.0, 1600.0), pressure=1.0e9)
    )
    return bm.phase_diagram(
        START,
        binary_phases(),
        settings=settings or section_settings(),
        diagram=diagram,
        composition_end=end,
        **domains,
    )


def check_boundaries(result):
    assert len(result.boundaries) == 2
    for line in result.boundaries:
        for point in line.points:
            # Equality of both ideal chemical potentials yields these two
            # coexistence compositions, independently of bulk or phase amounts.
            rt = 8.31446261815324 * point.temperature
            a = np.exp((5000.0 - 1.0e-6 * point.pressure) / rt)
            b = np.exp(-5000.0 / rt)
            x_b = (1.0 - a) / (b - a)
            y = x_b if line.zero_phase == 0 else b * x_b
            start = result.composition_start["Mg"]
            end = result.composition_end["Fe"]
            expected = start * y / (end * (1.0 - y) + start * y)
            assert point.composition_coordinate == pytest.approx(expected, abs=2.0e-8)
            if result.diagram_type == "PX":
                assert point.temperature == pytest.approx(1000.0, abs=1.0e-7)
            else:
                assert point.pressure == pytest.approx(1.0e9, abs=0.3)
            amounts = np.zeros(3)
            for phase in point.phases:
                x = phase.composition[1]
                amounts += phase.amount * np.array([1.0 - x, x, 1.0])
            np.testing.assert_allclose(
                amounts,
                [
                    start * (1.0 - expected),
                    end * expected,
                    start * (1.0 - expected) + end * expected,
                ],
                atol=3.0e-8,
                rtol=0.0,
            )
            assert point.mass_balance_error < 1.0e-8
            assert point.minimum_affinity >= -0.2


@pytest.fixture(scope="module", params=["PX", "TX"])
def diagram(request):
    return calculate(request.param)


def test_composition_sections_trace_exact_coexistence_and_close_fields(diagram):
    assert diagram.resolved, diagram.diagnostics
    assert all(state.success for state in diagram.samples)
    check_boundaries(diagram)
    geometry = bm.phase_diagram_field_polygons(diagram)
    assert not geometry.diagnostics
    assert sorted(p.n_phases for p in geometry.polygons) == [1, 1, 2]
    assert sum(p.area for p in geometry.polygons) == pytest.approx(1.0, abs=1.0e-8)
    assert all(not p.has_open_boundary for p in geometry.polygons)
    for field in geometry.polygons:
        coordinate = field.label_position
        bulk = {"Mg": 1.0 - coordinate[1], "Fe": coordinate[1], "O": 1.0}
        pressure, temperature = (
            (coordinate[0], 1000.0)
            if diagram.diagram_type == "PX"
            else (1.0e9, coordinate[0])
        )
        equilibrium = bm.stable_equilibrium(
            bulk, binary_phases(), pressure, temperature, section_settings()
        )
        assert equilibrium.success, equilibrium.message
        assert {phase.name for phase in equilibrium.phases} == {
            diagram.phase_names[index] for index in field.phases
        }


def test_composition_metadata_round_trips_and_plots_all_fields(diagram):
    data = json.loads(json.dumps(diagram.to_dict()))
    restored = bm.PhaseDiagramResult.from_dict(data)
    assert restored.diagram_type == diagram.diagram_type
    assert restored.composition_range == [0.0, 1.0]
    check_boundaries(restored)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = bm.plot_phase_diagram(
        data,
        pressure_unit="GPa",
        temperature_unit="C",
        composition_label="Fe/(Mg+Fe)",
        label_key_path=None,
        label_fontsize=9.0,
        show_nodes=True,
    )
    assert ax.get_xlabel() == "Fe/(Mg+Fe)"
    assert ax.get_xlim() == pytest.approx((0.0, 1.0))
    if diagram.diagram_type == "PX":
        assert ax.get_ylabel() == "Pressure (GPa)"
        assert ax.get_ylim() == pytest.approx((0.0, 2.0))
    else:
        assert ax.get_ylabel() == "Temperature (°C)"
        assert ax.get_ylim() == pytest.approx((326.85, 1326.85))
    assert len(ax.phase_diagram_geometry.polygons) == 3
    assert all(text.get_fontsize() == 9.0 for text in ax.texts)
    plt.close(fig)


@pytest.mark.parametrize("diagram", ["PX", "TX"])
def test_unequal_atom_totals_preserve_supplied_amounts(diagram):
    result = calculate(diagram, end={"Fe": 2.0, "O": 2.0})
    assert result.resolved, result.diagnostics
    check_boundaries(result)
    check_boundaries(bm.PhaseDiagramResult.from_dict(result.to_dict()))
    geometry = bm.phase_diagram_field_polygons(result)
    assert not geometry.diagnostics
    assert sorted(p.n_phases for p in geometry.polygons) == [1, 1, 2]
    assert sum(p.area for p in geometry.polygons) == pytest.approx(1.0, abs=1e-8)


def test_absent_element_restricts_a_multicomponent_solution_face():
    members = [
        (bm.Mineral(oxide_params(element, shift)), f"[{element}]O")
        for element, shift in [("Mg", 0.0), ("Fe", 0.0), ("Ca", -100000.0)]
    ]
    solution = bm.Solution(bm.IdealSolution(members), [0.4, 0.6, 0.0], name="oxide")
    unavailable = bm.Mineral(oxide_params("Ca", -1000000.0))
    bulk = {"Mg": 0.4, "Fe": 0.6, "O": 1.0}
    state = bm.stable_equilibrium(bulk, [solution, unavailable], 1.0e9, 1000.0)
    assert state.success, state.message
    assert len(state.phases) == 1
    np.testing.assert_allclose(state.phases[0].composition, [0.4, 0.6, 0.0], atol=1e-9)
    solution.set_state(1.0e9, 1000.0)
    assert state.gibbs == pytest.approx(solution.molar_gibbs, abs=1e-7)
    assert state.minimum_affinity >= -0.2


def test_composition_subrange_clips_coexistence_to_its_frame():
    result = bm.phase_diagram(
        START,
        binary_phases(),
        diagram="PX",
        composition_end=END,
        composition_range=(0.32, 0.6),
        pressure_range=(0.0, 2.0e9),
        temperature=1000.0,
        settings=section_settings(),
    )
    assert result.resolved, result.diagnostics
    check_boundaries(result)
    assert all(
        0.32 - 1e-10 <= p.composition_coordinate <= 0.6 + 1e-10
        for line in result.boundaries
        for p in line.points
    )
    endpoints = [
        p.composition_coordinate
        for line in result.boundaries
        for p in [line.points[0], line.points[-1]]
    ]
    assert min(endpoints) == pytest.approx(0.32, abs=1e-10)
    assert max(endpoints) == pytest.approx(0.6, abs=1e-10)
    geometry = bm.phase_diagram_field_polygons(result)
    assert not geometry.diagnostics
    assert sorted(p.n_phases for p in geometry.polygons) == [1, 1, 2]
    assert sum(p.area for p in geometry.polygons) == pytest.approx(1.0, abs=1e-8)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"composition_end": {}}, "endpoint"),
        ({"composition_end": START}, "differ"),
        ({"composition_end": {"Fe": 0.0, "O": 0.0}}, "positive amounts"),
        ({"composition_range": (-0.1, 1.0)}, "between 0 and 1"),
        ({"composition_range": (0.5, 0.5)}, "increasing"),
        (
            {"temperature_range": (900.0, 1100.0), "temperature": None},
            "fixed thermodynamic coordinate",
        ),
        ({"temperature": None}, "Specify both"),
        ({"diagram": "YX"}, "diagram must"),
    ],
)
def test_composition_section_rejects_invalid_paths_and_domains(kwargs, message):
    options = dict(
        diagram="PX",
        pressure_range=(0.0, 2.0e9),
        temperature=1000.0,
        composition_end=END,
        settings=section_settings(),
    )
    options.update(kwargs)
    with pytest.raises(ValueError, match=message):
        bm.phase_diagram(START, binary_phases(), **options)
