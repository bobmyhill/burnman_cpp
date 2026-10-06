"""Composition sections against independently known binary coexistence curves."""

import json
from pathlib import Path
import subprocess
import sys

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
    settings = bm.PseudosectionSettings()
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
    return bm.pseudosection(
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
    geometry = bm.pseudosection_field_polygons(diagram)
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
    restored = bm.PseudosectionResult.from_dict(data)
    assert restored.diagram_type == diagram.diagram_type
    assert restored.composition_range == [0.0, 1.0]
    check_boundaries(restored)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = bm.plot_pseudosection(
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
    assert len(ax.pseudosection_geometry.polygons) == 3
    assert all(text.get_fontsize() == 9.0 for text in ax.texts)
    plt.close(fig)


@pytest.mark.parametrize("diagram", ["PX", "TX"])
def test_unequal_atom_totals_preserve_supplied_amounts(diagram):
    result = calculate(diagram, end={"Fe": 2.0, "O": 2.0})
    assert result.resolved, result.diagnostics
    check_boundaries(result)
    check_boundaries(bm.PseudosectionResult.from_dict(result.to_dict()))
    geometry = bm.pseudosection_field_polygons(result)
    assert not geometry.diagnostics
    assert sorted(p.n_phases for p in geometry.polygons) == [1, 1, 2]
    assert sum(p.area for p in geometry.polygons) == pytest.approx(1.0, abs=1e-8)


@pytest.mark.parametrize("diagram", ["PX", "TX"])
def test_composition_section_resumes_saved_partial_lines(diagram):
    settings = section_settings()
    settings.max_trace_steps = 2
    settings.max_recovery_passes = 0
    partial = calculate(diagram, settings)
    assert not partial.resolved
    accepted = [
        (p.pressure, p.composition_coordinate)
        for line in partial.boundaries
        for p in line.points
    ]
    completed = bm.refine_pseudosection(
        START,
        binary_phases(),
        json.loads(json.dumps(partial.to_dict())),
        section_settings(),
    )
    assert completed.resolved, completed.diagnostics
    check_boundaries(completed)
    new_points = np.array(
        [
            (p.pressure, p.composition_coordinate)
            for line in completed.boundaries
            for p in line.points
        ]
    )
    for point in accepted:
        assert (
            np.min(np.linalg.norm((new_points - point) / [2.0e9, 1.0], axis=1)) < 1.0e-8
        )


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


def test_olivine_example_closes_endmember_edges_and_uses_only_native_thermodynamics():
    example = Path(__file__).resolve().parents[2] / "examples/example_olivine_px.py"
    script = """
import json, runpy, sys
example = runpy.run_path(sys.argv[1])
result = example['calculate'](quick=True)
assert not {'burnman', 'scipy', 'cvxpy', 'sympy', 'cdd'} & sys.modules.keys()
print(json.dumps(result.to_dict()))
"""
    output = subprocess.run(
        [sys.executable, "-I", "-c", script, str(example)],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    result = bm.PseudosectionResult.from_dict(json.loads(output.stdout))
    assert result.resolved, result.diagnostics
    geometry = bm.pseudosection_field_polygons(result)
    assert not geometry.diagnostics
    assert sorted(p.n_phases for p in geometry.polygons) == [1, 1, 1, 2, 2, 2]
    assert sum(p.area for p in geometry.polygons) == pytest.approx(1.0, abs=1e-8)
    assert all(not p.has_open_boundary for p in geometry.polygons)
    from burnman_cpp.minerals import SLB11

    phases = [
        SLB11.mg_fe_olivine(),
        SLB11.mg_fe_wadsleyite(),
        SLB11.mg_fe_ringwoodite(),
    ]
    endpoints = []
    for line in result.boundaries:
        for point in [line.points[0], line.points[-1]]:
            x = point.composition_coordinate
            if abs(x) < 1e-8 or abs(x - 1.0) < 1e-8:
                endpoints.append(x)
                energies = []
                for state in point.phases:
                    # At a pure bulk, the boundary must solve the independent
                    # equality of the two pure polymorph Gibbs energies.
                    phase = phases[state.candidate_index]
                    phase.set_composition([1.0 - x, x])
                    phase.set_state(point.pressure, point.temperature)
                    energies.append(phase.molar_gibbs)
                    np.testing.assert_allclose(
                        state.composition, [1.0 - x, x], atol=1e-8
                    )
                assert max(energies) - min(energies) < 0.02
    assert min(endpoints) == pytest.approx(0.0, abs=1e-10)
    assert max(endpoints) == pytest.approx(1.0, abs=1e-10)


def test_composition_subrange_clips_coexistence_to_its_frame():
    result = bm.pseudosection(
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
    geometry = bm.pseudosection_field_polygons(result)
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
            "fixed temperature",
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
        bm.pseudosection(START, binary_phases(), **options)
