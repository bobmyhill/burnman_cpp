"""Saved phase lines are refined to subdivisions of the full plotted axes."""

import json
from pathlib import Path
import runpy

import numpy as np
import pytest
import burnman_cpp as bm

from conftest import oxide_params
from test_pseudosection import crossing_phases, curved_phases, pure, settings
from test_composition_pseudosection import (
    START,
    binary_phases,
    calculate,
    check_boundaries,
)


def check_spacing(result, resolution, **units):
    from burnman_cpp.tools.pseudosection import _coordinate_transforms

    diagram, names, _, _, _, coordinates, _ = _coordinate_transforms(result, **units)
    limits = coordinates(np.asarray(result.coordinate_ranges).T)
    spacing = np.abs(limits[1] - limits[0]) / (np.asarray(resolution) - 1)
    for line in result.boundaries:
        values = coordinates(
            [[getattr(p, names[c]) for c in diagram] for p in line.points]
        )
        assert (np.abs(np.diff(values, axis=0)) <= spacing * (1.0 + 1.0e-10)).all()
        for point in line.points:
            assert point.mass_balance_error < 1.0e-8
            assert point.minimum_affinity >= -result.settings.affinity_tolerance
            assert point.residual < 0.01


@pytest.fixture(scope="module")
def crossing():
    bulk = {"Mg": 1.0, "Fe": 1.0, "O": 2.0}
    result = bm.pseudosection(
        bulk, crossing_phases(), (0.0, 2.0e9), (600.0, 1400.0), settings()
    )
    assert result.resolved, result.diagnostics
    saved = json.loads(json.dumps(result.to_dict()))
    for line in saved["boundaries"]:
        line["points"] = [line["points"][0], line["points"][-1]]
    return bulk, saved


@pytest.mark.parametrize("swap_axes", [False, True])
def test_all_straight_lines_refine_from_json_and_keep_original_points(
    crossing, tmp_path, swap_axes
):
    bulk, saved = crossing
    path = tmp_path / "previous.json"
    text = json.dumps(saved)
    path.write_text(text)
    units = dict(pressure_unit="GPa", temperature_unit="K", swap_axes=swap_axes)
    resolution = (101, 201)
    result = bm.refine_pseudosection(
        bulk, crossing_phases(), path, resolution=resolution, **units
    )
    assert result.resolved, result.diagnostics
    assert len(result.boundaries) == 4
    check_spacing(result, resolution, **units)
    for line in result.boundaries:
        assert np.max(np.abs(np.diff([p.temperature for p in line.points]))) <= (
            4.0 if swap_axes else 8.0
        ) * (1 + 1e-10)
        assert np.max(np.abs(np.diff([p.pressure for p in line.points]))) <= (
            2e7 if swap_axes else 1e7
        ) * (1 + 1e-10)
    for old, new in zip(saved["boundaries"], result.to_dict()["boundaries"]):
        assert len(new["points"]) > 2
        assert new["start_node"] == old["start_node"]
        assert new["end_node"] == old["end_node"]
        assert all(point in new["points"] for point in old["points"])
        for point in new["points"]:
            assert (
                abs(point["pressure"] - 1.0e9) < 1.0
                or abs(point["temperature"] - 1000.0) < 1.0e-6
            )
    assert path.read_text() == text
    assert json.dumps(saved) == text


def test_axis_counts_are_independent_of_linear_display_units(crossing):
    bulk, saved = crossing
    default = bm.refine_pseudosection(
        bulk, crossing_phases(), saved, resolution=(17, 21)
    )
    changed_units = bm.refine_pseudosection(
        bulk,
        crossing_phases(),
        saved,
        resolution=(17, 21),
        pressure_unit="GPa",
        temperature_unit="K",
    )
    assert default.resolved and changed_units.resolved
    assert default.to_dict() == changed_units.to_dict()


def test_single_axis_division_is_valid(crossing):
    bulk, saved = crossing
    result = bm.refine_pseudosection(bulk, crossing_phases(), saved, resolution=(2, 2))
    assert result.resolved, result.diagnostics
    check_spacing(result, (2, 2))


def test_completed_refinement_skips_junction_search_and_repeated_solves(
    crossing, capfd
):
    bulk, saved = crossing
    opts = bm.PseudosectionSettings.from_dict(saved["settings"])
    opts.verbose = True
    result = bm.refine_pseudosection(
        bulk, crossing_phases(), saved, opts, resolution=(17, 21)
    )
    assert result.resolved, result.diagnostics
    check_spacing(result, (17, 21))
    assert len(result.nodes) == len(saved["nodes"])
    assert len(result.boundaries) == len(saved["boundaries"])
    assert result.equilibrium_solves > saved["equilibrium_solves"]
    messages = capfd.readouterr().err
    assert "Refining 4 phase lines" in messages
    assert "Line 4/4" in messages
    assert "points added" in messages
    assert "equilibrium solves in this refinement" in messages
    assert "Checking phase field consistency" in messages
    assert "Rejected boundary corrector" not in messages
    assert "seeds remaining" not in messages
    repeated = bm.refine_pseudosection(
        bulk, crossing_phases(), result, resolution=(17, 21)
    )
    assert repeated.to_dict() == result.to_dict()
    messages = capfd.readouterr().err
    assert "Rejected boundary corrector" not in messages
    assert "seeds remaining" not in messages


def test_field_conflicts_recover_a_missing_line_in_a_completed_saved_diagram(crossing):
    bulk, data = crossing
    saved = json.loads(json.dumps(data))
    saved["boundaries"].pop()
    # The saved completion flag cannot replace the field consistency checks.
    assert saved["resolved"]
    result = bm.refine_pseudosection(
        bulk, crossing_phases(), saved, resolution=(17, 21)
    )
    assert result.resolved, result.diagnostics
    assert len(result.boundaries) == 4
    check_spacing(result, (17, 21))
    geometry = bm.pseudosection_field_polygons(result)
    assert not geometry.diagnostics
    assert len(geometry.polygons) == 4
    assert sum(p.area for p in geometry.polygons) == pytest.approx(1.0)


def test_pyrolite_refinement_crosses_pure_to_mixed_solution_compositions():
    saved = json.loads(
        (
            Path(__file__).parent / "data" / "pyrolite_refinement_segment.json"
        ).read_text()
    )["previous"]
    example = runpy.run_path(
        str(
            Path(__file__).parents[2] / "examples" / "example_pyrolite_pseudosection.py"
        )
    )
    opts = bm.PseudosectionSettings.from_dict(saved["settings"])
    opts.verbose = False
    opts.max_lines = 1
    opts.max_recovery_passes = 0
    result = bm.refine_pseudosection(
        saved["composition_start"],
        example["candidate_phases"](),
        saved,
        opts,
        resolution=(401, 401),
    )
    assert len(result.boundaries) == 1
    assert not any("requested line resolution" in d for d in result.diagnostics)
    check_spacing(result, (401, 401))
    points = result.boundaries[0].points
    assert any(32.0 < p.temperature < 47.0 for p in points)
    refined = result.to_dict()["boundaries"][0]["points"]
    assert all(p in refined for p in saved["boundaries"][0]["points"])


def test_curved_line_refinement_checks_actual_equilibrium_and_default_plot_units():
    bulk = {"Mg": 1.0, "O": 1.0}
    phases = curved_phases()
    result = bm.pseudosection(bulk, phases, (0.0, 5.0e9), (600.0, 1400.0), settings())
    saved = result.to_dict()
    refined = bm.refine_pseudosection(bulk, phases, saved, resolution=(81, 101))
    assert refined.resolved, refined.diagnostics
    check_spacing(refined, (81, 101))
    assert sum(len(line.points) for line in refined.boundaries) > sum(
        len(line.points) for line in result.boundaries
    )
    for line in refined.boundaries:
        for point in line.points:
            for phase in phases:
                phase.set_state(point.pressure, point.temperature)
            assert abs(phases[0].molar_gibbs - phases[1].molar_gibbs) < 0.01
    geometry = bm.pseudosection_field_polygons(refined)
    assert not geometry.diagnostics
    assert sum(p.area for p in geometry.polygons) == pytest.approx(1.0)


@pytest.mark.parametrize("diagram", ["PX", "TX"])
@pytest.mark.parametrize("end_amount", [1.0, 2.0])
def test_composition_lines_keep_analytic_coexistence_after_refinement(
    diagram, end_amount
):
    previous = calculate(diagram, end={"Fe": end_amount, "O": end_amount})
    resolution = (201, 51)
    units = dict(pressure_unit="GPa", temperature_unit="K")
    result = bm.refine_pseudosection(
        START, binary_phases(), previous, resolution=resolution, **units
    )
    assert result.resolved, result.diagnostics
    check_spacing(result, resolution, **units)
    check_boundaries(result)


@pytest.mark.parametrize("density", [False, True])
def test_extensive_axes_refine_as_totals_or_density_and_entropy_per_atom(density):
    first = bm.Mineral(oxide_params())
    second = bm.CombinedMineral([first], [1.0], [5000.0, 0.0, -1.0e-6], name="dense")
    first.set_state(5.0e9, 1000.0)
    opts = settings()
    opts.entropy_seeds = opts.volume_seeds = 3
    opts.max_phase_instances = 1
    options = dict(
        volume_range=(
            2.0 * (first.molar_volume - 1.2e-6),
            2.0 * (first.molar_volume + 0.2e-6),
        )
    )
    options["entropy_range"] = (
        2.0 * (first.molar_entropy - 0.3),
        2.0 * (first.molar_entropy + 0.3),
    )
    bulk = {"Mg": 2.0, "O": 2.0}
    previous = bm.pseudosection(
        bulk,
        [first, second],
        (4.0e9, 6.0e9),
        (980.0, 1020.0),
        opts,
        diagram="SV",
        **options,
    )
    assert previous.resolved, previous.diagnostics
    assert previous.boundaries
    resolution = (101, 101)
    units = dict(
        volume_unit="kg/m3" if density else "cm3",
        entropy_unit="kB/atom" if density else "kJ/K",
    )
    result = bm.refine_pseudosection(
        bulk, [first, second], previous.to_dict(), resolution=resolution, **units
    )
    assert result.resolved, result.diagnostics
    check_spacing(result, resolution, **units)
    assert sum(len(line.points) for line in result.boundaries) > sum(
        len(line.points) for line in previous.boundaries
    )
    for line in result.boundaries:
        for point in line.points:
            assert point.pressure == pytest.approx(5.0e9, abs=0.3)
            assert sum(phase.amount for phase in point.phases) == pytest.approx(
                2.0, abs=1.0e-8
            )


@pytest.mark.parametrize(
    "resolution",
    [
        (0, 201),
        (-1, 201),
        (1, 201),
        (2.5, 201),
        (np.nan, 201),
        (np.inf, 201),
        (101,),
        (101, 201, 301),
    ],
)
def test_invalid_axis_counts_are_rejected(crossing, resolution):
    bulk, saved = crossing
    with pytest.raises(ValueError, match="resolution"):
        bm.refine_pseudosection(bulk, crossing_phases(), saved, resolution=resolution)


def test_unmet_resolution_is_reported_without_losing_verified_boundaries(crossing):
    bulk, saved = crossing
    opts = bm.PseudosectionSettings.from_dict(saved["settings"])
    opts.max_trace_steps = 2
    opts.max_refinement_iterations = 1
    opts.max_recovery_passes = 0
    result = bm.refine_pseudosection(
        bulk, crossing_phases(), saved, opts, resolution=(801, 2001)
    )
    assert not result.resolved
    assert any(
        "requested line resolution" in d and "limit reached" in d
        for d in result.diagnostics
    )
    boundaries = result.to_dict()["boundaries"]
    assert len(boundaries) == len(saved["boundaries"])
    assert len(result.nodes) == len(saved["nodes"])
    for old in saved["boundaries"]:
        retained = next(
            line
            for line in boundaries
            if line["assemblage"] == old["assemblage"]
            and all(point in line["points"] for point in old["points"])
        )
        assert retained["termination"] == old["termination"]


def test_solution_lines_refine_from_coarse_points_at_the_critical_endpoint():
    mg = pure("Mg oxide", {"Mg": 1.0, "O": 1.0})
    fe = pure("Fe oxide", {"Fe": 1.0, "O": 1.0})
    phase = bm.Solution(
        bm.SymmetricRegularSolution(
            [(mg, "[Mg]O"), (fe, "[Fe]O")], energy_interaction=[[13000.0]]
        ),
        [0.5, 0.5],
        name="oxide",
    )
    opts = settings()
    opts.temperature_seeds = opts.composition_seeds = 5
    bulk = {"Mg": 1.0, "O": 1.0}
    previous = bm.pseudosection(
        bulk,
        [phase],
        settings=opts,
        diagram="TX",
        pressure=1.0e9,
        temperature_range=(600.0, 900.0),
        composition_end={"Fe": 1.0, "O": 1.0},
    )
    assert previous.resolved, previous.diagnostics
    saved = previous.to_dict()
    for line in saved["boundaries"]:
        line["points"] = [line["points"][0], line["points"][-1]]
    result = bm.refine_pseudosection(
        bulk, [phase], saved, resolution=(51, 61), temperature_unit="K"
    )
    assert result.resolved, result.diagnostics
    check_spacing(result, (51, 61), temperature_unit="K")
    critical = next(n for n in result.nodes if n.kind == "critical_point")
    assert critical.temperature == pytest.approx(
        13000.0 / (2.0 * 8.31446261815324), abs=1.0e-4
    )
    assert critical.composition_coordinate == pytest.approx(0.5, abs=1.0e-6)
    for line in result.boundaries:
        for point in line.points:
            low, high = sorted(p.composition[1] for p in point.phases)
            assert (
                abs(
                    13000.0 * (high - low)
                    - 8.31446261815324 * point.temperature * np.log(high / low)
                )
                < 0.01
            )
