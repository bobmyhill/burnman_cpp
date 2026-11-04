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

"""Saved phase lines are refined to subdivisions of the full plotted axes."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import burnman_cpp as bm
from conftest import oxide_params
from test_phase_diagram import crossing_phases, curved_phases, pure, settings
from test_composition_phase_diagram import (
    START,
    binary_phases,
    calculate,
    check_boundaries,
)


def check_boundary_equilibria(result):
    settings = result.settings
    # An X section interpolates bulks whose norms cannot exceed these endpoints.
    bulk_norm = max(
        np.linalg.norm(list(bulk.values()))
        for bulk in (result.composition_start, result.composition_end)
    )
    for line in result.boundaries:
        for point in line.points:
            assert point.mass_balance_error <= settings.mass_balance_tolerance
            assert point.minimum_affinity >= -settings.affinity_tolerance
            # The saved residual is the Euclidean norm of reaction and mass
            # equations. Engine.solve limits each reaction to 0.1*affinity_tol;
            # the endmember count bounds their number, including reduced faces.
            n_endmembers = sum(len(phase.composition) for phase in point.phases)
            residual_bound = np.hypot(
                np.sqrt(n_endmembers) * 0.1 * settings.affinity_tolerance,
                bulk_norm * settings.mass_balance_tolerance,
            )
            assert 0.0 <= point.residual <= residual_bound


def check_spacing(result, resolution, **units):
    from burnman_cpp.tools.phase_diagram import _coordinate_transforms

    diagram, names, _, _, _, coordinates, _ = _coordinate_transforms(result, **units)
    limits = coordinates(np.asarray(result.coordinate_ranges).T)
    spacing = np.abs(limits[1] - limits[0]) / (np.asarray(resolution) - 1)
    for line in result.boundaries:
        values = coordinates(
            [[getattr(p, names[c]) for c in diagram] for p in line.points]
        )
        assert (np.abs(np.diff(values, axis=0)) <= spacing * (1.0 + 1.0e-10)).all()
    check_boundary_equilibria(result)


@pytest.fixture(scope="module")
def crossing():
    bulk = {"Mg": 1.0, "Fe": 1.0, "O": 2.0}
    result = bm.phase_diagram(
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
    result = bm.refine_phase_diagram(
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
    default = bm.refine_phase_diagram(
        bulk, crossing_phases(), saved, resolution=(17, 21)
    )
    changed_units = bm.refine_phase_diagram(
        bulk,
        crossing_phases(),
        saved,
        resolution=(17, 21),
        pressure_unit="GPa",
        temperature_unit="K",
    )
    assert default.resolved and changed_units.resolved
    assert default.to_dict() == changed_units.to_dict()


@pytest.mark.parametrize("scope", ["none", "all", "region"])
@pytest.mark.parametrize("unresolved", [False, True])
def test_refinement_preserves_graph_ids_labels_and_saved_evidence(
    crossing, scope, unresolved
):
    bulk, original = crossing
    saved = deepcopy(original)
    if unresolved:
        saved["resolved"] = False
        saved["diagnostics"] = ["Boundary 7: saved unresolved connection."]
        saved["settings"]["max_recovery_passes"] = 5
    # Saved IDs are identities, even when no longer contiguous after editing.
    ids = {b["id"]: 7 + 10 * i for i, b in enumerate(saved["boundaries"])}
    for boundary in saved["boundaries"]:
        boundary["id"] = ids[boundary["id"]]
    for node in saved["nodes"]:
        node["incident_lines"] = [ids[i] for i in node["incident_lines"]]
    before = deepcopy(saved)
    options = {} if scope == "none" else {"resolution": (25, 31)}
    if scope == "region":
        options["region"] = ((0.5e9, 1.5e9), (800.0, 900.0))
    result = bm.refine_phase_diagram(
        bulk,
        crossing_phases(),
        saved,
        pressure_unit="Pa",
        temperature_unit="K",
        swap_axes=True,
        **options,
    )
    after = result.to_dict()
    assert result.resolved == saved["resolved"]
    assert result.diagnostics == saved["diagnostics"]
    assert result.spacing_resolved and not result.spacing_diagnostics
    for key in ("nodes", "fields", "samples", "excluded_regions", "phase_names"):
        assert after[key] == saved[key]
    for old, new in zip(saved["boundaries"], after["boundaries"]):
        assert {
            k: v for k, v in new.items() if k not in ("points", "limiting_curves")
        } == {k: v for k, v in old.items() if k not in ("points", "limiting_curves")}
        assert all(point in new["points"] for point in old["points"])
    if scope == "none":
        assert after == saved
    else:
        assert after["equilibrium_solves"] > saved["equilibrium_solves"]
    assert saved == before


def test_single_axis_division_is_valid(crossing):
    bulk, saved = crossing
    result = bm.refine_phase_diagram(bulk, crossing_phases(), saved, resolution=(2, 2))
    assert result.resolved, result.diagnostics
    check_spacing(result, (2, 2))


@pytest.mark.parametrize("swap_axes", [False, True])
def test_regional_refinement_preserves_unrelated_lines_and_plot_units(
    crossing, swap_axes
):
    bulk, saved = crossing
    region = ((800.0, 900.0), (0.5e9, 1.5e9))
    resolution = (11, 21)
    if swap_axes:
        region, resolution = region[::-1], resolution[::-1]
    result = bm.refine_phase_diagram(
        bulk,
        crossing_phases(),
        saved,
        resolution=resolution,
        region=region,
        swap_axes=swap_axes,
    )
    assert result.resolved, result.diagnostics
    assert result.pressure_range == saved["pressure_range"]
    assert result.temperature_range == saved["temperature_range"]
    refined = result.to_dict()
    changed = []
    for old, new in zip(saved["boundaries"], refined["boundaries"]):
        if all(p["temperature"] == 1000.0 for p in old["points"]):
            assert new == old  # Horizontal arcs never intersect the window.
        elif new != old:
            changed.append(new)
            points = new["points"]
            for a, b in zip(points[:-1], points[1:]):
                lo, hi = sorted([a["temperature"], b["temperature"]])
                if lo <= 900.0 and hi >= 800.0:
                    assert hi - lo <= 10.0 * (1.0 + 1.0e-10)
    assert len(changed) == 1
    # The same physical window expressed in Celsius and kbar is identical.
    alternate = ((526.85, 626.85), (5.0, 15.0))
    if swap_axes:
        alternate = alternate[::-1]
    equivalent = bm.refine_phase_diagram(
        bulk,
        crossing_phases(),
        saved,
        resolution=resolution,
        region=alternate,
        temperature_unit="C",
        pressure_unit="kbar",
        swap_axes=swap_axes,
    )
    assert equivalent.to_dict() == refined
    repeated = bm.refine_phase_diagram(
        bulk,
        crossing_phases(),
        result,
        resolution=resolution,
        region=region,
        swap_axes=swap_axes,
    )
    assert repeated.to_dict() == refined


@pytest.mark.parametrize(
    "region",
    [
        ((800.0, 900.0), (-0.1, 1.5)),
        ((900.0, 800.0), (0.5, 1.5)),
        ((800.0, float("nan")), (0.5, 1.5)),
    ],
)
def test_regional_refinement_rejects_invalid_windows(crossing, region):
    bulk, saved = crossing
    with pytest.raises(ValueError, match="region"):
        bm.refine_phase_diagram(
            bulk,
            crossing_phases(),
            saved,
            resolution=(11, 21),
            region=region,
            temperature_unit="K",
            pressure_unit="GPa",
        )
    with pytest.raises(ValueError, match="requires resolution"):
        bm.refine_phase_diagram(bulk, crossing_phases(), saved, region=region)


def test_completed_refinement_skips_junction_search_and_repeated_solves(
    crossing, capfd
):
    bulk, saved = crossing
    opts = bm.PhaseDiagramSettings.from_dict(saved["settings"])
    opts.verbose = True
    result = bm.refine_phase_diagram(
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
    assert "Checking phase field consistency" not in messages
    assert "Rejected boundary corrector" not in messages
    assert "seeds remaining" not in messages
    repeated = bm.refine_phase_diagram(
        bulk, crossing_phases(), result, resolution=(17, 21)
    )
    assert repeated.to_dict() == result.to_dict()
    messages = capfd.readouterr().err
    assert "Rejected boundary corrector" not in messages
    assert "seeds remaining" not in messages


def test_missing_line_remains_diagnosed_with_provisional_field_labels(crossing):
    bulk, data = crossing
    saved = json.loads(json.dumps(data))
    saved["boundaries"].pop()
    # Removing a connection invalidates the original completion flag.
    saved["resolved"] = False
    opts = bm.PhaseDiagramSettings.from_dict(saved["settings"])
    opts.max_lines = len(saved["boundaries"])  # Diagnose without tracing the gap.
    result = bm.refine_phase_diagram(
        bulk, crossing_phases(), saved, opts, resolution=(17, 21)
    )
    assert not result.resolved
    assert len(result.boundaries) == 3
    check_spacing(result, (17, 21))
    geometry = bm.phase_diagram_field_polygons(result)
    assert geometry.diagnostics
    assert len(geometry.polygons) == 3
    # Existing equilibrium limits can supply a provisional label, while the
    # missing physical boundary must still prevent a resolved result.
    assert all(p.phases for p in geometry.polygons)
    assert any(p.is_provisional for p in geometry.polygons)
    assert sum(p.area for p in geometry.polygons) == pytest.approx(1.0)


def test_refinement_reports_an_ordering_jump_without_repair():
    from burnman_cpp.minerals import model_sets

    # The old ky-out line jumped between ordered/disordered ilmenite roots.
    # Its endpoints verify, but the midpoint favours another ilmenite instance.
    saved = json.loads(
        (Path(__file__).parent / "data/metasediment_ordering_segment.json").read_text()
    )
    saved["resolved"] = False  # The known ordering jump remains unresolved.
    candidate_names = saved["phase_names"][:: saved["settings"]["max_phase_instances"]]
    by_name = {phase.name: phase for phase in model_sets.metapelite().phases}
    phases = [by_name[name] for name in candidate_names]
    result = bm.refine_phase_diagram(
        saved["composition_start"],
        phases,
        saved,
        resolution=(21, 21),
    )
    assert not result.resolved
    assert result.diagnostics == saved["diagnostics"]
    assert not result.spacing_resolved
    assert any("equilibrium refinement failed" in d for d in result.spacing_diagnostics)
    after = result.to_dict()
    for key in ("nodes", "fields", "samples"):
        assert after[key] == saved[key]
    assert len(after["boundaries"]) == len(saved["boundaries"])


def test_curved_line_refinement_checks_actual_equilibrium_and_default_plot_units():
    bulk = {"Mg": 1.0, "O": 1.0}
    phases = curved_phases()
    result = bm.phase_diagram(bulk, phases, (0.0, 5.0e9), (600.0, 1400.0), settings())
    saved = result.to_dict()
    refined = bm.refine_phase_diagram(bulk, phases, saved, resolution=(81, 101))
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
    geometry = bm.phase_diagram_field_polygons(refined)
    assert not geometry.diagnostics
    assert sum(p.area for p in geometry.polygons) == pytest.approx(1.0)


def test_metasediment_explicit_refinement_preserves_saved_samples():
    from burnman_cpp.minerals import model_sets

    # Chl-out and Mt-out from the fresh metasediment calculation. Refining
    # boundary 38 used to join two regions and incorrectly add Mt at this
    # known equilibrium sample, despite retaining every original point.
    data = json.loads(
        (Path(__file__).parent / "data" / "metasediment_curve_edit.json").read_text()
    )
    opts = bm.PhaseDiagramSettings.from_dict(data["settings"])
    opts.verbose = False
    opts.max_lines = 4
    location = [data["samples"][0][c] for c in ("pressure", "temperature")]
    expected = sorted(q["id"] for q in data["samples"][0]["phases"])
    candidate_names = data["phase_names"][:: data["settings"]["max_phase_instances"]]
    by_name = {phase.name: phase for phase in model_sets.metapelite().phases}
    phases = [by_name[name] for name in candidate_names]
    result = bm.refine_phase_diagram(
        data["composition_start"],
        phases,
        data,
        opts,
        resolution=(101, 201),
    )
    geometry = bm.phase_diagram_field_polygons(result)
    containing = [
        p
        for p in geometry.polygons
        if p.contains_rectangle(location, [1.0e-5, 1.0e-10])
    ]
    assert len(containing) == 1
    assert containing[0].phases == expected
    assert all("Curve edit rejected" not in d for d in result.diagnostics)
    assert result.to_dict()["samples"][: len(data["samples"])] == data["samples"]
    assert sum(len(line.points) for line in result.boundaries) > sum(
        len(line["points"]) for line in data["boundaries"]
    )


@pytest.mark.parametrize("diagram", ["PX", "TX"])
@pytest.mark.parametrize("end_amount", [1.0, 2.0])
def test_composition_lines_keep_analytic_coexistence_after_refinement(
    diagram, end_amount
):
    previous = calculate(diagram, end={"Fe": end_amount, "O": end_amount})
    resolution = (201, 51)
    units = dict(pressure_unit="GPa", temperature_unit="K")
    result = bm.refine_phase_diagram(
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
    previous = bm.phase_diagram(
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
    result = bm.refine_phase_diagram(
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
        bm.refine_phase_diagram(bulk, crossing_phases(), saved, resolution=resolution)


def test_unmet_resolution_is_reported_without_losing_verified_boundaries(
    crossing, tmp_path, capsys
):
    bulk, saved = crossing
    opts = bm.PhaseDiagramSettings.from_dict(saved["settings"])
    opts.max_trace_steps = 2
    opts.max_refinement_iterations = 1
    opts.max_recovery_passes = 0
    result = bm.refine_phase_diagram(
        bulk, crossing_phases(), saved, opts, resolution=(801, 2001)
    )
    assert result.resolved
    assert not result.diagnostics
    assert not result.spacing_resolved
    assert any(
        "requested line resolution" in d and "limit reached" in d
        for d in result.spacing_diagnostics
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
    for key in ("nodes", "fields", "samples"):
        assert result.to_dict()[key] == saved[key]

    path = bm.save_phase_diagram(result, tmp_path / "spacing.json")
    loaded = bm.load_phase_diagram(path)
    assert loaded == result.to_dict()
    assert loaded["resolved"] and not loaded["spacing_resolved"]
    assert bm.PhaseDiagramResult.from_dict(loaded).to_dict() == loaded
    from burnman_cpp.tools.phase_diagram_helpers import report_phase_diagram

    report_phase_diagram(result, tmp_path)
    output = capsys.readouterr().out
    assert "Resolved: True" in output
    assert "Requested line spacing was not reached:" in output
    retried = bm.refine_phase_diagram(
        bulk,
        crossing_phases(),
        loaded,
        bm.PhaseDiagramSettings.from_dict(saved["settings"]),
        resolution=(17, 21),
    )
    assert retried.resolved and retried.spacing_resolved
    assert not retried.diagnostics and not retried.spacing_diagnostics


@pytest.mark.parametrize("swap_instances", [False, True])
def test_solution_lines_refine_from_coarse_points_at_the_critical_endpoint(
    swap_instances,
):
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
    previous = bm.phase_diagram(
        bulk,
        [phase],
        settings=opts,
        diagram="TX",
        pressure=1.0e9,
        temperature_range=(600.0, 900.0),
        composition_end={"Fe": 1.0, "O": 1.0},
    )
    assert all(
        "label is provisional" in d for d in previous.diagnostics
    ), previous.diagnostics
    saved = previous.to_dict()
    for line in saved["boundaries"]:
        line["points"] = [line["points"][0], line["points"][-1]]
    if swap_instances:
        # Copy numbers do not identify the Mg-rich and Fe-rich branches.
        # Renumber the coexistence states while retaining each zero-amount
        # constraint's physical phase and the canonical field assemblages.
        def renumber(ids):
            return sorted(1 - i for i in ids)

        for node in saved["nodes"]:
            node["assemblage"] = renumber(node["assemblage"])
            node["zero_phases"] = renumber(node["zero_phases"])
        for line in saved["boundaries"]:
            line["zero_phase"] = 1 - line["zero_phase"]
            line["assemblage"] = renumber(line["assemblage"])
            for point in line["points"]:
                for state in point["phases"]:
                    state["id"] = 1 - state["id"]
                    state["name"] = saved["phase_names"][state["id"]]
    result = bm.refine_phase_diagram(
        bulk, [phase], saved, resolution=(51, 61), temperature_unit="K"
    )
    assert all(
        "label is provisional" in d for d in result.diagnostics
    ), result.diagnostics
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


def test_refinement_rejects_a_chord_joining_opposite_solvus_arms():
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
    previous = bm.phase_diagram(
        bulk,
        [phase],
        settings=opts,
        diagram="TX",
        pressure=1.0e9,
        temperature_range=(600.0, 900.0),
        composition_end={"Fe": 1.0, "O": 1.0},
    )
    assert all(
        "label is provisional" in d for d in previous.diagnostics
    ), previous.diagnostics
    saved = previous.to_dict()
    ordered = sorted(
        saved["boundaries"],
        key=lambda line: min(line["points"], key=lambda p: p["temperature"])[
            "composition_coordinate"
        ],
    )
    arms = [ordered[0], ordered[-1]]
    endpoints = [min(line["points"], key=lambda p: p["temperature"]) for line in arms]
    # Both endpoints are valid coexistence equilibria. Renumber the second
    # endpoint so its zero phase retains its physical role on that arm.
    # This does not make the chord across the miscibility gap a phase boundary.
    if arms[0]["zero_phase"] != arms[1]["zero_phase"]:
        for state in endpoints[1]["phases"]:
            state["id"] = 1 - state["id"]
            state["name"] = saved["phase_names"][state["id"]]
    line = dict(arms[0], id=0, start_node=-1, end_node=-1, points=endpoints)
    saved.update(boundaries=[line], nodes=[], fields=[], samples=[], resolved=True)
    opts.max_recovery_passes = 0
    opts.max_lines = 1
    result = bm.refine_phase_diagram(
        bulk, [phase], saved, opts, resolution=(21, 21), temperature_unit="K"
    )
    assert result.resolved and not result.diagnostics
    assert not result.spacing_resolved
    assert any("requested line resolution" in d for d in result.spacing_diagnostics)
    assert result.to_dict()["boundaries"][0]["points"] == endpoints
