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
import runpy

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
        for point in line.points:
            assert point.mass_balance_error < 1.0e-8
            assert point.minimum_affinity >= -result.settings.affinity_tolerance
            assert point.residual < 0.01


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
def test_resolved_refinement_preserves_graph_ids_labels_and_saved_evidence(
    crossing, scope
):
    bulk, original = crossing
    saved = deepcopy(original)
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
    assert result.resolved and result.spacing_resolved
    assert not result.diagnostics and not result.spacing_diagnostics
    for key in ("nodes", "fields", "samples", "excluded_regions", "phase_names"):
        assert after[key] == saved[key]
    for old, new in zip(saved["boundaries"], after["boundaries"]):
        assert {k: v for k, v in new.items() if k != "points"} == {
            k: v for k, v in old.items() if k != "points"
        }
        assert all(point in new["points"] for point in old["points"])
    if scope == "none":
        assert after == saved
    else:
        assert after["equilibrium_solves"] > saved["equilibrium_solves"]
    assert saved == before


@pytest.mark.parametrize(
    "name,finite", [("univariant_segments", False), ("thin_finite_field", True)]
)
def test_metasediment_reaction_segments_preserve_the_thin_finite_field(name, finite):
    from burnman_cpp.minerals import model_sets

    # Accepted full-diagram arcs, including different zero-amount constraints
    # on the same reaction. No synthetic equilibrium states or label votes.
    data = json.loads(
        (Path(__file__).parent / "data" / f"metasediment_{name}.json").read_text()
    )
    before = bm.phase_diagram_field_polygons(data, close_domain=False)
    assert len(before.polygons) == 1
    if not finite:
        assert not before.polygons[0].phases
    result = bm.refine_phase_diagram(
        data["composition_start"],
        model_sets.metapelite().phases,
        data,
        resolution=(2, 2),
        region=((573.15, 1173.15), (1.0e5, 2.0e9)),
        temperature_unit="K",
        pressure_unit="Pa",
    )
    geometry = bm.phase_diagram_field_polygons(result, close_domain=False)
    assert result.settings.affinity_tolerance == data["settings"]["affinity_tolerance"]
    assert result.pressure_range == data["pressure_range"]
    assert result.temperature_range == data["temperature_range"]
    if finite:
        assert len(geometry.polygons) == 1
        field = geometry.polygons[0]
        assert field.phases == [0, 3, 6, 21, 33, 36, 39, 45, 48, 63]
        assert field.n_phases == 10 and not field.is_provisional
        assert field.area > 0.0
        # The saved phase-out trace overshot its verified junction by
        # 34 Pa before doubling back. Preserve the finite field while
        # removing that excursion, rather than preserving its chord area.
        edge = next(
            b for b in result.boundaries if {b.start_node, b.end_node} == {2, 3}
        )
        assert (
            max(p.temperature for p in edge.points)
            <= result.nodes[2].temperature + 1.0e-6
        )
    else:
        assert not geometry.polygons
        assert len(result.boundaries) == 3
        assert {frozenset((b.start_node, b.end_node)) for b in result.boundaries} == {
            frozenset((0, 3)),
            frozenset((3, 1)),
            frozenset((1, 2)),
        }
    for line in result.boundaries:
        for point in line.points:
            assert point.mass_balance_error <= result.settings.mass_balance_tolerance
            assert point.minimum_affinity >= -result.settings.affinity_tolerance


@pytest.mark.parametrize("name", ["parallel_fields", "curved_node_connections"])
def test_metasediment_curved_loops_keep_finite_fields_and_stable_connections(name):
    from burnman_cpp.minerals import model_sets

    path = Path(__file__).parent / "data" / f"metasediment_{name}.json"
    data = json.loads(path.read_text())
    result = bm.refine_phase_diagram(
        data["composition_start"],
        model_sets.metapelite().phases,
        data,
        resolution=(2, 2),
        region=((573.15, 1173.15), (1.0e5, 2.0e9)),
        temperature_unit="K",
        pressure_unit="Pa",
    )
    geometry = bm.phase_diagram_field_polygons(result, close_domain=False)
    assert all(not p.is_provisional for p in geometry.polygons)
    if name == "parallel_fields":
        # The original chords gave the intervening eight-phase field negative
        # area. Its saved minimization must belong only to that finite strip.
        expected = [0, 3, 6, 21, 36, 45, 48, 63]
        target = [1.0881852213541665e8, 773.15]
        assert len(geometry.polygons) == 3
        matches = [
            p for p in geometry.polygons if p.contains_rectangle(target, [1.0, 1.0e-8])
        ]
        assert len(matches) == 1 and matches[0].phases == expected
        assert matches[0].area > 0.0
        # Its neighbours retain distinct seven- and nine-phase fields.
        assert {p.n_phases for p in geometry.polygons} == {7, 8, 9}
    else:
        assert {tuple(p.phases) for p in geometry.polygons} == {
            (0, 1, 3, 18, 21, 36, 45, 60),
            (0, 1, 3, 18, 21, 36, 37, 45, 60),
        }
        nodes = data["fixture_original_nodes"]
        assert not any(
            b.zero_phase == 51
            and {b.start_node, b.end_node} == {nodes.index(172), nodes.index(173)}
            for b in result.boundaries
        )
        full = bm.phase_diagram_field_polygons(result)
        matches = [
            p
            for p in full.polygons
            if p.contains_rectangle(
                [1.5683486180088599e9, 689.1695969015594], [1.0, 1.0e-8]
            )
        ]
        assert len(matches) == 1 and not matches[0].is_provisional
        assert matches[0].phases == [0, 1, 3, 18, 21, 36, 45, 51, 60]
    assert result.settings.affinity_tolerance == data["settings"]["affinity_tolerance"]
    assert result.pressure_range == data["pressure_range"]
    assert result.temperature_range == data["temperature_range"]
    for boundary in result.boundaries:
        for point in boundary.points:
            assert point.mass_balance_error <= result.settings.mass_balance_tolerance
            assert point.minimum_affinity >= -result.settings.affinity_tolerance


def transform_saved_traces(data, transform):
    if transform == "reverse":
        for line in data["boundaries"]:
            line["start_node"], line["end_node"] = line["end_node"], line["start_node"]
            line["side_a"], line["side_b"] = line["side_b"], line["side_a"]
            line["points"].reverse()
            line["termination"] = "; ".join(reversed(line["termination"].split("; ")))
    elif transform == "renumber":
        copies = data["settings"]["max_phase_instances"]

        def remap(i):
            return i if i < 0 else i // copies * copies + copies - 1 - i % copies

        for node in data["nodes"]:
            for key in ["assemblage", "zero_phases"]:
                node[key] = sorted(map(remap, node[key]))
        for line in data["boundaries"]:
            line["zero_phase"] = remap(line["zero_phase"])
            for key in ["assemblage", "side_a", "side_b"]:
                line[key] = sorted(map(remap, line[key]))
            for point in line["points"]:
                for phase in point["phases"]:
                    phase["id"] = remap(phase["id"])
                point["phases"].sort(key=lambda p: p["id"])


@pytest.mark.parametrize("transform", ["original", "reverse"])
@pytest.mark.parametrize("name", ["duplicate_invariant", "reaction_extents"])
def test_pyrolite_reaction_extents_do_not_enclose_artificial_fields(name, transform):
    path = Path(__file__).parent / "data" / f"pyrolite_{name}.json"
    data = json.loads(path.read_text())
    # This extracted network contains a known duplicate and requests repair.
    data["resolved"] = False
    transform_saved_traces(data, transform)
    factory = runpy.run_path(
        str(
            Path(__file__).parents[2] / "examples" / "example_pyrolite_phase_diagram.py"
        )
    )["candidate_phases"]
    result = bm.refine_phase_diagram(
        data["composition_start"],
        factory(),
        data,
        resolution=(2, 2),
        region=(data["temperature_range"], data["pressure_range"]),
        temperature_unit="K",
        pressure_unit="Pa",
    )
    geometry = bm.phase_diagram_field_polygons(result, close_domain=False)
    assert all(p.phases and not p.is_provisional for p in geometry.polygons)
    if name == "duplicate_invariant":
        # Splitting at verified nodes can increase the number of segments.
        # The duplicate invariant must retire without enclosing a field.
        original = data["fixture_original_nodes"]
        assert result.nodes[original.index(83)].incident_lines
        assert not result.nodes[original.index(328)].incident_lines
        assert not geometry.polygons
    else:
        assert len(result.boundaries) < len(data["boundaries"])
    for line in result.boundaries:
        for point in line.points:
            assert point.mass_balance_error <= result.settings.mass_balance_tolerance
            assert point.minimum_affinity >= -result.settings.affinity_tolerance


def refine_pyrolite_fixture(data):
    factory = runpy.run_path(
        str(
            Path(__file__).parents[2] / "examples" / "example_pyrolite_phase_diagram.py"
        )
    )["candidate_phases"]
    return bm.refine_phase_diagram(
        data["composition_start"],
        factory(),
        dict(data, resolved=False),  # These fixtures deliberately request repair.
        resolution=(2, 2),
        region=(data["temperature_range"], data["pressure_range"]),
        temperature_unit="K",
        pressure_unit="Pa",
    )


@pytest.mark.parametrize("transform", ["original", "reverse", "renumber"])
def test_pyrolite_zoom_truncates_events_and_preserves_pyroxene_coexistence(transform):
    data = json.loads(
        (
            Path(__file__).parent / "data" / "pyrolite_zoom_reaction_limits.json"
        ).read_text()
    )
    transform_saved_traces(data, transform)
    result = refine_pyrolite_fixture(data)
    geometry = bm.phase_diagram_field_polygons(result)
    assert all(p.phases and not p.is_provisional for p in geometry.polygons)
    coexistence = [0, 9, 12, 13, 15, 18, 27, 39]
    fields = [p for p in geometry.polygons if p.phases == coexistence]
    assert len(fields) == 1
    # Independent correctors locate a finite strip, about 695 Pa wide here.
    assert fields[0].contains_rectangle([3933525200.0, 96.5], [1.0, 1.0e-8])
    for line in result.boundaries:
        if line.zero_phase // result.settings.max_phase_instances != 7:
            continue
        p = np.asarray([q.pressure for q in line.points])
        for n in [line.start_node, line.end_node]:
            if result.nodes[n].kind == "junction":
                event = result.nodes[n].pressure
                # These spinel-out branches end at the coexistence limits.
                if n == 1:
                    assert p.max() <= event + 0.2
                elif n == 3:
                    assert p.min() >= event - 0.2
    for line in result.boundaries:
        for point in line.points:
            assert point.mass_balance_error <= result.settings.mass_balance_tolerance
            assert point.minimum_affinity >= -result.settings.affinity_tolerance


@pytest.mark.parametrize("transform", ["original", "reverse", "renumber"])
def test_equivalent_junctions_use_the_actual_zero_amount_solution_copies(transform):
    data = json.loads(
        (
            Path(__file__).parent / "data" / "pyrolite_exchanged_junction_zeros.json"
        ).read_text()
    )
    transform_saved_traces(data, transform)
    original = json.dumps(data)
    result = refine_pyrolite_fixture(data)
    assert json.dumps(data) == original
    for a, b in data["fixture_duplicate_nodes"]:
        assert result.nodes[a].incident_lines
        assert not result.nodes[b].incident_lines
    for representation in [result, result.to_dict()]:
        geometry = bm.phase_diagram_field_polygons(representation, close_domain=False)
        assert not [p for p in geometry.polygons if not p.outside_model_domain]
    assert (
        result.settings.composition_tolerance
        == data["settings"]["composition_tolerance"]
    )


@pytest.mark.parametrize(
    "name,phases",
    [
        ("small40", [3, 12, 18, 19, 20, 39, 60, 72]),
        ("large69", None),
        ("large92", [3, 6, 12, 15, 18, 19, 39]),
        ("large94", None),
    ],
)
def test_localized_univariant_junctions_preserve_finite_pyrolite_fields(name, phases):
    path = Path(__file__).parent / "data/pyrolite_duplicate_junctions.json"
    data = json.loads(path.read_text())[name]
    for pair in data["fixture_duplicate_nodes"]:
        assert all(
            data["nodes"][n]["gibbs_variance"] == 1
            and data["nodes"][n]["pt_nullity"] == 0
            for n in pair
        )
    before = bm.phase_diagram_field_polygons(data, close_domain=False)
    assert any(not p.phases and not p.outside_model_domain for p in before.polygons)
    result = refine_pyrolite_fixture(data)
    geometry = bm.phase_diagram_field_polygons(result, close_domain=False)
    fields = [p for p in geometry.polygons if not p.outside_model_domain]
    if phases is None:
        assert not fields
    else:
        assert len(fields) == 1
        field = fields[0]
        assert field.phases == phases
        assert (
            field.area > 0.0
            and not field.is_provisional
            and not field.has_open_boundary
        )
        if name == "small40":
            assert field.contains_rectangle([10.84e9, 45.0], [1.0, 1.0e-8])
            assert field.area == pytest.approx(
                max(p.area for p in before.polygons if not p.outside_model_domain),
                rel=1.0e-6,
            )
    for pair in data["fixture_duplicate_nodes"]:
        assert result.nodes[min(pair)].incident_lines
        assert not result.nodes[max(pair)].incident_lines
    assert len(result.boundaries) < len(data["boundaries"])
    for line in result.boundaries:
        for point in line.points:
            assert point.mass_balance_error <= result.settings.mass_balance_tolerance
            assert point.minimum_affinity >= -result.settings.affinity_tolerance


@pytest.mark.parametrize("transform", ["reverse", "renumber"])
@pytest.mark.parametrize("name", ["small40", "large69", "large92", "large94"])
def test_duplicate_junction_identity_ignores_direction_and_instance_numbers(
    name, transform
):
    data = json.loads(
        (Path(__file__).parent / "data/pyrolite_duplicate_junctions.json").read_text()
    )[name]
    transform_saved_traces(data, transform)
    result = refine_pyrolite_fixture(data)
    # Junction identity is independent of the later curve/side-label repair.
    for pair in data["fixture_duplicate_nodes"]:
        assert result.nodes[min(pair)].incident_lines
        assert not result.nodes[max(pair)].incident_lines


@pytest.mark.parametrize("transform", ["original", "reverse", "renumber"])
def test_nearby_bivariant_junctions_keep_the_finite_basalt_field(transform):
    from burnman_cpp.minerals import model_sets

    path = Path(__file__).parent / "data/basalt_nearby_bivariant_junctions.json"
    data = json.loads(path.read_text())
    transform_saved_traces(data, transform)
    for pair in data["fixture_nearby_nodes"]:
        assert all(
            data["nodes"][n]["gibbs_variance"] == 2
            and data["nodes"][n]["pt_nullity"] == 0
            for n in pair
        )
    result = bm.refine_phase_diagram(
        data["composition_start"],
        model_sets.metabasite().phases,
        data,
        resolution=(2, 2),
        region=(data["temperature_range"], data["pressure_range"]),
        temperature_unit="K",
        pressure_unit="Pa",
    )
    geometry = bm.phase_diagram_field_polygons(result, close_domain=False)
    fields = [
        p
        for p in geometry.polygons
        if p.contains_rectangle([945.0e6, 740.0], [1.0, 1.0e-8])
    ]
    assert len(fields) == 1
    assert fields[0].phases == [0, 1, 2, 15, 18, 21, 24, 33, 42]
    assert not fields[0].is_provisional and not fields[0].has_open_boundary
    for pair in data["fixture_nearby_nodes"]:
        assert all(result.nodes[n].incident_lines for n in pair)


@pytest.mark.parametrize("transform", ["original", "reverse", "renumber"])
def test_nearby_univariant_junctions_keep_distinct_solution_compositions(transform):
    path = Path(__file__).parent / "data/pyrolite_nearby_univariant_junctions.json"
    data = json.loads(path.read_text())
    transform_saved_traces(data, transform)
    before = bm.phase_diagram_field_polygons(data, close_domain=False)
    target = [203896066.0346335, 545.4829281751656]
    factory = runpy.run_path(
        str(
            Path(__file__).parents[2] / "examples" / "example_pyrolite_phase_diagram.py"
        )
    )["candidate_phases"]
    result = bm.refine_phase_diagram(
        data["composition_start"],
        factory(),
        data,
        resolution=(2, 2),
        region=(data["temperature_range"], data["pressure_range"]),
        temperature_unit="K",
        pressure_unit="Pa",
    )
    geometry = bm.phase_diagram_field_polygons(result, close_domain=False)
    fields = [
        p for p in geometry.polygons if p.contains_rectangle(target, [1.0, 1.0e-8])
    ]
    original = [
        p for p in before.polygons if p.contains_rectangle(target, [1.0, 1.0e-8])
    ]
    assert len(fields) == len(original) == 1
    assert fields[0].phases == [0, 9, 12, 21, 22, 24, 51, 54]
    assert not fields[0].is_provisional and not fields[0].has_open_boundary
    assert fields[0].area == pytest.approx(original[0].area, rel=1.0e-5)


def test_crossed_pyrolite_chords_require_verified_connections():
    path = Path(__file__).parent / "data/pyrolite_crossed_loops.json"
    data = json.loads(path.read_text())
    data["resolved"] = False  # Repair the known metastable crossing.
    factory = runpy.run_path(
        str(
            Path(__file__).parents[2] / "examples" / "example_pyrolite_phase_diagram.py"
        )
    )["candidate_phases"]
    result = bm.refine_phase_diagram(
        data["composition_start"],
        factory(),
        data,
        resolution=(2, 2),
        region=(data["temperature_range"], data["pressure_range"]),
        temperature_unit="K",
        pressure_unit="Pa",
    )
    # Correcting equivalent junctions resolves the six-phase loop. The other
    # crossing is a metastable chord whose interior fails the stable-equilibrium
    # probe, so it is dropped and no unlabelled polygon remains.
    geometry = bm.phase_diagram_field_polygons(result, close_domain=False)
    assert len(result.nodes) == len(data["nodes"])
    assert all(p.phases and not p.is_provisional for p in geometry.polygons)
    assert any(
        p.phases == [6, 12, 18, 19, 39, 60] and not p.is_provisional
        for p in geometry.polygons
    )
    for line in result.boundaries:
        for point in line.points:
            assert point.mass_balance_error <= result.settings.mass_balance_tolerance
            assert point.minimum_affinity >= -result.settings.affinity_tolerance


def test_regional_retrace_recovers_missing_metasediment_connection():
    from burnman_cpp.minerals import model_sets

    # Counterexample assembled from accepted local states and the actual coarse
    # melt-out arc that skipped the second-ilmenite junction in the full diagram.
    path = Path(__file__).parent / "data/metasediment_missing_melt_connection.json"
    saved = json.loads(path.read_text())
    before = bm.phase_diagram_field_polygons(saved)
    target = [1.107e9, 950.5]
    assert any(
        p.is_provisional and p.contains_rectangle(target, [1.0, 1.0e-8])
        for p in before.polygons
    )
    assert not any(
        b["zero_phase"] == 42 and {b["start_node"], b["end_node"]} == {6, 11}
        for b in saved["boundaries"]
    )
    result = bm.refine_phase_diagram(
        saved["composition_start"],
        model_sets.metapelite().phases,
        saved,
        resolution=(21, 21),
        region=((944.0, 955.0), (1.09, 1.12)),
        pressure_unit="GPa",
        temperature_unit="K",
    )
    assert result.pressure_range == saved["pressure_range"]
    assert result.temperature_range == saved["temperature_range"]
    assert any(
        b.zero_phase == 42 and {b.start_node, b.end_node} == {6, 11}
        for b in result.boundaries
    )
    after = bm.phase_diagram_field_polygons(result)
    fields = [p for p in after.polygons if p.contains_rectangle(target, [1.0, 1.0e-8])]
    assert len(fields) == 1 and not fields[0].is_provisional
    assert list(fields[0].phases) == [0, 3, 18, 21, 36, 45, 51, 60, 63]
    check_spacing(result, (21, 21), pressure_unit="GPa", temperature_unit="K")


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
    region = ((800.0, 900.0), (0.5, 1.5))
    resolution = (11, 21)
    if swap_axes:
        region, resolution = region[::-1], resolution[::-1]
    result = bm.refine_phase_diagram(
        bulk,
        crossing_phases(),
        saved,
        resolution=resolution,
        region=region,
        temperature_unit="K",
        pressure_unit="GPa",
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
        swap_axes=swap_axes,
    )
    assert equivalent.to_dict() == refined
    repeated = bm.refine_phase_diagram(
        bulk,
        crossing_phases(),
        result,
        resolution=resolution,
        region=region,
        temperature_unit="K",
        pressure_unit="GPa",
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
    assert any("conflicting perimeter assemblages" in d for d in geometry.diagnostics)
    assert len(geometry.polygons) == 3
    # Existing equilibrium limits can supply a provisional label, while the
    # missing physical boundary must still prevent a resolved result.
    assert all(p.phases for p in geometry.polygons)
    assert any(p.is_provisional for p in geometry.polygons)
    assert sum(p.area for p in geometry.polygons) == pytest.approx(1.0)


def test_completed_snapshot_recovers_a_dangling_internal_junction(crossing):
    bulk, original = crossing
    saved = json.loads(json.dumps(original))
    saved["boundaries"] = saved["boundaries"][:1]
    saved["samples"] = []
    saved["fields"] = []
    # Deliberately removing branches requests recovery of an unresolved graph.
    saved["resolved"] = False
    assert "junction" in saved["boundaries"][0]["termination"]
    opts = bm.PhaseDiagramSettings.from_dict(saved["settings"])
    opts.max_recovery_passes = 0
    opts.max_lines = 1
    incomplete = bm.refine_phase_diagram(
        bulk, crossing_phases(), saved, opts, resolution=(17, 21)
    )
    assert not incomplete.resolved
    assert any("Unpaired internal junction" in d for d in incomplete.diagnostics)
    opts.max_recovery_passes = 2
    opts.max_lines = 500
    repaired = bm.refine_phase_diagram(
        bulk, crossing_phases(), saved, opts, resolution=(17, 21)
    )
    assert repaired.resolved, repaired.diagnostics
    assert len(repaired.boundaries) == 4
    assert len(bm.phase_diagram_field_polygons(repaired).polygons) == 4
    check_spacing(repaired, (17, 21))


@pytest.mark.parametrize("recovery", [False, True])
def test_saved_ordering_jump_recovers_the_intervening_two_ilmenite_fields(recovery):
    from burnman_cpp.minerals import model_sets

    # The old ky-out line jumped between ordered/disordered ilmenite roots.
    # Its endpoints verify, but the midpoint favours another ilmenite instance.
    saved = json.loads(
        (Path(__file__).parent / "data/metasediment_ordering_segment.json").read_text()
    )
    saved["resolved"] = False  # The known ordering jump requires recovery.
    if not recovery:
        saved["settings"]["max_recovery_passes"] = 0
    result = bm.refine_phase_diagram(
        saved["composition_start"],
        model_sets.metapelite().phases,
        saved,
        resolution=(21, 21),
    )
    if not recovery:
        assert result.resolved and not result.diagnostics
        assert not result.spacing_resolved
        assert any(
            "equilibrium refinement failed" in d for d in result.spacing_diagnostics
        )
        return
    assert all(
        "label is provisional" in d for d in result.diagnostics
    ), result.diagnostics
    polygons = bm.phase_diagram_field_polygons(result).polygons
    assert any(
        "ilm #2" in [result.phase_names[i] for i in field.phases] for field in polygons
    )
    assert all(p.phases for p in polygons)
    check_spacing(result, (21, 21))


@pytest.mark.parametrize("crowded", [False, True])
@pytest.mark.parametrize("previously_verified", [False, True])
def test_perimeter_labels_survive_crowded_or_previously_verified_warm_roots(
    crowded, previously_verified
):
    from copy import deepcopy
    from burnman_cpp.minerals import model_sets

    # Perimeter and warm roots captured immediately before interior checking
    # changed a correct basalt field to a metastable plagioclase-free root.
    saved = json.loads(
        (Path(__file__).parent / "data/basalt_field_verification.json").read_text()
    )
    saved["samples"][0]["is_field_verification"] = previously_verified
    if crowded:
        # The physically competing state must be tried even when more than
        # 32 closer starting states all supply the higher-energy assemblage.
        for i in range(40):
            sample = deepcopy(saved["samples"][0])
            sample["pressure"] += (i + 1) * 10.0
            saved["samples"].append(sample)
    result = bm.refine_phase_diagram(
        saved["composition_start"], model_sets.metabasite().phases, saved
    )
    geometry = bm.phase_diagram_field_polygons(result, merge_fields=False)
    polygons = [
        polygon
        for polygon in geometry.polygons
        if polygon.contains_rectangle(
            [906391997.4521586, 725.3100445735436], [1.0e-5, 1.0e-10]
        )
    ]
    assert len(polygons) == 1
    assert not polygons[0].has_open_boundary
    assert [result.phase_names[i] for i in polygons[0].phases] == [
        "hb",
        "hb #2",
        "hb #3",
        "pl",
        "ep",
        "bi",
        "q",
        "sph",
        "ab",
    ]


@pytest.mark.parametrize(
    "name, point, expected",
    [
        (
            "plagioclase",
            [906391997.4521586, 725.3100445735436],
            ["hb", "hb #2", "hb #3", "pl", "ep", "bi", "q", "sph", "ab"],
        ),
        (
            "amphibole",
            [1478685047.4731708, 808.0422314933551],
            ["hb", "hb #2", "hb #3", "g", "ep", "q", "ru", "sph", "H2O", "ms"],
        ),
    ],
)
def test_local_basalt_fields_use_saved_evidence_without_new_interior_roots(
    name, point, expected
):
    from burnman_cpp.minerals import model_sets

    # Small independently seeded diagrams around the two basalt regressions.
    # The amphibole case also needs a new chlorite-bearing root: both supplied
    # assemblages at its ambiguous interior fail tighter stability checks.
    saved = json.loads(
        (Path(__file__).parent / f"data/basalt_local_{name}.json").read_text()
    )
    result = bm.refine_phase_diagram(
        saved["composition_start"], model_sets.metabasite().phases, saved
    )
    geometry = bm.phase_diagram_field_polygons(result, merge_fields=False)
    assert all(p.phases for p in geometry.polygons)
    assert all("label is provisional" in d for d in geometry.diagnostics)
    containing = [
        polygon
        for polygon in geometry.polygons
        if polygon.contains_rectangle(point, [1.0e-5, 1.0e-10])
    ]
    assert len(containing) == 1
    assert not containing[0].has_open_boundary
    assert [result.phase_names[i] for i in containing[0].phases] == expected


def test_local_pyrolite_neighbours_escape_the_phase_amount_cutoff():
    saved = json.loads(
        (Path(__file__).parent / "data/pyrolite_local_opx.json").read_text()
    )
    example = runpy.run_path(
        str(Path(__file__).parents[2] / "examples/example_pyrolite_phase_diagram.py")
    )
    result = bm.refine_phase_diagram(
        saved["composition_start"], example["candidate_phases"](), saved
    )
    assert result.resolved, result.diagnostics
    geometry = bm.phase_diagram_field_polygons(result, merge_fields=False)
    assert not geometry.diagnostics
    assert {tuple(p.phases) for p in geometry.polygons} == {
        tuple(f["phases"]) for f in saved["fields"]
    }
    assert all(not p.has_open_boundary for p in geometry.polygons)
    for line in result.boundaries:
        assert line.side_a != line.side_b
    polygon = next(
        p
        for p in geometry.polygons
        if p.contains_rectangle(
            [5403210373.62519, 690.4929506484826], [1.0e-5, 1.0e-10]
        )
    )
    assert "opx" in [result.phase_names[i] for i in polygon.phases]


def test_pyrolite_refinement_crosses_pure_to_mixed_solution_compositions():
    saved = json.loads(
        (
            Path(__file__).parent / "data" / "pyrolite_refinement_segment.json"
        ).read_text()
    )["previous"]
    example = runpy.run_path(
        str(
            Path(__file__).parents[2] / "examples" / "example_pyrolite_phase_diagram.py"
        )
    )
    opts = bm.PhaseDiagramSettings.from_dict(saved["settings"])
    opts.verbose = False
    opts.max_lines = 1
    opts.max_recovery_passes = 0
    result = bm.refine_phase_diagram(
        saved["composition_start"],
        example["candidate_phases"](),
        saved,
        opts,
        resolution=(401, 401),
    )
    assert len(result.boundaries) == 1
    assert not result.spacing_diagnostics
    check_spacing(result, (401, 401))
    points = result.boundaries[0].points
    assert any(32.0 < p.temperature < 47.0 for p in points)
    refined = result.to_dict()["boundaries"][0]["points"]
    assert all(p in refined for p in saved["boundaries"][0]["points"])


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


def test_near_chord_sample_does_not_trigger_automatic_geometry_repair():
    phases = curved_phases()
    bulk = {"Mg": 1.0, "O": 1.0}
    opts = settings()
    previous = bm.phase_diagram(bulk, phases, (0.0, 5.0e9), (600.0, 1400.0), opts)
    data = previous.to_dict()
    points = data["boundaries"][0]["points"]
    a, b = points[len(points) // 2 : len(points) // 2 + 2]
    fraction = 0.01
    temperature = (1.0 - fraction) * a["temperature"] + fraction * b["temperature"]
    chord = (1.0 - fraction) * a["pressure"] + fraction * b["pressure"]
    # Locate the actual coexistence pressure independently from tracing.
    low, high = data["pressure_range"]
    for _ in range(60):
        middle = (low + high) / 2.0
        for phase in phases:
            phase.set_state(middle, temperature)
        if phases[0].molar_gibbs < phases[1].molar_gibbs:
            low = middle
        else:
            high = middle
    pressure = (chord + (low + high) / 2.0) / 2.0

    def verified(p, t):
        tight = bm.PhaseDiagramSettings.from_dict(data["settings"])
        tight.affinity_tolerance = 1.0e-5
        state = bm.stable_equilibrium(bulk, phases, p, t, tight)
        assert state.success, state.message
        return dict(
            data["samples"][0],
            pressure=p,
            temperature=t,
            entropy=state.entropy,
            volume=state.volume,
            success=True,
            is_field_verification=True,
            message=state.message,
            gibbs=state.gibbs,
            mass_balance_error=state.mass_balance_error,
            minimum_affinity=state.minimum_affinity,
            equilibrium_error=state.equilibrium_error,
            phases=[
                dict(
                    id=q.id,
                    candidate_index=q.candidate_index,
                    name=q.name,
                    amount=q.amount,
                    composition=list(q.composition),
                )
                for q in state.phases
            ],
        )

    data["samples"] = [
        verified(1.0e9, 1000.0),
        verified(4.8e9, 1000.0),
        verified(pressure, temperature),
    ]
    assert not bm.phase_diagram_field_polygons(data).diagnostics
    opts.max_lines = 1
    result = bm.refine_phase_diagram(bulk, phases, data, opts)
    assert result.resolved, result.diagnostics
    assert len(result.boundaries) == 1
    assert len(result.boundaries[0].points) == len(points)
    assert result.equilibrium_solves == data["equilibrium_solves"]
    assert result.to_dict()["samples"] == data["samples"]


@pytest.mark.parametrize("recovery_passes", [0, 2])
def test_metasediment_explicit_refinement_preserves_saved_samples(
    recovery_passes,
):
    from burnman_cpp.minerals import model_sets

    # Chl-out and Mt-out from the fresh metasediment calculation. Refining
    # boundary 38 used to join two regions and incorrectly add Mt at this
    # known equilibrium sample, despite retaining every original point.
    data = json.loads(
        (Path(__file__).parent / "data" / "metasediment_curve_edit.json").read_text()
    )
    opts = bm.PhaseDiagramSettings.from_dict(data["settings"])
    opts.verbose = False
    opts.max_recovery_passes = recovery_passes
    opts.max_lines = 4
    location = [data["samples"][0][c] for c in ("pressure", "temperature")]
    expected = sorted(q["id"] for q in data["samples"][0]["phases"])
    result = bm.refine_phase_diagram(
        data["composition_start"],
        model_sets.metapelite().phases,
        data,
        opts,
        resolution=(101, 201),
    )
    geometry = bm.phase_diagram_field_polygons(result, merge_fields=False)
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


@pytest.mark.parametrize("structural", [[], ["Field 1 label is provisional."]])
def test_legacy_spacing_diagnostics_migrate_without_losing_metadata(
    crossing, tmp_path, structural
):
    _, original = crossing
    saved = deepcopy(original)
    warning = (
        "Boundary 0: requested line resolution was not reached on "
        "1 segment(s) (refinement limit reached)."
    )
    saved.update(schema_version=3, resolved=False, diagnostics=structural + [warning])
    del saved["spacing_diagnostics"], saved["spacing_resolved"]
    expected_resolved = not structural
    native = bm.PhaseDiagramResult.from_dict(saved)
    assert native.resolved == expected_resolved
    assert native.diagnostics == structural
    assert not native.spacing_resolved
    assert native.spacing_diagnostics == [warning]

    saved["description"] = "Preserve extra saved metadata."
    source = tmp_path / "legacy.json"
    source.write_text(json.dumps(saved))
    loaded = bm.load_phase_diagram(source)
    assert loaded["resolved"] == expected_resolved
    assert loaded["diagnostics"] == structural
    assert loaded["spacing_diagnostics"] == [warning]
    assert not loaded["spacing_resolved"]
    assert loaded["description"] == saved["description"]
    destination = bm.save_phase_diagram(saved, tmp_path / "updated.json")
    assert bm.load_phase_diagram(destination) == loaded


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
