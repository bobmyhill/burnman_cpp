# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Reaction topology can verify an assemblage without resolving its width."""

from copy import deepcopy
import json
from pathlib import Path

import pytest
import burnman_cpp as bm
from conftest import diagnostic
from burnman_cpp.minerals import model_sets


@pytest.fixture
def source():
    return json.loads(
        (Path(__file__).parent / "data/taylor_subpascal_perimeters.json").read_text()
    )


@pytest.fixture(scope="module")
def models():
    return model_sets.slb2024().phases


def unresolved_fields(data):
    return [
        p
        for p in bm.phase_diagram_field_polygons(data).polygons
        if p.has_unresolved_interior
    ]


def test_stable_reaction_perimeters_remove_only_label_uncertainty(
    source, models, tmp_path
):
    assert len(unresolved_fields(source)) == 2
    assert all(
        p.has_closed_reaction_ring and not p.is_provisional
        for p in unresolved_fields(source)
    )
    result = bm.verify_phase_diagram_field_perimeters(source, models)
    certificates = result.to_dict()["field_verifications"]
    assert len(certificates) == 2
    assert all(c["success"] for c in certificates), certificates
    assert all(c["checked_junctions"] == 3 for c in certificates)
    assert all(c["checked_boundary_points"] >= 9 for c in certificates)
    # The first strip's coarse chords cross, while the corrected reactions do not.
    assert certificates[0]["checked_crossings"] >= 2
    limits = source["settings"]
    assert all(
        c["maximum_mass_balance_error"] <= limits["mass_balance_tolerance"]
        and c["minimum_affinity"] >= -limits["affinity_tolerance"]
        for c in certificates
    ), certificates
    fields = unresolved_fields(result)
    assert len(fields) == 2
    assert all(
        p.has_verified_reaction_perimeter and not p.is_provisional for p in fields
    )
    # The cropped fixture omits the surrounding fields, which remain a real
    # blocking issue. Its four interior-coordinate notes are informational.
    assert list(result.diagnostics) == [
        "Field 3 has conflicting perimeter assemblages; no consistent label was found."
    ]
    assert len(result.coordinate_diagnostics) == 4
    path = bm.save_phase_diagram(result, tmp_path / "verified.json")
    restored = bm.load_phase_diagram(path)
    assert restored["field_verifications"] == certificates
    assert restored["coordinate_diagnostics"] == result.coordinate_diagnostic_records
    assert all(
        p.has_verified_reaction_perimeter and not p.is_provisional
        for p in unresolved_fields(restored)
    )
    # Native restoration retains the same certificates too.
    assert (
        bm.PhaseDiagramResult.from_dict(restored).to_dict()["field_verifications"]
        == certificates
    )


def test_common_vertices_do_not_certify_metastable_reactions(source, models):
    lower_silica = bm.CombinedMineral(
        [models[20]], [1.0], [-1.0e5, 0.0, 0.0], name="lower silica"
    )
    result = bm.verify_phase_diagram_field_perimeters(source, [*models, lower_silica])
    assert not any(c["success"] for c in result.to_dict()["field_verifications"])
    assert all(p.is_provisional for p in unresolved_fields(result))


def test_reaction_constraints_identify_a_side_missed_by_probes(source, models):
    line = next(b for b in source["boundaries"] if b["id"] == 360)
    line["side_a"] = []
    result = bm.verify_phase_diagram_field_perimeters(source, models)
    certificate = next(
        c
        for c in result.to_dict()["field_verifications"]
        if c["boundary_ids"] == [292, 360, 368]
    )
    assert certificate["success"], certificate
    assert certificate["checked_junctions"] == 3
    assert certificate["checked_boundary_points"] >= 9


@pytest.mark.parametrize("at_saved_point", [False, True])
def test_intervening_line_requires_a_verified_crossing(source, models, at_saved_point):
    line = next(b for b in source["boundaries"] if b["id"] == 368)
    middle = deepcopy(line["points"][1])
    if not at_saved_point:
        first, last = line["points"][:2]
        middle["pressure"] = 0.5 * (first["pressure"] + last["pressure"])
        middle["temperature"] = 0.5 * (first["temperature"] + last["temperature"])
    points = []
    for offset in [-1.0e5, 0.0, 1.0e5]:
        point = deepcopy(middle)
        point["pressure"] += offset
        point["phases"] = []
        points.append(point)
    source["boundaries"].append(
        dict(
            id=400,
            zero_phase=-1,
            start_node=-1,
            end_node=-1,
            assemblage=[],
            side_a=[],
            side_b=[],
            reaction_kind="unknown",
            termination="unresolved; unresolved",
            points=points,
        )
    )
    result = bm.verify_phase_diagram_field_perimeters(source, models)
    certificate = next(
        c
        for c in result.to_dict()["field_verifications"]
        if c["boundary_ids"] == [292, 360, 368]
    )
    assert not certificate["success"], certificate
    assert certificate["checked_crossings"] > 0
    assert "crossing" in certificate["message"]


@pytest.mark.parametrize("change", ["move_point", "add_line"])
def test_graph_changes_invalidate_saved_certificates(source, models, change):
    saved = bm.verify_phase_diagram_field_perimeters(source, models).to_dict()
    assert all(p.has_verified_reaction_perimeter for p in unresolved_fields(saved))
    # A changed graph must be checked again even if it retains the old proof.
    if change == "move_point":
        saved["boundaries"][0]["points"][1]["temperature"] += 1.0e-5
    else:
        saved["boundaries"].append(
            dict(
                id=400,
                points=[
                    dict(pressure=0.0, temperature=2200.0),
                    dict(pressure=0.0, temperature=2250.0),
                ],
            )
        )
    assert all(
        not p.has_verified_reaction_perimeter
        and p.has_closed_reaction_ring
        and not p.is_provisional
        for p in unresolved_fields(saved)
    )


def test_complete_metasediment_ring_identifies_a_field_without_an_interior():
    source = json.loads(
        (
            Path(__file__).parent / "data/metasediment_thin_reaction_ring.json"
        ).read_text()
    )
    (field,) = unresolved_fields(source)
    assert field.phases == [0, 3, 21, 33, 36, 39, 45, 60, 72, 75]
    assert field.has_closed_reaction_ring
    assert not field.has_verified_reaction_perimeter
    assert not field.is_provisional
    assert not field.has_open_boundary
    assert any(
        "complete reaction ring" in note
        for note in bm.phase_diagram_field_polygons(source).coordinate_diagnostics
    )


@pytest.mark.parametrize(
    "change", ["unknown_reaction", "wrong_vertex", "wrong_assemblage"]
)
def test_incomplete_or_incompatible_metasediment_ring_stays_provisional(change):
    source = json.loads(
        (
            Path(__file__).parent / "data/metasediment_thin_reaction_ring.json"
        ).read_text()
    )
    line = source["boundaries"][0]
    if change == "unknown_reaction":
        line["reaction_kind"] = "unknown"
    elif change == "wrong_vertex":
        line["end_node"] = -1
    else:
        line["assemblage"] = [0, 3]
    assert not any(p.has_closed_reaction_ring for p in unresolved_fields(source))
    assert all(p.is_provisional for p in unresolved_fields(source))


def test_ring_fallback_requires_other_lines_to_meet_at_common_vertices():
    source = json.loads(
        (
            Path(__file__).parent / "data/metasediment_thin_reaction_ring.json"
        ).read_text()
    )
    failed = source["samples"][0]
    source["boundaries"].append(
        dict(
            id=500,
            reaction_kind="unknown",
            points=[
                dict(
                    pressure=failed["pressure"] + offset,
                    temperature=failed["temperature"],
                )
                for offset in [-1.0e5, 1.0e5]
            ],
        )
    )
    (field,) = unresolved_fields(source)
    assert not field.has_closed_reaction_ring
    assert field.is_provisional


def test_common_vertex_identity_is_required(source, models):
    next(b for b in source["boundaries"] if b["id"] == 292)["end_node"] = 223
    result = bm.verify_phase_diagram_field_perimeters(source, models)
    assert not any(
        c["success"] and c["boundary_ids"] == [292, 360, 368]
        for c in result.to_dict()["field_verifications"]
    )


def test_overlapping_unresolved_line_cannot_be_ignored(source, models):
    line = deepcopy(next(b for b in source["boundaries"] if b["id"] == 368))
    line.update(
        id=400,
        start_node=-1,
        end_node=-1,
        zero_phase=-1,
        assemblage=[],
        side_a=[],
        side_b=[],
        reaction_kind="unknown",
    )
    source["boundaries"].append(line)
    result = bm.verify_phase_diagram_field_perimeters(source, models)
    assert not any(
        c["success"] and {292, 360, 368}.issubset(c["boundary_ids"])
        for c in result.to_dict()["field_verifications"]
    )
    field = next(
        p
        for p in bm.phase_diagram_field_polygons(result).polygons
        if p.phases == [6, 18, 27, 28, 39, 45, 60, 78]
    )
    assert field.is_provisional
    assert not field.has_verified_reaction_perimeter


def test_verified_precision_notes_do_not_block_resolution(tmp_path):
    phase = bm.minerals.HGP_2018_ds633.per()
    original = bm.phase_diagram(
        {"Mg": 1.0, "O": 1.0}, [phase], [1.0e9, 2.0e9], [800.0, 1000.0]
    ).to_dict()
    state = deepcopy(original["samples"][0])
    state.update(pressure=1.5e9, temperature=900.0, is_field_verification=True)
    original["samples"] = [state]
    failed = deepcopy(state)
    failed.update(
        pressure=1.6e9,
        temperature=910.0,
        success=False,
        is_field_verification=True,
        phases=[],
        message="Field width could not be resolved above the equilibrium coordinate accuracy",
    )
    original["samples"].append(failed)
    warning = "Unresolved seed P=1.6e+09 Pa, T=910 K: " + failed["message"]
    original["diagnostics"] = [
        diagnostic(warning, "sample", 1),
        diagnostic(
            " has conflicting perimeter reactions; its label is provisional.",
            "field",
            999,
        ),
        diagnostic(category="missing_assemblage", entity_id=999),
    ]
    original["resolved"] = False
    result = bm.verify_phase_diagram_field_perimeters(original, [phase])
    assert result.resolved
    assert not result.diagnostics
    assert list(result.coordinate_diagnostics) == [warning]
    path = bm.save_phase_diagram(result, tmp_path / "precision.json")
    saved = bm.load_phase_diagram(path)
    assert saved["resolved"]
    assert saved["coordinate_diagnostics"] == [diagnostic(warning, "sample", 1)]
    original["diagnostics"].append(diagnostic("A genuine unrelated equilibrium failed"))
    result = bm.verify_phase_diagram_field_perimeters(original, [phase])
    assert not result.resolved
    assert list(result.diagnostics) == ["A genuine unrelated equilibrium failed"]
