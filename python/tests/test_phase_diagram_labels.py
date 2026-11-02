# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Secondary labels follow moving composition branches, rather than copy IDs."""

from copy import deepcopy

import numpy as np
import pytest
import burnman_cpp as bm


def moving_solvus(*, swapped=False, three=False):
    """A two/three-branch strip between fields with opposite surviving roots."""
    count = 3 if three else 2

    def phases(y, branches, boundary=False):
        records = []
        for i, branch in enumerate(branches):
            phase_id = count - 1 - i if swapped and boundary else i
            x = 0.05 + 0.15 * branch + 0.6 * y
            records.append(
                dict(
                    id=phase_id,
                    candidate_index=0,
                    name="hb",
                    amount=1.0,
                    composition=[x, 1.0 - x],
                )
            )
        records.append(
            dict(id=3, candidate_index=1, name="q", amount=1.0, composition=[1.0])
        )
        return records

    lines = []
    for index, x in enumerate((0.3, 0.7)):
        # At the first edge branch 0 survives; at the other, the last branch.
        # For the triple case use replacement evidence for the two dropped roots.
        zero = count - 1 if index == 0 else 0
        if swapped:
            zero = count - 1 - zero
        left, right = (
            ([0, 3], list(range(count)) + [3])
            if index == 0
            else (list(range(count)) + [3], [0, 3])
        )
        points = []
        for y in np.linspace(0, 1, 9):
            records = phases(y, range(count), boundary=True)
            if not three:
                next(p for p in records if p["id"] == zero)["amount"] = 0.0
            points.append(dict(pressure=x, temperature=float(y), phases=records))
        lines.append(
            dict(
                id=index,
                zero_phase=zero,
                assemblage=list(range(count)) + [3],
                side_b=left,
                side_a=right,
                start_node=-1,
                end_node=-1,
                reaction_kind="replacement" if three else "phase_out",
                is_solution_replacement=three,
                points=points,
                termination="frame; frame",
            )
        )
    samples = [
        dict(pressure=x, temperature=0.5, success=True, phases=phases(0.5, roots))
        for x, roots in ((0.15, [0]), (0.5, range(count)), (0.85, [count - 1]))
    ]
    return dict(
        diagram_type="PT",
        pressure_range=[0.0, 1.0],
        temperature_range=[0.0, 1.0],
        phase_names=["hb", "hb #2", "hb #3", "q", "q #2", "q #3"],
        settings=dict(max_phase_instances=3),
        samples=samples,
        boundaries=lines,
        fields=[
            dict(id=0, phases=[0, 3], sample_indices=[0, 2]),
            dict(id=1, phases=list(range(count)) + [3], sample_indices=[1]),
        ],
        nodes=[],
    )


def field_labels(data):
    annotations = data["secondary_phase_labels"]
    geometry = bm.phase_diagram_field_polygons(data)
    fields = {f["field_number"]: f for f in annotations["fields"]}
    return [
        [
            p["label"]
            for p in fields[polygon.field_number]["phases"]
            if p["phase_id"] < 3
        ]
        for polygon in sorted(geometry.polygons, key=lambda p: p.label_position[0])
    ]


@pytest.mark.parametrize("swapped", [False, True])
def test_drifting_solvus_tracks_survivors_despite_copy_renumbering(swapped):
    original = moving_solvus(swapped=swapped)
    snapshot = deepcopy(original)
    labelled = bm.label_phase_diagram_solutions(original)
    assert original == snapshot
    assert field_labels(labelled) == [["hb_1"], ["hb_1", "hb_2"], ["hb_2"]]
    groups = labelled["secondary_phase_labels"]["groups"]
    assert len(groups) == 2
    # The composition ranges overlap: a single fixed cutoff cannot do this.
    assert groups[0]["maximum_composition"][0] > groups[1]["minimum_composition"][0]
    assert not labelled["secondary_phase_labels"]["diagnostics"]


def test_three_coexisting_roots_remain_distinct_across_replacement_boundaries():
    labelled = bm.label_phase_diagram_solutions(moving_solvus(three=True, swapped=True))
    assert field_labels(labelled) == [["hb_1"], ["hb_1", "hb_2", "hb_3"], ["hb_3"]]


def test_inconsistent_continuity_never_merges_coexisting_instances():
    data = moving_solvus()
    corners = [(0.3, 0.3), (0.7, 0.3), (0.7, 0.7), (0.3, 0.7), (0.3, 0.3)]
    source = data["boundaries"][0]["points"][4]["phases"]
    boundaries = []
    for i, (a, b) in enumerate(zip(corners[:-1], corners[1:])):
        phases = deepcopy(source)
        zero = i % 2
        for phase in phases:
            phase["amount"] = 0.0 if phase["id"] == zero else 1.0
        boundaries.append(
            dict(
                id=i,
                zero_phase=zero,
                assemblage=[0, 1, 3],
                side_b=[0, 1, 3],
                side_a=[0, 3],
                start_node=-1,
                end_node=-1,
                reaction_kind="phase_out",
                points=[
                    dict(pressure=p, temperature=t, phases=deepcopy(phases))
                    for p, t in (a, b)
                ],
                termination="closed loop",
            )
        )
    data["boundaries"] = boundaries
    data["samples"] = data["samples"][:2]
    data["fields"][0]["sample_indices"] = [0]
    labelled = bm.label_phase_diagram_solutions(data)
    assert labelled["secondary_phase_labels"]["diagnostics"]
    assert len(labelled["secondary_phase_labels"]["groups"]) == 2
    for field in labelled["secondary_phase_labels"]["fields"]:
        labels = [p["label"] for p in field["phases"]]
        assert len(set(labels)) == len(labels)


def test_signed_original_endmember_coefficients_are_preserved():
    data = moving_solvus()
    states = data["samples"] + [p for b in data["boundaries"] for p in b["points"]]
    for state in states:
        for phase in state["phases"]:
            if phase["candidate_index"] == 0:
                x = 1.4 * phase["composition"][0] - 0.2
                phase["composition"] = [x, 1.0 - x]
    labelled = bm.label_phase_diagram_solutions(data)
    assert field_labels(labelled) == [["hb_1"], ["hb_1", "hb_2"], ["hb_2"]]
    assert labelled["secondary_phase_labels"]["groups"][0]["minimum_composition"][0] < 0


def test_uniform_two_phase_field_can_use_frame_seeds_without_reaction_lines():
    data = moving_solvus()
    data["boundaries"] = []
    sample = data["samples"][1]
    data["samples"] = [
        dict(deepcopy(sample), pressure=p, temperature=t)
        for p, t in [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]
    ]
    data["fields"] = [dict(id=0, phases=[0, 1, 3], sample_indices=[0, 1, 2, 3])]
    labelled = bm.label_phase_diagram_solutions(data)
    assert field_labels(labelled) == [["hb_1", "hb_2"]]


def test_native_non_pt_coordinates_are_used_for_continuity():
    data = moving_solvus()
    data.update(diagram_type="PS", entropy_range=[0.0, 1.0])
    states = data["samples"] + [p for b in data["boundaries"] for p in b["points"]]
    for state in states:
        state["entropy"] = state["temperature"]
        state["temperature"] = 1000.0
    labelled = bm.label_phase_diagram_solutions(data)
    assert field_labels(labelled) == [["hb_1"], ["hb_1", "hb_2"], ["hb_2"]]


def test_boundary_direction_and_input_order_do_not_change_labels():
    original = moving_solvus()
    reordered = deepcopy(original)
    reordered["boundaries"].reverse()
    for boundary in reordered["boundaries"]:
        boundary["points"].reverse()
        boundary["side_a"], boundary["side_b"] = boundary["side_b"], boundary["side_a"]
    assert field_labels(bm.label_phase_diagram_solutions(reordered)) == field_labels(
        bm.label_phase_diagram_solutions(original)
    )


def test_saved_annotations_roundtrip_and_refresh_after_composition_edits(tmp_path):
    labelled = bm.label_phase_diagram_solutions(moving_solvus())
    path = bm.save_phase_diagram(labelled, tmp_path / "labelled.json")
    assert bm.label_phase_diagram_solutions(path) == labelled
    previous_digest = labelled["secondary_phase_labels"]["source_digest"]
    labelled["samples"][0]["phases"][0]["composition"] = [0.4, 0.6]
    refreshed = bm.label_phase_diagram_solutions(labelled)
    assert refreshed["secondary_phase_labels"]["source_digest"] != previous_digest


def test_missing_compositions_raise_instead_of_inventing_branch_identity():
    data = moving_solvus()
    for sample in data["samples"]:
        for phase in sample["phases"]:
            phase.pop("composition")
    with pytest.raises(ValueError, match="saved phase compositions"):
        bm.label_phase_diagram_solutions(data)


def test_secondary_plot_labels_are_optional_and_respect_aliases(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = moving_solvus()
    options = dict(
        pressure_unit="Pa",
        temperature_unit="K",
        phase_aliases={"hb": "amp"},
        colorbar=False,
        label_key_path=tmp_path / "labels.md",
    )
    plain, ax = bm.plot_phase_diagram(data, **options)
    assert any("2amp" in text for text in ax.phase_diagram_field_labels.values())
    assert ax.phase_diagram_secondary_labels is None
    labelled, ax = bm.plot_phase_diagram(data, secondary_labels=True, **options)
    assert any("amp_1 amp_2" in text for text in ax.phase_diagram_field_labels.values())
    assert len(ax.phase_diagram_secondary_labels["groups"]) == 2
    plt.close(plain)
    plt.close(labelled)
    with pytest.raises(TypeError, match="secondary_labels"):
        bm.plot_phase_diagram(data, secondary_labels="yes")
