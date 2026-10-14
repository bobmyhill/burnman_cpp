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

"""Local calculation, seam topology and reusable saved patch results."""

from copy import deepcopy
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest

import burnman_cpp as bm
from conftest import oxide_params
from test_phase_diagram import crossing_phases, settings


def diagram(points, low=0.0, high=1.0, *, closed=False):
    points = [dict(pressure=p, temperature=t, residual=0.0) for p, t in points]
    return dict(
        diagram_type="PT",
        pressure_range=[low, high],
        temperature_range=[low, high],
        composition_start={"Mg": 1.0, "O": 1.0},
        composition_end={},
        phase_names=["A", "A #2", "A #3", "B", "B #2", "B #3"],
        nodes=(
            []
            if closed
            else [
                dict(p, id=i, kind="domain_edge", assemblage=[0, 3], zero_phases=[0])
                for i, p in enumerate((points[0], points[-1]))
            ]
        ),
        boundaries=[
            dict(
                id=0,
                points=points,
                start_node=-1 if closed else 0,
                end_node=-1 if closed else 1,
                zero_phase=0,
                assemblage=[0, 3],
                side_a=[0],
                side_b=[3],
                termination=(
                    "closed loop; closed loop" if closed else "domain edge; domain edge"
                ),
            )
        ],
        samples=[],
        fields=[],
        diagnostics=[],
        resolved=True,
        equilibrium_solves=10,
        minimization_calls=20,
        metadata="retained",
    )


def reverse(data):
    for b in data["boundaries"]:
        b["points"].reverse()
        b["start_node"], b["end_node"] = b["end_node"], b["start_node"]
        b["side_a"], b["side_b"] = b["side_b"], b["side_a"]


def assert_indices(data):
    assert [n["id"] for n in data["nodes"]] == list(range(len(data["nodes"])))
    assert [b["id"] for b in data["boundaries"]] == list(range(len(data["boundaries"])))
    for n in data["nodes"]:
        assert n["incident_lines"] == [
            b["id"]
            for b in data["boundaries"]
            if n["id"] in (b["start_node"], b["end_node"])
        ]
    assert all(
        0 <= i < len(data["samples"])
        for f in data["fields"]
        for i in f["sample_indices"]
    )


@pytest.mark.parametrize("reverse_base", [False, True])
@pytest.mark.parametrize("reverse_patch", [False, True])
def test_coarse_chord_crossing_box_retains_orientation_and_inputs(
    reverse_base, reverse_patch
):
    base = diagram([(0.5, 0), (0.5, 1)])  # Neither saved vertex is inside.
    patch = diagram([(0.5, 0.25), (0.5, 0.5), (0.5, 0.75)], 0.25, 0.75)
    if reverse_base:
        reverse(base)
    if reverse_patch:
        reverse(patch)
    originals = deepcopy((base, patch))
    result, report = bm.merge_phase_diagrams(base, patch)
    assert (base, patch) == originals
    assert result["resolved"], result["diagnostics"]
    assert "Joined 2 line ends" in report[0]
    assert len(result["boundaries"]) == 1
    assert len(result["boundaries"][0]["points"]) == 5
    geometry = bm.phase_diagram_field_polygons(result)
    assert {tuple(p.phases) for p in geometry.polygons} == {(0,), (3,)}
    assert sorted(p.area for p in geometry.polygons) == pytest.approx([0.5, 0.5])
    assert_indices(result)


def test_closed_loop_can_join_back_to_itself():
    base = diagram(
        [(0.8, 0.2), (0.8, 0.8), (0.2, 0.8), (0.2, 0.2), (0.8, 0.2)], closed=True
    )
    patch = diagram([(0.4, 0.2), (0.8, 0.2), (0.8, 0.8), (0.4, 0.8)], 0.1, 0.9)
    patch["pressure_range"] = [0.4, 0.9]
    result, report = bm.merge_phase_diagrams(base, patch)
    assert "Joined 2" in report[0]
    assert result["resolved"], result["diagnostics"]
    assert not result["nodes"]
    assert len(result["boundaries"]) == 1
    line = result["boundaries"][0]
    assert line["points"][0] == line["points"][-1]
    assert line["termination"] == "closed loop; closed loop"
    assert sorted(
        p.area for p in bm.phase_diagram_field_polygons(result).polygons
    ) == pytest.approx([0.36, 0.64])


def test_unmatched_ports_and_open_native_endpoints_stay_unresolved():
    base = diagram([(0.5, 0), (0.5, 1)])
    patch = diagram([(0.5, 0.25), (0.5, 0.5)], 0.25, 0.75)
    patch["boundaries"][0].update(
        end_node=-1, termination="domain edge; corrector failed"
    )
    result, _ = bm.merge_phase_diagrams(base, patch)
    assert not result["resolved"]
    assert any("unresolved patch seam" in d for d in result["diagnostics"])
    assert any("corrector failed" in d for d in result["diagnostics"])
    assert_indices(result)
    patch = diagram([(0.6, 0.25), (0.6, 0.75)], 0.25, 0.75)
    result, _ = bm.merge_phase_diagrams(base, patch, join_tolerance=0.01)
    assert not result["resolved"]
    assert any(n["kind"] == "patch_seam" for n in result["nodes"])


def test_reaction_identity_does_not_join_different_assemblages():
    base = diagram([(0.5, 0), (0.5, 1)])
    patch = diagram([(0.5, 0.25), (0.5, 0.75)], 0.25, 0.75)
    patch["boundaries"][0]["assemblage"] = [0, 3, 4]
    result, report = bm.merge_phase_diagrams(base, patch)
    assert "Joined 0" in report[0]
    assert not result["resolved"]


def test_samples_and_excluded_domain_are_replaced_and_overlap_last_wins(tmp_path):
    base = diagram([(0.5, 0), (0.5, 1)])
    base["samples"] = [
        dict(pressure=p, temperature=0.5, success=True, phases=[dict(id=3, amount=1)])
        for p in (0.1, 0.4)
    ]
    base["fields"] = [dict(id=0, phases=[3], sample_indices=[0, 1])]
    base["excluded_regions"] = [[[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]]]
    patch = diagram([(0.5, 0.25), (0.5, 0.75)], 0.25, 0.75)
    patch["samples"] = [
        dict(
            pressure=0.4,
            temperature=0.5,
            success=True,
            phases=[dict(id=3, amount=1)],
            marker="new",
        )
    ]
    patch["fields"] = [dict(id=0, phases=[3], sample_indices=[0])]
    first, _ = bm.merge_phase_diagrams(base, patch)
    assert len(first["samples"]) == 2
    assert first["samples"][1]["marker"] == "new"
    assert not first["excluded_regions"]
    patch["samples"][0]["marker"] = "newest"
    path = bm.save_phase_diagram(first, tmp_path / "first.json")
    result, _ = bm.merge_phase_diagrams(path, patch)
    assert len(result["samples"]) == 2
    assert result["samples"][1]["marker"] == "newest"
    assert result["metadata"] == "retained"
    assert result["equilibrium_solves"] == 30
    assert_indices(result)


@pytest.mark.parametrize(
    "key,value",
    [
        ("composition_start", {"Fe": 1}),
        ("phase_names", ["different"]),
        ("diagram_type", "TX"),
        ("fixed_value", 1.0),
    ],
)
def test_incompatible_inputs_are_rejected(key, value):
    base = diagram([(0.5, 0), (0.5, 1)])
    patch = diagram([(0.5, 0.25), (0.5, 0.75)], 0.25, 0.75)
    patch[key] = value
    with pytest.raises(ValueError, match="Incompatible"):
        bm.merge_phase_diagrams(base, patch)


def test_fresh_patch_and_saved_merge_can_resume_native_checks(tmp_path):
    phases = crossing_phases()
    opts = settings()
    base = bm.phase_diagram(
        {"Mg": 1, "Fe": 1, "O": 2}, phases, (0, 2.0e9), (600, 1400), opts
    )
    assert base.resolved, base.diagnostics
    patch = bm.calculate_phase_diagram_patch(
        base,
        phases,
        ((800, 1200), (0.5, 1.5)),
        opts,
        pressure_unit="GPa",
        temperature_unit="K",
        resolution=(11, 13),
    )
    assert patch.resolved, patch.diagnostics
    merged, _ = bm.merge_phase_diagrams(base, patch, phases=phases)
    assert merged["resolved"], merged["diagnostics"]
    assert len(bm.phase_diagram_field_polygons(merged).polygons) == 4
    assert_indices(merged)
    path = bm.save_phase_diagram(merged, tmp_path / "merged.json")
    opts.max_recovery_passes = 0
    resumed = bm.refine_phase_diagram(merged["composition_start"], phases, path, opts)
    assert resumed.resolved, resumed.diagnostics
    assert len(bm.phase_diagram_field_polygons(resumed).polygons) == 4


@pytest.mark.parametrize(
    "diagram_type", ["PT", "PS", "PV", "TS", "TV", "SV", "PX", "TX", "SX", "VX"]
)
def test_local_calculation_preserves_thermodynamic_axes_and_fixed_coordinate(
    diagram_type,
):
    mineral = bm.Mineral(oxide_params())
    mineral.set_state(4.0e9, 1000)
    ranges = dict(
        P=(3.9e9, 4.1e9),
        T=(999, 1001),
        S=(mineral.molar_entropy - 0.05, mineral.molar_entropy + 0.05),
        V=(mineral.molar_volume - 1.0e-9, mineral.molar_volume + 1.0e-9),
        X=(0, 1),
    )
    names = dict(
        P="pressure", T="temperature", S="entropy", V="volume", X="composition"
    )
    options = {names[a] + "_range": ranges[a] for a in diagram_type}
    if "X" in diagram_type:
        options.update(composition_end={"Mg": 1.001, "O": 1.001})
        options["pressure" if diagram_type[0] == "T" else "temperature"] = (
            4.0e9 if diagram_type[0] == "T" else 1000
        )
    opts = bm.PhaseDiagramSettings()
    opts.max_phase_instances = 1
    for name in names.values():
        setattr(opts, name + "_seeds", 3)
    base = bm.phase_diagram(
        {"Mg": 1, "O": 1}, [mineral], settings=opts, diagram=diagram_type, **options
    )
    display = dict(pressure_unit="Pa", temperature_unit="K", swap_axes=True)
    region = tuple(
        (lo + 0.25 * (hi - lo), hi - 0.25 * (hi - lo))
        for lo, hi in (ranges[a] for a in diagram_type)
    )
    patch = bm.calculate_phase_diagram_patch(base, [mineral], region, **display)
    assert patch.resolved, patch.diagnostics
    np.testing.assert_allclose(patch.coordinate_ranges, region)
    assert patch.fixed_coordinate == base.fixed_coordinate
    assert patch.fixed_value == base.fixed_value
    merged, _ = bm.merge_phase_diagrams(base, patch, phases=[mineral])
    assert merged["resolved"], merged["diagnostics"]
    assert merged["diagram_type"] == diagram_type


def test_field_windows_use_plot_numbers_and_clip_padding():
    base = diagram([(0.5, 0), (0.5, 1)])
    fields = bm.phase_diagram_field_regions(
        base, [1], padding=0.05, temperature_unit="K", pressure_unit="Pa"
    )
    assert list(fields) == [1]
    assert fields[1][0] == pytest.approx((0, 1))
    assert 0 <= fields[1][1][0] < fields[1][1][1] <= 1
    assert bm.phase_diagram_field_regions(base) == {}
    with pytest.raises(ValueError, match="Unknown Field"):
        bm.phase_diagram_field_regions(base, [0])


def test_problem_regions_include_boundary_diagnostics_without_problem_fields(tmp_path):
    base = diagram([(0.5, 0), (0.5, 1)])
    base["diagnostics"] = [
        "Boundary 0: neighbouring fields could not be distinguished within tolerances.",
        "Boundary 0: corrector failed; junction",
        "Boundary 999: stale diagnostic",
        "No closed region resolved for sampled assemblage 4.",
    ]
    original = deepcopy(base)
    assert bm.phase_diagram_field_regions(base) == {}
    path = bm.save_phase_diagram(base, tmp_path / "base.json")
    regions = bm.phase_diagram_problem_regions(
        path, pressure_unit="Pa", temperature_unit="K"
    )
    assert list(regions) == ["Boundary 0"]
    assert regions["Boundary 0"][0] == pytest.approx((0, 1))
    assert regions["Boundary 0"][1] == pytest.approx((0.4875, 0.5125))
    assert base == original


def test_problem_regions_keep_fields_and_boundary_ids_separate_and_include_nodes():
    base = diagram([(0.5, 0.2), (0.5, 0.8)])
    line = base["boundaries"][0]
    line.update(id=1, side_a=[], side_b=[])
    base["nodes"][0].update(pressure=0.6, temperature=0)
    base["nodes"][1]["temperature"] = 1
    base["diagnostics"] = [
        "Boundary 1: neighbouring fields could not be distinguished within tolerances."
    ]
    display = dict(temperature_unit="K", pressure_unit="Pa")
    fields = bm.phase_diagram_field_regions(base, padding=0.1, **display)
    assert fields
    regions = bm.phase_diagram_problem_regions(base, padding=0.1, **display)
    assert {label: r for label, r in regions.items() if label.startswith("Field ")} == {
        f"Field {n}": r for n, r in fields.items()
    }
    assert regions["Boundary 1"][0] == pytest.approx((0, 1))
    assert regions["Boundary 1"][1] == pytest.approx((0.49, 0.61))


@pytest.mark.parametrize("pressure", [0.0, -1.0e-24])
def test_problem_regions_give_frame_boundaries_finite_width(pressure):
    base = diagram([(pressure, 0), (pressure, 1)])
    base["boundaries"][0]["start_node"] = -1
    base["diagnostics"] = ["Boundary 0: corrector failed; domain edge"]
    region = bm.phase_diagram_problem_regions(
        base, padding=0, pressure_unit="Pa", temperature_unit="K"
    )["Boundary 0"]
    assert region[0] == pytest.approx((0, 1))
    assert region[1] == pytest.approx((0, 0.0125))


@pytest.mark.parametrize(
    "diagram_type", ["PT", "PS", "PV", "TS", "TV", "SV", "PX", "TX", "SX", "VX"]
)
def test_problem_regions_follow_all_diagram_axes(diagram_type):
    base = diagram([(0.5, 0), (0.5, 1)])
    base.update(
        diagram_type=diagram_type,
        entropy_range=[0, 1],
        volume_range=[0, 1],
        composition_range=[0, 1],
    )
    names = dict(
        P="pressure",
        T="temperature",
        S="entropy",
        V="volume",
        X="composition_coordinate",
    )
    for records in (base["nodes"], base["boundaries"][0]["points"]):
        for i, record in enumerate(records):
            record.update(entropy=0, volume=0, composition_coordinate=0)
            record[names[diagram_type[0]]] = 0.5
            record[names[diagram_type[1]]] = i
    base["diagnostics"] = [
        "Boundary 0: neighbouring fields could not be distinguished within tolerances."
    ]
    region = bm.phase_diagram_problem_regions(
        base, swap_axes=True, pressure_unit="Pa", temperature_unit="K"
    )["Boundary 0"]
    assert region[0] == pytest.approx((0.4875, 0.5125))
    assert region[1] == pytest.approx((0, 1))


def test_general_example_accepts_python_file_and_existing_patch(tmp_path, monkeypatch):
    example = (
        Path(__file__).resolve().parents[2] / "examples/example_patch_phase_diagram.py"
    )
    module = runpy.run_path(str(example))
    models = tmp_path / "custom_models.py"
    models.write_text("def candidate_phases():\n    return ['loaded']\n")
    assert module["candidate_phases"](str(models)) == ["loaded"]
    base = bm.save_phase_diagram(diagram([(0.5, 0), (0.5, 1)]), tmp_path / "base.json")
    patch = bm.save_phase_diagram(
        diagram([(0.5, 0.25), (0.5, 0.75)], 0.25, 0.75), tmp_path / "patch.json"
    )
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(example),
            "patch",
            str(base),
            str(patch),
            "--output-dir",
            str(output),
            "--no-plots",
        ],
    )
    module["main"]()
    assert bm.load_phase_diagram(output / "final/phase_diagram.json")["resolved"]
    assert "Joined 2" in (output / "merge_report.txt").read_text()


@pytest.mark.parametrize(
    "mode,named",
    [
        ("make-for-fields", True),
        ("make-for-problems", False),
        ("make-and-patch", True),
        ("make-and-patch", False),
    ],
)
def test_example_calculation_modes_save_patches_and_merge_only_when_requested(
    mode, named, tmp_path, monkeypatch
):
    example = (
        Path(__file__).resolve().parents[2] / "examples/example_patch_phase_diagram.py"
    )
    module = runpy.run_path(str(example))
    opts = settings()
    base = bm.phase_diagram(
        {"Mg": 1, "Fe": 1, "O": 2}, crossing_phases(), (0, 2.0e9), (600, 1400), opts
    ).to_dict()
    # Removing one arm of the four-field intersection makes a provisional
    # loop, so automatic selection exercises real polygon diagnostics.
    if not named:
        base["boundaries"].pop()
        assert bm.phase_diagram_field_regions(base)
    source = bm.save_phase_diagram(base, tmp_path / "base.json")
    models = tmp_path / "custom_models.py"
    models.write_text(
        "from test_phase_diagram import crossing_phases as candidate_phases\n"
    )
    output = tmp_path / mode
    argv = [
        str(example),
        mode,
        str(source),
        "--models",
        str(models),
        "--seeds",
        "4",
        "5",
        "--output-dir",
        str(output),
        "--no-plots",
    ]
    if named:
        argv += ["--fields", "1"]
    monkeypatch.setattr(sys, "argv", argv)
    module["main"]()
    saved_patches = list(output.glob("field_*_patch/phase_diagram.json"))
    expected = bm.phase_diagram_field_regions(base, [1] if named else None)
    assert len(saved_patches) == len(expected)
    for path in saved_patches:
        patch = bm.load_phase_diagram(path)
        assert patch["settings"]["temperature_seeds"] == 4
        assert patch["settings"]["pressure_seeds"] == 5
        assert patch["settings"]["step"] == opts.step
    assert (output / "final/phase_diagram.json").exists() == (mode == "make-and-patch")
    assert (output / "merge_report.txt").exists() == (mode == "make-and-patch")
    assert bm.load_phase_diagram(source) == base
