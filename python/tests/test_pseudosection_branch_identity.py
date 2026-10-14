"""A failed removal of one solution copy must not suppress another branch."""

import json
from pathlib import Path
import runpy

import pytest
import burnman_cpp as bm


def transform(data, mode):
    if mode == "reverse":
        for line in data["boundaries"]:
            line["points"].reverse()
            line["start_node"], line["end_node"] = line["end_node"], line["start_node"]
            line["side_a"], line["side_b"] = line["side_b"], line["side_a"]
            line["termination"] = "; ".join(reversed(line["termination"].split("; ")))
    if mode == "renumber":
        copies = data["settings"]["max_phase_instances"]

        def number(i):
            return i // copies * copies + copies - 1 - i % copies

        for node in data["nodes"]:
            for key in ["assemblage", "zero_phases"]:
                node[key] = sorted(map(number, node[key]))
        for line in data["boundaries"]:
            for key in ["assemblage", "side_a", "side_b"]:
                line[key] = sorted(map(number, line[key]))
            line["zero_phase"] = number(line["zero_phase"])
            for point in line["points"]:
                for phase in point["phases"]:
                    phase["id"] = number(phase["id"])
                    phase["name"] = data["phase_names"][phase["id"]]
                point["phases"].sort(key=lambda phase: phase["id"])


@pytest.mark.parametrize("mode", ["original", "reverse", "renumber"])
def test_missing_olivine_wadsleyite_branch_is_recovered(mode):
    data = json.loads(
        (
            Path(__file__).parent / "data" / "pyrolite_missing_solution_branch.json"
        ).read_text()
    )
    transform(data, mode)
    original = json.dumps(data)
    root = Path(__file__).parents[2]
    models = runpy.run_path(str(root / "examples/example_pyrolite_pseudosection.py"))
    result = bm.refine_pseudosection(
        data["composition_start"], models["candidate_phases"](), data, resolution=(2, 2)
    )
    assert json.dumps(data) == original
    assert (
        result.settings.composition_tolerance
        == data["settings"]["composition_tolerance"]
    )
    assert result.settings.affinity_tolerance == data["settings"]["affinity_tolerance"]

    olivine = (0, 12, 15, 18, 19, 39, 72)
    wadsleyite = (3, 12, 15, 18, 19, 39, 72)
    reaction = {olivine, wadsleyite}
    recovered = [
        line
        for line in result.boundaries
        if {tuple(line.side_a), tuple(line.side_b)} == reaction
    ]
    assert recovered, result.diagnostics
    assert any(
        line.start_node >= 0
        and line.end_node >= 0
        and line.termination == "junction; junction"
        for line in recovered
    )
    for line in recovered:
        for point in line.points:
            assert point.mass_balance_error <= result.settings.mass_balance_tolerance
            assert point.minimum_affinity >= -result.settings.affinity_tolerance

    # Both true fields have seven phases and the same colour. They must still
    # be separate polygons with their own assemblages and verified perimeters.
    for representation in [result, json.loads(json.dumps(result.to_dict()))]:
        geometry = bm.pseudosection_field_polygons(representation)
        for pressure, expected in [(9.52e9, olivine), (9.61e9, wadsleyite)]:
            fields = [
                p
                for p in geometry.polygons
                if p.contains_rectangle([pressure, 150.0], [1.0, 1.0e-8])
            ]
            assert len(fields) == 1
            assert tuple(fields[0].phases) == expected
            assert not fields[0].is_provisional
            assert not fields[0].has_open_boundary
