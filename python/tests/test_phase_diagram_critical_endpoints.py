# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""A phase-out solvus can terminate at a genuine intrinsic critical point."""

import json
from pathlib import Path

import pytest
import burnman_cpp as bm
from burnman_cpp.minerals import load_dataset, model_sets


def thermodynamic_section(data, phases, diagram):
    """Express the saved physical equilibria in another thermodynamic section."""

    def coordinates(point, states):
        entropy, volume = 0.0, 0.0
        for state in states:
            phase = phases[state["candidate_index"]].copy()
            if isinstance(phase, bm.Solution):
                phase.set_composition(state["composition"])
            phase.set_state(point["pressure"], point["temperature"])
            entropy += state["amount"] * phase.molar_entropy
            volume += state["amount"] * phase.molar_volume
        point.update(entropy=entropy, volume=volume)

    for line in data["boundaries"]:
        for point in line["points"]:
            coordinates(point, point["phases"])
    for node in data["nodes"]:
        line = next(
            line
            for line in data["boundaries"]
            if node["id"] in (line["start_node"], line["end_node"])
        )
        point = line["points"][0 if line["start_node"] == node["id"] else -1]
        coordinates(node, point["phases"])
    data.update(
        diagram_type=diagram, entropy_range=[0.0, 1000.0], volume_range=[1.0e-6, 1.0e-3]
    )
    return data


@pytest.mark.parametrize("diagram", ["PT", "PS", "PV", "TS", "TV", "SV"])
def test_phase_out_critical_endpoint_is_certified(diagram):
    source = Path(__file__).parent / "data/slb2024_phaseout_critical.json"
    data = json.loads(source.read_text())
    phases = model_sets.slb2024().phases
    thermodynamic_section(data, phases, diagram)
    assert data["boundaries"][0]["reaction_kind"] == "phase_out"
    assert data["nodes"][0]["gibbs_variance"] == 2

    result = bm.verify_phase_diagram_critical_endpoints(data, phases)
    assert result.nodes[0].is_verified_critical_endpoint
    assert not any(
        "Unpaired solution critical endpoint" in s for s in result.diagnostics
    )

    # A physical critical point does not certify an inconsistent S/V location
    # on the saved diagram. Keep this independent of the critical-mode check.
    if diagram != "PT":
        coordinate = "entropy" if "S" in diagram else "volume"
        original = data["nodes"][0][coordinate]
        data["nodes"][0][coordinate] += 0.01 * original
        shifted = bm.verify_phase_diagram_critical_endpoints(data, phases)
        assert not shifted.nodes[0].is_verified_critical_endpoint
        data["nodes"][0][coordinate] = original

    data["nodes"][0]["critical_mode"] = [0.0] * len(data["nodes"][0]["critical_mode"])
    invalid = bm.verify_phase_diagram_critical_endpoints(data, phases)
    assert not invalid.nodes[0].is_verified_critical_endpoint


@pytest.mark.parametrize("diagram", ["PT", "PS", "PV", "TS", "TV", "SV"])
@pytest.mark.parametrize("reverse_boundaries", [False, True])
def test_two_phase_out_arms_continue_through_a_solvus_coalescence(
    reverse_boundaries, diagram
):
    source = Path(__file__).parent / "data/taylor_solvus_continuation.json"
    data = json.loads(source.read_text())
    if reverse_boundaries:
        data["boundaries"].reverse()
    phases = model_sets.slb2024().phases
    thermodynamic_section(data, phases, diagram)
    checked = bm.verify_phase_diagram_critical_endpoints(data, phases)
    assert checked.nodes[0].kind == "boundary_continuation"
    assert not checked.nodes[0].is_verified_critical_endpoint
    assert len(checked.boundaries) == 2
    assert all(
        "boundary continuation" in line.termination for line in checked.boundaries
    )


def test_basalt_ps_critical_endpoints_with_higher_gibbs_variance():
    # Cropped from the 81x81 basalt discovery grid at 0-40 GPa. Gibbs phase
    # variance counts chemical degrees of freedom, not the number of plotted
    # directions. These ordinary critical endpoints have variance five/three.
    source = Path(__file__).parent / "data/basalt_ps_critical_endpoints.json"
    data = json.loads(source.read_text())
    phases, _ = load_dataset("SLB_2024", data["models"]["phase_names"])
    assert [n["gibbs_variance"] for n in data["nodes"]] == [5, 3]

    result = bm.verify_phase_diagram_critical_endpoints(data, phases)
    assert all(n.is_verified_critical_endpoint for n in result.nodes)
    assert not any(
        "Unpaired solution critical endpoint" in d for d in result.diagnostics
    )
    assert (
        result.to_dict()["boundaries"]
        == bm.PhaseDiagramResult.from_dict(data).to_dict()["boundaries"]
    )
