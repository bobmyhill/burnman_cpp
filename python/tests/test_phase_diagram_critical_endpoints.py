# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""A phase-out solvus can terminate at a genuine intrinsic critical point."""

import json
from pathlib import Path

import pytest
import burnman_cpp as bm
from burnman_cpp.minerals import model_sets


def test_phase_out_critical_endpoint_is_certified():
    source = Path(__file__).parent / "data/slb2024_phaseout_critical.json"
    data = json.loads(source.read_text())
    phases = model_sets.slb2024().phases
    assert data["boundaries"][0]["reaction_kind"] == "phase_out"
    assert data["nodes"][0]["gibbs_variance"] == 2

    result = bm.verify_phase_diagram_critical_endpoints(data, phases)
    assert result.nodes[0].is_verified_critical_endpoint
    assert not any(
        "Unpaired solution critical endpoint" in s for s in result.diagnostics
    )

    data["nodes"][0]["critical_mode"] = [0.0] * len(data["nodes"][0]["critical_mode"])
    invalid = bm.verify_phase_diagram_critical_endpoints(data, phases)
    assert not invalid.nodes[0].is_verified_critical_endpoint


@pytest.mark.parametrize("reverse_boundaries", [False, True])
def test_two_phase_out_arms_continue_through_a_solvus_coalescence(reverse_boundaries):
    source = Path(__file__).parent / "data/taylor_solvus_continuation.json"
    data = json.loads(source.read_text())
    if reverse_boundaries:
        data["boundaries"].reverse()
    phases = model_sets.slb2024().phases
    checked = bm.verify_phase_diagram_critical_endpoints(data, phases)
    assert checked.nodes[0].kind == "boundary_continuation"
    assert not checked.nodes[0].is_verified_critical_endpoint
    assert len(checked.boundaries) == 2
    assert all(
        "boundary continuation" in line.termination for line in checked.boundaries
    )
