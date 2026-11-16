# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Weak entries require validated Gibbs improvements, not a global tighter tolerance."""

import json
from pathlib import Path

import pytest
import burnman_cpp as bm
from burnman_cpp.minerals import model_sets


@pytest.mark.parametrize(
    "pressure,temperature,candidate,count,previous_gibbs",
    [
        (22.87539909936502e9, 373.6926625565959, 2, 1, -674583.1201403165),
        (0.8159899944660127e9, 645.5742290605776, 4, 2, -1258926.134823311),
        (18.328623660689857e9, 511.0579941343082, 6, 2, -788939.8922610441),
    ],
)
def test_weak_mineral_or_solution_copy_lowers_gibbs_at_default_tolerance(
    pressure, temperature, candidate, count, previous_gibbs
):
    source = Path(__file__).parent / "data/taylor_solvus_continuation.json"
    bulk = json.loads(source.read_text())["composition_start"]
    settings = bm.PhaseDiagramSettings()
    assert settings.affinity_tolerance == 0.2
    state = bm.stable_equilibrium(
        bulk, model_sets.slb2024().phases, pressure, temperature, settings
    )
    assert state.success, state.message
    total = sum(p.amount for p in state.phases)
    assert (
        sum(
            p.candidate_index == candidate
            and p.amount > settings.amount_tolerance * total
            for p in state.phases
        )
        == count
    )
    assert state.gibbs < previous_gibbs - 1.0e-8
    assert state.mass_balance_error < settings.mass_balance_tolerance
