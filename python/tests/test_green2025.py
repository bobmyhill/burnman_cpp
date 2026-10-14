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

"""Independent assemblage landmarks in Green et al. (2025), Supplement 3.

The reference is the published MAGEMin figures, not Python BurnMan output.
The six examples retain the oxide amounts and manually read field labels.
"""

from collections import Counter
import importlib
from pathlib import Path

import pytest
import burnman_cpp as bm

NAMES = ("klb1", "re46", "at1", "mix1g", "ton101u", "ton101s")


@pytest.mark.parametrize(
    "pressure, temperature, lines, status, match",
    [
        (15.0, 1200.0, True, "match", True),
        (15.0, 1210.0, True, "mismatch", False),
        (15.0, 1200.0, False, "missing_intersection", False),
        (25.0, 1200.0, True, "not_applicable", None),
        (15.0, 1800.0, True, "not_applicable", None),
    ],
)
def test_liquidus_comparison_distinguishes_missing_and_out_of_window_references(
    monkeypatch, pressure, temperature, lines, status, match
):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "examples"))
    helper = importlib.import_module("green2025")
    data = dict(
        pressure_range=[1.0e9, 2.0e9],
        temperature_range=[1000.0, 2000.0],
        phase_names=["melt", "solid"],
        boundaries=(
            [
                dict(
                    side_a=[0],
                    side_b=[0, 1],
                    points=[
                        dict(pressure=1.0e9, temperature=1473.15),
                        dict(pressure=2.0e9, temperature=1473.15),
                    ],
                )
            ]
            if lines
            else []
        ),
    )
    result = helper.compare_liquidus(data, pressure, temperature)
    assert result["status"] == status
    assert result["match"] is match
    if status in {"not_applicable", "missing_intersection"}:
        assert result["calculated_temperature_C"] is None
        assert result["difference_C"] is None
        assert result["reason"]


@pytest.fixture(scope="module", params=NAMES)
def benchmark(request):
    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(Path(__file__).parents[2] / "examples"))
        example = importlib.import_module(f"example_green2025_{request.param}")
        runner = importlib.import_module("green2025")
    composition = bm.Composition(example.OXIDES, unit_type="molar")
    return example, composition, runner.candidate_phases(composition)


def test_published_interior_assemblages(benchmark):
    example, composition, phases = benchmark
    for temperature_c, pressure_kbar, expected in example.REFERENCE_POINTS:
        state = bm.stable_equilibrium(
            composition.atomic_composition,
            phases,
            pressure_kbar * 1.0e8,
            temperature_c + 273.15,
        )
        assert state.success, state.message
        assert state.mass_balance_error < 1.0e-8
        assert state.minimum_affinity >= -0.2
        actual = Counter(p.name.split(" #")[0] for p in state.phases)
        assert actual == Counter(expected.split()), (temperature_c, pressure_kbar)
