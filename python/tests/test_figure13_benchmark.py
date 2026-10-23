# ------------------------------------------------------
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
#
# This file is part of BurnMan.
#
# Detailed license information governing the source code
# and contributions can be found in the LICENSE document.
#
# ------------------------------------------------------
"""The Figure 13 bulk and spin ratio retain their physical definitions."""

from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import pytest

import burnman_cpp as bm
from burnman_cpp.minerals import SLB24


@pytest.fixture(scope="module")
def benchmark():
    pytest.importorskip("matplotlib")
    return runpy.run_path(
        str(
            Path(__file__).parents[2] / "benchmarks/stixrude_2024_figure13/benchmark.py"
        )
    )


def test_figure13_bulk_preserves_table1_cations_and_sets_two_percent_ferric(benchmark):
    table = bm.Composition(
        dict(
            SiO2=45.0,
            MgO=38.88,
            FeO=8.05,
            Fe2O3=0.18,
            CaO=3.18,
            Al2O3=4.0,
            Na2O=0.13,
            Cr2O3=0.57,
        ),
        "mass",
    )
    bulk = benchmark["pyrolite_composition"]()
    assert sum(bulk.mass_composition.values()) == pytest.approx(0.1, abs=1e-15)
    assert 2 * bulk.molar_composition["Fe2O3"] / bulk.atomic_composition[
        "Fe"
    ] == pytest.approx(0.02, abs=1e-15)
    for element in table.atomic_composition.keys() - {"O"}:
        assert bulk.atomic_composition[element] / bulk.atomic_composition[
            "Si"
        ] == pytest.approx(
            table.atomic_composition[element] / table.atomic_composition["Si"],
            rel=1e-14,
        )


def test_spin_ratio_weights_fp_copies_and_excludes_ferric_fe_and_absent_fp(benchmark):
    fp = SLB24.ferropericlase()
    names = list(fp.endmember_names)
    hi = names.index(SLB24.wustite().name)
    lo = names.index(SLB24.wustite_low_spin().name)
    magnetite = names.index(SLB24.magnetite().name)

    def phase(amount, high, low, ferric=0.0):
        x = np.zeros(len(names))
        x[hi] = high
        x[lo] = low
        x[magnetite] = ferric
        x[0] = 1 - high - low - ferric
        return SimpleNamespace(candidate_index=0, amount=amount, composition=x)

    def state(phases, success=True):
        return SimpleNamespace(
            phases=phases, success=success, outside_model_domain=False
        )

    states = (
        state([phase(0.2, 0.8, 0.1), phase(0.8, 0.1, 0.2, 0.1)]),
        state([]),
        state([phase(1.0, 0.0, 0.0, 0.1)]),
        state([phase(1.0, 0.0, 1.0)], success=False),
    )
    path = bm.EquilibriumPath(states, (fp.name,), np.zeros((4, 1)), np.zeros((4, 1)))
    fractions = benchmark["ferropericlase_low_spin_fraction"](path, fp)
    assert fractions[0] == pytest.approx(
        (0.2 * 0.1 + 0.8 * 0.2) / (0.2 * (0.8 + 0.1) + 0.8 * (0.1 + 0.2))
    )
    assert np.isnan(fractions[1:]).all()
