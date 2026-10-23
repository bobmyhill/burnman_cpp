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
"""Figure 14 bulk inventories and the prescribed basal thermal conditions."""

from pathlib import Path
import runpy

import numpy as np
import pytest
from scipy.special import erfc

import burnman_cpp as bm

BENCHMARKS = Path(__file__).parents[2] / "benchmarks"


def test_native_prem_keeps_the_mantle_cmb_and_pressure_height_coordinates():
    pytest.importorskip("matplotlib")
    source = runpy.run_path(str(BENCHMARKS / "stixrude_2024_figure14/benchmark.py"))
    prem, cmb_depth = source["PREM"], source["CMB_DEPTH"]
    assert isinstance(prem, bm.seismic.PREM)
    assert cmb_depth == 2891e3
    assert prem.pressure(cmb_depth) == pytest.approx(135.751e9, abs=1e-3)
    height = np.linspace(0.0, 600.0, 601)
    pressure = prem.pressure(cmb_depth - height * 1000)
    np.testing.assert_allclose(
        source["height_at_pressure"](pressure), height, rtol=0.0, atol=1e-9
    )
    assert source["height_at_pressure"](135.751e9) == 0.0
    assert source["height_at_pressure"](102.10632e9) == pytest.approx(600.0)


@pytest.mark.parametrize("lithology", ["pyrolite", "harzburgite", "basalt"])
def test_table1_bulks_conserve_cation_ratios_and_initial_ferric_fraction(lithology):
    source = runpy.run_path(str(BENCHMARKS / "stixrude_2024.py"))
    table = bm.Composition(source["TABLE1"][lithology], "mass")
    bulk = source["bulk_composition"](lithology)
    assert sum(bulk.mass_composition.values()) == pytest.approx(0.1, abs=1.0e-15)
    assert 2 * bulk.molar_composition["Fe2O3"] / bulk.atomic_composition[
        "Fe"
    ] == pytest.approx(source["FERRIC_FRACTIONS"][lithology], abs=1.0e-15)
    for element in table.atomic_composition.keys() - {"O"}:
        assert bulk.atomic_composition[element] / bulk.atomic_composition[
            "Si"
        ] == pytest.approx(
            table.atomic_composition[element] / table.atomic_composition["Si"],
            rel=1.0e-14,
        )


@pytest.mark.parametrize("basal_isentrope_temperature", [2100.0, 2500.0])
def test_erfc_layer_uses_calculated_basal_isentrope_and_requested_cmb_temperature(
    basal_isentrope_temperature,
):
    pytest.importorskip("matplotlib")
    source = runpy.run_path(str(BENCHMARKS / "stixrude_2024_figure14/geotherm.py"))
    height = np.array([0.0, 180.0, 8000.0])
    isentrope = np.array([basal_isentrope_temperature, 2200.0, 2000.0])
    temperature = source["boundary_layer_temperature"](
        height, isentrope, basal_isentrope_temperature, basal_temperature=4100.0
    )
    assert temperature[0] == 4100.0
    assert temperature[1] - isentrope[1] == pytest.approx(
        (4100.0 - basal_isentrope_temperature) * erfc(1.0)
    )
    assert temperature[-1] == isentrope[-1]
