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


import numpy as np
import pytest

from burnman_cpp.minerals import (
    HP_2011_ds62,
    SLB_2011,
    JH_2015,
    HGP_2018_ds633,
    mp50MnNCKFMASHTO,
)

import burnman_cpp as bm
from conftest import EXTENSIVE_PROPERTIES as TOTALS, assert_material_amount_scaling

FACTORIES = [
    HGP_2018_ds633.iron,
    HGP_2018_ds633.wu,
    HGP_2018_ds633.mt,
    HGP_2018_ds633.hem,
    HP_2011_ds62.sill,
    HP_2011_ds62.andalusite,
    HP_2011_ds62.ky,
    SLB_2011.mg_fe_olivine,
    SLB_2011.mg_fe_wadsleyite,
    SLB_2011.mg_fe_ringwoodite,
    SLB_2011.mg_fe_bridgmanite,
    SLB_2011.post_perovskite,
    SLB_2011.ferropericlase,
    SLB_2011.orthopyroxene,
    SLB_2011.garnet,
    SLB_2011.ca_perovskite,
    SLB_2011.pyrope_grossular,
    SLB_2011.mg_fe_bridgmanite_binary,
    JH_2015.orthopyroxene,
    JH_2015.mg_fe_orthopyroxene,
    mp50MnNCKFMASHTO.mu,
    mp50MnNCKFMASHTO.bi,
    mp50MnNCKFMASHTO.g,
    mp50MnNCKFMASHTO.ilmm,
    mp50MnNCKFMASHTO.st,
]


@pytest.mark.parametrize("factory", FACTORIES, ids=lambda f: f.__name__)
def test_native_factories_independent_and_finite(factory):
    a, b = factory(), factory()
    a.set_state(3.0e9, 1200.0)
    gibbs = a.molar_gibbs
    formula = a.formula
    b.set_state(10.0e9, 1800.0)
    a.reset_cache()
    assert a.molar_gibbs == pytest.approx(gibbs, rel=1.0e-12)
    assert a.formula == formula
    assert np.isfinite([a.molar_gibbs, a.molar_entropy, a.molar_volume]).all()
    assert a.molar_volume > 0.0


def test_reduced_models_span_the_required_elements():
    opx = JH_2015.mg_fe_orthopyroxene()
    garnet = SLB_2011.pyrope_grossular()
    bdg = SLB_2011.mg_fe_bridgmanite_binary()
    assert opx.n_endmembers == 3
    assert garnet.n_endmembers == bdg.n_endmembers == 2
    assert set(opx.elements) == {"Mg", "Fe", "Si", "O"}
    assert set(garnet.elements) == {"Mg", "Ca", "Al", "Si", "O"}
    assert set(bdg.elements) == {"Mg", "Fe", "Si", "O"}
    opx.set_composition([0.55, -0.1, 0.55])
    opx.set_state(1.0e5, 800.0)
    assert np.isfinite(opx.molar_gibbs)
    with pytest.raises(ValueError, match="occupancies"):
        opx.set_composition([1.1, -1.2, 1.1])


def _reference_material(package, kind):
    dataset = package.minerals.SLB_2011
    if kind == "mineral":
        return dataset.forsterite()
    if kind == "solution":
        phase = dataset.mg_fe_olivine()
        phase.set_composition([0.3, 0.7])
        return phase
    composite = bm.Assemblage if package is bm else package.Composite
    return composite([dataset.forsterite(), dataset.fayalite()], [0.3, 0.7])


@pytest.mark.parametrize("kind", ["mineral", "solution", "composite"])
def test_total_properties_match_python_burnman_and_survive_evaluation(kind):
    reference = _reference_material(pytest.importorskip("burnman"), kind)
    native = _reference_material(bm, kind)
    assert native.number_of_moles == 1.0
    native.set_state(4e9, 900.0)
    intensive_names = (
        "density",
        "thermal_expansivity",
        "isothermal_bulk_modulus_reuss",
    )
    intensives = [getattr(native, name) for name in intensive_names]
    reference.number_of_moles = native.number_of_moles = 2.5
    assert [getattr(native, name) for name in intensive_names] == pytest.approx(
        intensives
    )
    for temperature in (900.0, 1500.0):
        reference.set_state(4e9, temperature)
        native.set_state(4e9, temperature)
        for name in TOTALS:
            assert getattr(native, name) == pytest.approx(
                getattr(reference, name), rel=1e-9
            )
    expected = reference.evaluate(TOTALS, [4e9, 5e9], [1100.0, 1600.0])
    np.testing.assert_allclose(
        native.evaluate(TOTALS, [4e9, 5e9], [1100.0, 1600.0]), expected, rtol=1e-9
    )
    assert (native.pressure, native.temperature) == (4e9, 1500.0)
    copied = native.copy()
    native.n_moles = 0.0
    assert copied.number_of_moles == 2.5
    for name in TOTALS:
        assert getattr(native, name) == 0.0
        assert getattr(copied, name) == pytest.approx(getattr(reference, name))


@pytest.mark.parametrize("kind", ["mineral", "combined", "solution"])
def test_native_material_intensive_and_extensive_properties(kind):
    if kind == "combined":
        phase = bm.CombinedMineral(
            [SLB_2011.enstatite(), SLB_2011.periclase()], [0.5, 1.0]
        )
    else:
        phase = SLB_2011.forsterite() if kind == "mineral" else SLB_2011.mg_fe_olivine()
    phase.set_state(20e9, 1000.0)
    assert_material_amount_scaling(phase)
