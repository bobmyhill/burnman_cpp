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

"""Default tolerances for material equation-of-state fitting."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

import burnman_cpp as bm
from burnman_cpp.optimize.eos_fitting import default_mle_tolerances


def test_fixed_tolerances_and_standard_state():
    material = bm.minerals.HP_2011_ds62.fo()
    material.set_state(2.0e9, 1500.0)
    flags = [
        "H",
        "energy",
        "molar_internal_energy",
        "helmholtz",
        "molar_helmholtz",
        "gibbs",
        "molar_gibbs",
        "enthalpy",
        "molar_enthalpy",
        "P",
        "pressure",
    ]
    assert bm.eos_fitting.default_mle_tolerances is default_mle_tolerances
    assert_array_equal(default_mle_tolerances(material, flags), [1.0] * 9 + [100.0] * 2)
    assert material.pressure == 1.0e5
    assert material.temperature == 300.0


def test_relative_tolerances_preserve_order_and_duplicates():
    material = bm.minerals.SLB_2011.forsterite()
    flags = ["V", "T", "S", "alpha", "molar_volume", "T", "C_p", "K_T", "rho"]
    actual = default_mle_tolerances(material, flags)
    properties = [
        "molar_volume",
        "temperature",
        "molar_entropy",
        "thermal_expansivity",
        "molar_volume",
        "temperature",
        "molar_heat_capacity_p",
        "isothermal_bulk_modulus_reuss",
        "density",
    ]
    expected = [1.0e-5 * getattr(material, name) for name in properties]
    assert_array_equal(actual, expected)


def test_empty_flags_still_set_standard_state():
    material = bm.minerals.HP_2011_ds62.fo()
    material.set_state(2.0e9, 1500.0)
    actual = default_mle_tolerances(material, [])
    assert actual.shape == (0,)
    assert actual.dtype == np.float64
    assert material.pressure == 1.0e5
    assert material.temperature == 300.0


def test_nonpositive_relative_tolerance_raises():
    # Python's fixed energy flag list does not include the F alias.
    material = bm.minerals.HP_2011_ds62.fo()
    with pytest.raises(ValueError, match="All MLE tolerances should be > 0"):
        default_mle_tolerances(material, ["F"])
    assert material.molar_helmholtz < 0.0


@pytest.mark.parametrize(
    "dataset,name,composition",
    [
        ("HP_2011_ds62", "fo", None),
        ("SLB_2011", "forsterite", None),
        ("SLB_2011", "mg_fe_olivine", [0.8, 0.2]),
    ],
)
def test_optional_python_burnman_parity(dataset, name, composition):
    reference = pytest.importorskip("burnman")
    from burnman.optimize.eos_fitting import default_mle_tolerances as expected

    native_material = getattr(getattr(bm.minerals, dataset), name)()
    python_material = getattr(getattr(reference.minerals, dataset), name)()
    if composition is not None:
        native_material.set_composition(composition)
        python_material.set_composition(composition)
    flags = ["P", "H", "V", "T", "S", "alpha", "C_p", "K_T", "rho", "molar_volume"]
    assert_allclose(
        default_mle_tolerances(native_material, flags),
        expected(python_material, flags),
        rtol=1.0e-8,
        atol=0.0,
    )
