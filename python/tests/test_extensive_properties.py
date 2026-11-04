# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Total thermodynamic properties on an explicit formula-unit mole scale."""

import numpy as np
import pytest

import burnman_cpp as bm
from burnman_cpp.minerals import SLB_2011

TOTALS = (
    "mass",
    "internal_energy",
    "helmholtz",
    "gibbs",
    "enthalpy",
    "volume",
    "entropy",
    "heat_capacity_p",
    "heat_capacity_v",
)


def material(package, kind):
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
    reference = material(pytest.importorskip("burnman"), kind)
    native = material(bm, kind)
    assert native.number_of_moles == 1.0
    native.set_state(4e9, 900.0)
    intensive_names = (
        "density",
        "thermal_expansivity",
        "isothermal_bulk_modulus_reuss",
    )
    intensives = [getattr(native, name) for name in intensive_names]
    native.number_of_moles = 2.5
    assert [getattr(native, name) for name in intensive_names] == pytest.approx(
        intensives
    )
    # Keep the reference at one mole: its Solution alpha/K_T currently depend
    # on number_of_moles, which also contaminates its molar Cv at other amounts.
    for temperature in (900.0, 1500.0):
        reference.set_state(4e9, temperature)
        native.set_state(4e9, temperature)
        for name in TOTALS:
            assert getattr(native, name) == pytest.approx(
                2.5 * getattr(reference, name), rel=1e-9
            )
    expected = 2.5 * reference.evaluate(TOTALS, [4e9, 5e9], [1100.0, 1600.0])
    np.testing.assert_allclose(
        native.evaluate(TOTALS, [4e9, 5e9], [1100.0, 1600.0]), expected, rtol=1e-9
    )
    assert (native.pressure, native.temperature) == (4e9, 1500.0)
    copied = native.copy()
    native.n_moles = 0.0
    assert copied.number_of_moles == 2.5
    for name in TOTALS:
        assert getattr(native, name) == 0.0
        assert getattr(copied, name) == pytest.approx(2.5 * getattr(reference, name))


def test_amount_validation_and_state_independent_mass():
    mineral = SLB_2011.forsterite()
    mass = mineral.mass
    mineral.n_moles = 3.0
    assert not mineral.has_state()
    assert mineral.number_of_moles == 3.0
    assert mineral.mass == pytest.approx(3.0 * mass)
    for invalid in (-1.0, np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            mineral.number_of_moles = invalid
        assert mineral.number_of_moles == 3.0
