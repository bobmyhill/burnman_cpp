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

"""SLB2024 thermodynamics, redox and EOS-domain regressions."""

import numpy as np
import pytest

import burnman_cpp as bm
from conftest import assert_material_amount_scaling
from burnman_cpp.minerals import SLB_2024

SOLUTIONS = [
    "c2c_pyroxene",
    "calcium_ferrite_structured_phase",
    "clinopyroxene",
    "garnet",
    "ilmenite",
    "ferropericlase",
    "new_aluminous_phase",
    "olivine",
    "orthopyroxene",
    "plagioclase",
    "post_perovskite",
    "bridgmanite",
    "ringwoodite",
    "mg_fe_aluminous_spinel",
    "wadsleyite",
]


@pytest.mark.parametrize(
    "factory",
    [
        SLB_2024.fea,
        SLB_2024.feg,
        SLB_2024.fee,
        SLB_2024.mag,
        SLB_2024.smag,
        SLB_2024.hmag,
    ],
)
def test_zero_temperature_magnetic_and_electronic_limits_are_finite(factory):
    phase = factory()
    phase.set_state(30.0e9, 0.0)
    values = [
        phase.molar_gibbs,
        phase.molar_entropy,
        phase.molar_volume,
        phase.molar_heat_capacity_p,
    ]
    assert np.isfinite(values).all()
    assert phase.molar_heat_capacity_p == pytest.approx(0.0, abs=1.0e-12)
    phase.set_state(30.0e9, 1.0e-5)
    assert phase.molar_gibbs == pytest.approx(values[0], abs=0.01)


def test_spinodal_bracket_recovers_the_stable_root_and_rejects_the_other():
    # A 4000 K forsterite root exists just above its thermal pressure minimum.
    phase = SLB_2024.fo()
    phase.set_state(3.0e9, 4000.0)
    assert phase.isothermal_bulk_modulus_reuss > 0.0
    volume = phase.molar_volume
    phase.set_state(3.01e9, 4000.0)
    assert phase.molar_volume < volume
    phase.set_state(0.0, 4000.0)
    with pytest.raises(ValueError, match="limiting pressure"):
        _ = phase.molar_volume


def test_zero_site_multiplicity_survives_polytope_basis_transformation():
    original = SLB_2024.ferropericlase()
    basis = np.eye(original.n_endmembers)[[0, 4]]
    reduced = bm.transform_solution_to_new_basis(original, basis, [0.75, 0.25])
    original.set_composition([0.75, 0.0, 0.0, 0.0, 0.25])
    for phase in [original, reduced]:
        phase.set_state(30.0e9, 1600.0)
    assert reduced.molar_gibbs == pytest.approx(original.molar_gibbs, abs=1.0e-7)


@pytest.mark.parametrize(
    "name",
    SOLUTIONS + ["fea", "feg", "fee", "st", "qtz", "neph", "mag", "smag", "hmag"],
)
def test_sLB2024_catalogue_matches_python_reference(name):
    reference = pytest.importorskip("burnman").minerals.SLB_2024
    native = getattr(SLB_2024, name)()
    pure = getattr(reference, name)()
    if isinstance(native, bm.Solution):
        fractions = np.arange(1.0, native.n_endmembers + 1)
        fractions /= fractions.sum()
        native.set_composition(fractions)
        pure.set_composition(fractions)
    for pressure, temperature in [(20.0e9, 1200.0), (80.0e9, 3000.0)]:
        native.set_state(pressure, temperature)
        pure.set_state(pressure, temperature)
        properties = [
            "molar_gibbs",
            "molar_entropy",
            "molar_volume",
            "molar_heat_capacity_p",
            "shear_modulus",
        ]
        if isinstance(native, bm.Solution):
            properties += ["partial_gibbs", "gibbs_hessian"]
        for prop in properties:
            np.testing.assert_allclose(
                getattr(native, prop),
                getattr(pure, prop),
                rtol=2.0e-10,
                atol=2.0e-5,
                err_msg=f"{name}, {pressure}, {temperature}: {prop}",
            )


def test_relaxed_solution_intensive_and_extensive_properties():
    phase = SLB_2024.ferropericlase_relaxed()
    phase.set_composition([0.75, 0.2, 0.03, 0.02])
    phase.set_state(20e9, 1000.0)
    assert_material_amount_scaling(phase)
