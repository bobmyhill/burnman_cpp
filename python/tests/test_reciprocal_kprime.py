# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

"""RKprime factory, reference data and adapter support."""

import pytest

import burnman_cpp as bm
from burnman_cpp.adapters import from_burnman


def rkprime_params():
    return dict(
        equation_of_state="rkprime",
        V_0=11.24e-6,
        K_0=161e9,
        Kprime_0=4.21796,
        Kprime_inf=2.1,
        G_0=80e9,
        Gprime_inf=1.2,
        molar_mass=0.0403,
    )


@pytest.mark.parametrize("method", ["rkprime", "RKprime", bm.EOSType.RKprime])
def test_factory_and_mineral_properties(method):
    native = bm.Mineral(dict(rkprime_params(), equation_of_state=method))
    assert native.params.equation_of_state == bm.EOSType.RKprime
    assert native.params.F_0 == native.params.P_0 == 0.0
    assert native.params.E_0 is None
    # xi=.1, with 60-digit independent quadrature for the Helmholtz energy.
    pressure, volume, bulk, energy = (
        25849207070.478264,
        0.8828480152934618 * native.params.V_0,
        258492070704.78262,
        15336.122850211423,
    )
    for t in (0.0, 300.0, 2000.0):
        native.set_state(pressure, t)
        assert native.molar_volume == pytest.approx(volume, rel=1e-13)
        assert native.isothermal_bulk_modulus_reuss == pytest.approx(bulk, rel=1e-13)
        assert native.isentropic_bulk_modulus_reuss == pytest.approx(bulk, rel=1e-13)
        assert native.molar_helmholtz == pytest.approx(energy, rel=1e-11)
        assert native.molar_internal_energy == native.molar_helmholtz
        assert native.molar_gibbs == pytest.approx(
            energy + pressure * volume, rel=1e-12
        )
        assert native.molar_enthalpy == native.molar_gibbs
        assert native.molar_heat_capacity_p == native.molar_heat_capacity_v == 1e-99
        assert (
            native.molar_entropy
            == native.thermal_expansivity
            == native.grueneisen_parameter
            == 0.0
        )
        expected_shear = (
            native.params.G_0 / native.params.K_0 * bulk
            - (
                native.params.G_0 / native.params.K_0 * native.params.Kprime_inf
                - native.params.Gprime_inf
            )
            * pressure
        )
        assert native.shear_modulus == pytest.approx(expected_shear, rel=1e-13)
    model = bm.make_eos(method)
    assert model.compute_pressure(
        300.0, native.molar_volume, native.params
    ) == pytest.approx(pressure, rel=1e-13)


@pytest.mark.parametrize("field", ["V_0", "K_0", "Kprime_0", "Kprime_inf"])
def test_required_parameters(field):
    p = rkprime_params()
    del p[field]
    with pytest.raises(ValueError, match=field):
        bm.Mineral(p)


def test_reference_pressure_and_helmholtz_convention():
    native = bm.Mineral(dict(rkprime_params(), P_0=2.5e9, F_0=-125000.0))
    native.set_state(native.params.P_0, 300.0)
    assert native.shear_modulus == pytest.approx(native.params.G_0, rel=1e-15)
    assert native.molar_helmholtz == native.params.F_0
    assert (
        native.molar_gibbs == native.params.F_0 + native.params.P_0 * native.params.V_0
    )
    with pytest.raises(ValueError, match="F_0.*E_0"):
        bm.Mineral(dict(rkprime_params(), E_0=0.0))


@pytest.mark.parametrize("pressure", [-10e9, 0.0, 1e5, 10e9, 100e9, 1e12])
def test_optional_python_reference_and_adapter(pressure):
    burnman = pytest.importorskip("burnman")
    original = burnman.Mineral(dict(rkprime_params(), F_0=-125000.0))
    original.set_state(pressure, 300.0)
    native = from_burnman(original)
    native.set_state(pressure, 300.0)
    assert native.params.equation_of_state == bm.EOSType.RKprime
    for attribute in (
        "molar_volume",
        "isothermal_bulk_modulus_reuss",
        "shear_modulus",
        "molar_gibbs",
        "molar_helmholtz",
    ):
        assert getattr(native, attribute) == pytest.approx(
            getattr(original, attribute), rel=2e-9, abs=5e-5
        ), attribute
