# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

"""MACAW factory, native mineral properties and Python adapter."""

import pytest

import burnman_cpp as bm
from burnman_cpp.adapters import from_burnman


def macaw_params():
    return dict(
        equation_of_state="macaw",
        V_0=11.24e-6,
        K_0=161e9,
        Kprime_0=4.21796,
        Kprime_inf=3.0,
        molar_mass=0.0403,
    )


@pytest.mark.parametrize("method", ["macaw", "MACAW", bm.EOSType.MACAW])
def test_factory_and_native_mineral_properties(method):
    native = bm.Mineral(dict(macaw_params(), equation_of_state=method))
    assert native.params.equation_of_state == bm.EOSType.MACAW
    assert native.params.F_0 == 0.0
    assert native.params.P_0 == 1e5
    assert native.params.E_0 is None
    # 60-digit independent MACAW values at V/V_0=.8, including default P_0.
    pressure = 58089267503.034828 + native.params.P_0
    energy = 53508.360730554698 + native.params.P_0 * 0.2 * native.params.V_0
    for t in (0.0, 300.0, 2000.0):
        native.set_state(pressure, t)
        assert native.molar_volume == pytest.approx(0.8 * native.params.V_0, rel=1e-13)
        assert native.isothermal_bulk_modulus_reuss == pytest.approx(
            387494292131.85278, rel=1e-13
        )
        assert native.isentropic_bulk_modulus_reuss == pytest.approx(
            native.isothermal_bulk_modulus_reuss, rel=1e-15
        )
        assert native.molar_helmholtz == pytest.approx(energy, rel=1e-12)
        assert native.molar_internal_energy == native.molar_helmholtz
        assert native.molar_gibbs == pytest.approx(
            energy + pressure * native.molar_volume, rel=1e-12
        )
        assert native.molar_enthalpy == native.molar_gibbs
        assert (
            native.molar_entropy
            == native.thermal_expansivity
            == native.grueneisen_parameter
            == 0.0
        )
        assert native.molar_heat_capacity_p == native.molar_heat_capacity_v == 1e-99
        assert native.shear_modulus == 1e99
    assert bm.make_eos(method).compute_pressure(
        300.0, native.molar_volume, native.params
    ) == pytest.approx(pressure, rel=1e-13)


@pytest.mark.parametrize("field", ["V_0", "K_0", "Kprime_0", "Kprime_inf"])
def test_required_parameters(field):
    p = macaw_params()
    del p[field]
    with pytest.raises(ValueError, match=field):
        bm.Mineral(p)


def test_helmholtz_reference_and_pressure_convention():
    native = bm.Mineral(dict(macaw_params(), P_0=2.5e9, F_0=-125000.0))
    native.set_state(native.params.P_0, 300.0)
    assert native.molar_volume == native.params.V_0
    assert native.molar_helmholtz == native.params.F_0
    assert (
        native.molar_gibbs == native.params.F_0 + native.params.P_0 * native.params.V_0
    )
    with pytest.raises(ValueError, match="F_0.*E_0"):
        bm.Mineral(dict(macaw_params(), E_0=0.0))


@pytest.mark.parametrize("ratio", [0.4, 0.8, 0.99, 1.0, 1.2])
def test_optional_python_reference_and_adapter(ratio):
    burnman = pytest.importorskip("burnman")
    original = burnman.Mineral(dict(macaw_params(), P_0=2.5e9, F_0=-125000.0))
    native = from_burnman(original)
    volume = ratio * original.params["V_0"]
    pressure = original.method.pressure(300.0, volume, original.params)
    native.set_state(pressure, 300.0)
    assert native.params.equation_of_state == bm.EOSType.MACAW
    assert native.molar_volume == pytest.approx(volume, rel=1e-13)
    # Use known volumes, avoiding upstream's loose absolute inversion tolerance.
    assert native.isothermal_bulk_modulus_reuss == pytest.approx(
        original.method.isothermal_bulk_modulus_reuss(
            pressure, 300.0, volume, original.params
        ),
        rel=1e-12,
    )
    assert native.molar_gibbs == pytest.approx(
        original.method.gibbs_energy(pressure, 300.0, volume, original.params),
        rel=1e-11,
        abs=1e-7,
    )
