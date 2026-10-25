# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

"""Morse factory, native Mineral properties and optional Python parity."""

import pytest

import burnman_cpp as bm
from burnman_cpp.adapters import from_burnman


def morse_params():
    return dict(
        equation_of_state="morse",
        V_0=11.24e-6,
        K_0=161e9,
        Kprime_0=3.8,
        molar_mass=0.0403,
    )


@pytest.mark.parametrize("method", ["morse", "Morse", bm.EOSType.Morse])
def test_factory_and_mineral_properties(method):
    native = bm.Mineral(dict(morse_params(), equation_of_state=method))
    assert native.params.equation_of_state == bm.EOSType.Morse
    assert native.params.F_0 == 0.0
    assert native.params.P_0 == 0.0
    assert native.params.E_0 is None
    # Python Morse at V/V_0=0.8 and P_0=F_0=0.
    pressure, energy = 54380737902.413048, 51315.973318865021
    for temperature in (0.0, 300.0, 2000.0):
        native.set_state(pressure, temperature)
        assert native.molar_volume == pytest.approx(0.8 * native.params.V_0, rel=1e-13)
        assert native.isothermal_bulk_modulus_reuss == pytest.approx(
            342468842218.10699, rel=1e-13
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
        assert native.molar_heat_capacity_p == native.molar_heat_capacity_v == 1e-99
        assert native.molar_entropy == native.thermal_expansivity == 0.0
        assert native.shear_modulus == native.grueneisen_parameter == 0.0
    model = bm.make_eos(method)
    assert model.compute_pressure(
        300.0, native.molar_volume, native.params
    ) == pytest.approx(pressure, rel=1e-13)


@pytest.mark.parametrize("field", ["V_0", "K_0", "Kprime_0"])
def test_required_parameters(field):
    params = morse_params()
    del params[field]
    with pytest.raises(ValueError, match=field):
        bm.Mineral(params)


def test_energy_reference_and_pressure_work():
    native = bm.Mineral(dict(morse_params(), P_0=2.5e9, F_0=-125000.0))
    pressure = 54380737902.413048 + native.params.P_0
    native.set_state(pressure, 300.0)
    expected = (
        51315.973318865021
        + native.params.F_0
        + native.params.P_0 * (native.params.V_0 - native.molar_volume)
    )
    assert native.molar_helmholtz == pytest.approx(expected, rel=1e-12)
    with pytest.raises(ValueError, match="F_0.*E_0"):
        bm.Mineral(dict(morse_params(), E_0=0.0))


@pytest.mark.parametrize("ratio", [0.4, 0.8, 0.99, 1.0, 1.2])
def test_optional_python_reference_and_adapter(ratio):
    burnman = pytest.importorskip("burnman")
    params = dict(morse_params(), F_0=-125000.0)
    original = burnman.Mineral(params)
    native = from_burnman(original)
    volume = ratio * params["V_0"]
    pressure = original.method.pressure(300.0, volume, original.params)
    native.set_state(pressure, 300.0)
    assert native.params.equation_of_state == bm.EOSType.Morse
    assert native.molar_volume == pytest.approx(volume, rel=1e-13)
    # Compare at a known volume: upstream's volume solver has a loose absolute
    # tolerance, and its Helmholtz energy omits the work of nonzero P_0.
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
