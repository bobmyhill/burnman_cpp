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

"""SPOCK factory, mineral integration and optional Python BurnMan comparisons."""

import pytest

import burnman_cpp as bm


def spock_params():
    return dict(
        equation_of_state="spock",
        V_0=11.24e-6,
        K_0=161.0e9,
        Kprime_0=3.8,
        Kdprime_0=-0.02e-9,
        Kprime_inf=5.0 / 3.0,
        molar_mass=0.0403,
    )


@pytest.mark.parametrize("method", ["spock", "SPOCK", bm.EOSType.SPOCK])
def test_spock_factory_and_reference_state(method):
    params = spock_params()
    params["equation_of_state"] = method
    mineral = bm.Mineral(params)
    validated = mineral.params
    assert validated.F_0 == 0.0
    assert validated.P_0 == 1.0e5
    assert validated.E_0 is None
    assert validated.equation_of_state == bm.EOSType.SPOCK
    eos = bm.make_eos(method)
    assert eos.compute_pressure(300.0, params["V_0"], validated) == validated.P_0
    mineral.set_state(validated.P_0, 300.0)
    assert mineral.molar_volume == pytest.approx(params["V_0"], rel=2.0e-13)
    assert mineral.isothermal_bulk_modulus_reuss == pytest.approx(params["K_0"])
    assert mineral.molar_gibbs == pytest.approx(validated.P_0 * params["V_0"])
    assert mineral.molar_heat_capacity_p == 1.0e-99
    assert mineral.molar_heat_capacity_v == 1.0e-99
    assert mineral.shear_modulus == 1.0e99
    assert mineral.grueneisen_parameter == 0.0


@pytest.mark.parametrize(
    "missing", ["V_0", "K_0", "Kprime_0", "Kdprime_0", "Kprime_inf"]
)
def test_spock_requires_parameters(missing):
    params = spock_params()
    del params[missing]
    with pytest.raises(ValueError, match=missing):
        bm.Mineral(params)


def test_spock_rejects_internal_energy_reference():
    with pytest.raises(ValueError, match="F_0, not internal energy E_0"):
        bm.Mineral(dict(spock_params(), E_0=0.0))


@pytest.mark.parametrize("ratio", [0.4, 0.8, 0.95, 0.99, 1.0, 1.2])
@pytest.mark.parametrize("temperature", [300.0, 2000.0])
@pytest.mark.parametrize("iron", [False, True])
def test_spock_python_reference(ratio, temperature, iron):
    reference = pytest.importorskip("burnman.eos.spock").SPOCK()
    params = dict(spock_params(), F_0=-125000.0, P_0=2.5e9)
    if iron:
        params.update(
            V_0=6.75e-6,
            K_0=163.4e9,
            Kprime_0=5.38,
            Kdprime_0=-0.05e-9,
            Kprime_inf=2.0,
        )
    volume = params["V_0"] * ratio
    pressure = reference.pressure(temperature, volume, params)
    native = bm.Mineral(params)
    native.set_state(pressure, temperature)
    eos = bm.make_eos("spock")
    assert eos.compute_pressure(temperature, volume, native.params) == pytest.approx(
        pressure, rel=2.0e-11, abs=1.0e-3
    )
    assert native.molar_volume == pytest.approx(volume, rel=2.0e-12)
    assert native.isothermal_bulk_modulus_reuss == pytest.approx(
        reference.isothermal_bulk_modulus_reuss(pressure, temperature, volume, params),
        rel=2.0e-12,
    )
    assert native.molar_helmholtz == pytest.approx(
        reference._molar_helmholtz_energy(pressure, temperature, volume, params),
        rel=2.0e-12,
        abs=3.0e-7,
    )
    assert native.molar_gibbs == pytest.approx(
        reference.gibbs_energy(pressure, temperature, volume, params),
        rel=2.0e-12,
        abs=3.0e-7,
    )


def test_spock_burnman_adapter():
    reference = pytest.importorskip("burnman")
    from burnman_cpp.adapters import from_burnman

    original = reference.Mineral(spock_params())
    native = from_burnman(original)
    native.set_state(1.0e5, 300.0)
    assert native.params.equation_of_state == bm.EOSType.SPOCK
    assert native.molar_volume == pytest.approx(original.params["V_0"], rel=2.0e-13)
