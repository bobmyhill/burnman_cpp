# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

"""HP98 bindings, analytic heat capacity and optional Python parity."""

import pytest

import burnman_cpp as bm
from burnman_cpp.adapters import from_burnman


def hp98_params():
    return dict(
        equation_of_state="hp98",
        V_0=2.445e-5,
        K_0=251e9,
        Kprime_0=4.14,
        Cp=[149.3, 0.002918, -2983000.0, -799.1],
        a_0=1.87e-5,
        dKdT_0=-1e7,
        H_0=-1443030.0,
        S_0=62.6,
        n=5.0,
        molar_mass=0.1003887,
    )


@pytest.mark.parametrize("method", ["hp98", "HP98", bm.EOSType.HP98])
def test_factory_and_native_mineral_properties(method):
    native = bm.Mineral(dict(hp98_params(), equation_of_state=method))
    assert native.params.equation_of_state == bm.EOSType.HP98
    assert native.params.T_0 == 298.15
    assert native.params.P_0 == 0.0
    assert native.params.E_0 is None
    assert native.params.Kdprime_0 is None
    assert native.params.T_einstein is None
    native.set_state(1e10, 3000.0)
    # Independent 60-digit G derivatives at 10 GPa and 3000 K.
    for attribute, expected in (
        ("molar_gibbs", -1853010.1928389042),
        ("molar_volume", 2.4325135752288525e-5),
        ("molar_entropy", 331.7511078176048),
        ("molar_heat_capacity_p", 142.7985457998513),
        ("thermal_expansivity", 1.3065256505458962e-5),
        ("isothermal_bulk_modulus_reuss", 265381500000.0),
    ):
        assert getattr(native, attribute) == pytest.approx(
            expected, rel=1e-13
        ), attribute
    assert native.shear_modulus == 0.0
    assert native.molar_enthalpy == pytest.approx(
        native.molar_gibbs + 3000.0 * native.molar_entropy, rel=1e-14
    )
    assert native.molar_internal_energy == pytest.approx(
        native.molar_enthalpy - 1e10 * native.molar_volume, rel=1e-14
    )
    assert bm.make_eos(method).compute_pressure(
        3000.0, native.molar_volume, native.params
    ) == pytest.approx(1e10, rel=1e-12)


@pytest.mark.parametrize(
    "field", ["V_0", "K_0", "Kprime_0", "Cp", "a_0", "dKdT_0", "n", "molar_mass"]
)
def test_required_parameters(field):
    p = hp98_params()
    del p[field]
    with pytest.raises(ValueError):
        bm.Mineral(p)


def test_zero_pressure_convention_and_independent_instances():
    first = bm.Mineral(hp98_params())
    second = bm.Mineral(dict(hp98_params(), P_0=2.5e9))
    for phase in (first, second):
        phase.set_state(1e10, 1200.0)
    assert first.molar_volume == second.molar_volume
    assert first.molar_gibbs == second.molar_gibbs
    gibbs = first.molar_gibbs
    second.set_state(1e5, 300.0)
    first.reset_cache()
    assert first.molar_gibbs == gibbs
    with pytest.raises(ValueError, match="H_0.*E_0"):
        bm.Mineral(dict(hp98_params(), E_0=0.0))


@pytest.mark.parametrize(
    "pressure,t",
    [(0.0, 298.15), (1e5, 1000.0), (1e10, 1200.0), (1e10, 3000.0), (1e11, 2000.0)],
)
def test_optional_python_reference_and_adapter(pressure, t):
    burnman = pytest.importorskip("burnman")
    original = burnman.Mineral(hp98_params())
    original.set_state(pressure, t)
    native = from_burnman(original)
    native.set_state(pressure, t)
    assert native.params.equation_of_state == bm.EOSType.HP98
    for attribute in (
        "molar_gibbs",
        "molar_volume",
        "molar_entropy",
        "thermal_expansivity",
        "isothermal_bulk_modulus_reuss",
    ):
        assert getattr(native, attribute) == pytest.approx(
            getattr(original, attribute),
            rel=3e-11,
            abs=1e-8 if attribute in ("molar_gibbs", "molar_entropy") else 0.0,
        ), attribute
    # Upstream differentiates S over 0.1 K for Cp; native HP98 is analytic.
    assert native.molar_heat_capacity_p == pytest.approx(
        original.molar_heat_capacity_p, rel=2e-6
    )
