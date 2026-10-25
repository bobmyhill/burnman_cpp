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

"""Murnaghan factory, mineral properties and Python BurnMan comparisons."""

import pytest

import burnman_cpp as bm
from burnman_cpp.adapters import from_burnman


def murnaghan_params():
    return dict(
        equation_of_state="murnaghan",
        V_0=11.24e-6,
        K_0=161.0e9,
        Kprime_0=3.8,
        molar_mass=0.0403,
    )


@pytest.mark.parametrize("method", ["murnaghan", "Murnaghan", bm.EOSType.Murnaghan])
def test_murnaghan_factory_and_reference_state(method):
    params = dict(murnaghan_params(), equation_of_state=method)
    mineral = bm.Mineral(params)
    validated = mineral.params
    assert validated.equation_of_state == bm.EOSType.Murnaghan
    assert validated.E_0 == 0.0
    assert validated.P_0 == 0.0
    eos = bm.make_eos(method)
    assert eos.compute_pressure(300.0, params["V_0"], validated) == 0.0
    mineral.set_state(0.0, 300.0)
    assert mineral.molar_volume == params["V_0"]
    assert mineral.isothermal_bulk_modulus_reuss == params["K_0"]
    assert mineral.isentropic_bulk_modulus_reuss == params["K_0"]
    assert mineral.molar_internal_energy == 0.0
    assert mineral.molar_helmholtz == 0.0
    assert mineral.molar_gibbs == 0.0
    assert mineral.molar_enthalpy == 0.0
    assert mineral.molar_heat_capacity_p == 1.0e99
    assert mineral.molar_heat_capacity_v == 1.0e99
    assert mineral.shear_modulus == 0.0
    assert mineral.thermal_expansivity == 0.0
    assert mineral.grueneisen_parameter == 0.0
    assert mineral.molar_entropy == 0.0


@pytest.mark.parametrize("missing", ["V_0", "K_0", "Kprime_0"])
def test_murnaghan_requires_parameters(missing):
    params = murnaghan_params()
    del params[missing]
    with pytest.raises(ValueError, match=missing):
        bm.Mineral(params)


@pytest.mark.parametrize("energy_key", ["E_0", "F_0"])
@pytest.mark.parametrize("temperature", [0.0, 300.0, 2000.0])
def test_murnaghan_compressed_mineral_and_energy_reference(energy_key, temperature):
    params = dict(murnaghan_params(), P_0=2.5e9, **{energy_key: -125000.0})
    mineral = bm.Mineral(params)
    # Python BurnMan e2cfac31052273b3fa9d5ce905978e97723005e9 at V/V_0 = 0.8,
    # with the reference pressure and energy shifted consistently.
    pressure = 56555273737.019379 + params["P_0"]
    volume = 0.8 * params["V_0"]
    energy = 52363.221944027995 - 125000.0 + params["P_0"] * (params["V_0"] - volume)
    mineral.set_state(pressure, temperature)
    assert mineral.molar_volume == pytest.approx(volume, rel=1e-12)
    assert mineral.density == pytest.approx(params["molar_mass"] / volume)
    assert mineral.isothermal_bulk_modulus_reuss == pytest.approx(
        375910040200.67358, rel=1e-12
    )
    assert mineral.molar_internal_energy == pytest.approx(energy, rel=1e-12, abs=1e-8)
    assert mineral.molar_helmholtz == pytest.approx(energy, rel=1e-12, abs=1e-8)
    assert mineral.molar_gibbs == pytest.approx(energy + pressure * volume, rel=1e-12)
    assert mineral.molar_enthalpy == pytest.approx(
        energy + pressure * volume, rel=1e-12
    )


def test_murnaghan_rejects_pressure_outside_stable_branch():
    params = bm.mineral_params(dict(murnaghan_params(), K_0=160e9, Kprime_0=4.0))
    with pytest.raises(ValueError, match="positive bulk modulus branch"):
        bm.make_eos("murnaghan").compute_volume(-40e9, 300.0, params)


@pytest.mark.parametrize("ratio", [0.4, 0.8, 0.99, 1.0, 1.2])
def test_murnaghan_python_reference_and_adapter(ratio):
    burnman = pytest.importorskip("burnman")
    params = dict(murnaghan_params(), F_0=-125000.0)
    reference = burnman.Mineral(params)
    volume = ratio * params["V_0"]
    pressure = reference.method.pressure(300.0, volume, reference.params)
    reference.set_state(pressure, 300.0)
    native = from_burnman(reference)
    assert native.params.equation_of_state == bm.EOSType.Murnaghan
    native.set_state(pressure, 300.0)
    assert native.molar_volume == pytest.approx(volume, rel=1e-12)
    assert native.isothermal_bulk_modulus_reuss == pytest.approx(
        reference.isothermal_bulk_modulus_reuss, rel=1e-12
    )
    assert native.molar_gibbs == pytest.approx(
        reference.molar_gibbs, rel=1e-12, abs=1e-7
    )
    assert native.molar_helmholtz == pytest.approx(
        reference.molar_helmholtz, rel=1e-12, abs=1e-7
    )
