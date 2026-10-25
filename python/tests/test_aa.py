# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

"""Native AA: corrected Python parity, parameter conversion and thermo identities."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

import burnman_cpp as bm
from burnman_cpp.adapters import from_burnman
from burnman_cpp.minerals import other

REFERENCE = json.loads(
    (Path(__file__).resolve().parents[2] / "tests/reference/aa1994.json").read_text()
)


@pytest.mark.parametrize("method", ["aa", "AA", bm.EOSType.AA])
def test_factory_and_python_parameter_names(method):
    mineral = bm.Mineral(dict(REFERENCE["params"], equation_of_state=method))
    p = mineral.params
    assert p.equation_of_state == bm.EOSType.AA
    assert p.K_0 == REFERENCE["params"]["K_S"]
    assert p.Kprime_0 == REFERENCE["params"]["Kprime_S"]
    assert p.Kdprime_0 == REFERENCE["params"]["Kprime_prime_S"]
    for key in ("a", "b", "Theta", "lmda", "F"):
        assert getattr(p, "aa_" + key) == REFERENCE["params"][key]
    assert p.a is None
    model = bm.make_eos(method)
    assert model.compute_pressure(p.T_0, p.V_0, p) == p.P_0


@pytest.mark.parametrize(
    "row",
    REFERENCE["states"],
    ids=lambda row: f"rho{row['density']:g}-T{row['temperature']:g}",
)
def test_corrected_python_reference_and_native_liquid_factory(row):
    mineral = other.liquid_iron()
    assert mineral.params.equation_of_state == bm.EOSType.AA
    assert mineral.formula == {"Fe": 1.0}
    p, t, v = row["pressure"], row["temperature"], row["volume"]
    model = bm.make_eos("aa")
    assert model.compute_pressure(t, v, mineral.params) == pytest.approx(
        p, rel=3e-12, abs=0.01
    )
    for method, key in [
        ("compute_molar_heat_capacity_v", "cv"),
        ("compute_molar_heat_capacity_p", "cp"),
        ("compute_grueneisen_parameter", "gamma"),
        ("compute_isothermal_bulk_modulus_reuss", "kt"),
        ("compute_thermal_expansivity", "alpha"),
    ]:
        assert getattr(model, method)(p, t, v, mineral.params) == pytest.approx(
            row[key], rel=3e-12
        )
    mineral.set_state(p, t)
    assert mineral.molar_volume == pytest.approx(v, rel=3e-13)
    assert mineral.density == pytest.approx(row["density"], rel=3e-13)
    for attribute, key in [
        ("molar_helmholtz", "helmholtz"),
        ("molar_internal_energy", "energy"),
        ("molar_entropy", "entropy"),
        ("molar_heat_capacity_v", "cv"),
        ("molar_heat_capacity_p", "cp"),
        ("thermal_expansivity", "alpha"),
        ("grueneisen_parameter", "gamma"),
        ("isothermal_bulk_modulus_reuss", "kt"),
        ("isentropic_bulk_modulus_reuss", "ks"),
    ]:
        assert getattr(mineral, attribute) == pytest.approx(
            row[key], rel=3e-11, abs=1e-7
        )
    assert mineral.shear_modulus == 0.0
    assert mineral.molar_gibbs == pytest.approx(row["helmholtz"] + p * v, rel=3e-12)
    assert mineral.molar_enthalpy == pytest.approx(row["energy"] + p * v, rel=3e-12)


def test_adapter_copies_all_aa_parameters_from_optional_burnman():
    burnman = pytest.importorskip("burnman")
    # The pinned upstream class still has the old numerical implementation;
    # copy its data, and compare native behavior to the corrected reference.
    phase = burnman.minerals.other.liquid_iron()
    native = from_burnman(phase)
    native.set_state(320e9, 7000.0)
    direct = other.liquid_iron()
    direct.set_state(320e9, 7000.0)
    assert native.molar_gibbs == pytest.approx(direct.molar_gibbs, rel=1e-12)
    assert native.molar_volume == pytest.approx(direct.molar_volume, rel=1e-12)
    # A copy owns its parameters; later edits to Python input arrays do not leak.
    phase.params["a"][0] *= 2.0
    native.reset_cache()
    assert native.molar_gibbs == pytest.approx(direct.molar_gibbs, rel=1e-12)


def test_aa_array_aliases_do_not_change_dks_bonding_matrix():
    p = bm.mineral_params(a=[[1.0, 2.0], [3.0, 4.0]])
    assert p.a == [[1.0, 2.0], [3.0, 4.0]]
    assert p.aa_a is None
    for duplicate in [{"K_0": 109.7e9}, {"aa_a": [1.0, 2.0]}]:
        with pytest.raises(ValueError, match="more than once"):
            bm.mineral_params(dict(REFERENCE["params"], **duplicate))


@pytest.mark.parametrize("n", [0.0, -2 / 3, -4 / 3, -2.0, -8 / 3, 1.0])
def test_reference_isentrope_against_independent_temperature_integral(n):
    p = dict(REFERENCE["params"], grueneisen_n=n, P_0=2.5e9)
    model = bm.make_eos("aa")
    native = bm.Mineral(p)
    xi1 = 0.75 * (4.0 - p["Kprime_S"])
    xi2 = (
        0.375 * (p["K_S"] * p["Kprime_prime_S"] + p["Kprime_S"] * (p["Kprime_S"] - 7.0))
        + 143 / 24
    )

    def reference_energy(ell):
        q = np.expm1(2 * ell / 3)
        return 4.5 * p["V_0"] * p["K_S"] * q * q * (
            0.25 - xi1 * q / 6 + xi2 * q * q / 8
        ) - p["P_0"] * p["V_0"] * np.expm1(-ell)

    for ratio in [0.6, 1.0, 1.3]:
        ell = -np.log(ratio)
        exponent = quad(
            lambda z: p["grueneisen_0"]
            + p["grueneisen_prime"] * np.exp(n * z) * reference_energy(z),
            0.0,
            ell,
            epsabs=1e-12,
        )[0]
        ts = p["T_0"] * np.exp(exponent)
        v = ratio * p["V_0"]
        q = np.expm1(2 * ell / 3)
        ps = p["P_0"] + 1.5 * p["K_S"] * np.exp(5 * ell / 3) * q * (
            1 - xi1 * q + xi2 * q * q
        )
        assert model.compute_pressure(ts, v, native.params) == pytest.approx(
            ps, rel=2e-12, abs=0.01
        )
        # Evaluate directly in the natural variables; some tension states have
        # no pressure root bracketed by the retained global volume interval.
        assert model.compute_entropy(ps, ts, v, native.params) == pytest.approx(
            p["S_0"], abs=1e-9
        )


def test_material_instances_have_independent_state():
    first, second = other.liquid_iron(), other.liquid_iron()
    first.set_state(130e9, 5000.0)
    gibbs = first.molar_gibbs
    second.set_state(320e9, 7000.0)
    first.reset_cache()
    assert first.molar_gibbs == pytest.approx(gibbs, rel=1e-14)
