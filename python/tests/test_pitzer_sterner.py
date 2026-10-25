# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""General PS coefficients, both caloric references and Python comparisons."""

import json
from pathlib import Path

import numpy as np
import pytest

import burnman_cpp as bm


def parameters(mode):
    table = json.loads(
        (
            Path(__file__).parents[2]
            / "benchmarks/pitzer_sterner_1995/reference/figure1.json"
        ).read_text()
    )["fluids"]["CO2" if mode == "co2" else "H2O"]
    p = dict(
        equation_of_state="pitzer-sterner",
        formula=table["formula"],
        n=3.0,
        molar_mass=table["molar_mass_kg_mol"],
        c_coeffs=table["c_coeffs"],
    )
    if mode == "debye":
        p.update(
            F_0=-249677.75163034006 + 233.2550 * 298.15,
            Cv_0=8.31446 * 2.25,
            Debye_0=3800.0,
            Debye_n=1.2,
        )
    else:
        # Arbitrary ideal-gas reference; exercise nondefault T_0 and P_0.
        p.update(
            Cp=[40.1, 0.008656, 487500.0, -251.2],
            H_0=-241810.0,
            S_0=188.8,
            T_0=300.0,
            P_0=2.0e5,
        )
    return p


@pytest.mark.parametrize("mode", ["debye", "hp", "co2"])
def test_general_factory_and_fixed_volume_properties(mode):
    p = bm.mineral_params(parameters(mode))
    eos = bm.make_eos("pitzer-sterner")
    assert p.equation_of_state == bm.EOSType.PitzerSterner
    for pressure, t in [(1e5, 300.0), (1e9, 1600.0), (8e10, 5000.0)]:
        v = eos.compute_volume(pressure, t, p)
        assert eos.compute_pressure(t, v, p) == pytest.approx(pressure, rel=1e-8)
        f = eos.compute_helmholtz_free_energy(pressure, t, v, p)
        s = eos.compute_entropy(pressure, t, v, p)
        cv = eos.compute_molar_heat_capacity_v(pressure, t, v, p)
        cp = eos.compute_molar_heat_capacity_p(pressure, t, v, p)
        bulk = eos.compute_isothermal_bulk_modulus_reuss(pressure, t, v, p)
        alpha = eos.compute_thermal_expansivity(pressure, t, v, p)
        assert cp == pytest.approx(cv + t * v * alpha**2 * bulk, rel=1e-13)
        assert eos.compute_gibbs_free_energy(pressure, t, v, p) == pytest.approx(
            f + pressure * v, rel=1e-13
        )
        assert eos.compute_enthalpy(pressure, t, v, p) == pytest.approx(
            f + t * s + pressure * v, rel=1e-13
        )


def test_original_water_factory_and_optional_bounds():
    first = bm.minerals.Pitzer_Sterner_1994.H2O_Pitzer_Sterner()
    second = bm.Mineral(parameters("debye"))
    for phase in (first, second):
        phase.set_state(1e10, 2000.0)
    assert first.molar_gibbs == pytest.approx(second.molar_gibbs, rel=1e-13)
    second.set_state(1e5, 300.0)
    assert first.temperature == 2000.0
    restricted = bm.Mineral(
        dict(parameters("hp"), T_min=500.0, T_max=1700.0, P_max=5e9)
    )
    for t in (500.0, 1700.0):
        restricted.set_state(1e5, t)
        assert np.isfinite(restricted.molar_heat_capacity_p)
    for pressure, t in [(1e5, 499.0), (1e5, 1701.0), (5.1e9, 1000.0)]:
        with pytest.raises(ValueError):
            restricted.set_state(pressure, t)
            _ = restricted.molar_volume  # State-dependent properties are lazy.


@pytest.mark.parametrize("mode", ["debye", "hp", "co2"])
@pytest.mark.parametrize(
    "pressure,t",
    [
        (1e3, 300.0),
        (1e5, 300.0),
        (3e6, 500.0),
        (8e6, 550.0),
        (1e5, 1000.0),
        (1e9, 1500.0),
        (1e10, 2000.0),
        (8e10, 5000.0),
    ],
)
def test_optional_python_reference(mode, pressure, t):
    burnman = pytest.importorskip("burnman")
    p = parameters(mode)
    from burnman_cpp.adapters import from_burnman

    original = burnman.Mineral(p)
    native = from_burnman(original)
    for phase in (original, native):
        phase.set_state(pressure, t)
    for attribute in (
        "molar_volume",
        "molar_helmholtz",
        "molar_gibbs",
        "molar_entropy",
        "molar_heat_capacity_v",
        "molar_heat_capacity_p",
        "isothermal_bulk_modulus_reuss",
        "isentropic_bulk_modulus_reuss",
        "thermal_expansivity",
        "grueneisen_parameter",
    ):
        # Python uses finite differences for several response functions.
        assert getattr(native, attribute) == pytest.approx(
            getattr(original, attribute), rel=3e-6, abs=1e-8
        ), attribute
