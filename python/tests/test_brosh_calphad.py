# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

"""Brosh CALPHAD bindings and corrected Python SE_2015 reference values."""

import json
from pathlib import Path

import pytest
from scipy.optimize import brentq

import burnman_cpp as bm
from burnman_cpp.adapters import from_burnman
from burnman_cpp.minerals import SE_2015

REFERENCE = json.loads(
    (
        Path(__file__).resolve().parents[2] / "tests/reference/brosh_calphad.json"
    ).read_text()
)
PHASES = ("bcc", "fcc", "hcp", "liquid")


@pytest.mark.parametrize(
    "row",
    REFERENCE["states"],
    ids=lambda row: f"{row['phase']}-{row['pressure']:g}-{row['temperature']:g}",
)
def test_corrected_python_reference(row):
    mineral = getattr(SE_2015, row["phase"] + "_iron")()
    assert mineral.params.equation_of_state == bm.EOSType.BroshCalphad
    mineral.set_state(row["pressure"], row["temperature"])
    for attribute, expected in row.items():
        if attribute in ("phase", "pressure", "temperature"):
            continue
        # The Python cold-curve antiderivatives lose a few microjoules/mol.
        absolute = 5e-6 if attribute == "molar_gibbs" else 0.0
        assert getattr(mineral, attribute) == pytest.approx(
            expected, rel=3e-11, abs=absolute
        ), attribute


@pytest.mark.parametrize(
    "method", ["brosh_calphad", "BroshCalphad", bm.EOSType.BroshCalphad]
)
def test_parameter_conversion_and_factory(method):
    p = SE_2015.hcp_iron().params
    fields = (
        "name",
        "formula",
        "molar_mass",
        "V_0",
        "K_0",
        "Kprime_0",
        "napfu",
        "theta_0",
        "grueneisen_0",
        "gibbs_coefficients",
    )
    values = {field: getattr(p, field) for field in fields}
    values.update(
        equation_of_state=method,
        a=p.brosh_a,
        c=p.brosh_c,
        b=[p.b_0, p.b_1],
        delta=[p.delta_0, p.delta_1],
    )
    # SGTE input may include unused zero padding, as in Python liquid Fe.
    values["gibbs_coefficients"] = [
        (upper, list(coefficients) + [0.0])
        for upper, coefficients in values["gibbs_coefficients"]
    ]
    native = bm.Mineral(values)
    assert native.params.brosh_a == p.brosh_a
    assert native.params.brosh_c == p.brosh_c
    assert native.params.b_1 == p.b_1
    assert native.params.delta_1 == p.delta_1
    assert native.params.a is None
    native.set_state(1e11, 3000.0)
    model = bm.make_eos(method)
    assert model.compute_pressure(
        3000.0, native.molar_volume, native.params
    ) == pytest.approx(1e11, abs=0.01)
    for extra in ({"b_0": 1.0}, {"brosh_a": p.brosh_a}):
        with pytest.raises(ValueError, match="more than once"):
            bm.mineral_params(dict(values, **extra))
    with pytest.raises(ValueError, match="two coefficients"):
        bm.mineral_params(dict(values, b=[1.0]))
    values["gibbs_coefficients"][0][1][-1] = 1.0
    with pytest.raises(ValueError, match="13 coefficients"):
        bm.mineral_params(values)


def test_native_iron_melting_and_independent_state():
    solid, liquid = SE_2015.hcp_iron(), SE_2015.liquid_iron()

    def difference(temperature):
        solid.set_state(363.85e9, temperature)
        liquid.set_state(363.85e9, temperature)
        return solid.molar_gibbs - liquid.molar_gibbs

    melting = brentq(difference, 5000.0, 8000.0)
    assert melting == pytest.approx(6453.78756, abs=1e-4)
    solid.set_state(363.85e9, 6453.0)
    assert solid.density == pytest.approx(13709.16, abs=0.01)
    gibbs = solid.molar_gibbs
    another = SE_2015.hcp_iron()
    another.set_state(1e5, 300.0)
    solid.reset_cache()
    assert solid.molar_gibbs == gibbs
    phases, metadata = bm.minerals.load_dataset("SE_2015")
    assert len(phases) == 4
    assert set(metadata["phase_names"]) == {phase + "_iron" for phase in PHASES}


@pytest.mark.parametrize("phase", PHASES)
def test_adapter_from_optional_burnman(phase):
    burnman = pytest.importorskip("burnman")
    original = getattr(burnman.minerals.SE_2015, phase + "_iron")()
    native = from_burnman(original)
    original.set_state(1e10, 1000.0)
    native.set_state(1e10, 1000.0)
    for attribute in (
        "molar_gibbs",
        "molar_volume",
        "molar_entropy",
        "molar_heat_capacity_p",
    ):
        assert getattr(native, attribute) == pytest.approx(
            getattr(original, attribute), rel=1e-6, abs=5e-6
        )
