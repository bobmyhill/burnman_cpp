# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Gibbs derivatives, optional shear checks and the upstream checking protocol."""

import warnings

import numpy as np
import pytest

import burnman_cpp as bm
from burnman_cpp.tools.eos import check_eos_consistency

CASES = [
    ("HP_2011_ds62", "per", False),
    ("HP_2011_ds62", "foL", False),
    ("HP_2011_fluids", "CO2", False),
    ("SLB_2011", "periclase", True),
    ("SLB_2022", "almandine", True),
    ("DKS_2013_liquids", "Mg3Si2O7_liquid", False),
    ("DKS_2013_solids", "periclase", False),
    ("Sundman_1991", "bcc_iron", False),
    ("SLB_2011", "garnet", True),
    ("custom", "hp98", False),
    ("custom", "dks_shear", True),
]


def make_model(module, dataset, name):
    if dataset == "custom":
        if name == "hp98":
            params = module.minerals.HP_2011_ds62.per().params
            updates = {"equation_of_state": "hp98", "dKdT_0": -1e7}
            if not isinstance(params, dict):
                params.E_0 = None
        else:
            params = module.minerals.DKS_2013_solids.periclase().params
            reference = module.minerals.SLB_2011.periclase().params
            updates = {"power_law_grueneisen": False}
            updates.update(
                (
                    key,
                    (
                        reference[key]
                        if isinstance(reference, dict)
                        else getattr(reference, key)
                    ),
                )
                for key in ("G_0", "Gprime_0", "eta_s_0")
            )
        if isinstance(params, dict):
            params.update(updates)
        else:
            for key, value in updates.items():
                setattr(
                    params,
                    key,
                    bm.eos_type(value) if key == "equation_of_state" else value,
                )
        return module.Mineral(params)
    model = getattr(getattr(module.minerals, dataset), name)()
    if name == "garnet":
        model.set_composition([0.2] * 5)
    return model


def state(dataset):
    return (2e9, 500.0) if dataset == "SLB_2022" else (10e9, 3000.0)


def checks(output):
    return [
        line.strip()
        for line in output.splitlines()
        if line.rsplit(":", 1)[-1].strip() in ("True", "False")
    ]


@pytest.mark.parametrize("dataset,name,shear", CASES)
def test_native_models(dataset, name, shear):
    assert check_eos_consistency(
        make_model(bm, dataset, name),
        *state(dataset),
        including_shear_properties=shear,
    )


@pytest.mark.parametrize("dataset,name,shear", CASES)
def test_optional_python_reference(dataset, name, shear, capsys):
    reference = pytest.importorskip("burnman")
    from burnman.tools.eos import check_eos_consistency as original

    results = []
    reports = []
    for module, checker in [(bm, check_eos_consistency), (reference, original)]:
        results.append(
            checker(
                make_model(module, dataset, name),
                *state(dataset),
                including_shear_properties=shear,
                verbose=True,
            )
        )
        reports.append(checks(capsys.readouterr().out))
    assert results[0] == results[1]
    assert reports[0] == reports[1]


class PropertyOverride:
    def __init__(self, model, name, value):
        self.model, self.property_name, self.value = model, name, value

    def __getattr__(self, name):
        return self.value if name == self.property_name else getattr(self.model, name)


@pytest.mark.parametrize(
    "name",
    [
        "molar_entropy",
        "thermal_expansivity",
        "molar_heat_capacity_p",
        "grueneisen_parameter",
        "p_wave_velocity",
        "shear_wave_velocity",
    ],
)
def test_detects_inconsistent_or_nonfinite_properties(name, capsys):
    model = bm.minerals.SLB_2011.periclase()
    model.set_state(1e9, 1000.0)
    wrong = PropertyOverride(model, name, 1.1 * getattr(model, name))
    assert not check_eos_consistency(wrong, T=1000.0, verbose=True)
    assert any(line.endswith("False") for line in checks(capsys.readouterr().out))
    assert not check_eos_consistency(PropertyOverride(model, name, np.nan), T=1000.0)


def test_callback_states_and_verbose_diagnostics(capsys):
    model = bm.minerals.SLB_2011.periclase()
    states = []
    assert check_eos_consistency(
        model,
        T=1000.0,
        verbose=True,
        equilibration_function=lambda m: states.append((m.pressure, m.temperature)),
    )
    assert states == [
        (1e9, 1000.0),
        (1e9, 1001.0),
        (1e9 + 1000, 1000.0),
        (1e9, 1000.5),
        (1e9 + 500, 1000.0),
    ]
    assert (model.pressure, model.temperature) == states[-1]
    output = capsys.readouterr().out
    assert len(checks(output)) == 14
    assert "All EoS consistency constraints satisfied" in output


def test_composite_thermodynamics_and_optional_shear(capsys):
    model = bm.Assemblage(
        [bm.minerals.SLB_2011.periclase(), bm.minerals.SLB_2011.forsterite()],
        [0.3, 0.7],
    )

    class NoShear:
        def __getattr__(self, name):
            if name in {
                "shear_modulus",
                "p_wave_velocity",
                "shear_wave_velocity",
                "bulk_sound_velocity",
            }:
                raise AssertionError(
                    "Shear and velocity properties must not be queried"
                )
            return getattr(model, name)

    assert check_eos_consistency(
        NoShear(), T=1000.0, including_shear_properties=False, verbose=True
    )
    assert "(not including shear properties)" in capsys.readouterr().out


def test_single_shear_warning_prints_upstream_guidance(capsys):
    model = bm.minerals.SLB_2011.periclase()

    class ShearWarning:
        def __getattr__(self, name):
            if name == "p_wave_velocity":
                warnings.warn("Missing shear parametrization", UserWarning)
            return getattr(model, name)

    assert check_eos_consistency(ShearWarning(), T=1000.0)
    output = capsys.readouterr().out
    assert "Missing shear parametrization" in output
    assert "including_shear_properties to False" in output
