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

"""Reaction chemistry checked against energy balance and independent models."""

from functools import reduce
from math import gcd

import numpy as np
from numpy.testing import assert_allclose
import pytest

import burnman_cpp as bm
from burnman_cpp.tools import chemistry, partitioning


def shifted_minerals():
    mineral = bm.minerals.HP_2011_ds62.fo()
    thermal = bm.CombinedMineral([mineral], [1.0], [1000.0, 2.0, 0.0])
    pressure = bm.CombinedMineral([mineral], [1.0], [-2000.0, 0.0, 1.0e-6])
    return mineral, thermal, pressure


def test_reaction_roots_and_invariant_point():
    mineral, thermal, pressure = shifted_minerals()
    temperature = chemistry.equilibrium_temperature([mineral, thermal], [-1, 1], 2.0e9)
    assert_allclose(temperature, 500.0, atol=1.0e-5, rtol=0)
    assert thermal.temperature == temperature
    p = chemistry.equilibrium_pressure([mineral, pressure], [-1, 1], temperature)
    assert_allclose(p, 2.0e9, atol=2.0, rtol=0)
    assert_allclose(
        chemistry.equilibrium_pressure(
            [mineral, pressure], [-1.0e-12, 1.0e-12], temperature
        ),
        p,
        atol=2.0,
        rtol=0,
    )
    assert pressure.pressure == p
    pt = chemistry.invariant_point(
        [mineral, thermal], [-1, 1], [mineral, pressure], [-1, 1]
    )
    assert_allclose(pt, [2.0e9, 500.0], rtol=1.0e-8)
    assert_allclose([mineral.pressure, mineral.temperature], pt)
    assert_allclose(thermal.molar_gibbs - mineral.molar_gibbs, 0.0, atol=1.0e-6)
    assert_allclose(pressure.molar_gibbs - mineral.molar_gibbs, 0.0, atol=1.0e-6)


def test_root_failures_and_input_validation():
    mineral, thermal, pressure = shifted_minerals()
    constant = bm.CombinedMineral([mineral], [1.0], [1000.0, 0.0, 0.0])
    with pytest.raises(RuntimeError, match="derivative"):
        chemistry.equilibrium_temperature([mineral, constant], [-1, 1], 1.0e9)
    with pytest.raises(RuntimeError, match="independent|isolated"):
        chemistry.invariant_point(
            [mineral, thermal], [-1, 1], [mineral, thermal], [-1, 1]
        )
    for amounts in ([1.0], [0.0, 0.0], [np.nan, 1.0]):
        with pytest.raises(ValueError):
            chemistry.equilibrium_pressure([mineral, pressure], amounts, 1000.0)
    with pytest.raises(ValueError):
        chemistry.equilibrium_temperature([mineral, thermal], [-1, 1], 1.0e9, -1.0)


def test_component_chemical_potentials_and_fugacities():
    pv = bm.minerals.SLB_2011.mg_perovskite()
    per = bm.minerals.SLB_2011.periclase()
    rock = bm.Assemblage([pv, per], [0.5, 0.5])
    with pytest.raises(RuntimeError, match="state"):
        rock.chemical_potential([{"Si": 1.0, "O": 2.0}])
    rock.set_state(25.0e9, 2000.0)
    rock.set_components([{"Si": 1.0, "O": 2.0}, {"Mg": 1.0, "Si": -1.0, "O": -1.0}])
    assert_allclose(
        rock.chemical_potential(None),
        [pv.molar_gibbs - per.molar_gibbs, 2.0 * per.molar_gibbs - pv.molar_gibbs],
    )
    rock.set_state(30.0e9, 1800.0)
    assert_allclose(rock.chemical_potential()[0], pv.molar_gibbs - per.molar_gibbs)
    for formula in ({"Si": 1.0}, {"H": 1.0}, {"Si": np.nan}):
        with pytest.raises(ValueError, match="defined"):
            rock.set_components([formula])
    reference = bm.minerals.SLB_2011.stishovite()
    reference.set_state(30.0e9, 1800.0)
    expected = np.exp(
        (pv.molar_gibbs - per.molar_gibbs - reference.molar_gibbs)
        / (bm.constants.gas_constant * 1800.0)
    )
    assert_allclose(chemistry.fugacity(reference, rock), expected)
    assert_allclose(chemistry.relative_fugacity(reference.formula, rock, rock), 1.0)
    reference_rock = bm.Assemblage([pv, reference], [0.5, 0.5])
    reference_rock.set_state(30.0e9, 1800.0)
    assert_allclose(chemistry.fugacity(reference, reference_rock), 1.0)
    assert_allclose(
        chemistry.relative_fugacity(reference.formula, rock, reference_rock), expected
    )
    reference.set_state(30.0e9, 1900.0)
    with pytest.raises(ValueError, match="same pressure and temperature"):
        chemistry.fugacity(reference, rock)


def test_duplicate_components_and_disequilibrium():
    mineral, thermal, _ = shifted_minerals()
    rock = bm.Assemblage([mineral, mineral.copy()], [0.5, 0.5])
    rock.set_state(1.0e9, 1000.0)
    assert_allclose(rock.chemical_potential([mineral.formula])[0], mineral.molar_gibbs)
    rock = bm.Assemblage([mineral, thermal], [0.5, 0.5])
    rock.set_state(1.0e9, 1000.0)
    with pytest.raises(RuntimeError, match="equilibrated"):
        rock.chemical_potential([mineral.formula])
    rock.set_state(1.0e9, 500.0)
    assert_allclose(rock.chemical_potential([mineral.formula])[0], mineral.molar_gibbs)


def test_chemical_potentials_require_defined_components_at_every_state():
    material = bm.EquilibratedComposite(
        {"Mg": 2.0, "Si": 1.0, "O": 4.0},
        "SLB_2024",
        phase_names=["forsterite", "mg_perovskite", "periclase"],
    )
    silica = {"Si": 1.0, "O": 2.0}
    material.set_state(1.0e5, 300.0)
    for scale in [1.0e-20, 1.0, 1.0e20]:
        assert_allclose(
            material.chemical_potential(
                [
                    {
                        element: scale * amount
                        for element, amount in material.formula.items()
                    }
                ]
            ),
            [scale * material.molar_gibbs],
        )
        for formula in [silica, {"Mg": 1.0}, {"O": 1.0}]:
            with pytest.raises(ValueError, match="not defined"):
                material.chemical_potential(
                    [{element: scale * amount for element, amount in formula.items()}]
                )
    material.set_state(50.0e9, 1500.0)
    potential = material.chemical_potential([silica])[0]
    phases = {phase.name: phase for phase in material.phases}
    # SLB_2024 periclase is Mg4O4, so a quarter mole contributes one MgO.
    assert_allclose(
        potential,
        phases["Mg-Perovskite"].molar_gibbs - 0.25 * phases["Periclase"].molar_gibbs,
    )
    with pytest.raises(ValueError, match="not defined"):
        material.chemical_potential([{"Mg": 1.0}])
    material.set_state(1.0e5, 300.0)
    with pytest.raises(ValueError, match="not defined"):
        material.chemical_potential([silica])
    with pytest.raises(RuntimeError, match="set_components"):
        material.chemical_potential()


def test_endmember_partials_are_gibbs_derivatives():
    solution = bm.minerals.SLB_2011.mg_fe_olivine()
    solution.set_composition([0.8, 0.2])
    rock = bm.Assemblage([solution, bm.minerals.SLB_2011.periclase()], [0.8, 0.2])
    p, t = 20.0e9, 1700.0
    rock.set_state(p, t)
    volumes, entropies = (
        rock.endmember_partial_volumes,
        rock.endmember_partial_entropies,
    )
    rock.set_state(p + 1.0e5, t)
    plus = rock.endmember_partial_gibbs
    rock.set_state(p - 1.0e5, t)
    assert_allclose((plus - rock.endmember_partial_gibbs) / 2.0e5, volumes, rtol=1.0e-7)
    rock.set_state(p, t + 0.01)
    plus = rock.endmember_partial_gibbs
    rock.set_state(p, t - 0.01)
    assert_allclose(
        -(plus - rock.endmember_partial_gibbs) / 0.02, entropies, rtol=1.0e-7
    )


@pytest.mark.parametrize("use_reference", [False, True])
def test_hugoniot_energy_balance(use_reference):
    mineral = bm.minerals.SLB_2011.periclase()
    reference = (
        bm.CombinedMineral([mineral], [1.0], [-100.0, 0.0, 0.0])
        if use_reference
        else mineral
    )
    p0, t0 = 1.0e5, 300.0
    reference.set_state(p0, t0)
    u0, v0 = reference.molar_internal_energy, reference.molar_volume
    pressures = np.array([p0, 10.0e9, 30.0e9])
    temperatures, volumes = chemistry.hugoniot(
        mineral, p0, t0, pressures, reference if use_reference else None
    )
    for p, t, v in zip(pressures, temperatures, volumes):
        mineral.set_state(p, t)
        assert_allclose(
            mineral.molar_internal_energy - u0,
            0.5 * (p - p0) * (v0 - v),
            atol=2.0e-6,
            rtol=1.0e-9,
        )
    if not use_reference:
        assert temperatures[0] == t0
    assert_allclose(volumes[-1], mineral.molar_volume)


def test_hugoniot_is_independent_of_reference_formula_units():
    mineral = bm.minerals.SLB_2011.periclase()
    reference = bm.CombinedMineral([mineral], [2.0])
    pressures = np.array([1.0e5, 10.0e9, 30.0e9])
    expected = chemistry.hugoniot(mineral, 1.0e5, 300.0, pressures)
    actual = chemistry.hugoniot(mineral, 1.0e5, 300.0, pressures, reference)
    assert_allclose(actual, expected, rtol=1.0e-10, atol=1.0e-8)


def test_equilibrated_hugoniot_preserves_mass_across_dissociation():
    material = bm.EquilibratedComposite(
        {"Mg": 2.0, "Si": 1.0, "O": 4.0},
        "SLB_2024",
        phase_names=["forsterite", "mg_perovskite", "periclase"],
    )
    p0, t0 = 1.0e5, 300.0
    material.set_state(p0, t0)
    u0 = material.molar_internal_energy / material.molar_mass
    v0 = material.molar_volume / material.molar_mass
    initial_moles = material.n_moles
    pressures = np.array([p0, 50.0e9])
    temperatures, volumes = chemistry.hugoniot(material, p0, t0, pressures)
    assert temperatures[0] == pytest.approx(t0)
    assert temperatures[1] > t0
    assert material.n_moles != pytest.approx(initial_moles)
    for p, t, v in zip(pressures, temperatures, volumes):
        material.set_state(p, t)
        for element, amount in {"Mg": 2.0, "Si": 1.0, "O": 4.0}.items():
            assert_allclose(material.n_moles * material.formula[element], amount)
        assert_allclose(
            material.molar_internal_energy / material.molar_mass - u0,
            0.5 * (p - p0) * (v0 - v / material.molar_mass),
            atol=2.0e-5,
            rtol=1.0e-9,
        )


@pytest.mark.parametrize("warm_start", [True, False])
def test_equilibrated_hugoniot_phase_changes(warm_start):
    bulk = {"Mg": 2.0, "Si": 1.0, "O": 4.0}
    material = bm.EquilibratedComposite(
        bulk,
        "SLB_2024",
        phase_names=[
            "forsterite",
            "mg_wadsleyite",
            "mg_ringwoodite",
            "mg_perovskite",
            "periclase",
        ],
        warm_start=warm_start,
    )
    p0, t0 = 1.0e5, 300.0
    material.set_state(p0, t0)
    reference = material.snapshot()
    u0 = reference.molar_internal_energy / reference.molar_mass
    v0 = reference.molar_volume / reference.molar_mass
    for pressure in [10.0e9, 12.000088e9, 12.5e9, 50.0e9, 12.000088e9]:
        temperatures, volumes = chemistry.hugoniot(
            material, p0, t0, [pressure], reference
        )
        assert material.equilibrium_state.success
        assert_allclose(temperatures[0], material.temperature)
        assert_allclose(volumes[0], material.molar_volume)
        assert_allclose(
            material.molar_internal_energy / material.molar_mass - u0,
            0.5 * (pressure - p0) * (v0 - volumes[0] / material.molar_mass),
            atol=2.0e-5,
            rtol=1.0e-9,
        )
        for element, amount in bulk.items():
            assert_allclose(material.n_moles * material.formula[element], amount)
        assert np.all(material.molar_fractions >= 0.0)
        if pressure == 12.000088e9:
            assert [phase.name for phase in material.phases] == ["Mg-Ringwoodite"]


@pytest.mark.parametrize("warm_start", [True, False])
def test_equilibrated_hugoniot_determines_coexisting_phase_amounts(warm_start):
    base = bm.minerals.SLB_2011.forsterite()
    dense = bm.CombinedMineral([base], [1.0], [2000.0, 2.0, -1.0e-6])
    bulk = {"Mg": 2.0, "Si": 1.0, "O": 4.0}
    material = bm.EquilibratedComposite(bulk, [base, dense], warm_start=warm_start)
    p0, t0, pressure = 1.0e5, 300.0, 1.4e9
    material.set_state(p0, t0)
    reference = material.snapshot()
    u0 = reference.molar_internal_energy
    v0 = reference.molar_volume
    temperatures, volumes = chemistry.hugoniot(material, p0, t0, [pressure], reference)
    # The prescribed Gibbs offset gives exact coexistence at 300 K.
    # The lever rule for shock energy fixes the dense phase's amount.
    base.set_state(pressure, 300.0)
    dense.set_state(pressure, 300.0)
    residuals = np.array(
        [
            phase.molar_internal_energy
            - u0
            - 0.5 * (pressure - p0) * (v0 - phase.molar_volume)
            for phase in [base, dense]
        ]
    )
    fraction = -residuals[0] / (residuals[1] - residuals[0])
    assert 0.01 < fraction < 0.99
    assert_allclose(temperatures, [300.0], atol=1.0e-5)
    assert_allclose(material.molar_fractions, [1.0 - fraction, fraction], atol=1.0e-7)
    assert_allclose(material.reaction_affinities, 0.0, atol=1.0e-5)
    assert_allclose(
        material.molar_internal_energy - u0,
        0.5 * (pressure - p0) * (v0 - volumes[0]),
        atol=2.0e-6,
    )


def test_balanced_reactions_and_strings():
    matrix = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    reactions = chemistry.reactions_from_stoichiometric_matrix(matrix)
    assert reactions.shape == (6, 4)
    assert_allclose(reactions @ matrix, 0.0)
    assert_allclose(
        chemistry.reactions_from_stoichiometric_matrix(matrix * 1.0e-20), reactions
    )
    assert_allclose(
        chemistry.reactions_from_stoichiometric_matrix(matrix * [1.0e-20, 1.0e20]),
        reactions,
    )
    rows = {tuple(row) for row in reactions}
    for row in reactions:
        assert tuple(-row) in rows
        assert reduce(gcd, abs(row.astype(int))) == 1
    names = ["per", "fper", "fs", "fa"]
    strings = chemistry.reactions_from_formulae(
        ["MgO", "FeO", "Fe2Si2O6", "Fe2SiO4"], names
    )
    assert sorted(strings) == ["2 fa = 2 fper + 1 fs", "2 fper + 1 fs = 2 fa"]
    numeric = chemistry.reactions_from_formulae([{"Mg": 1.0}, {"Mg": 0.5}], None, False)
    assert_allclose(numeric, [[-1, 2], [1, -2]])
    assert chemistry.reactions_from_stoichiometric_matrix(np.eye(25)).shape == (0, 25)
    for matrix in ([], [[0.0]], [[np.nan]]):
        with pytest.raises(ValueError):
            chemistry.reactions_from_stoichiometric_matrix(matrix)
    with pytest.raises(ValueError):
        chemistry.reactions_from_formulae(["Xx2O"], ["unknown"])


@pytest.mark.parametrize("Kd", [0.01, 0.5, 1.0, 2.0, 100.0])
def test_partitioning_bulk_balance(Kd):
    bulk = {"Mg": 0.8, "Fe": 0.2, "Si": 0.6}
    fp, pv = partitioning.calculate_nakajima_fp_pv_partition_coefficient(
        25.0e9, 2000.0, bulk, Kd
    )
    assert_allclose(0.4 * fp + 0.6 * pv, 0.2, atol=1.0e-14)
    assert_allclose(pv * (1.0 - fp) / (fp * (1.0 - pv)), Kd, rtol=1.0e-12)
    assert (
        bm.calculate_nakajima_fp_pv_partition_coefficient
        is partitioning.calculate_nakajima_fp_pv_partition_coefficient
    )


@pytest.mark.parametrize("fe", [0.0, 1.0e-12, 0.2, 1.0 - 1.0e-12, 1.0])
@pytest.mark.parametrize("log_K", [-1000.0, -50.0, 0.0, 50.0, 1000.0])
def test_partitioning_limits(fe, log_K):
    bulk = {"Mg": 1.0 - fe, "Fe": fe, "Si": 0.6}
    p = 25.0e9 - log_K * bm.constants.gas_constant * 2000.0 / 2.0e-7
    fp, pv = bm.calculate_nakajima_fp_pv_partition_coefficient(p, 2000.0, bulk, 1.0)
    assert 0.0 <= fp <= 1.0 and 0.0 <= pv <= 1.0
    assert_allclose(0.4 * fp + 0.6 * pv, fe, atol=2.0e-15)


def test_partitioning_validation():
    function = partitioning.calculate_nakajima_fp_pv_partition_coefficient
    for bulk in (
        {},
        {"Mg": 1.0},
        {"Mg": 1.0, "Si": 1.0},
        {"Mg": -1.0, "Fe": 2.0, "Si": 0.5},
        {"Mg": 1.0, "Fe": np.nan, "Si": 0.5},
    ):
        with pytest.raises(ValueError):
            function(25.0e9, 2000.0, bulk, 0.5)
    bulk = {"Mg": 0.8, "Fe": 0.2, "Si": 0.6}
    for p, t, K in ((np.nan, 2000.0, 0.5), (25.0e9, 0.0, 0.5), (25.0e9, 2000.0, 0.0)):
        with pytest.raises(ValueError):
            function(p, t, bulk, K)


def test_python_burnman_reference():
    burnman = pytest.importorskip("burnman")
    from burnman.tools import chemistry as reference
    from burnman.tools.partitioning import (
        calculate_nakajima_fp_pv_partition_coefficient as reference_partition,
    )

    for temperature in (600.0, 700.0, 800.0):
        p = chemistry.equilibrium_pressure(
            [bm.minerals.HP_2011_ds62.sill(), bm.minerals.HP_2011_ds62.ky()],
            [-1, 1],
            temperature,
        )
        expected = reference.equilibrium_pressure(
            [burnman.minerals.HP_2011_ds62.sill(), burnman.minerals.HP_2011_ds62.ky()],
            [-1, 1],
            temperature,
        )
        assert_allclose(p, expected, atol=2.0, rtol=1.0e-9)
    pressures = np.array([1.0e5, 10.0e9, 30.0e9])
    actual = chemistry.hugoniot(
        bm.minerals.SLB_2011.periclase(), 1.0e5, 300.0, pressures
    )
    expected = reference.hugoniot(
        burnman.minerals.SLB_2011.periclase(), 1.0e5, 300.0, pressures
    )
    assert_allclose(actual, expected, rtol=1.0e-8, atol=1.0e-9)
    bulk = {"Mg": 0.8, "Fe": 0.2, "Si": 0.6}
    for p in (10.0e9, 25.0e9, 100.0e9):
        assert_allclose(
            bm.calculate_nakajima_fp_pv_partition_coefficient(p, 2000.0, bulk, 0.5),
            reference_partition(p, 2000.0, bulk, 0.5),
            rtol=1.0e-12,
        )
    matrix = np.array([[1, 0], [0, 1], [1, 1], [2, 1], [1, 2]], dtype=float)
    actual = chemistry.reactions_from_stoichiometric_matrix(matrix)
    expected = np.asarray(
        reference.reactions_from_stoichiometric_matrix(matrix), dtype=float
    )
    assert {tuple(row) for row in actual} == {tuple(row) for row in expected}
