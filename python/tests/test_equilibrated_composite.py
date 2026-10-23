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

"""Full phase discovery through a Composite-compatible material interface."""

from types import ModuleType

import numpy as np
from numpy.testing import assert_allclose
import pytest

import burnman_cpp as bm
from burnman_cpp.classes.equilibratedcomposite import EquilibratedComposite
from burnman_cpp.constants import gas_constant
from burnman_cpp.minerals import SLB11, SLB24, available_phase_names, load_dataset
from conftest import oxide_params


def oxide_dataset():
    low = bm.Mineral(oxide_params())
    params = oxide_params(energy_shift=100000.0)
    params["name"] = "high energy MgO"
    return [bm.Mineral(params), low]


def test_single_phase_matches_node_and_composite_properties_without_mutating_candidates():
    phases = oxide_dataset()
    bulk = bm.Composition({"MgO": 2.0}, "molar")
    rock = EquilibratedComposite(bulk, phases)
    assert isinstance(rock, bm.Assemblage)
    assert isinstance(rock, bm.Material)
    assert not rock.has_state()
    assert not rock.equilibrated
    with pytest.raises(RuntimeError, match="set_state"):
        _ = rock.molar_mass
    rock.set_state(4e9, 1000.0)
    node = bm.stable_equilibrium(bulk.atomic_composition, phases, 4e9, 1000.0)
    assert node.success, node.message
    assert rock.equilibrated
    assert rock.n_phases == 1
    assert rock.n_moles == pytest.approx(2.0)
    assert rock.equilibrium_state.phases[0].candidate_index == 1
    assert rock.equilibrium_state.gibbs == pytest.approx(node.gibbs)
    assert rock.n_moles * rock.S == pytest.approx(node.entropy)
    assert rock.n_moles * rock.V == pytest.approx(node.volume)
    assert rock.molar_fractions == pytest.approx([1.0])
    assert all(not phase.has_state() for phase in phases)
    ordinary = rock.snapshot()
    names = bm.thermodynamic_property_names()
    expected = bm.assemblage_properties(ordinary, names)
    for name in names:
        assert getattr(rock, name) == pytest.approx(expected[name], rel=1e-10)
    assert rock.rho == rock.density
    assert rock.K_S == rock.isentropic_bulk_modulus_reuss
    assert rock.phase_volumes.sum() == pytest.approx(rock.V)
    assert rock.phase_masses.sum() == pytest.approx(rock.molar_mass)
    assert rock.atomic_composition == bulk.atomic_composition
    bulk.renormalize("molar", "total", 5.0)
    assert rock.atomic_composition["Mg"] == 2.0
    phases[1].set_state(7e9, 1500.0)
    assert rock.pressure == 4e9
    assert ordinary.get_phase(0) is not rock.get_phase(0)
    ordinary.set_state(5e9, 1100.0)
    assert rock.temperature == 1000.0


@pytest.mark.parametrize("warm_start, searches", [(True, 1), (False, 2)])
def test_set_state_reuses_equilibrium_when_enabled(capfd, warm_start, searches):
    settings = bm.PhaseDiagramSettings()
    settings.verbose = True
    rock = EquilibratedComposite(
        {"MgO": 1.0}, oxide_dataset(), settings=settings, warm_start=warm_start
    )
    rock.set_state(4e9, 1000.0)
    rock.set_state(4.01e9, 1001.0)
    assert rock.warm_start is warm_start
    assert capfd.readouterr().err.count("LP 0 ") == searches
    assert rock.equilibrium_state.mass_balance_error < 1e-8
    assert rock.equilibrium_state.minimum_affinity >= -settings.affinity_tolerance
    rock.warm_start = False
    rock.set_state(4.02e9, 1002.0)
    assert capfd.readouterr().err.count("LP 0 ") == 1


def test_new_phases_enter_and_leave_in_both_directions():
    phases = [SLB24.qtz(), SLB24.coes(), SLB24.st()]
    rock = EquilibratedComposite({"SiO2": 1.0}, phases)
    names = []
    for pressure in [1e9, 20e9, 1e9]:
        rock.set_state(pressure, 1000.0)
        node = bm.stable_equilibrium(rock.atomic_composition, phases, pressure, 1000.0)
        assert node.success, node.message
        assert [p.name for p in rock.phases] == [p.name for p in node.phases]
        assert rock.n_moles * rock.molar_gibbs == pytest.approx(node.gibbs, rel=1e-10)
        names.append(rock.phases[0].name)
    assert names[0] != names[1]
    assert names[0] == names[2]


def test_dataset_modules_strings_and_factory_selection():
    for dataset in [SLB11, "SLB11"]:
        rock = EquilibratedComposite(
            {"Mg2SiO4": 1.0},
            dataset,
            phase_names=["mg_fe_olivine", "mg_fe_wadsleyite", "mg_fe_ringwoodite"],
        )
        rock.set_state(1e9, 1400.0)
        assert rock.n_phases == 1
        assert rock.equilibrium_state.mass_balance_error < 1e-8
        assert len(rock.candidate_phases) == 3
        assert len(rock.dataset_metadata["phase_names"]) == 3
    module = ModuleType("custom")
    module.phase = lambda: bm.Mineral(oxide_params())
    rock = EquilibratedComposite({"MgO": 1.0}, module, phase_names=["phase"])
    rock.set_state(1e9, 1000.0)
    assert rock.n_moles == pytest.approx(1.0)


@pytest.mark.parametrize("dataset", [SLB11, "SLB11"])
def test_available_phase_names_can_be_selected_without_a_material_state(dataset):
    names = EquilibratedComposite.available_phase_names(dataset)
    assert names == sorted(set(names))
    assert {
        "mg_fe_olivine",
        "mg_fe_wadsleyite",
        "mg_fe_ringwoodite",
        "forsterite",
        "fo",
    }.issubset(names)
    phases, metadata = load_dataset(dataset, phase_names=names)
    assert metadata["phase_names"] == names
    assert len(phases) == len(names)
    assert all(isinstance(phase, bm.Material) for phase in phases)
    assert all(not phase.has_state() for phase in phases)


def test_available_phase_names_exclude_relaxed_materials():
    names = available_phase_names(SLB24)
    assert {"ferropericlase", "bridgmanite"}.issubset(names)
    assert "ferropericlase_relaxed" not in names
    assert "bridgmanite_relaxed" not in names
    phases, _ = load_dataset(SLB24, phase_names=names)
    assert all(not isinstance(phase, bm.RelaxedSolution) for phase in phases)
    _, default_selection = load_dataset(SLB24)
    assert set(default_selection["phase_names"]).issubset(names)


def test_coexisting_solution_copies_and_composition_changes():
    mg = bm.Mineral(oxide_params())
    fe = bm.Mineral(oxide_params("Fe"))
    solution = bm.Solution(
        bm.SymmetricRegularSolution(
            [(mg, "[Mg]O"), (fe, "[Fe]O")], energy_interaction=[[13000.0]]
        ),
        [0.5, 0.5],
        name="oxide solution",
    )
    rock = EquilibratedComposite({"MgO": 0.5, "FeO": 0.5}, [solution])
    rock.set_state(1e9, 500.0)
    assert rock.n_phases == 2
    compositions = [phase.molar_fractions for phase in rock.phases]
    assert abs(compositions[0][0] - compositions[1][0]) > 0.5
    assert_allclose(
        rock.n_moles * np.array([rock.formula[el] for el in ["Mg", "Fe", "O"]]),
        [0.5, 0.5, 1.0],
        atol=1e-8,
    )
    assert len(rock.equilibrium_state.phases) == 2
    rock.set_state(1e9, 1500.0)
    assert rock.n_phases == 1
    assert_allclose(rock.phases[0].molar_fractions, [0.5, 0.5], atol=1e-8)
    rock.set_state(1e9, 500.0)
    assert rock.n_phases == 2
    assert_allclose(
        sorted(phase.molar_fractions[0] for phase in rock.phases),
        sorted(composition[0] for composition in compositions),
        atol=1e-8,
    )


def test_warm_start_rejects_a_weak_solution_saddle():
    rt = gas_constant * 1000.0
    solution = bm.Solution(
        bm.SymmetricRegularSolution(
            [
                (bm.Mineral(oxide_params()), "[Mg]O"),
                (bm.Mineral(oxide_params("Fe")), "[Fe]O"),
            ],
            energy_interaction=[[2.0 * rt - 1000.0]],
            volume_interaction=[[1e-6]],
        ),
        [0.5, 0.5],
    )
    rock = EquilibratedComposite({"MgO": 0.5, "FeO": 0.5}, [solution])
    rock.set_state(0.9e9, 1000.0)
    assert rock.n_phases == 1
    rock.set_state(1.08e9, 1000.0)
    assert rock.n_phases == 2
    solution.set_state(1.08e9, 1000.0)
    assert rock.molar_gibbs < solution.molar_gibbs - 1e-5
    cold = EquilibratedComposite(rock.composition, [solution], warm_start=False)
    cold.set_state(1.08e9, 1000.0)
    assert_allclose(
        sorted(phase.molar_fractions[0] for phase in rock.phases),
        sorted(phase.molar_fractions[0] for phase in cold.phases),
        atol=1e-8,
    )


def test_relaxed_response_matches_equilibrium_finite_difference_and_frozen_snapshot():
    rock = EquilibratedComposite({"MgO": 0.9, "FeO": 0.1}, [SLB24.ferropericlase()])
    volume = []
    for temperature in [1999.9, 2000.0, 2000.1]:
        rock.set_state(110e9, temperature)
        volume.append(rock.n_moles * rock.molar_volume)
        if temperature == 2000.0:
            alpha = rock.alpha
            assert bm.assemblage_properties(rock, ["thermal_expansivity"])[
                "thermal_expansivity"
            ] == pytest.approx(alpha)
            frozen = rock.snapshot()
            basis = rock.equilibrium_state.phases[0].composition_basis
            assert basis is not None
    assert alpha == pytest.approx((volume[2] - volume[0]) / 0.2 / volume[1], rel=1e-5)
    rock.set_state(110e9, 2000.0)
    rock.relaxed = False
    assert rock.alpha == pytest.approx(frozen.thermal_expansivity)
    assert bm.assemblage_properties(rock, ["thermal_expansivity"])[
        "thermal_expansivity"
    ] == pytest.approx(alpha)
    rock.set_averaging_scheme("Reuss")
    rock.set_state(110e9, 2000.0)
    assert rock.averaging_scheme == bm.AveragingType.Reuss
    rock.relaxed = True
    assert rock.alpha == pytest.approx(alpha, rel=1e-9)


def test_evaluate_restores_state_and_accepts_original_burnman_property_names():
    rock = EquilibratedComposite({"MgO": 1.0}, oxide_dataset())
    pressure = np.array([[1e9, 2e9], [3e9, 4e9]])
    temperature = np.full_like(pressure, 1000.0)
    values = rock.evaluate(["rho", "S", "v_p"], pressure, temperature)
    assert values.shape == (3, 2, 2)
    assert np.isfinite(values).all()
    assert not rock.has_state()
    assert rock.equilibrium_state is None
    rock.set_state(5e9, 1100.0)
    expected = rock.rho
    assert_allclose(rock.evaluate(["density"], pressure, temperature)[0], values[0])
    assert rock.pressure == 5e9
    assert rock.temperature == 1100.0
    assert rock.rho == expected
    assert rock.evaluate(["rho"], [], []).shape == (1, 0)
    with pytest.raises(ValueError):
        rock.evaluate(["bad"], [1e9], [1000.0])
    with pytest.raises(ValueError):
        rock.evaluate(["rho"], [1e9], [1000.0, 1100.0])
    with pytest.raises(ValueError):
        rock.evaluate(["rho"], [1e9, -1.0], [1000.0, 1000.0])
    assert rock.pressure == 5e9


def test_evaluate_warm_starts_match_independent_equilibria(capfd):
    settings = bm.PhaseDiagramSettings()
    settings.verbose = True
    rock = EquilibratedComposite({"MgO": 1.0}, oxide_dataset(), settings=settings)
    pressures = np.array([[1e9, 1.01e9], [1.02e9, 1.03e9]])
    temperatures = np.array([[1000.0, 1001.0], [1002.0, 1003.0]])
    names = ["rho", "S", "K_S", "alpha"]
    warm = rock.evaluate(names, pressures, temperatures)
    assert capfd.readouterr().err.count("LP 0 ") == 1
    assert not rock.equilibrated
    # An evaluation from an unset material must not leave a hidden warm seed.
    rock.warm_start = False
    cold = rock.evaluate(names, pressures, temperatures)
    assert capfd.readouterr().err.count("LP 0 ") == pressures.size
    assert_allclose(warm, cold, rtol=1e-10)
    rock.warm_start = True
    rock.set_state(1e9, 1000.0)
    assert capfd.readouterr().err.count("LP 0 ") == 1


def test_failed_solve_preserves_accepted_state_and_failure_diagnostics():
    settings = bm.PhaseDiagramSettings()
    settings.required_eos_phases = [SLB11.forsterite().name]
    rock = EquilibratedComposite(
        {"Mg2SiO4": 1.0}, [SLB11.forsterite()], settings=settings
    )
    rock.set_state(1e9, 1000.0)
    density = rock.density
    with pytest.raises(RuntimeError):
        rock.set_state(0.0, 100000.0)
    assert rock.last_equilibrium_attempt.temperature == 100000.0
    assert (
        not rock.last_equilibrium_attempt.success
        or rock.last_equilibrium_attempt.outside_model_domain
    )
    assert rock.equilibrium_state.success
    assert rock.temperature == 1000.0
    assert rock.density == density
    rock.set_state(1.01e9, 1001.0)
    assert rock.equilibrium_state.success
    assert rock.pressure == 1.01e9
    assert rock.temperature == 1001.0
    with pytest.raises(TypeError):
        rock.set_fractions([1.0])
    with pytest.raises(AttributeError):
        rock.n_moles = 3.0
    with pytest.raises(RuntimeError):
        rock.set_method(bm.EOSType.SLB3)


def test_native_geotherm_calls_full_equilibration_and_restores_anchor():
    rock = EquilibratedComposite({"MgO": 2.0}, oxide_dataset())
    adiabat = bm.geotherm.AdiabaticGeotherm(rock, 1500.0, 100e9)
    pressures = [100e9, 150e9]
    temperatures = adiabat.temperatures_from_pressures(pressures)
    reference = bm.geotherm.adiabatic_profile(
        pressures, bm.Mineral(oxide_params()), 1500.0
    )
    assert_allclose(temperatures, reference, rtol=1e-10)
    assert rock.pressure == 100e9
    assert rock.temperature == 1500.0
    assert rock.n_moles == pytest.approx(2.0)


def test_invalid_bulk_and_candidates_are_rejected():
    for bulk in [{}, {"MgO": -1.0}, {"MgO": float("nan")}]:
        with pytest.raises(ValueError):
            EquilibratedComposite(bulk, oxide_dataset())
    with pytest.raises(ValueError):
        EquilibratedComposite({"MgO": 1.0}, [])
    with pytest.raises(ValueError):
        EquilibratedComposite({"MgO": 1.0}, oxide_dataset(), phase_names=["a"])
