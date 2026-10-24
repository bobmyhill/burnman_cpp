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

"""Published SLB catalogues, parameter metadata and legacy data limits."""

import importlib

import numpy as np
import pytest

import burnman_cpp as bm
from burnman_cpp import minerals

DATASETS = {"SLB_2005": 7, "SLB_2011_ZSB_2013": 7, "SLB_2022": 153}


@pytest.mark.parametrize("dataset,count", DATASETS.items())
def test_public_names_and_independent_materials(dataset, count):
    module = importlib.import_module(f"burnman_cpp.minerals.{dataset}")
    assert module is getattr(minerals, dataset)
    names = minerals.available_phase_names(dataset)
    assert len(names) == count
    for name in names:
        first, second = getattr(module, name)(), getattr(module, name)()
        first.set_state(10e9, 800)
        volume = first.molar_volume
        second.set_state(80e9, 2000)
        first.reset_cache()
        assert first.molar_volume == pytest.approx(volume, rel=1e-12)
        assert np.isfinite(
            [
                first.density,
                first.molar_heat_capacity_p,
                first.molar_heat_capacity_v,
                first.thermal_expansivity,
                first.isothermal_bulk_modulus_reuss,
                first.isentropic_bulk_modulus_reuss,
                first.shear_modulus,
            ]
        ).all(), f"{dataset}.{name}"
        assert volume > 0, f"{dataset}.{name}"
        assert np.isfinite(
            [
                first.p_wave_velocity,
                first.bulk_sound_velocity,
                first.shear_wave_velocity,
            ]
        ).all(), f"{dataset}.{name}"
        if dataset != "SLB_2022":
            assert np.isnan(first.params.F_0)
            assert np.isnan(first.molar_gibbs)
            assert np.isnan(first.molar_enthalpy)
        else:
            assert np.isfinite(first.molar_gibbs)
        if dataset == "SLB_2011_ZSB_2013":
            assert first.params.formula is None
            with pytest.raises(ValueError, match="no formula"):
                _ = first.formula


def test_legacy_aliases_and_uncertainty_copies():
    for module in [minerals.SLB_2005, minerals.SLB_2011_ZSB_2013]:
        for element in ["mg", "fe"]:
            original = getattr(module, f"{element}_perovskite")()
            alias = getattr(module, f"{element}_bridgmanite")()
            assert original.name == alias.name
            assert original.params.K_0 == alias.params.K_0
    phase = minerals.SLB_2011_ZSB_2013.mg_perovskite()
    assert phase.uncertainties["err_K_0"] == 3e9
    assert phase.uncertainties["err_Debye_0"] == 5.0
    view = phase.uncertainties
    view["err_K_0"] = -1
    assert phase.uncertainties["err_K_0"] == 3e9
    copied = phase.copy()
    copied.uncertainties = {"err_K_0": 1e9}
    assert phase.uncertainties["err_K_0"] == 3e9
    copied.uncertainties = None
    assert not hasattr(copied, "uncertainties")
    assert not hasattr(minerals.SLB_2005.stishovite(), "uncertainties")


def test_slb2022_cell_metadata_does_not_rescale_molar_properties():
    phase = minerals.SLB_2022.ab()
    assert phase.params.Z == 4.0
    params = phase.params
    params.Z = 8.0
    other = bm.Mineral(params)
    for mineral in [phase, other]:
        mineral.set_state(10e9, 1000)
    assert other.molar_volume == phase.molar_volume
    assert other.molar_gibbs == phase.molar_gibbs
    assert bm.mineral_params({"Z": 4.0}).Z == 4.0


def test_slb2022_solution_selection():
    phases, metadata = minerals.load_dataset("SLB_2022")
    assert metadata["dataset"].endswith("SLB_2022")
    names = [phase.name for phase in phases]
    assert len(names) == len(set(names))
    assert any(isinstance(phase, bm.Solution) for phase in phases)
    olivine = minerals.SLB_2022.olivine()
    olivine.set_composition([0.9, 0.1])
    olivine.set_state(10e9, 1200)
    np.testing.assert_allclose(
        [olivine.formula["Mg"], olivine.formula["Fe"]], [1.8, 0.2]
    )


@pytest.mark.parametrize(
    "name,critical_volume,entropy,critical_temperature",
    [("neph", 0.8, 10.0, 467.0), ("qtz", 1.35936, 5.76, 847.0)],
)
def test_landau_volume_matches_hefesto_units(
    name, critical_volume, entropy, critical_temperature
):
    phase = getattr(minerals.SLB_2022, name)()
    phase.set_state(10e9, 800)
    # HeFESTo uses GPa and cm^3/mol; both volumes are per chemical formula mole.
    tc = critical_temperature + critical_volume * 10 / (0.001 * entropy)
    q_squared = np.sqrt((tc - 800) / critical_temperature)
    expected_excess_volume = -critical_volume / 1e6 * (q_squared - 1)
    assert phase.get_property_modifiers()["dGdP"] == pytest.approx(
        expected_excess_volume, abs=1e-16
    )
    assert phase.molar_volume > 0
    assert phase.density > 0
    assert np.isfinite(phase.shear_wave_velocity)


@pytest.mark.parametrize("dataset", DATASETS)
def test_adapter_retains_metadata_and_undefined_energies(dataset):
    burnman = pytest.importorskip("burnman")
    from burnman_cpp.adapters import from_burnman

    name = "ab" if dataset == "SLB_2022" else "mg_perovskite"
    reference = getattr(getattr(burnman.minerals, dataset), name)()
    adapted = from_burnman(reference)
    native = getattr(getattr(minerals, dataset), name)()
    for phase in [adapted, native]:
        phase.set_state(25e9, 1500)
    assert adapted.molar_volume == native.molar_volume
    np.testing.assert_allclose(adapted.molar_gibbs, native.molar_gibbs)
    if "Z" in reference.params:
        assert adapted.params.Z == reference.params["Z"]
    if hasattr(reference, "uncertainties"):
        assert adapted.uncertainties == reference.uncertainties
