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

"""Holland-Powell endmembers, fluids, dry igneous and metapelite catalogues."""

import importlib

import numpy as np
import pytest

import burnman_cpp as bm
from burnman_cpp import minerals

DATASETS = {
    "HHPH_2013": 43,
    "HP_2011_fluids": 7,
    "ig50NCKFMASTOCr": 45,
    "mp50KFMASH": 23,
}


@pytest.mark.parametrize("dataset,count", DATASETS.items())
def test_public_names_and_independent_materials(dataset, count):
    module = importlib.import_module(f"burnman_cpp.minerals.{dataset}")
    assert module is getattr(minerals, dataset)
    names = minerals.available_phase_names(dataset)
    assert len(names) == count
    for name in names:
        first, second = getattr(module, name)(), getattr(module, name)()
        first.set_state(1.0e5, 800.0)
        gibbs = first.molar_gibbs
        volume = first.molar_volume
        if isinstance(second, bm.Solution):
            fractions = np.arange(1.0, second.n_endmembers + 1)
            second.set_composition(fractions / fractions.sum())
        second.set_state(1.0e9, 1200.0)
        first.reset_cache()
        assert first.molar_gibbs == pytest.approx(gibbs, rel=1e-12)
        assert first.molar_volume == pytest.approx(volume, rel=1e-12)
        assert np.isfinite(
            [
                first.molar_gibbs,
                first.molar_entropy,
                first.molar_volume,
                first.molar_heat_capacity_p,
            ]
        ).all(), f"{dataset}.{name}"


def test_high_pressure_endmember_uncertainties():
    phase = minerals.HHPH_2013.fo()
    assert phase.formula == {"Mg": 2.0, "Si": 1.0, "O": 4.0}
    assert phase.params.H_0 == -2172450.0
    assert phase.uncertainties == {"err_H_0": 530.0}
    modified = phase.uncertainties
    modified["err_H_0"] = 0.0
    assert phase.uncertainties["err_H_0"] == 530.0


@pytest.mark.parametrize("name", ["CO2", "CH4", "O2", "H2", "S2", "H2S"])
def test_gas_coefficients_and_fluid_shear(name):
    phase = getattr(minerals.HP_2011_fluids, name)()
    assert phase.params.equation_of_state == bm.EOSType.CORK
    assert isinstance(phase.params.cork_params, bm.CorkParams)
    assert phase.params.cork_params.b == 0.918301
    assert len(phase.params.dcorkdT) == 4
    phase.set_state(1.0e5, 800.0)
    assert phase.molar_volume > 0.0
    assert phase.shear_modulus == 0.0
    assert phase.shear_wave_velocity == 0.0


def test_water_uses_hp2011_thermal_reference():
    phase = minerals.HP_2011_fluids.H2O()
    reference = bm.water_fluid()
    assert phase.name == "H2O"
    for material in (phase, reference):
        material.set_state(1.0e8, 800.0)
    assert phase.formula == {"H": 2.0, "O": 1.0}
    assert phase.molar_gibbs == reference.molar_gibbs
    assert phase.molar_volume == reference.molar_volume


@pytest.mark.parametrize(
    "dataset,names",
    [
        ("HHPH_2013", ["fo", "fa"]),
        ("HP_2011_fluids", ["H2O", "CO2"]),
        ("ig50NCKFMASTOCr", ["ol", "opx", "cpx"]),
        ("mp50KFMASH", ["g", "liq"]),
    ],
)
def test_named_phase_selection(dataset, names):
    phases, metadata = minerals.load_dataset(dataset, phase_names=names)
    assert metadata["dataset"].endswith(f".{dataset}")
    assert metadata["phase_names"] == names
    assert len(phases) == len(names)
    for phase in phases:
        phase.set_state(1.0e8, 800.0)
        assert np.isfinite(phase.molar_gibbs)


def test_metapelite_and_dry_igneous_solution_definitions():
    garnet = minerals.mp50KFMASH.g()
    assert garnet.n_endmembers == 2
    garnet.set_composition([0.4, 0.6])
    assert garnet.formula["Mg"] == pytest.approx(1.2)
    assert garnet.formula["Fe"] == pytest.approx(1.8)
    assert minerals.mp50KFMASH.liq().n_endmembers == 6
    opx = minerals.ig50NCKFMASTOCr.opx()
    assert opx.n_endmembers == 9
    assert opx.endmember_names == [
        "en",
        "fs",
        "fm",
        "odi",
        "mgts",
        "cren",
        "obuf",
        "mess",
        "ojd",
    ]
    phases, metadata = minerals.load_dataset("mp50KFMASH")
    assert "g" in metadata["phase_names"]
    assert "liq" in metadata["phase_names"]
    assert len({phase.name for phase in phases}) == len(phases)
