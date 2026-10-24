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

"""Geotherm tabulations, pressure conversion and constant-entropy profiles."""

import gc
from pathlib import Path
import weakref

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

import burnman_cpp as bm
from burnman_cpp.classes import geotherm

TABLES = {
    "BrownShankland": "brown_81",
    "Anderson": "anderson_82",
    "StaceyContinental": "Stacey_1977_continents",
    "StaceyOceanic": "Stacey_1977_oceans",
    "Katsura2022": "Katsura_2022",
    "Plesa2022Mars6cm3": "Plesa_2022_Mars_V_6cm3",
    "Anzellini2013": "Anzellini_2013_PT",
}


def periclase_params():
    return dict(
        equation_of_state="slb3",
        formula={"Mg": 1.0, "O": 1.0},
        V_0=11.24e-6,
        K_0=161.0e9,
        Kprime_0=3.8,
        G_0=131.0e9,
        Gprime_0=2.1,
        molar_mass=0.0403,
        n=2.0,
        Debye_0=773.0,
        grueneisen_0=1.5,
        q_0=1.5,
        eta_s_0=2.8,
    )


@pytest.mark.parametrize(
    "name,depths,expected",
    [
        ("BrownShankland", [600e3], [1825.0]),
        ("Anderson", [621e3], [1805.0]),
        ("StaceyContinental", [11e3, 620e3], [540.0, 2225.0]),
        ("StaceyOceanic", [11e3, 620e3], [550.0, 2225.0]),
        ("Katsura2022", [100e3, 400e3], [1672.0, 1796.0]),
        ("Plesa2022Mars6cm3", [173e3, 320e3], [1028.0, 1489.0]),
        ("Anzellini2013", [100e3, 700e3], [1603.017, 1952.184]),
    ],
)
def test_reference_values(name, depths, expected):
    assert_allclose(
        getattr(geotherm, name)().temperatures(depths), expected, atol=0.001, rtol=0
    )


@pytest.mark.parametrize("name", TABLES)
def test_bundled_tables_and_metadata(name):
    assert bm.geotherm is geotherm
    model = getattr(geotherm, name)()
    resource = (
        Path(__file__).resolve().parents[2]
        / "src/data/input_geotherm"
        / (TABLES[name] + ".txt")
    )
    table = np.loadtxt(resource)
    coordinates = (
        model.pressures_array if name == "Anzellini2013" else model.depths_array
    )
    assert_array_equal(coordinates, table[:, 0])
    assert_array_equal(model.temperatures_array, table[:, 1])
    if name == "Anzellini2013":
        assert model.minimum_pressure == table[0, 0]
        assert model.maximum_pressure == table[-1, 0]
        assert model.depth_to_pressure_model is bm.seismic.prem_model
    else:
        assert model.minimum_depth == table[0, 0]
        assert model.maximum_depth == table[-1, 0]


def test_custom_depth_table_discontinuities_and_array_shapes():
    model = geotherm.Geotherm([0, 10, 10, 20], [100.0, 200.0, 400.0, 601.0])
    assert model.temperatures(10) == 200.0
    assert model.temperatures(10.001) > 400.0
    assert_allclose(
        model.temperatures(np.array([[0, 5], [15, 20]])),
        [[100.0, 150.0], [500.5, 601.0]],
    )
    assert model.temperatures([]).shape == (0,)
    assert model.temperatures(np.empty((2, 0))).shape == (2, 0)
    assert model.temperatures([15]).dtype.kind == "f"
    for value in [-1.0, 21.0, np.inf, np.nan]:
        with pytest.raises(ValueError):
            model.temperatures(value)
    assert geotherm.Geotherm([0], [300]).temperatures(0) == 300.0
    with pytest.raises(ValueError):
        geotherm.Geotherm([], [])
    with pytest.raises(ValueError):
        geotherm.Geotherm([1, 0], [300, 400])
    with pytest.raises(ValueError):
        geotherm.Geotherm([0, 1], [300])


def test_pressure_conversion_custom_models_and_lifetime():
    class LinearPressure:
        def pressure(self, depths):
            return np.asarray(depths) * 1e6

    conversion = LinearPressure()
    reference = weakref.ref(conversion)
    model = geotherm.GeothermFromPressures([0.0, 1e9], [300.0, 1300.0], conversion)
    assert model.depth_to_pressure_model is conversion
    del conversion
    gc.collect()
    assert reference() is not None
    assert model.temperatures(500.0) == 800.0
    assert_allclose(
        model.temperatures_from_pressures([[0, 0.5e9], [0.75e9, 1e9]]),
        [[300, 800], [1050, 1300]],
    )
    with pytest.raises(ValueError):
        model.temperatures_from_pressures(1.1e9)
    model.depth_to_pressure_model = bm.seismic.prem_model
    assert model.depth_to_pressure_model is bm.seismic.prem_model
    assert_allclose(
        model.temperatures([0, 1e3]),
        model.temperatures_from_pressures(bm.seismic.prem_model.pressure([0, 1e3])),
    )
    with pytest.raises(TypeError):
        model.depth_to_pressure_model = object()
    with pytest.raises(TypeError):
        geotherm.Anzellini2013(object())

    class WrongShape:
        def pressure(self, depths):
            return np.array([0.0])

    with pytest.raises(ValueError, match="wrong number"):
        geotherm.Anzellini2013(WrongShape()).temperatures([0.0, 1.0])


def test_python_subclasses_dispatch_through_native_base():
    class Custom(geotherm.GeothermFromPressures):
        def temperatures_from_pressures(self, pressures):
            return 1000.0 + np.asarray(pressures) / 1e9

    model = Custom([0.0, 1e9], [300.0, 1300.0])
    assert_allclose(
        model.temperatures([0.0, 1e6]),
        1000.0 + bm.seismic.prem_model.pressure([0.0, 1e6]) / 1e9,
    )

    class Modified(geotherm.BrownShankland):
        def temperatures(self, depths):
            return super().temperatures(depths) + 10.0

    assert_allclose(geotherm.Geotherm.temperatures(Modified(), [600e3]), [1835.0])

    class CustomSeismic(bm.seismic.Seismic1DModel):
        def pressure(self, depths):
            return np.asarray(depths) * 1e6

    model.depth_to_pressure_model = CustomSeismic()
    gc.collect()
    assert model.temperatures(500.0) == 1000.5


def test_adiabat_reference_and_anchor_state():
    rock = bm.Mineral(periclase_params())
    rock.set_state(1.0e9, 300.0)
    result = geotherm.adiabatic_profile([100e9, 150e9], rock, 1500.0)
    # Historical upstream regression; live-reference parity below is stricter.
    assert_allclose(result, [1500.0, 1650.22034002], rtol=0, atol=1e-4)
    assert rock.pressure == 100e9
    assert rock.temperature == 1500.0
    lower_anchor = geotherm.adiabatic_profile([100e9, 50e9], rock, 1500.0)[-1]
    assert_allclose(
        geotherm.adiabatic_profile([100e9, 150e9], rock, lower_anchor, 50e9),
        result,
        rtol=1e-11,
    )
    assert geotherm.adiabatic_profile(100e9, rock, 1500.0) == 1500.0
    assert geotherm.adiabatic_profile([], rock, 1500.0).size == 0
    assert rock.pressure == 100e9
    with pytest.raises(ValueError):
        geotherm.adiabatic_profile([100e9], rock, -1.0)
    with pytest.raises(ValueError):
        geotherm.adiabatic_profile([np.nan], rock, 1500.0)


@pytest.mark.parametrize("kind", ["mineral", "assemblage"])
def test_adiabats_conserve_entropy_and_allow_unordered_pressures(kind):
    rock = bm.Mineral(periclase_params())
    if kind == "assemblage":
        from burnman_cpp.minerals import SLB11

        rock = bm.Assemblage([rock, SLB11.forsterite()], [0.4, 0.6])
    pressures = np.array([100e9, 150e9, 50e9, 100e9, 125e9, 125e9])
    model = geotherm.AdiabaticGeotherm(rock, 1500.0, 100e9)
    temperatures = model.temperatures_from_pressures(pressures)
    assert rock.pressure == 100e9
    assert rock.temperature == 1500.0
    entropy = rock.molar_entropy
    for pressure, temperature in zip(pressures, temperatures):
        rock.set_state(pressure, temperature)
        assert_allclose(rock.molar_entropy, entropy, rtol=1e-12, atol=1e-10)
    depths = bm.seismic.prem_model.depth(pressures)
    assert_allclose(model.temperatures(depths), temperatures, rtol=1e-11)
    assert model.material is rock
    assert model.T_0 == 1500.0
    assert model.P_0 == 100e9


@pytest.mark.parametrize("name", TABLES)
def test_optional_upstream_tabulated_parity(name):
    burnman = pytest.importorskip("burnman")
    model = getattr(geotherm, name)()
    reference = getattr(burnman.geotherm, name)()
    if name == "Anzellini2013":
        x = model.pressures_array
        query = np.concatenate((x, np.linspace(x[0], x[-1], 201)))
        assert_allclose(
            model.temperatures_from_pressures(query),
            reference.temperatures_from_pressures(query),
            rtol=1e-14,
        )
    else:
        x = model.depths_array
        query = np.concatenate((x, np.linspace(x[0], x[-1], 201)))
        assert_allclose(
            model.temperatures(query), reference.temperatures(query), rtol=1e-14
        )


def test_optional_upstream_adiabat_parity():
    burnman = pytest.importorskip("burnman")
    native = bm.Mineral(periclase_params())
    reference = burnman.Mineral(periclase_params())
    pressures = np.array([25e9, 50e9, 100e9, 150e9, 100e9, 5e9])
    expected = burnman.geotherm.adiabatic_profile(pressures, reference, 1500.0, 100e9)
    assert_allclose(
        geotherm.adiabatic_profile(pressures, native, 1500.0, 100e9),
        expected,
        rtol=1e-10,
    )
    assert native.pressure == reference.pressure == 100e9
    assert native.temperature == reference.temperature == 1500.0
