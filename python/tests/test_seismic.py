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

"""Seismic model API, physical integrals and optional upstream comparisons."""

from importlib.resources import files

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.integrate import cumulative_trapezoid

import burnman_cpp as bm
from burnman_cpp.classes import seismic
from burnman_cpp.tools.seismic import attenuation_correction

MODELS = ["PREM", "Slow", "Fast", "STW105", "IASP91", "AK135"]


@pytest.mark.parametrize("name", MODELS)
def test_reference_values(name):
    expected = {
        "PREM": [12817.6924, 6932.8549, 5120.6291],
        "Slow": [12795.360742611414, 6904.329188001315, 5120.6291],
        "Fast": [12795.360742611414, 6941.422556820191, 5120.6291],
        "STW105": [12817.85012987013, 6932.934206525182, 5120.674925562243],
        "IASP91": [12794.4, 6921.0],
        "AK135": [12798.468686868688, 6920.521212121212, 5103.89696969697],
    }
    model = getattr(seismic, name)()
    properties = ["v_p", "v_s"] + ([] if name == "IASP91" else ["density"])
    assert_allclose(
        model.evaluate(properties, [2.0e6])[:, 0], expected[name], rtol=1e-12
    )
    assert isinstance(model.v_p(2.0e6), float)
    assert model.v_p(2.0e6) == model.v_p([2.0e6])[0]
    coordinates = np.array([[1.0e6, 2.0e6], [2.5e6, 2.6e6]])
    assert model.v_s(coordinates).shape == coordinates.shape
    assert model.v_p([]).shape == (0,)
    assert model.evaluate([], []).shape == (0, 0)
    assert model.evaluate(["v_p"]).shape == (1, len(model.internal_depth_list()))


def test_import_paths_and_bundled_tables():
    assert bm.seismic is seismic
    assert isinstance(seismic.prem_model, seismic.PREM)
    for name in [
        "prem",
        "STW105",
        "iasp91",
        "ak135",
        "pwave_fast",
        "pwave_slow",
        "swave_fast",
        "swave_slow",
    ]:
        resource = files("burnman_cpp").joinpath("data", "input_seismic", name + ".txt")
        assert resource.is_file()
    prem = seismic.PREM()
    data = np.loadtxt(files("burnman_cpp").joinpath("data/input_seismic/prem.txt"))
    assert_array_equal(prem.table_depth, data[:, 0])
    assert_array_equal(prem.table_QK, data[:, 6])


def test_discontinuities_bounds_and_moduli():
    model = seismic.PREM()
    depths = model.table_depth
    shifted = model.internal_depth_list()
    duplicate = np.flatnonzero(np.diff(depths) == 0)
    assert_array_equal(shifted[duplicate], depths[duplicate] - 1.0)
    assert_array_equal(shifted[duplicate + 1], depths[duplicate + 1] + 1.0)
    query = np.concatenate(([-1.0], depths, shifted, [7.0e6]))
    assert_allclose(
        model.v_p(query), np.interp(query, depths, model.table_vp), rtol=1e-14
    )
    for interval in [0.0, 0.1, 2.0]:
        bounded = model.internal_depth_list(660e3, 800e3, interval)
        assert np.all(bounded >= 660e3 - interval)
        assert np.all(bounded <= 800e3 + interval)
    assert model.internal_depth_list(1e8, 2e8).size == 0
    z = [1.0e6, 2.0e6, 3.0e6]
    assert_allclose(model.G(z), model.density(z) * model.v_s(z) ** 2)
    assert_allclose(model.K(z), model.density(z) * model.v_phi(z) ** 2)
    assert_allclose(model.depth(model.pressure(z)), z)
    assert_allclose(model.radius(model.pressure(z)), model.earth_radius - np.array(z))
    with pytest.raises(ValueError, match="outside range"):
        model.depth([-1.0])
    with pytest.raises(ValueError, match="Unknown seismic property"):
        model.evaluate(["unknown"])


@pytest.mark.parametrize("name", ["PREM", "STW105", "AK135"])
def test_lazy_gravity_and_pressure_integrals(name):
    model = getattr(seismic, name)()
    r = model.table_radius[::-1]
    density = model.table_density[::-1]
    g = cumulative_trapezoid(
        bm.constants.G * 4.0 * np.pi * density * r**2, x=r, initial=0
    )
    g[1:] /= r[1:] ** 2
    assert_allclose(
        model.gravity(model.table_depth),
        np.interp(model.table_depth, model.table_depth, g[::-1]),
        rtol=2e-14,
    )
    # Repeated depths select the deeper row; integration uses the same lookup.
    z = model.earth_radius - model.table_radius
    expected = cumulative_trapezoid(
        model.gravity(z) * model.table_density, x=z, initial=0
    )
    model.table_pressure = []
    result = model.pressure(z)
    assert_allclose(result, np.interp(z, z, expected), rtol=2e-14)
    assert_allclose(model.table_pressure, expected, rtol=2e-14)
    assert np.isfinite(model.depth([model.table_pressure[-1]])).all()


def test_bullen_matches_nonuniform_numpy_gradient():
    model = seismic.PREM()
    z = np.array([800e3, 900e3, 1150e3, 1500e3, 1900e3, 2200e3, 2600e3])
    phi = model.v_phi(z) ** 2
    kappa = phi * model.density(z)
    expected = np.gradient(kappa, model.pressure(z), edge_order=2) - np.gradient(
        phi, z, edge_order=2
    ) / model.gravity(z)
    assert_allclose(model.bullen(z), expected, rtol=1e-12)
    assert_allclose(model.bullen(z[::-1]), expected[::-1], rtol=1e-12)
    with pytest.raises(ValueError, match="at least four"):
        model.bullen(z[:3])


def test_missing_and_malformed_tables():
    model = seismic.IASP91()
    with pytest.raises(ValueError, match="Density"):
        model.density(100e3)
    with pytest.raises(ValueError):
        model.gravity([0.0])
    with pytest.raises(ValueError):
        model.QK([0.0])
    model.table_vp = [1.0]
    with pytest.raises(ValueError, match="length"):
        model.v_p(0.0)
    model.table_depth = [1.0, 0.0]
    with pytest.raises(ValueError, match="increasing"):
        model.v_p(0.0)
    with pytest.raises(NotImplementedError):
        seismic.Seismic1DModel().v_p(0.0)


def test_custom_python_models_dispatch_through_native_base():
    class Custom(seismic.Seismic1DModel):
        def internal_depth_list(
            self, mindepth=0, maxdepth=1e99, discontinuity_interval=1
        ):
            return np.array([0.0, 1.0, 2.0])

        def v_p(self, depth):
            return np.full_like(depth, 6000.0)

        def v_s(self, depth):
            return np.full_like(depth, 3000.0)

        def density(self, depth):
            return np.full_like(depth, 3300.0)

    model = Custom()
    assert_allclose(model.evaluate(["v_p", "v_s"]), [[6000.0] * 3, [3000.0] * 3])
    assert_allclose(model.G([0.0, 1.0]), 3300.0 * 3000.0**2)

    class CustomTable(seismic.SeismicTable):
        def __init__(self):
            super().__init__()
            self.table_depth = [0.0, 1.0, 2.0]
            self.table_vp = [6000.0, 7000.0, 8000.0]
            self.table_vs = [3000.0, 3500.0, 4000.0]

        def _lookup(self, depth, value_table):
            return np.interp(depth, self.table_depth, value_table) + 10.0

    assert_allclose(CustomTable().evaluate(["v_p"], [0.5, 1.5]), [[6510.0, 7510.0]])

    class ModifiedPREM(seismic.PREM):
        def v_p(self, depth):
            return super().v_p(depth) + 100.0

    assert_allclose(ModifiedPREM().evaluate(["v_p"], [2e6]), [[12917.6924]])


def test_attenuation_scalar_and_broadcasting():
    vp = np.array([[12000.0], [13000.0]])
    vs = np.array([6000.0, 7000.0, 7500.0])
    inputs = (vp.copy(), vs.copy())
    corrected = attenuation_correction(vp, vs, 9000.0, 200.0, 1000.0)
    cot = 1.0 / np.tan(0.3 * np.pi / 2.0)
    qp = 0.75 * (vp / vs) ** 2 * 200.0
    assert_allclose(corrected[0], vp * (1.0 - 0.5 * cot / qp))
    assert_allclose(
        corrected[1], np.broadcast_to(vs * (1.0 - 0.5 * cot / 200.0), (2, 3))
    )
    assert_allclose(corrected[2], 9000.0 * (1.0 - 0.5 * cot / 1000.0))
    assert_array_equal(vp, inputs[0])
    assert_array_equal(vs, inputs[1])
    scalar = attenuation_correction(12000.0, 7000.0, 9000.0, 200.0, 1000.0)
    assert all(np.ndim(value) == 0 for value in scalar)


@pytest.mark.parametrize("name", MODELS)
def test_optional_upstream_parity(name):
    burnman = pytest.importorskip("burnman")
    native = getattr(seismic, name)()
    reference = getattr(burnman.seismic, name)()
    assert_allclose(
        native.internal_depth_list(), reference.internal_depth_list(), atol=0, rtol=0
    )
    for attr in [
        "table_depth",
        "table_radius",
        "table_pressure",
        "table_density",
        "table_vp",
        "table_vs",
        "table_QK",
        "table_QG",
    ]:
        assert_allclose(getattr(native, attr), getattr(reference, attr), rtol=2e-15)
    z = np.linspace(native.table_depth[0], native.table_depth[-1], 301)
    properties = ["v_p", "v_s", "v_phi"]
    if name != "IASP91":
        properties += ["density", "G", "K"]
    if name in ["PREM", "STW105", "AK135"]:
        properties += ["pressure", "gravity", "QK", "QG"]
    assert_allclose(
        native.evaluate(properties, z), reference.evaluate(properties, z), rtol=3e-14
    )
