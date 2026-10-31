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

"""Radial structures, hydrostatic balance, interfaces and independent states."""

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

import burnman_cpp as bm
from conftest import oxide_params
from test_geotherm import periclase_params


def make_layer(n=21, material=None, inner=660e3, outer=2890e3):
    layer = bm.Layer("mantle", np.linspace(inner, outer, n))
    layer.set_material(material or bm.Mineral(dict(periclase_params(), F_0=-569444.6)))
    layer.set_temperature_mode("user-defined", np.full(n, 2000.0))
    layer.set_pressure_mode("self-consistent", gravity_bottom=10.0, pressure_top=22.5e9)
    return layer


def iron(liquid=True):
    params = dict(name="iron", formula={"Fe": 1.0}, molar_mass=0.055845)
    if liquid:
        params.update(
            equation_of_state="bm4",
            V_0=7.95626e-6,
            K_0=109.7e9,
            Kprime_0=4.66,
            Kprime_prime_0=-0.043e-9,
        )
    else:
        params.update(
            equation_of_state="vinet", V_0=6.75e-6, K_0=163.4e9, Kprime_0=5.38, n=1.0
        )
    return bm.Mineral(params)


def make_planet(n=41):
    core = bm.Layer("core", np.linspace(0.0, 3480e3, n))
    core.set_material(iron())
    core.set_temperature_mode("user-defined", np.full(n, 300.0))
    mantle = bm.Layer("mantle", np.linspace(3480e3, 6371e3, n))
    mantle.set_material(bm.minerals.SLB_2011.mg_perovskite())
    mantle.set_temperature_mode("adiabatic", temperature_top=1200.0)
    planet = bm.Planet("earth_like", [mantle, core])
    return planet, core, mantle


@pytest.mark.parametrize(
    "radii", [[], [1.0], [-1.0, 2.0], [2.0, 1.0], [1.0, 1.0], [0.0, np.nan]]
)
def test_layer_rejects_invalid_radii(radii):
    with pytest.raises(ValueError):
        bm.Layer("invalid", radii)


def test_class_modules_and_configuration():
    from burnman_cpp.classes.layer import Layer, BoundaryLayerPerturbation
    from burnman_cpp.classes.planet import Planet

    assert Layer is bm.Layer
    assert Planet is bm.Planet
    assert BoundaryLayerPerturbation is bm.BoundaryLayerPerturbation
    layer = bm.Layer(radii=[0.0, 1.0])
    with pytest.raises(RuntimeError, match="set_material"):
        layer.make()
    layer.set_material(bm.Mineral(periclase_params()))
    with pytest.raises(RuntimeError):
        layer.make()
    with pytest.raises(ValueError):
        layer.set_temperature_mode("wrong")
    with pytest.raises(ValueError):
        layer.set_temperature_mode("user-defined", [300.0])
    with pytest.raises(ValueError):
        layer.set_pressure_mode(gravity_bottom=-1.0, pressure_top=0.0)
    with pytest.raises(ValueError):
        layer.set_pressure_mode(gravity_bottom=0.0, pressure_top=0.0, max_delta=0.0)
    with pytest.raises(RuntimeError, match="make"):
        _ = layer.density


def test_layer_pressure_profiles_aliases_and_repeated_make():
    layer = make_layer()
    layer.make()
    assert layer.converged and layer.made
    assert layer.n_iterations < layer.n_max_iterations
    assert layer.pressures[-1] == 22.5e9
    assert_allclose(layer.gravity[0], 10.0)
    assert np.all(np.diff(layer.pressures) < 0.0)
    assert_array_equal(layer.P, layer.pressures)
    assert_array_equal(layer.rho, layer.density)
    assert_array_equal(layer.v_p, layer.p_wave_velocity)
    for prop in ["mass", "moment_of_inertia"]:
        assert getattr(layer, prop) > 0.0
    for prop in [
        "bullen",
        "brunt_vasala",
        "energy",
        "gibbs",
        "helmholtz",
        "H",
        "V",
        "molar_mass",
        "K_T",
        "K_S",
        "beta_T",
        "beta_S",
        "G",
        "v_phi",
        "gr",
        "alpha",
        "C_p",
        "C_v",
        "isentropic_thermal_gradient",
    ]:
        assert np.isfinite(getattr(layer, prop)).all()
    pressure = layer.pressures.copy()
    layer.make()
    assert_allclose(layer.pressures, pressure, rtol=1e-13)
    copy = layer.radii
    copy[0] = -1.0
    assert layer.radii[0] == 660e3
    layer.reset()
    assert not layer.made
    with pytest.raises(RuntimeError):
        _ = layer.pressures


def test_prescribed_profiles_survive_reset_and_interpolate_states():
    layer = make_layer(n=9)
    pressure = np.linspace(100e9, 20e9, 9)
    temperature = np.linspace(2500.0, 1900.0, 9)
    layer.set_temperature_mode("user-defined", temperature)
    layer.set_pressure_mode("user-defined", pressure, gravity_bottom=10.0)
    layer.make()
    query = np.array([2.5e6, 1.0e6, 1.5e6, 1.0e6])
    values = layer.evaluate(
        ["density", "pressure", "temperature", "depth", "gravity"], query, 6371e3
    )
    assert values.shape == (5, 4)
    assert_array_equal(values[3], 6371e3 - query)
    assert_allclose(values[1], np.interp(query, layer.radii, pressure))
    assert_allclose(values[2], np.interp(query, layer.radii, temperature))
    rock = bm.Mineral(periclase_params())
    for i in range(len(query)):
        rock.set_state(values[1, i], values[2, i])
        assert_allclose(values[0, i], rock.density, rtol=1e-13)
    assert layer.evaluate(["density"]).shape == (9,)
    assert layer.evaluate(["density"], []).shape == (0,)
    assert layer.evaluate([], []).shape == (0, 0)
    with pytest.raises(ValueError):
        layer.evaluate(["depth"])
    with pytest.raises(LookupError):
        layer.evaluate(["density"], [1.0])
    with pytest.raises(ValueError):
        layer.evaluate(["unknown"])
    layer.reset()
    layer.make()
    assert_array_equal(layer.pressures, pressure)
    assert_array_equal(layer.temperatures, temperature)


def test_adiabatic_and_perturbed_temperature_anchors():
    layer = make_layer()
    layer.set_temperature_mode("adiabatic", temperature_top=1900.0)
    layer.make()
    assert_allclose(layer.S, np.full(layer.n_slices, layer.S[-1]), rtol=1e-10)
    assert_allclose(layer.T[-1], 1900.0, atol=1e-8)
    perturbation = np.linspace(200.0, -50.0, layer.n_slices)
    layer.set_temperature_mode(
        "perturbed-adiabatic", perturbation, temperature_top=1900.0
    )
    layer.make()
    assert_allclose(layer.T[-1], 1900.0, atol=1e-8)
    rock = layer.material.copy()
    reference = bm.geotherm.adiabatic_profile(layer.P[::-1], rock, 1950.0)[::-1]
    assert_allclose(layer.T, reference + perturbation, rtol=1e-11)


def test_material_copies_and_sublayers_are_independent():
    a = bm.minerals.SLB_2011.mg_perovskite()
    b = bm.minerals.SLB_2011.periclase()
    mixture = bm.Assemblage([a, b], [0.8, 0.2])
    mixture.set_state(25e9, 1900.0)
    reference = mixture.density
    duplicate = mixture.copy()
    duplicate.set_state(100e9, 3000.0)
    mixture.reset_cache()
    assert_allclose(mixture.density, reference, rtol=1e-14)
    assert mixture.get_phase(0) is not duplicate.get_phase(0)
    layer = make_layer(material=mixture)
    layer.make()
    assert layer.sublayers[0] is not layer.sublayers[1]
    old_last = layer.sublayers[-1].density
    layer.sublayers[0].set_state(5e9, 1000.0)
    layer.sublayers[-1].reset_cache()
    assert_allclose(layer.sublayers[-1].density, old_last, rtol=1e-14)
    solution = bm.minerals.SLB_2024.olivine()
    solution.set_composition([0.7, 0.3])
    solution.set_state(10e9, 1800.0)
    old_density = solution.density
    other = solution.copy()
    other.set_state(20e9, 2500.0)
    solution.reset_cache()
    assert_allclose(solution.density, old_density, rtol=1e-14)


def test_planet_interfaces_order_and_arbitrary_queries():
    planet, core, mantle = make_planet()
    assert not planet.made
    assert planet.layers == [core, mantle]
    assert list(planet) == [core, mantle]
    assert planet.n_slices == core.n_slices + mantle.n_slices
    assert core.n_end == mantle.n_start
    planet.make()
    assert planet.converged
    assert planet.get_layer("core") is core
    assert planet.get_layer_by_radius(3480e3) is core
    assert planet.get_layer_by_radius(5000e3) is mantle
    assert planet.gravity[0] == 0.0
    assert core.gravity[0] == 0.0
    assert planet.P[-1] == 0.0
    assert planet.P[core.n_end - 1] == planet.P[mantle.n_start]
    assert planet.gravity[core.n_end - 1] == planet.gravity[mantle.n_start]
    assert core.pressure_mode == "set-in-planet"
    assert mantle.pressure_mode == "set-in-planet"
    assert_allclose(mantle.S, mantle.S[-1], rtol=1e-10)
    assert_allclose(planet.mass, core.mass + mantle.mass, rtol=1e-14)
    assert_allclose(planet.average_density, planet.mass / planet.volume, rtol=1e-14)
    assert_allclose(
        planet.moment_of_inertia_factor,
        planet.moment_of_inertia / (planet.mass * planet.radius_planet**2),
        rtol=1e-14,
    )
    query = [6000e3, 2000e3, 3480e3, 4000e3, 2000e3]
    rho, depth = planet.evaluate(["rho", "depth"], query)
    for i, radius in enumerate(query):
        expected = planet.get_layer_by_radius(radius).evaluate(["rho"], [radius])
        assert_allclose(rho[i], expected[0], rtol=1e-14)
    assert_array_equal(depth, planet.radius_planet - np.array(query))
    assert planet.evaluate(["rho"], []).shape == (0,)
    with pytest.raises(LookupError):
        planet.get_layer("missing")
    for radius in [-1.0, np.nan, 7000e3]:
        with pytest.raises(LookupError):
            planet.get_layer_by_radius(radius)
    assert "2 layers" in str(planet)
    assert "mantle" in str(mantle)


def test_planet_prescribed_pressures_and_temperature_inheritance():
    lower = bm.Layer("lower", np.linspace(0.0, 1e6, 7))
    upper = bm.Layer("upper", np.linspace(1e6, 2e6, 7))
    for layer in [lower, upper]:
        layer.set_material(bm.Mineral(periclase_params()))
        layer.set_temperature_mode(
            "adiabatic", temperature_top=1600.0 if layer is upper else None
        )
    planet = bm.Planet("prescribed", [upper, lower])
    pressure = np.concatenate([np.linspace(40e9, 20e9, 7), np.linspace(20e9, 1e9, 7)])
    planet.set_pressure_mode("user-defined", pressure)
    planet.make()
    assert_array_equal(planet.pressures, pressure)
    assert_allclose(lower.temperatures[-1], upper.temperatures[0], rtol=1e-12)
    assert_allclose(planet.S, planet.S[-1], rtol=1e-10)
    assert np.isfinite(planet.gravity).all()
    assert planet.gravity[0] == 0.0
    planet.make()
    assert_array_equal(planet.pressures, pressure)
    pressure[7] += 1.0
    with pytest.raises(ValueError, match="continuous"):
        planet.set_pressure_mode("user-defined", pressure)
    bad = bm.Layer("gap", np.linspace(2.1e6, 3e6, 7))
    with pytest.raises(ValueError, match="adjoin"):
        bm.Planet("gap", [lower, bad])
    with pytest.raises(ValueError):
        bm.Planet("empty", [])
    with pytest.raises(ValueError):
        bm.Planet("invalid", [lower], n_max_iterations=0)


@pytest.mark.parametrize("rayleigh", [1e7, 1e16])
def test_boundary_perturbation_values_and_gradients(rayleigh):
    model = bm.BoundaryLayerPerturbation(3480e3, 5711e3, rayleigh, 1000.0, 0.25)
    assert_allclose(model.temperature([3480e3, 5711e3]), [800.0, -200.0], atol=1e-9)
    model.set_model_thermal_gradients(-0.018, -0.0006)
    assert_allclose(model.dTdr([3480e3, 5711e3]), [-0.018, -0.0006], atol=1e-14)
    assert isinstance(model.temperature(3480e3), float)
    assert model.temperature(np.zeros((0, 2))).shape == (0, 2)
    assert model.temperature(np.full((2, 3), 4000e3)).shape == (2, 3)


def test_fourth_order_iron_matches_pressure_bulk_modulus_and_energy_derivatives():
    rock = iron()
    method = bm.make_eos("bm4")
    params = rock.params
    for pressure in [0.0, 50e9, 200e9, 500e9]:
        rock.set_state(pressure, 300.0)
        volume = rock.molar_volume
        assert_allclose(
            method.compute_pressure(300.0, volume, params),
            pressure,
            rtol=1e-11,
            atol=0.01,
        )
        step = volume * 1e-5
        derivative = (
            method.compute_pressure(300.0, volume + step, params)
            - method.compute_pressure(300.0, volume - step, params)
        ) / (2 * step)
        assert_allclose(
            rock.isothermal_bulk_modulus_reuss, -volume * derivative, rtol=1e-8
        )
        step = volume * 1e-4
        energies = []
        for offset in [-2, -1, 1, 2]:
            rock.set_state(
                method.compute_pressure(300.0, volume + offset * step, params), 300.0
            )
            energies.append(rock.molar_internal_energy)
        energy_derivative = (
            energies[0] - 8 * energies[1] + 8 * energies[2] - energies[3]
        ) / (12 * step)
        assert_allclose(-energy_derivative, pressure, rtol=1e-8, atol=1.0)
        assert rock.shear_modulus == 0.0


@pytest.mark.parametrize("n", [10, 41, 101])
def test_layer_and_planet_optional_burnman_reference(n):
    reference = pytest.importorskip("burnman")
    native = make_layer(n=n)
    native.make()
    params = periclase_params()
    rock = reference.Mineral(params)
    layer = reference.Layer("mantle", native.radii)
    layer.set_material(rock)
    layer.set_temperature_mode("user-defined", np.full(native.n_slices, 2000.0))
    layer.set_pressure_mode("self-consistent", pressure_top=22.5e9, gravity_bottom=10.0)
    layer.make()
    assert_allclose(native.pressures, layer.pressures, rtol=1e-9, atol=0.05)
    assert_allclose(native.gravity, layer.gravity, rtol=1e-10)
    assert_allclose(native.density, layer.density, rtol=1e-10)
    assert_allclose(native.mass, layer.mass, rtol=1e-10)
    assert_allclose(native.moment_of_inertia, layer.moment_of_inertia, rtol=1e-10)
    assert_allclose(native.bullen, layer.bullen, rtol=1e-8, atol=1e-9)
    assert_allclose(native.brunt_vasala, layer.brunt_vasala, rtol=1e-8, atol=1e-13)
    planet, core, mantle = make_planet(n=n)
    planet.make()
    ref_core = reference.Layer("core", core.radii)
    ref_core.set_material(reference.minerals.other.Liquid_Fe_Anderson())
    ref_core.set_temperature_mode("user-defined", np.full(core.n_slices, 300.0))
    ref_mantle = reference.Layer("mantle", mantle.radii)
    ref_mantle.set_material(reference.minerals.SLB_2011.mg_perovskite())
    ref_mantle.set_temperature_mode("adiabatic", temperature_top=1200.0)
    ref_planet = reference.Planet("earth_like", [ref_mantle, ref_core])
    ref_planet.make()
    assert_allclose(planet.pressures, ref_planet.pressures, rtol=1e-9, atol=0.05)
    assert_allclose(planet.gravity, ref_planet.gravity, rtol=1e-10, atol=1e-12)
    assert_allclose(planet.temperatures, ref_planet.temperatures, rtol=1e-9)
    assert_allclose(planet.density, ref_planet.density, rtol=1e-10)
    assert_allclose(planet.mass, ref_planet.mass, rtol=1e-10)
    assert_allclose(
        planet.moment_of_inertia_factor, ref_planet.moment_of_inertia_factor, rtol=1e-10
    )


@pytest.mark.parametrize("relaxed", [False, True])
def test_equilibrated_material_copy_and_layer_preserve_warm_states(capfd, relaxed):
    settings = bm.PhaseDiagramSettings()
    settings.verbose = True
    material = bm.EquilibratedComposite(
        {"MgO": 2.0},
        [bm.Mineral(oxide_params())],
        settings=settings,
        relaxed=relaxed,
    )
    material.set_state(4e9, 1000.0)
    density = material.density
    capfd.readouterr()
    duplicate = material.copy()
    assert "LP 0 " not in capfd.readouterr().err
    assert duplicate.has_state()
    assert duplicate.relaxed is relaxed
    assert_allclose(duplicate.density, density, rtol=1e-13)
    assert duplicate.get_phase(0) is not material.get_phase(0)
    duplicate.set_state(4.01e9, 1001.0)
    material.reset_cache()
    assert material.pressure == 4e9
    assert_allclose(material.density, density, rtol=1e-13)
    layer = bm.Layer("equilibrium", np.linspace(1e6, 2e6, 3))
    layer.set_material(material)
    layer.set_temperature_mode("user-defined", [1002.0, 1001.0, 1000.0])
    layer.set_pressure_mode("user-defined", [4.02e9, 4.01e9, 4e9], gravity_bottom=1.0)
    layer.make()
    assert np.isfinite(layer.density).all()
    assert layer.sublayers[0] is not layer.sublayers[1]
    assert layer.sublayers[0].get_phase(0) is not layer.sublayers[1].get_phase(0)
    assert material.pressure == 4e9
    assert_allclose(material.density, density, rtol=1e-13)


def test_fourth_order_iron_optional_burnman_reference():
    reference = pytest.importorskip("burnman")
    from burnman_cpp.adapters import from_burnman

    native = iron()
    python = reference.minerals.other.Liquid_Fe_Anderson()
    adapted = from_burnman(python)
    for pressure in [0.0, 5e9, 50e9, 200e9, 500e9]:
        native.set_state(pressure, 300.0)
        python.set_state(pressure, 300.0)
        adapted.set_state(pressure, 300.0)
        assert_allclose(adapted.density, native.density, rtol=1e-13)
        assert_allclose(native.molar_volume, python.molar_volume, rtol=1e-6)
        assert_allclose(native.density, python.density, rtol=1e-6)
        assert_allclose(
            native.isothermal_bulk_modulus_reuss,
            python.isothermal_bulk_modulus_reuss,
            rtol=3e-6,
        )


def test_optional_pinned_radial_interpolation():
    reference = pytest.importorskip("burnman")
    for x in (
        np.array([0.0, 1.0]),
        np.array([0.0, 0.2, 1.0]),
        np.array([0.0, 0.1, 0.4, 1.0]),
        np.array([0.0, 0.05, 0.2, 0.3, 0.45, 0.47, 0.6, 0.85, 1.0]),
    ):
        radii = 1e6 * (1.0 + x)
        native, python = bm.Layer("test", radii), reference.Layer("test", radii)
        for density in (
            8000.0 - 4000.0 * x**3,
            8000.0 - 4000.0 * (x - 0.53) ** 2,
            np.where(x <= 0.45, 8000.0, 4000.0),
        ):
            gravity = native._compute_gravity(density, 5.0)
            expected_gravity = python._compute_gravity(density, 5.0)
            assert_allclose(gravity, expected_gravity, rtol=2e-13)
            assert_allclose(
                native._compute_pressure(density, gravity, 1e5),
                python._compute_pressure(density, expected_gravity, 1e5),
                rtol=2e-13,
            )
