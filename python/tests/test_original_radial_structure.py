# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2012-2026 by the BurnMan Team.
"""Standalone ports of upstream hydrostatic reference checks.

Reference: tests/reference/burnman.json. Both boundary-value solutions use only
native material properties and SciPy; no Python BurnMan or saved profiles.
Analytical materials and spline limiting are covered by the C++ radial tests.
"""

import unittest
import numpy as np
from original_test_support import BurnManTest, parameter_dict

import burnman_cpp as burnman
from burnman_cpp import constants


class test_radial_structure(BurnManTest):
    def test_reference_core_eos_volume_precision(self):
        original = parameter_dict(burnman.minerals.other.Liquid_Fe_Anderson().params)
        method = burnman.make_eos("bm4")
        for scale in (1e-3, 1.0, 1e3):
            params = burnman.mineral_params(dict(original, V_0=original["V_0"] * scale))
            for pressure in (0.0, 1e9, 100e9, 500e9):
                volume = method.compute_volume(pressure, 300.0, params)
                result = method.compute_pressure(300.0, volume, params)
                self.assertFloatEqual(result, pressure, tol=2e-12, tol_zero=1e-3)

    def test_nonconvergence_is_reported(self):
        for planet in (False, True):
            radii = (
                np.linspace(0.0, 1e6, 17) if planet else np.linspace(660e3, 2890e3, 17)
            )
            layer = burnman.Layer("MgO", radii)
            layer.set_material(burnman.minerals.SLB_2011.periclase())
            layer.set_temperature_mode("user-defined", temperatures=np.full(17, 2000.0))
            if planet:
                model = burnman.Planet(
                    "test", [layer], n_max_iterations=1, max_delta=1e-12
                )
            else:
                model = layer
                model.set_pressure_mode(
                    gravity_bottom=10.0,
                    pressure_top=22.5e9,
                    n_max_iterations=1,
                    max_delta=1e-12,
                )
            with self.assertWarnsRegex(RuntimeWarning, "did not converge"):
                model.make()
            self.assertFalse(model.converged)
            self.assertEqual(model.n_iterations, 1)

    def test_layer_profile_against_independent_hydrostatic_solution(self):
        from scipy.integrate import solve_bvp

        n = 513
        layer = burnman.Layer("MgO", np.linspace(660e3, 2890e3, n))
        layer.set_material(burnman.minerals.SLB_2011.periclase())
        layer.set_temperature_mode("user-defined", temperatures=np.full(n, 2000.0))
        layer.set_pressure_mode(
            gravity_bottom=10.0,
            pressure_top=22.5e9,
            max_delta=1e-12,
            n_max_iterations=100,
        )
        layer.make()
        material = layer.material.copy()
        x = layer.radii / 1e6

        # Independently integrate pressure (GPa) and q=(r/1e6)^2*g.
        def derivatives(radius, state):
            density = material.evaluate(
                ["density"], state[0] * 1e9, np.full(len(radius), 2000.0)
            )[0]
            return np.array(
                [
                    -density * state[1] / (1000.0 * radius**2),
                    4.0 * np.pi * constants.G * 1e6 * density * radius**2,
                ]
            )

        def boundaries(inner, outer):
            return np.array([inner[1] - x[0] ** 2 * 10.0, outer[0] - 22.5])

        initial = np.array([layer.pressures / 1e9, layer.gravity * x**2])
        reference = solve_bvp(
            derivatives, boundaries, x, initial, tol=1e-9, max_nodes=10000
        )
        self.assertTrue(reference.success, reference.message)
        pressure, q = reference.sol(x)
        self.assertArraysAlmostEqual(
            layer.pressures, pressure * 1e9, tol=0.0, tol_zero=10.0
        )
        self.assertArraysAlmostEqual(layer.gravity, q / x**2, tol=0.0, tol_zero=1e-8)

    def test_planet_profile_against_independent_hydrostatic_solution(self):
        from scipy.integrate import solve_bvp

        n = 257
        core = burnman.Layer("core", np.linspace(0.0, 3480e3, n))
        core.set_material(burnman.minerals.other.Liquid_Fe_Anderson())
        core.set_temperature_mode("user-defined", temperatures=np.full(n, 300.0))
        mantle = burnman.Layer("mantle", np.linspace(3480e3, 6371e3, n))
        mantle.set_material(burnman.minerals.SLB_2011.mg_bridgmanite())
        mantle.set_temperature_mode("adiabatic", temperature_top=1200.0)
        planet = burnman.Planet(
            "earth_like", [core, mantle], max_delta=1e-12, n_max_iterations=100
        )
        planet.make()
        core_material, mantle_material = core.material.copy(), mantle.material.copy()
        core_radius, mantle_width = 3.480, 2.891  # Million metres.
        x = np.linspace(0.0, 1.0, n)

        # Separate coordinates retain the density jump at the interface.
        # State: core P (GPa), core g, mantle P (GPa), mantle g, mantle T.
        def derivatives(position, state):
            core_density = core_material.evaluate(
                ["density"], state[0] * 1e9, np.full(len(position), 300.0)
            )[0]
            density, gamma, bulk = mantle_material.evaluate(
                ["density", "grueneisen_parameter", "isentropic_bulk_modulus_reuss"],
                state[2] * 1e9,
                state[4],
            )
            pressure_gradient = -mantle_width / 1000.0 * density * state[3]
            return np.array(
                [
                    -core_radius / 1000.0 * core_density * state[1],
                    4.0 * np.pi * constants.G * core_radius * 1e6 * core_density,
                    pressure_gradient,
                    4.0 * np.pi * constants.G * mantle_width * 1e6 * density
                    - 2.0
                    * mantle_width
                    * state[3]
                    / (core_radius + mantle_width * position),
                    gamma * state[4] / bulk * pressure_gradient * 1e9,
                ]
            )

        def boundaries(inner, outer):
            return np.array(
                [
                    inner[1],
                    outer[2],
                    outer[4] - 1200.0,
                    outer[0] - inner[2],
                    outer[1] - inner[3],
                ]
            )

        initial = np.array(
            [
                core.pressures / 1e9,
                core.gravity,
                mantle.pressures / 1e9,
                mantle.gravity,
                mantle.temperatures,
            ]
        )
        singular = np.zeros((5, 5))
        singular[1, 1] = -2.0  # Regular central gravity: dg/dx = 4*pi*G*rho*Rc - 2g/x.
        reference = solve_bvp(
            derivatives, boundaries, x, initial, S=singular, tol=1e-8, max_nodes=10000
        )
        self.assertTrue(reference.success, reference.message)
        pressure_core, gravity_core, pressure_mantle, gravity_mantle, temperature = (
            reference.sol(x)
        )
        self.assertArraysAlmostEqual(
            core.pressures, pressure_core * 1e9, tol=0.0, tol_zero=10.0
        )
        self.assertArraysAlmostEqual(
            mantle.pressures, pressure_mantle * 1e9, tol=0.0, tol_zero=10.0
        )
        self.assertArraysAlmostEqual(core.gravity, gravity_core, tol=0.0, tol_zero=1e-8)
        self.assertArraysAlmostEqual(
            mantle.gravity, gravity_mantle, tol=0.0, tol_zero=1e-8
        )
        self.assertArraysAlmostEqual(
            mantle.temperatures, temperature, tol=0.0, tol_zero=1e-6
        )


if __name__ == "__main__":
    unittest.main()
