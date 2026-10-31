# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2012-2026 by the BurnMan Team.
"""Standalone port of upstream tests/test_planet.py.

Reference: tests/reference/burnman.json.
Original numerical inputs, assertions and tolerances are retained.
"""

from original_test_support import BurnManTest
import numpy as np
import burnman_cpp as burnman
from burnman_cpp import Planet
from burnman_cpp import Layer


def make_simple_planet():
    core = Layer("core", np.linspace(0.0, 3480000.0, 10))
    core.set_material(burnman.minerals.other.Liquid_Fe_Anderson())
    core.set_temperature_mode(
        "user-defined", temperatures=300.0 * np.ones_like(core.radii)
    )
    mantle = Layer("mantle", np.linspace(3480000.0, 6371000.0, 10))
    mantle.set_material(burnman.minerals.SLB_2011.mg_bridgmanite())
    mantle.set_temperature_mode("adiabatic", temperature_top=1200)
    myplanet = Planet("earth_like", [core, mantle])
    myplanet.make()
    return (myplanet, core, mantle)


class test_planet(BurnManTest):

    def test_sort_layers(self):
        myplanet, core, mantle = make_simple_planet()
        assert myplanet.layers == [core, mantle]

    def test_temperature(self):
        myplanet, core, mantle = make_simple_planet()
        self.assertArraysAlmostEqual(
            mantle.molar_entropy,
            [mantle.molar_entropy[-1] for S in mantle.molar_entropy],
        )

    def test_evaluate(self):
        myplanet, core, mantle = make_simple_planet()
        property_names = [
            "depth",
            "gravity",
            "temperature",
            "molar_internal_energy",
            "molar_gibbs",
            "molar_helmholtz",
            "molar_mass",
            "molar_volume",
            "molar_entropy",
            "molar_enthalpy",
            "isothermal_bulk_modulus_reuss",
            "isentropic_bulk_modulus_reuss",
            "isothermal_compressibility_reuss",
            "isentropic_compressibility_reuss",
            "bulk_sound_velocity",
            "grueneisen_parameter",
            "thermal_expansivity",
            "molar_heat_capacity_v",
            "molar_heat_capacity_p",
            "P",
            "T",
            "energy",
            "helmholtz",
            "gibbs",
            "V",
            "rho",
            "S",
            "H",
            "K_T",
            "K_S",
            "beta_T",
            "beta_S",
            "v_phi",
            "gr",
            "alpha",
            "C_v",
            "C_p",
        ]
        properties = myplanet.evaluate(property_names, myplanet.radii)
        for i, name in enumerate(property_names):
            self.assertFloatEqual(properties[i][-2], getattr(myplanet, name)[-2])
        property_names = [
            "average_density",
            "mass",
            "moment_of_inertia",
            "moment_of_inertia_factor",
        ]
        for name in property_names:
            self.assertTrue(isinstance(getattr(myplanet, name), float))
        property_names = [
            "bullen",
            "brunt_vasala",
            "shear_modulus",
            "p_wave_velocity",
            "shear_wave_velocity",
            "G",
            "v_p",
            "v_s",
        ]
        properties = myplanet.evaluate(property_names, myplanet.layers[1].radii[1:-1])
        for i, name in enumerate(property_names):
            self.assertFloatEqual(
                properties[i][-1], getattr(myplanet.layers[1], name)[-2]
            )

    def test_evaluate_solid_planet(self):
        myplanet, core, mantle = make_simple_planet()
        core.set_material(burnman.minerals.SLB_2011.mg_bridgmanite())
        myplanet.make()
        property_names = [
            "depth",
            "gravity",
            "temperature",
            "molar_internal_energy",
            "molar_gibbs",
            "molar_helmholtz",
            "molar_mass",
            "molar_volume",
            "molar_entropy",
            "molar_enthalpy",
            "isothermal_bulk_modulus_reuss",
            "isentropic_bulk_modulus_reuss",
            "isothermal_compressibility_reuss",
            "isentropic_compressibility_reuss",
            "bulk_sound_velocity",
            "grueneisen_parameter",
            "thermal_expansivity",
            "molar_heat_capacity_v",
            "molar_heat_capacity_p",
            "P",
            "T",
            "energy",
            "helmholtz",
            "gibbs",
            "V",
            "rho",
            "S",
            "H",
            "K_T",
            "K_S",
            "beta_T",
            "beta_S",
            "v_phi",
            "gr",
            "alpha",
            "C_v",
            "C_p",
            "bullen",
            "brunt_vasala",
            "shear_modulus",
            "p_wave_velocity",
            "shear_wave_velocity",
            "G",
            "v_p",
            "v_s",
        ]
        properties = myplanet.evaluate(property_names, myplanet.radii)
        for i, name in enumerate(property_names):
            self.assertFloatEqual(properties[i][-2], getattr(myplanet, name)[-2])
        property_names = [
            "average_density",
            "mass",
            "moment_of_inertia",
            "moment_of_inertia_factor",
        ]
        for name in property_names:
            self.assertTrue(isinstance(getattr(myplanet, name), float))
