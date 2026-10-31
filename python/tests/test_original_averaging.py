# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2012-2026 by the BurnMan Team.
"""Standalone port of upstream tests/test_averaging.py.

Reference: tests/reference/burnman.json.
Original numerical inputs, assertions and tolerances are retained.
"""

from original_test_support import (
    BurnManTest,
    make_assemblage,
    native_method,
    native_evaluate,
)
import burnman_cpp as burnman
from burnman_cpp import minerals


class mypericlase(burnman.Mineral):
    """
    Stixrude & Lithgow-Bertelloni 2005 and references therein
    """

    def __init__(self):
        params = {
            "equation_of_state": "slb3",
            "V_0": 1.124e-05,
            "K_0": 161000000000.0,
            "Kprime_0": 3.8,
            "G_0": 131000000000.0,
            "Gprime_0": 2.1,
            "molar_mass": 0.0403,
            "n": 2,
            "Debye_0": 773.0,
            "grueneisen_0": 1.5,
            "q_0": 1.5,
            "eta_s_0": 2.8,
        }
        params.setdefault("formula", {"Mg": 1.0, "O": 1.0})
        params.setdefault("name", "mypericlase")
        burnman.Mineral.__init__(self, params)


class VRH(BurnManTest):

    def test_1(self):
        rock = make_assemblage([mypericlase()], [1.0])
        rock.set_method(native_method("slb3"))
        rock.set_averaging_scheme(burnman.AveragingType.VRH)
        rho, v_p, v_s, v_phi, K_vrh, G_vrh = native_evaluate(
            rock,
            [
                "density",
                "p_wave_velocity",
                "shear_wave_velocity",
                "bulk_sound_velocity",
                "K_eff",
                "shear_modulus",
            ],
            [10000000000.0],
            [300.0],
        )
        self.assertFloatEqual(3791.392, rho[0])
        self.assertFloatEqual(10285.368, v_p[0])
        self.assertFloatEqual(6308.811, v_s[0])
        self.assertFloatEqual(7260.9, v_phi[0])
        self.assertFloatEqual(199.884, K_vrh[0] / 1000000000.0)
        self.assertFloatEqual(150.901, G_vrh[0] / 1000000000.0)

    def same(self, number):
        rock = make_assemblage([mypericlase()] * number, [1.0 / number] * number)
        rock.set_method(native_method("slb3"))
        rock.set_averaging_scheme(burnman.AveragingType.VRH)
        rho, v_p, v_s, v_phi, K_vrh, G_vrh = native_evaluate(
            rock,
            [
                "density",
                "p_wave_velocity",
                "shear_wave_velocity",
                "bulk_sound_velocity",
                "K_eff",
                "shear_modulus",
            ],
            [10000000000.0],
            [300.0],
        )
        self.assertFloatEqual(3791.392, rho[0])
        self.assertFloatEqual(10285.368, v_p[0])
        self.assertFloatEqual(6308.811, v_s[0])
        self.assertFloatEqual(7260.9, v_phi[0])
        self.assertFloatEqual(199.884, K_vrh[0] / 1000000000.0)
        self.assertFloatEqual(150.901, G_vrh[0] / 1000000000.0)

    def test_same(self):
        self.same(2)
        self.same(3)
        self.same(4)

    def test_two_different(self):
        rock = make_assemblage(
            [minerals.SLB_2005.periclase(), minerals.SLB_2005.fe_perovskite()],
            [0.5, 0.5],
        )
        rock.set_method(native_method("slb3"))
        rock.set_averaging_scheme(burnman.AveragingType.VRH)
        rho, v_p, v_s, v_phi, K_vrh, G_vrh = native_evaluate(
            rock,
            [
                "density",
                "p_wave_velocity",
                "shear_wave_velocity",
                "bulk_sound_velocity",
                "K_eff",
                "shear_modulus",
            ],
            [10000000000.0],
            [300.0],
        )
        self.assertFloatEqual(4881.469, rho[0])
        self.assertFloatEqual(9957.665, v_p[0])
        self.assertFloatEqual(5606.916, v_s[0])
        self.assertFloatEqual(7565.607, v_phi[0])
        self.assertFloatEqual(279.408, K_vrh[0] / 1000000000.0)
        self.assertFloatEqual(153.461, G_vrh[0] / 1000000000.0)


class Reuss(BurnManTest):

    def test_1(self):
        rock = make_assemblage([mypericlase()], [1.0])
        rock.set_averaging_scheme(burnman.AveragingType.Reuss)
        rho, v_p, v_s, v_phi, K, G = native_evaluate(
            rock,
            [
                "density",
                "p_wave_velocity",
                "shear_wave_velocity",
                "bulk_sound_velocity",
                "K_eff",
                "shear_modulus",
            ],
            [10000000000.0],
            [300.0],
        )
        self.assertFloatEqual(3791.392, rho[0])
        self.assertFloatEqual(10285.368, v_p[0])
        self.assertFloatEqual(6308.811, v_s[0])
        self.assertFloatEqual(7260.9, v_phi[0])
        self.assertFloatEqual(199.884, K[0] / 1000000000.0)
        self.assertFloatEqual(150.901, G[0] / 1000000000.0)

    def same(self, number):
        rock = make_assemblage([mypericlase()] * number, [1.0 / number] * number)
        rock.set_averaging_scheme(burnman.AveragingType.Reuss)
        rho, v_p, v_s, v_phi, K, G = native_evaluate(
            rock,
            [
                "density",
                "p_wave_velocity",
                "shear_wave_velocity",
                "bulk_sound_velocity",
                "K_eff",
                "shear_modulus",
            ],
            [10000000000.0],
            [300.0],
        )
        self.assertFloatEqual(3791.392, rho[0])
        self.assertFloatEqual(10285.368, v_p[0])
        self.assertFloatEqual(6308.811, v_s[0])
        self.assertFloatEqual(7260.9, v_phi[0])
        self.assertFloatEqual(199.884, K[0] / 1000000000.0)
        self.assertFloatEqual(150.901, G[0] / 1000000000.0)

    def test_same(self):
        self.same(2)
        self.same(3)
        self.same(4)

    def test_two_different(self):
        rock = make_assemblage(
            [minerals.SLB_2005.periclase(), minerals.SLB_2005.fe_perovskite()],
            [0.5, 0.5],
        )
        rock.set_averaging_scheme(burnman.AveragingType.Reuss)
        rho, v_p, v_s, v_phi, K, G = native_evaluate(
            rock,
            [
                "density",
                "p_wave_velocity",
                "shear_wave_velocity",
                "bulk_sound_velocity",
                "K_eff",
                "shear_modulus",
            ],
            [10000000000.0],
            [300.0],
        )
        self.assertFloatEqual(4881.469, rho[0])
        self.assertFloatEqual(9887.625, v_p[0])
        self.assertFloatEqual(5606.745, v_s[0])
        self.assertFloatEqual(7473.353, v_phi[0])
        self.assertFloatEqual(272.635, K[0] / 1000000000.0)
        self.assertFloatEqual(153.452, G[0] / 1000000000.0)


class Voigt(BurnManTest):

    def test_1(self):
        rock = make_assemblage([mypericlase()], [1.0])
        rock.set_averaging_scheme(burnman.AveragingType.Voigt)
        rho, v_p, v_s, v_phi, K, G = native_evaluate(
            rock,
            [
                "density",
                "p_wave_velocity",
                "shear_wave_velocity",
                "bulk_sound_velocity",
                "K_eff",
                "shear_modulus",
            ],
            [10000000000.0],
            [300.0],
        )
        self.assertFloatEqual(3791.392, rho[0])
        self.assertFloatEqual(10285.368, v_p[0])
        self.assertFloatEqual(6308.811, v_s[0])
        self.assertFloatEqual(7260.9, v_phi[0])
        self.assertFloatEqual(199.884, K[0] / 1000000000.0)
        self.assertFloatEqual(150.901, G[0] / 1000000000.0)

    def same(self, number):
        rock = make_assemblage([mypericlase()] * number, [1.0 / number] * number)
        rock.set_averaging_scheme(burnman.AveragingType.Voigt)
        rho, v_p, v_s, v_phi, K, G = native_evaluate(
            rock,
            [
                "density",
                "p_wave_velocity",
                "shear_wave_velocity",
                "bulk_sound_velocity",
                "K_eff",
                "shear_modulus",
            ],
            [10000000000.0],
            [300.0],
        )
        self.assertFloatEqual(3791.392, rho[0])
        self.assertFloatEqual(10285.368, v_p[0])
        self.assertFloatEqual(6308.811, v_s[0])
        self.assertFloatEqual(7260.9, v_phi[0])
        self.assertFloatEqual(199.884, K[0] / 1000000000.0)
        self.assertFloatEqual(150.901, G[0] / 1000000000.0)

    def test_same(self):
        self.same(2)
        self.same(3)
        self.same(4)

    def test_two_different(self):
        rock = make_assemblage(
            [minerals.SLB_2005.periclase(), minerals.SLB_2005.fe_perovskite()],
            [0.5, 0.5],
        )
        rock.set_averaging_scheme(burnman.AveragingType.Voigt)
        rho, v_p, v_s, v_phi, K, G = native_evaluate(
            rock,
            [
                "density",
                "p_wave_velocity",
                "shear_wave_velocity",
                "bulk_sound_velocity",
                "K_eff",
                "shear_modulus",
            ],
            [10000000000.0],
            [300.0],
        )
        self.assertFloatEqual(4881.469, rho[0])
        self.assertFloatEqual(10027.216, v_p[0])
        self.assertFloatEqual(5607.087, v_s[0])
        self.assertFloatEqual(7656.75, v_phi[0])
        self.assertFloatEqual(286.18, K[0] / 1000000000.0)
        self.assertFloatEqual(153.471, G[0] / 1000000000.0)


class HSLower(BurnManTest):

    def test_1(self):
        rock = make_assemblage([mypericlase()], [1.0])
        rock.set_averaging_scheme(burnman.AveragingType.HashinShtrikmanLower)
        rho, v_p, v_s, v_phi, K, G = native_evaluate(
            rock,
            [
                "density",
                "p_wave_velocity",
                "shear_wave_velocity",
                "bulk_sound_velocity",
                "K_eff",
                "shear_modulus",
            ],
            [10000000000.0],
            [300.0],
        )
        self.assertFloatEqual(3791.392, rho[0])
        self.assertFloatEqual(10285.368, v_p[0])
        self.assertFloatEqual(6308.811, v_s[0])
        self.assertFloatEqual(7260.9, v_phi[0])
        self.assertFloatEqual(199.884, K[0] / 1000000000.0)
        self.assertFloatEqual(150.901, G[0] / 1000000000.0)

    def same(self, number):
        rock = make_assemblage([mypericlase()] * number, [1.0 / number] * number)
        rock.set_averaging_scheme(burnman.AveragingType.HashinShtrikmanLower)
        rho, v_p, v_s, v_phi, K, G = native_evaluate(
            rock,
            [
                "density",
                "p_wave_velocity",
                "shear_wave_velocity",
                "bulk_sound_velocity",
                "K_eff",
                "shear_modulus",
            ],
            [10000000000.0],
            [300.0],
        )
        self.assertFloatEqual(3791.392, rho[0])
        self.assertFloatEqual(10285.368, v_p[0])
        self.assertFloatEqual(6308.811, v_s[0])
        self.assertFloatEqual(7260.9, v_phi[0])
        self.assertFloatEqual(199.884, K[0] / 1000000000.0)
        self.assertFloatEqual(150.901, G[0] / 1000000000.0)

    def test_same(self):
        self.same(2)
        self.same(3)
        self.same(4)

    def test_two_different(self):
        rock = make_assemblage(
            [minerals.SLB_2005.periclase(), minerals.SLB_2005.fe_perovskite()],
            [0.5, 0.5],
        )
        rock.set_averaging_scheme(burnman.AveragingType.HashinShtrikmanLower)
        rho, v_p, v_s, v_phi, K, G = native_evaluate(
            rock,
            [
                "density",
                "p_wave_velocity",
                "shear_wave_velocity",
                "bulk_sound_velocity",
                "K_eff",
                "shear_modulus",
            ],
            [10000000000.0],
            [300.0],
        )
        self.assertFloatEqual(4881.469, rho[0])
        self.assertFloatEqual(9951.957, v_p[0])
        self.assertFloatEqual(5606.916, v_s[0])
        self.assertFloatEqual(7558.094, v_phi[0])
        self.assertFloatEqual(278.853, K[0] / 1000000000.0)
        self.assertFloatEqual(153.461, G[0] / 1000000000.0)


class HSUpper(BurnManTest):

    def test_1(self):
        rock = make_assemblage([mypericlase()], [1.0])
        rock.set_averaging_scheme(burnman.AveragingType.HashinShtrikmanUpper)
        rho, v_p, v_s, v_phi, K, G = native_evaluate(
            rock,
            [
                "density",
                "p_wave_velocity",
                "shear_wave_velocity",
                "bulk_sound_velocity",
                "K_eff",
                "shear_modulus",
            ],
            [10000000000.0],
            [300.0],
        )
        self.assertFloatEqual(3791.392, rho[0])
        self.assertFloatEqual(10285.368, v_p[0])
        self.assertFloatEqual(6308.811, v_s[0])
        self.assertFloatEqual(7260.9, v_phi[0])
        self.assertFloatEqual(199.884, K[0] / 1000000000.0)
        self.assertFloatEqual(150.901, G[0] / 1000000000.0)

    def same(self, number):
        rock = make_assemblage([mypericlase()] * number, [1.0 / number] * number)
        rock.set_averaging_scheme(burnman.AveragingType.HashinShtrikmanUpper)
        rho, v_p, v_s, v_phi, K, G = native_evaluate(
            rock,
            [
                "density",
                "p_wave_velocity",
                "shear_wave_velocity",
                "bulk_sound_velocity",
                "K_eff",
                "shear_modulus",
            ],
            [10000000000.0],
            [300.0],
        )
        self.assertFloatEqual(3791.392, rho[0])
        self.assertFloatEqual(10285.368, v_p[0])
        self.assertFloatEqual(6308.811, v_s[0])
        self.assertFloatEqual(7260.9, v_phi[0])
        self.assertFloatEqual(199.884, K[0] / 1000000000.0)
        self.assertFloatEqual(150.901, G[0] / 1000000000.0)

    def test_same(self):
        self.same(2)
        self.same(3)
        self.same(4)

    def test_two_different(self):
        rock = make_assemblage(
            [minerals.SLB_2005.periclase(), minerals.SLB_2005.fe_perovskite()],
            [0.5, 0.5],
        )
        rock.set_averaging_scheme(burnman.AveragingType.HashinShtrikmanUpper)
        rho, v_p, v_s, v_phi, K, G = native_evaluate(
            rock,
            [
                "density",
                "p_wave_velocity",
                "shear_wave_velocity",
                "bulk_sound_velocity",
                "K_eff",
                "shear_modulus",
            ],
            [10000000000.0],
            [300.0],
        )
        self.assertFloatEqual(4881.469, rho[0])
        self.assertFloatEqual(9952.798, v_p[0])
        self.assertFloatEqual(5606.925, v_s[0])
        self.assertFloatEqual(7559.192, v_phi[0])
        self.assertFloatEqual(278.934, K[0] / 1000000000.0)
        self.assertFloatEqual(153.462, G[0] / 1000000000.0)


class HSAverage(BurnManTest):

    def test_1(self):
        rock = make_assemblage([mypericlase()], [1.0])
        rock.set_averaging_scheme(burnman.AveragingType.HashinShtrikman)
        rho, v_p, v_s, v_phi, K, G = native_evaluate(
            rock,
            [
                "density",
                "p_wave_velocity",
                "shear_wave_velocity",
                "bulk_sound_velocity",
                "K_eff",
                "shear_modulus",
            ],
            [10000000000.0],
            [300.0],
        )
        self.assertFloatEqual(10285.368, v_p[0])
        self.assertFloatEqual(6308.811, v_s[0])
        self.assertFloatEqual(7260.9, v_phi[0])
        self.assertFloatEqual(199.884, K[0] / 1000000000.0)
        self.assertFloatEqual(150.901, G[0] / 1000000000.0)

    def same(self, number):
        rock = make_assemblage([mypericlase()] * number, [1.0 / number] * number)
        rock.set_averaging_scheme(burnman.AveragingType.HashinShtrikman)
        rho, v_p, v_s, v_phi, K, G = native_evaluate(
            rock,
            [
                "density",
                "p_wave_velocity",
                "shear_wave_velocity",
                "bulk_sound_velocity",
                "K_eff",
                "shear_modulus",
            ],
            [10000000000.0],
            [300.0],
        )
        self.assertFloatEqual(3791.392, rho[0])
        self.assertFloatEqual(10285.368, v_p[0])
        self.assertFloatEqual(6308.811, v_s[0])
        self.assertFloatEqual(7260.9, v_phi[0])
        self.assertFloatEqual(199.884, K[0] / 1000000000.0)
        self.assertFloatEqual(150.901, G[0] / 1000000000.0)

    def test_same(self):
        self.same(2)
        self.same(3)
        self.same(4)

    def test_two_different(self):
        rock = make_assemblage(
            [minerals.SLB_2005.periclase(), minerals.SLB_2005.fe_perovskite()],
            [0.5, 0.5],
        )
        rock.set_averaging_scheme(burnman.AveragingType.HashinShtrikman)
        rho, v_p, v_s, v_phi, K, G = native_evaluate(
            rock,
            [
                "density",
                "p_wave_velocity",
                "shear_wave_velocity",
                "bulk_sound_velocity",
                "K_eff",
                "shear_modulus",
            ],
            [10000000000.0],
            [300.0],
        )
        self.assertFloatEqual(4881.469, rho[0])
        self.assertFloatEqual(9952.378, v_p[0])
        self.assertFloatEqual(5606.92, v_s[0])
        self.assertFloatEqual(7558.643, v_phi[0])
        self.assertFloatEqual(278.893, K[0] / 1000000000.0)
        self.assertFloatEqual(153.461, G[0] / 1000000000.0)
