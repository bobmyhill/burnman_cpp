# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2012-2026 by the BurnMan Team.
"""Standalone port of upstream tests/test_ig51_solutions.py.

Reference: tests/reference/burnman.json.
Original numerical inputs, assertions and tolerances are retained.
"""

from original_test_support import native_endmembers, is_native_factory, ideal_model_call
import unittest
import numpy as np
from burnman_cpp import Solution
from burnman_cpp import IdealSolution
from burnman_cpp.constants import gas_constant
from burnman_cpp.minerals import HPx_ds636, ig51W24, ig51G25
from burnman_cpp.tools.eos import check_eos_consistency


class ig51_solutions(unittest.TestCase):

    def test_complete_solution_sets(self):
        expected = {
            ig51W24: {
                "liq_W24d": 14,
                "ol_H18": 4,
                "g_W24": 6,
                "spl_T21": 8,
                "fsp_H22": 3,
                "cpx_W24": 10,
                "opx_W24": 9,
                "ilm_W24": 5,
                "nph_W24": 6,
                "kals_W24": 2,
                "lct_W24": 2,
                "mel_W24": 5,
            },
            ig51G25: {
                "liq_G25w": 12,
                "fl_G25": 11,
                "ol_H18": 4,
                "g_W24": 6,
                "spl_T21": 8,
                "fsp_H22": 3,
                "cpx_W24": 10,
                "opx_W24": 9,
                "ilm_W24": 5,
                "bi_G25": 6,
                "mu_W14": 6,
                "amp_G16": 11,
                "ep_H11": 3,
                "cd_G25": 3,
            },
        }
        for module, phases in expected.items():
            actual = {
                name: constructor
                for name, constructor in vars(module).items()
                if is_native_factory(constructor)
                and constructor.__module__ == module.__name__
                and is_native_factory(constructor, Solution)
            }
            self.assertEqual(set(actual), set(phases))
            for name, constructor in actual.items():
                with self.subTest(dataset=module.__name__, phase=name):
                    solution = constructor()
                    self.assertEqual(solution.n_endmembers, phases[name])
                    solution.set_composition(np.full(phases[name], 1.0 / phases[name]))
                    solution.set_state(1000000000.0, 1400.0)
                    self.assertTrue(np.isfinite(solution.molar_gibbs))
                    self.assertGreater(solution.molar_volume, 0.0)
                    self.assertTrue(np.isfinite(solution.molar_heat_capacity_p))

    def test_corrected_melt_activities(self):
        for constructor in (ig51W24.liq_W24d, ig51G25.liq_G25w):
            with self.subTest(melt=constructor.__name__):
                melt = constructor()
                model = melt.solution_model
                p = np.arange(1.0, melt.n_endmembers + 1.0)
                p /= sum(p)
                mg, fe, sl, wo = (3, 4, 1, 2)
                total = 4.0 * (p[mg] + p[fe]) + p[sl] + p[wo]
                expected = p.copy()
                expected[sl] *= p[sl] / total
                expected[wo] *= p[wo] / total
                for i in (mg, fe):
                    expected[i] = (p[mg] + p[fe]) * (4.0 * p[i] / total) ** 4
                if constructor is ig51G25.liq_G25w:
                    expected *= 1.0 - p[-1]
                    expected[-1] = p[-1] ** 2
                np.testing.assert_allclose(
                    ideal_model_call(
                        model, "compute_activities", 1000000000.0, 1400.0, p
                    ),
                    expected,
                    rtol=1e-12,
                    atol=1e-15,
                )
                np.testing.assert_allclose(
                    ideal_model_call(
                        model,
                        "compute_excess_partial_entropies",
                        1000000000.0,
                        1400.0,
                        p,
                    ),
                    -gas_constant * np.log(expected),
                    rtol=1e-12,
                )
                for pure in np.eye(melt.n_endmembers):
                    self.assertAlmostEqual(
                        model.compute_excess_gibbs_free_energy(
                            1000000000.0, 1400.0, pure
                        ),
                        0.0,
                    )
                melt.set_composition(p)
                self.assertTrue(
                    check_eos_consistency(melt, P=1000000000.0, T=1400.0, tol=0.0001)
                )

    def test_g25_fluid_water_and_activities(self):
        fluid = ig51G25.fl_G25()
        water_index = fluid.endmember_names.index("H2O")
        self.assertEqual(
            native_endmembers(fluid)[water_index][0].formula, HPx_ds636.H2O().formula
        )
        p = np.full(fluid.n_endmembers, 0.05)
        p[water_index] = 0.5
        expected = (1.0 - p[water_index]) * p
        expected[water_index] = p[water_index] ** 2
        np.testing.assert_allclose(
            ideal_model_call(
                fluid.solution_model, "compute_activities", 1000000000.0, 1200.0, p
            ),
            expected,
        )
        pure = np.zeros(fluid.n_endmembers)
        pure[water_index] = 1.0
        fluid.set_composition(pure)
        fluid.set_state(1000000000.0, 1200.0)
        water = HPx_ds636.H2O()
        water.set_state(1000000000.0, 1200.0)
        self.assertAlmostEqual(fluid.molar_gibbs, water.molar_gibbs)
        self.assertAlmostEqual(fluid.molar_volume, water.molar_volume)

    def test_temperature_and_pressure_interactions(self):
        for module in (ig51W24, ig51G25):
            model = module.g_W24().solution_model
            p = np.array([0.5, 0.0, 0.5, 0.0, 0.0, 0.0])
            self.assertEqual(model.alphas[0], 1.0)
            self.assertEqual(model.alphas[2], 2.5)
            weight = 10.0 / 49.0
            for pressure, temperature in ((100000.0, 1000.0), (1000000000.0, 1400.0)):
                excess = model.compute_excess_gibbs_free_energy(
                    pressure, temperature, p
                )
                ideal = IdealSolution(
                    native_endmembers(model)
                ).compute_excess_gibbs_free_energy(pressure, temperature, p)
                expected = weight * (45400.0 - 10.0 * temperature + 4e-07 * pressure)
                self.assertAlmostEqual(excess - ideal, expected)
                self.assertAlmostEqual(
                    model.compute_excess_entropy(pressure, temperature, p)
                    - IdealSolution(native_endmembers(model)).compute_excess_entropy(
                        pressure, temperature, p
                    ),
                    weight * 10.0,
                )
                self.assertAlmostEqual(
                    model.compute_excess_volume(pressure, temperature, p),
                    weight * 4e-07,
                )
