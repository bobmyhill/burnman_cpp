# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2012-2026 by the BurnMan Team.
"""Standalone port of upstream tests/test_tools.py.

Reference: tests/reference/burnman.json.
Original numerical inputs, assertions and tolerances are retained.
"""

from original_test_support import (
    BurnManTest,
    parameter_dict,
    set_parameter,
    make_assemblage,
    equilibrate,
)
import numpy as np
import matplotlib.pyplot as plt
import burnman_cpp as burnman
from burnman_cpp.tools.chemistry import equilibrium_temperature
from burnman_cpp.tools.chemistry import equilibrium_pressure
from burnman_cpp.tools.chemistry import hugoniot
from burnman_cpp.tools.chemistry import reactions_from_stoichiometric_matrix
from burnman_cpp.tools.chemistry import reactions_from_formulae
from burnman_cpp.tools.plot import plot_projected_elastic_properties, pretty_plot
from burnman_cpp.utils.chemistry import convert_fractions
from burnman_cpp.utils.math import bracket
from burnman_cpp.utils.math import smooth_array
from burnman_cpp.utils.math import interp_smoothed_array_and_derivatives
from burnman_cpp.utils.math import _pad_ndarray_inverse_mirror
from burnman_cpp.utils.math import is_positive_definite
from burnman_cpp.minerals import HP_2011_ds62, mp50MnNCKFMASHTO
from burnman_cpp.tools.thermobarometry import estimate_conditions
from burnman_cpp.tools.thermobarometry import get_reaction_matrix
from burnman_cpp.tools.thermobarometry import assemblage_affinity_misfit
from burnman_cpp.tools.thermobarometry import assemblage_state_misfit


def setup_assemblage(scale=1.0):
    standard_deviations = {}
    "\n    Set up a mineral assemblage for testing thermobarometry.\n    :param scale: Scale factor for compositional noise.\n    "
    np.random.seed(42)
    mu = mp50MnNCKFMASHTO.mu()
    bi = mp50MnNCKFMASHTO.bi()
    g = mp50MnNCKFMASHTO.g()
    ilmm = mp50MnNCKFMASHTO.ilmm()
    st = mp50MnNCKFMASHTO.st()
    q = HP_2011_ds62.q()
    compositions = [
        np.array(
            [0.55510672, 0.00393874, 0.00365819, 0.43232719, 0.00339884, 0.00157032]
        ),
        np.array(
            [
                0.12024528,
                0.55795845,
                -0.23022757,
                0.2928378,
                0.19457943,
                0.06260454,
                0.00200206,
            ]
        ),
        np.array([0.11996805, 0.75282947, 0.10051702, 0.02013999, 0.00654547]),
        np.array([-0.05461279, 0.84687573, 0.1062428, 0.05404236, 0.04745189]),
        np.array([0.06925327, 0.80706976, 0.01346876, 0.03074615, 0.07946205]),
    ]
    standard_deviations[mu] = [0.01, 0.001, 0.001, 0.01, 0.001, 0.0005]
    standard_deviations[bi] = [0.01, 0.01, 0.05, 0.01, 0.01, 0.01, 0.0005]
    standard_deviations[g] = [0.01, 0.01, 0.01, 0.01, 0.001]
    standard_deviations[ilmm] = [0.01, 0.01, 0.01, 0.01, 0.01]
    standard_deviations[st] = [0.01, 0.01, 0.01, 0.01, 0.01]
    assemblage = make_assemblage([mu, bi, g, ilmm, st, q], [1.0 / 6.0] * 6)
    for phase, comp in zip(assemblage.phases[:-1], compositions):
        noise = np.random.normal(0, 1.0, len(comp)) * standard_deviations[phase]
        comp_with_noise = comp + noise * scale
        comp_with_noise /= np.sum(comp_with_noise)
        phase.set_composition(comp_with_noise)
    for phase in assemblage.phases[:-1]:
        phase.compositional_covariances = np.diag(
            np.array(standard_deviations[phase]) ** 2
        )
    return assemblage


class test_tools(BurnManTest):

    def test_eqm_T(self):
        fo = burnman.minerals.HP_2011_ds62.fo()
        fo2 = burnman.minerals.HP_2011_ds62.fo()
        H_ex = 1000.0
        S_ex = 2.0
        set_parameter(fo2, "H_0", parameter_dict(fo.params)["H_0"] + H_ex)
        set_parameter(fo2, "S_0", parameter_dict(fo.params)["S_0"] + S_ex)
        T = H_ex / S_ex
        T_calc = equilibrium_temperature(
            [fo, fo2], [1.0, -1.0], parameter_dict(fo.params)["P_0"], 1200.0
        )
        self.assertArraysAlmostEqual([T], [T_calc])

    def test_eqm_P(self):
        fo = burnman.minerals.HP_2011_ds62.fo()
        fo2 = burnman.minerals.HP_2011_ds62.fo()
        set_parameter(fo, "K_0", 1e20)
        set_parameter(fo2, "K_0", 1e20)
        H_ex = 1
        V_ex = 1e-08
        set_parameter(fo2, "H_0", parameter_dict(fo.params)["H_0"] + H_ex)
        set_parameter(fo2, "V_0", parameter_dict(fo.params)["V_0"] - V_ex)
        P = parameter_dict(fo.params)["P_0"] + H_ex / V_ex
        P_calc = equilibrium_pressure(
            [fo, fo2], [1.0, -1.0], parameter_dict(fo.params)["T_0"]
        )
        self.assertArraysAlmostEqual([P], [P_calc])

    def test_hugoniot(self):
        fo = burnman.minerals.HP_2011_ds62.fo()
        T_ref = 298.15
        P_ref = 100000.0
        pressures = np.array([P_ref])
        temperatures, volumes = hugoniot(fo, P_ref, T_ref, pressures)
        self.assertArraysAlmostEqual(temperatures, np.array([T_ref]))

    def test_fraction_conversion_mass(self):
        pv = burnman.minerals.HP_2011_ds62.mpv()
        en = burnman.minerals.HP_2011_ds62.en()
        c = make_assemblage([pv, en], [0.5, 0.5])
        molar_fractions = convert_fractions(c, [0.5, 0.5], "mass", "molar")
        self.assertArraysAlmostEqual(molar_fractions, [2.0 / 3.0, 1.0 / 3.0])

    def test_fraction_conversion_mass_2(self):
        per = burnman.minerals.SLB_2011.periclase()
        stv = burnman.minerals.SLB_2011.stishovite()
        fo = burnman.minerals.SLB_2011.forsterite()
        c = make_assemblage([per, stv, fo], [0.5, 0.25, 0.25])
        mass_fractions = convert_fractions(c, [0.5, 0.25, 0.25], "molar", "mass")
        self.assertArraysAlmostEqual(
            [mass_fractions[0] + mass_fractions[1]], [mass_fractions[2]]
        )

    def test_fraction_conversion_volume(self):
        per = burnman.minerals.SLB_2011.periclase()
        c = make_assemblage([per, per, per], [0.25, 0.25, 0.5])
        c.set_state(100000.0, 300.0)
        mass_fractions = convert_fractions(c, [0.25, 0.25, 0.5], "molar", "volume")
        self.assertArraysAlmostEqual(
            [mass_fractions[0] + mass_fractions[1]], [mass_fractions[2]]
        )

    def test_bracket(self):

        def fn(x):
            return (x - 1.0) * (x - 2.0)

        sol = bracket(fn, 1.2, 0.01)
        self.assertTrue(sol[2] * sol[3] < 0.0)
        self.assertTrue(sol[1] < 1.0)
        self.assertTrue(sol[0] > 1.0)
        sol = bracket(fn, 0.6, 0.01)
        self.assertTrue(sol[2] * sol[3] < 0.0)
        self.assertTrue(sol[0] < 1.0)
        self.assertTrue(sol[1] > 1.0)
        sol = bracket(fn, 1.7, 0.01)
        self.assertTrue(sol[2] * sol[3] < 0.0)
        self.assertTrue(sol[0] < 2.0)
        self.assertTrue(sol[1] > 2.0)
        sol = bracket(fn, 2.5, 0.01)
        self.assertTrue(sol[2] * sol[3] < 0.0)
        self.assertTrue(sol[1] < 2.0)
        self.assertTrue(sol[0] > 2.0)
        sol = bracket(fn, 1.2, 2.0)
        self.assertTrue(sol[2] * sol[3] < 0.0)
        self.assertTrue(sol[1] < 1.0)
        self.assertTrue(sol[0] > 1.0)
        sol = bracket(fn, 1.8, 2.0)
        self.assertTrue(sol[2] * sol[3] < 0.0)
        self.assertTrue(sol[0] < 2.0)
        self.assertTrue(sol[1] > 2.0)

    def test_padding_1D(self):
        array = np.array([1.0, 2.0, 3.0, 5.0])
        padding = (1,)
        padded_array = _pad_ndarray_inverse_mirror(array, padding)
        self.assertArraysAlmostEqual(
            padded_array, np.array([0.0, 1.0, 2.0, 3.0, 5.0, 7.0])
        )

    def test_padding_2D(self):
        array = np.array([[1.0, 2.0], [3.0, 5.0]])
        padding = (1, 1)
        padded_array = _pad_ndarray_inverse_mirror(array, padding)
        self.assertArraysAlmostEqual(
            np.ndarray.flatten(padded_array),
            np.ndarray.flatten(
                np.array(
                    [
                        [-3.0, -1.0, -1.0, 1.0],
                        [0.0, 1.0, 2.0, 3.0],
                        [1.0, 3.0, 5.0, 7.0],
                        [4.0, 5.0, 8.0, 9.0],
                    ]
                )
            ),
        )

    def test_smoothing(self):
        array = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        smoothed_array = smooth_array(array, [1.0], [1.0])
        self.assertArraysAlmostEqual(array, smoothed_array)

    def test_interp_smoothing_ij(self):
        array = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
        axis_values = np.array([0.0, 1.0, 2.0])
        f, dfdx, dfdy = interp_smoothed_array_and_derivatives(
            array, axis_values, axis_values, 0, 0, indexing="ij"
        )
        self.assertArraysAlmostEqual(
            [f([0.0, 1.0])[0], dfdx([0.0, 1.0])[0], dfdy([0.0, 1.0])[0]],
            [array[0][1], 0.0, 1.0],
        )

    def test_interp_smoothing_xy(self):
        array = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
        axis_values = np.array([0.0, 1.0, 2.0])
        f, dfdx, dfdy = interp_smoothed_array_and_derivatives(
            array, axis_values, axis_values, 0, 0, indexing="xy"
        )
        self.assertArraysAlmostEqual(
            [f([0.0, 1.0])[0], dfdx([0.0, 1.0])[0], dfdy([0.0, 1.0])[0]],
            [array[1][0], 1.0, 0.0],
        )

    def test_n_reactions_from_stoichiometry(self):
        M = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
        R = reactions_from_stoichiometric_matrix(M)
        self.assertTrue(len(R) == 6.0)

    def test_reactions_from_formulae(self):
        formulae = ["MgO", "FeO", "Fe2Si2O6", "Fe2SiO4"]
        compound_names = ["per", "fper", "fs", "fa"]
        R = sorted(
            reactions_from_formulae(formulae, compound_names, return_strings=True)
        )
        self.assertTrue(R[0] == "2 fa = 2 fper + 1 fs")

    def test_positive_definite(self):
        arr = np.array([[2.0, 0.0], [0.0, 1.0]])
        self.assertTrue(is_positive_definite(arr))

    def test_plot_elastic(self):
        pretty_plot()
        stishovite_Q1 = burnman.Mineral(
            {
                "name": "Stishovite",
                "formula": {"Si": 1.0, "O": 2.0},
                "equation_of_state": "slb3",
                "F_0": -815061.947383201,
                "V_0": 1.3985103364121215e-05,
                "K_0": 302651021060.0,
                "Kprime_0": 4.231768387000001,
                "Debye_0": 1099.41845635,
                "grueneisen_0": 1.815179405,
                "q_0": 2.2095517465427,
                "G_0": 228000000000.0,
                "Gprime_0": 1.94045,
                "eta_s_0": 4.40394,
                "n": 3.0,
                "Z": 2.0,
                "molar_mass": 0.060085,
                "T_0": 300.0,
                "F_0": 0.0,
                "P_0": 0.0,
            }
        )
        a = 0.02754879
        b = 0.02839605
        c = 1.3985103364121215e-05 / (a * b)
        cell_parameters = [a, b, c, 90, 90, 90]
        c11 = 7.93215475
        c44 = 3.6433892
        c66 = 4.63773363
        d11 = 0.03176523
        d44 = 0.31665864
        d66 = 0.09046939
        carr = c11 * np.ones((6, 6))
        carr[3, 3] = c44
        carr[4, 4] = c44
        carr[5, 5] = c66
        darr = d11 * np.ones((6, 6))
        darr[3, 3] = d44
        darr[4, 4] = d44
        darr[5, 5] = d66
        anisotropic_parameters = {
            "a": np.array(
                [
                    [0.56655959, -0.08994629, -0.04885663, 0.0, 0.0, 0.0],
                    [-0.08994629, 0.71555477, -0.25117643, 0.0, 0.0, 0.0],
                    [-0.04885663, -0.25117643, 0.49784435, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.33005083, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.45154562, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.97764314],
                ]
            ),
            "b": np.array(
                [
                    [0.17631275, -0.06562267, -0.04243526, 0.0, 0.0, 0.0],
                    [-0.06562267, 0.17631275, -0.04243526, 0.0, 0.0, 0.0],
                    [-0.04243526, -0.04243526, -0.0516391, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, -0.70121757, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, -0.70121757, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, -0.18867047],
                ]
            ),
            "c": carr,
            "d": darr,
        }

        def psi_func(f, Pth, params):
            a = params["a"]
            b = params["b"]
            c = params["c"]
            d = params["d"]
            dPsidf = a + b * np.tanh(c * f + d)
            Psi = a * f + b / c * np.log(np.cosh(c * f + d) / np.cosh(d))
            dPsidPth = np.zeros((6, 6))
            return (Psi, dPsidf, dPsidPth)

        m = burnman.AnisotropicMineral(
            stishovite_Q1,
            cell_parameters,
            anisotropic_parameters,
            psi_function=psi_func,
            orthotropic=True,
        )
        m.set_state(1000000000.0, 1000.0)
        plot_types = [
            "vp",
            "vs1",
            "vs2",
            "linear compressibility",
            "minimum poisson ratio",
            "maximum poisson ratio",
            "vp/vs1",
            "vp/vs2",
            "s anisotropy",
            "linear compressibility",
            "youngs modulus",
        ]
        fig = plt.figure(figsize=(12, 7))
        ax = [fig.add_subplot(4, 3, i, projection="polar") for i in range(1, 12)]
        contour_sets, ticks, lines = plot_projected_elastic_properties(
            m, plot_types, ax, 4, 4, 5, 5
        )
        self.assertTrue(len(contour_sets) == 11)
        self.assertTrue(len(ticks) == 11)
        self.assertTrue(len(lines) == 11)

    def test_estimate_conditions(self):
        assemblage = setup_assemblage()
        theta = np.array([500000000.0, 500.0])
        dataset_covariances = HP_2011_ds62.cov()
        res = estimate_conditions(
            assemblage, dataset_covariances, guessed_conditions=theta
        )
        self.assertTrue(res.success)
        self.assertEqual(res.n_reactions, 18)

    def test_estimate_conditions_at_equilibrium(self):
        assemblage = setup_assemblage(scale=0.0)
        theta = np.array([500000000.0, 500.0])
        dataset_covariances = HP_2011_ds62.cov()
        res = estimate_conditions(
            assemblage, dataset_covariances, guessed_conditions=theta
        )
        self.assertTrue(res.success)
        self.assertArraysAlmostEqual(
            [res.x[0] / 1000000.0, res.x[1]], [400.0, 873.0], tol=0.1
        )

    def test_estimate_conditions_exclude_small_fractions(self):
        assemblage = setup_assemblage()
        theta = np.array([500000000.0, 500.0])
        dataset_covariances = HP_2011_ds62.cov()
        res = estimate_conditions(
            assemblage,
            dataset_covariances,
            guessed_conditions=theta,
            small_fraction_tol=0.01,
        )
        self.assertTrue(res.success)
        self.assertEqual(res.n_reactions, 11)

    def test_estimate_conditions_fixed_P(self):
        assemblage = setup_assemblage()
        theta = np.array([500000000.0, 500.0])
        dataset_covariances = HP_2011_ds62.cov()
        res = estimate_conditions(
            assemblage,
            dataset_covariances,
            guessed_conditions=theta,
            pressure_bounds=[1000000000.0, 1000000000.0],
        )
        self.assertTrue(res.success)
        self.assertEqual(res.x[0], 1000000000.0)
        self.assertArraysAlmostEqual(res.xcov[0], [0.0, 0.0])
        self.assertArraysAlmostEqual(res.xcov[:, 0], [0.0, 0.0])
        self.assertFloatEqual(res.var[0], 0.0)
        dadt = res.dadx[:, 1]
        self.assertFloatEqual(
            res.xcov[1, 1], 1.0 / dadt.dot(np.linalg.solve(res.acov, dadt))
        )
        self.assertTrue(np.all(np.isnan(res.xcorr[0])))
        self.assertTrue(np.all(np.isnan(res.xcorr[:, 0])))
        self.assertFloatEqual(res.xcorr[1, 1], 1.0)

    def test_estimate_conditions_fixed_T(self):
        assemblage = setup_assemblage()
        theta = np.array([500000000.0, 500.0])
        dataset_covariances = HP_2011_ds62.cov()
        res = estimate_conditions(
            assemblage,
            dataset_covariances,
            guessed_conditions=theta,
            temperature_bounds=[500.0, 500.0],
        )
        self.assertTrue(res.success)
        self.assertEqual(res.x[1], 500.0)
        self.assertArraysAlmostEqual(res.xcov[1], [0.0, 0.0])
        self.assertArraysAlmostEqual(res.xcov[:, 1], [0.0, 0.0])
        self.assertFloatEqual(res.var[1], 0.0)
        dadp = res.dadx[:, 0]
        self.assertFloatEqual(
            res.xcov[0, 0], 1.0 / dadp.dot(np.linalg.solve(res.acov, dadp))
        )
        self.assertTrue(np.all(np.isnan(res.xcorr[1])))
        self.assertTrue(np.all(np.isnan(res.xcorr[:, 1])))
        self.assertFloatEqual(res.xcorr[0, 0], 1.0)

    def test_estimate_conditions_fixed_state_with_correlated_priors(self):
        for fixed_index, bounds in enumerate(
            (
                {"pressure_bounds": [400000000.0, 400000000.0]},
                {"temperature_bounds": [873.0, 873.0]},
            )
        ):
            with self.subTest(fixed_index=fixed_index):
                assemblage = setup_assemblage()
                assemblage.state_priors = np.array([400000000.0, 873.0])
                assemblage.state_inverse_covariances = np.array(
                    [[1e-12 / 3600.0, -1e-06 / 2250.0], [-1e-06 / 2250.0, 1.0 / 900.0]]
                )
                res = estimate_conditions(
                    assemblage,
                    HP_2011_ds62.cov(),
                    include_state_misfit=True,
                    guessed_conditions=assemblage.state_priors,
                    **bounds,
                )
                self.assertTrue(res.success, res.message)
                free_index = 1 - fixed_index
                derivative = res.dadx[:, free_index]
                prior_variance = [3600000000000000.0, 900.0][free_index]
                expected = 1.0 / (
                    derivative.dot(np.linalg.solve(res.acov, derivative))
                    + 1.0 / prior_variance
                )
                self.assertFloatEqual(res.xcov[free_index, free_index], expected)
                self.assertArraysAlmostEqual(res.xcov[fixed_index], [0.0, 0.0])
                self.assertArraysAlmostEqual(res.xcov[:, fixed_index], [0.0, 0.0])

    def test_estimate_conditions_fixed_P_and_T_with_free_composition(self):
        assemblage = setup_assemblage(scale=0.0)
        biotite = assemblage.phases[1]
        biotite.free_compositional_vectors = make_assemblage([biotite]).reaction_basis
        res = estimate_conditions(
            assemblage,
            HP_2011_ds62.cov(),
            guessed_conditions=[400000000.0, 873.0],
            pressure_bounds=[400000000.0, 400000000.0],
            temperature_bounds=[873.0, 873.0],
        )
        self.assertTrue(res.success, res.message)
        self.assertEqual(res.n_params, 1)
        self.assertEqual(res.xcov.shape, (2, 2))
        self.assertArraysAlmostEqual(res.xcov.ravel(), [0.0] * 4)
        self.assertArraysAlmostEqual(res.var, [0.0, 0.0])
        self.assertTrue(np.all(np.isnan(res.xcorr)))

    def test_estimate_conditions_narrow_bounds_do_not_fix_state(self):
        res = estimate_conditions(
            setup_assemblage(),
            HP_2011_ds62.cov(),
            guessed_conditions=[400000000.0, 873.0],
            pressure_bounds=[400000000.0 - 1000.0, 400000000.0 + 1000.0],
            temperature_bounds=[873.0 - 0.001, 873.0 + 0.001],
        )
        self.assertTrue(res.success, res.message)
        self.assertEqual(res.n_params, 2)
        self.assertGreater(res.var[0], 0.0)
        self.assertGreater(res.var[1], 0.0)

    def test_estimate_conditions_fail_fixed_P_and_T(self):
        assemblage = setup_assemblage()
        theta = np.array([500000000.0, 500.0])
        dataset_covariances = HP_2011_ds62.cov()
        with self.assertRaises(Exception):
            _ = estimate_conditions(
                assemblage,
                dataset_covariances,
                guessed_conditions=theta,
                pressure_bounds=[1000000000.0, 1000000000.0],
                temperature_bounds=[500.0, 500.0],
            )

    def test_estimate_conditions_free_compositional_vector(self):
        for scale in (0.0, 1.0):
            with self.subTest(scale=scale):
                assemblage = setup_assemblage(scale=scale)
                theta = np.array([500000000.0, 500.0])
                dataset_covariances = HP_2011_ds62.cov()
                for phase in assemblage.phases:
                    r = make_assemblage([phase]).reaction_basis
                    if r.shape[0] > 0:
                        phase.free_compositional_vectors = r
                res = estimate_conditions(
                    assemblage, dataset_covariances, guessed_conditions=theta
                )
                self.assertTrue(res.success, res.message)
                self.assertEqual(res.n_reactions, 18)
                self.assertEqual(res.n_params, 4)
                for phase in assemblage.phases[:-1]:
                    self.assertGreaterEqual(min(phase.site_occupancies), -1e-10)
                self.assertLess(min(assemblage.phases[1].molar_fractions), 0.0)

    def test_estimate_conditions_automatic_scaling(self):
        reference = None
        for guess in (
            [12000000000.0, 1000.0],
            [14000000000.0, 1400.0],
            [16000000000.0, 1800.0],
        ):
            with self.subTest(guess=guess):
                ol = burnman.minerals.SLB_2024.olivine()
                wad = burnman.minerals.SLB_2024.wadsleyite()
                ol.set_composition([0.9, 0.1])
                wad.set_composition([0.82, 0.18])
                ol.compositional_covariances = np.eye(2) * 1e-10
                wad.compositional_covariances = np.eye(2) * 0.0001
                assemblage = make_assemblage([ol, wad], [0.5, 0.5])
                assemblage.state_priors = np.array([13000000000.0, 1400.0])
                assemblage.state_inverse_covariances = np.diag([1e-18, 0.0001])
                data = {
                    "endmember_names": [*ol.endmember_names, *wad.endmember_names],
                    "covariance_matrix": np.eye(4) * 70.0**2,
                }
                result = estimate_conditions(
                    assemblage,
                    data,
                    include_state_misfit=True,
                    guessed_conditions=guess,
                )
                self.assertTrue(result.success, result.message)
                self.assertFloatEqual(result.fun, 0.9517, tol=1e-05)
                self.assertArraysAlmostEqual(
                    result.x / [1000000000.0, 1.0], [12.3656, 1380.6], tol=0.0001
                )
                if reference is not None:
                    error = np.linalg.solve(
                        np.linalg.cholesky(reference.xcov), result.x - reference.x
                    )
                    self.assertLess(np.linalg.norm(error), 0.002)
                reference = result

    def test_estimate_conditions_covariance_preserves_pressure_information(self):
        assemblage = setup_assemblage()
        assemblage.state_priors = np.array([400000000.0, 873.0])
        assemblage.state_inverse_covariances = np.diag([1e-18, 1.0])
        result = estimate_conditions(
            assemblage,
            HP_2011_ds62.cov(),
            include_state_misfit=True,
            guessed_conditions=assemblage.state_priors,
        )
        self.assertTrue(result.success, result.message)
        units = np.diag([1000000000.0, 1.0])
        jacobian = result.dadx @ units
        information = jacobian.T @ np.linalg.solve(result.acov, jacobian)
        information += units @ assemblage.state_inverse_covariances @ units
        expected = units @ np.linalg.inv(information) @ units
        self.assertGreater(result.var[0], 1000000.0)
        self.assertArraysAlmostEqual(result.xcov.ravel(), expected.ravel())

    def test_estimate_conditions_with_condition_priors(self):
        assemblage = setup_assemblage()
        theta = np.array([400000000.0, 873.0])
        dataset_covariances = HP_2011_ds62.cov()
        res_without_priors = estimate_conditions(
            assemblage, dataset_covariances, guessed_conditions=theta
        )
        assemblage.state_priors = np.array([1000000000.0, 873.0])
        assemblage.state_inverse_covariances = np.diag(
            np.array([1.0 / 100000000.0, 1.0 / 50.0]) ** 2
        )
        res_with_priors = estimate_conditions(
            assemblage,
            dataset_covariances,
            include_state_misfit=True,
            guessed_conditions=res_without_priors.x[:2],
        )
        self.assertTrue(res_without_priors.success)
        self.assertEqual(res_without_priors.n_reactions, 18)
        self.assertTrue(res_without_priors.x[0] < res_with_priors.x[0])

    def test_assemblage_affinity_misfit(self):
        sill = HP_2011_ds62.sill()
        ky = HP_2011_ds62.ky()
        assemblage = make_assemblage([sill, ky], [0.5, 0.5])
        equality_constraints = [["P", 1000000000.0], ["phase_fraction", (sill, 0.5)]]
        equilibrate(assemblage.formula, assemblage, equality_constraints)
        P_true = assemblage.pressure
        T_true = assemblage.temperature
        P_unc = 100000000.0
        T_unc = 10.0
        R = get_reaction_matrix(assemblage)
        with self.assertRaises(ValueError):
            assemblage_affinity_misfit(assemblage, R, include_state_uncertainties=True)
        assemblage.state_covariances = np.diag(np.array([P_unc, T_unc]) ** 2)
        chisqr = assemblage_affinity_misfit(
            assemblage, R, include_state_uncertainties=True
        )
        self.assertAlmostEqual(chisqr, 0.0, places=5)
        assemblage.set_state(P_true + P_unc, T_true)
        chisqr = assemblage_affinity_misfit(
            assemblage, R, include_state_uncertainties=False
        )
        self.assertTrue(chisqr > 1000000000000.0)
        assemblage.set_state(P_true + P_unc, T_true)
        chisqr = assemblage_affinity_misfit(
            assemblage, R, include_state_uncertainties=True
        )
        self.assertAlmostEqual(chisqr, 0.956585, places=4)
        assemblage.set_state(P_true, T_true + T_unc)
        chisqr = assemblage_affinity_misfit(
            assemblage, R, include_state_uncertainties=True
        )
        self.assertAlmostEqual(chisqr, 0.043427, places=4)

    def test_assemblage_state_misfit(self):
        sill = HP_2011_ds62.sill()
        ky = HP_2011_ds62.ky()
        assemblage = make_assemblage([sill, ky], [0.5, 0.5])
        equality_constraints = [["P", 1000000000.0], ["phase_fraction", (sill, 0.5)]]
        equilibrate(assemblage.formula, assemblage, equality_constraints)
        P_true = assemblage.pressure
        T_true = assemblage.temperature
        P_unc = 100000000.0
        T_unc = 10.0
        with self.assertRaises(ValueError):
            assemblage_state_misfit(assemblage)
        assemblage.state_priors = np.array([P_true, T_true])
        assemblage.state_inverse_covariances = np.diag(
            1.0 / np.array([P_unc, T_unc]) ** 2
        )
        chisqr = assemblage_state_misfit(assemblage)
        self.assertAlmostEqual(chisqr, 0.0, places=5)
        assemblage.set_state(P_true + P_unc, T_true)
        chisqr = assemblage_state_misfit(assemblage)
        self.assertAlmostEqual(chisqr, 1.0, places=5)
        assemblage.set_state(P_true, T_true + T_unc)
        chisqr = assemblage_state_misfit(assemblage)
        self.assertAlmostEqual(chisqr, 1.0, places=5)
