# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2012-2026 by the BurnMan Team.
"""Standalone port of upstream tests/test_fitting.py.

Reference: tests/reference/burnman.json.
Original numerical inputs, assertions and tolerances are retained.
"""

from original_test_support import BurnManTest, parameter_dict, set_parameter
import os
import numpy as np
import burnman_cpp as burnman
from burnman_cpp.optimize.linear_fitting import weighted_constrained_least_squares
from burnman_cpp.optimize.nonlinear_fitting import (
    NonLinearModel,
    nonlinear_least_squares_fit,
    calculate_jacobian,
    find_mle,
)
from burnman_cpp.utils.misc import attribute_function, pretty_string_values
from burnman_cpp.optimize.composition_fitting import (
    DummyCompositionSolution,
    fit_composition_to_solution,
)

path = os.path.dirname(os.path.abspath(__file__))


class JacobianLineModel(NonLinearModel):

    def __init__(self, in_place=True):
        self.in_place = in_place
        self.params = np.array([2.0, -1.0])
        self.bounds = np.array([[-np.inf, np.inf], [-np.inf, np.inf]])
        self.delta_params = np.array([1e-05, 1e-05])
        x = np.array([-1.0, 0.0, 2.0])
        self.data = np.column_stack((x, 2.0 * x - 1.0 + [0.1, -0.2, 0.3]))
        self.data_covariances = np.array([np.diag([0.0, 0.25**2])] * len(x))
        self.mle_tolerances = np.full(len(x), 1e-12)
        self.flags = [None] * len(x)

    def get_params(self):
        return parameter_dict(self.params)

    def set_params(self, values):
        values = np.clip(values, self.bounds[:, 0], self.bounds[:, 1])
        if self.in_place:
            set_parameter(self, slice(None, None, None), values)
        else:
            self.params = values.copy()

    def function(self, x, flag=None):
        return np.array(
            [
                x[0],
                parameter_dict(self.params)[0] * x[0] + parameter_dict(self.params)[1],
            ]
        )

    def normal(self, x, flag=None):
        normal = np.array([-parameter_dict(self.params)[0], 1.0])
        return normal / np.linalg.norm(normal)


class test_fitting(BurnManTest):

    def test_jacobian_in_place_setter(self):
        model = JacobianLineModel()
        baseline = model.get_params().copy()
        calculate_jacobian(model)
        expected = np.column_stack((model.data[:, 0], np.ones(len(model.data)))) / 0.25
        self.assertArraysAlmostEqual(
            model.jacobian.flatten(), expected.flatten(), tol=1e-07, tol_zero=1e-08
        )
        self.assertArraysAlmostEqual(
            model.get_params(), baseline, tol=0.0, tol_zero=0.0
        )

    def test_jacobian_uses_actual_parameter_steps(self):
        for lower, upper in ((-np.inf, 0.0), (0.0, np.inf), (-0.25, 0.5)):
            for sign in (-1.0, 1.0):
                with self.subTest(lower=lower, upper=upper, sign=sign):
                    model = JacobianLineModel(in_place=False)
                    baseline = model.get_params().copy()
                    model.bounds[0] = baseline[0] + np.array([lower, upper]) * 1e-05
                    model.delta_params[0] *= sign
                    calculate_jacobian(model)
                    expected = (
                        np.column_stack((model.data[:, 0], np.ones(len(model.data))))
                        / 0.25
                    )
                    self.assertArraysAlmostEqual(
                        model.jacobian.flatten(),
                        expected.flatten(),
                        tol=1e-07,
                        tol_zero=1e-08,
                    )
                    self.assertArraysAlmostEqual(
                        model.get_params(), baseline, tol=0.0, tol_zero=0.0
                    )

    def test_jacobian_restores_parameters_after_failure(self):
        for direction in (-1.0, 1.0):
            with self.subTest(failing_trial=direction):
                model = JacobianLineModel()
                baseline = model.get_params().copy()
                original_function = model.function

                def function(x, flag=None):
                    if direction * (model.get_params()[0] - baseline[0]) > 0.0:
                        raise RuntimeError("trial evaluation failed")
                    return original_function(x, flag)

                model.function = function
                with self.assertRaisesRegex(RuntimeError, "trial evaluation failed"):
                    calculate_jacobian(model)
                self.assertArraysAlmostEqual(
                    model.get_params(), baseline, tol=0.0, tol_zero=0.0
                )

    def test_jacobian_rejects_fixed_parameter(self):
        model = JacobianLineModel()
        baseline = model.get_params().copy()
        model.bounds[0] = baseline[0]
        with self.assertRaisesRegex(ValueError, "Cannot perturb parameter 0"):
            calculate_jacobian(model)
        self.assertArraysAlmostEqual(
            model.get_params(), baseline, tol=0.0, tol_zero=0.0
        )

    def test_in_place_fit_covariance(self):
        model = JacobianLineModel()
        design = np.column_stack((model.data[:, 0], np.ones(len(model.data))))
        expected = np.linalg.solve(design.T @ design, design.T @ model.data[:, 1])
        residuals = model.data[:, 1] - design @ expected
        covariance = (
            np.linalg.inv(design.T @ design)
            * (residuals @ residuals)
            / (len(model.data) - len(model.get_params()))
        )
        nonlinear_least_squares_fit(model, param_tolerance=1e-09)
        self.assertArraysAlmostEqual(model.popt, expected, tol=1e-07)
        self.assertArraysAlmostEqual(
            model.pcov.flatten(), covariance.flatten(), tol=1e-07
        )

    def test_linear_fit(self):
        i, x, Wx, y, Wy = np.loadtxt(f"{path}/data/Pearson_York.dat", unpack=True)
        data = np.array([x, y]).T
        cov = np.array([[1.0 / Wx, 0.0 * Wx], [0.0 * Wy, 1.0 / Wy]]).T

        class m(NonLinearModel):

            def __init__(self, data, cov, guessed_params, delta_params):
                self.data = data
                self.data_covariances = cov
                self.set_params(guessed_params)
                self.delta_params = delta_params
                self.mle_tolerances = np.array([0.1] * len(data[:, 0]))

            def set_params(self, param_values):
                self.params = param_values

            def get_params(self):
                return parameter_dict(self.params)

            def function(self, x, flag):
                return np.array(
                    [
                        x[0],
                        parameter_dict(self.params)[0] * x[0]
                        + parameter_dict(self.params)[1],
                    ]
                )

            def normal(self, x, flag):
                n = np.array([parameter_dict(self.params)[0], -1.0])
                return n / np.linalg.norm(n)

        guessed_params = np.array([-0.5, 5.5])
        delta_params = np.array([0.001, 0.001])
        fitted_curve = m(data, cov, guessed_params, delta_params)
        nonlinear_least_squares_fit(model=fitted_curve, param_tolerance=1e-05)
        self.assertArraysAlmostEqual([fitted_curve.WSS], [11.8663531941])

    def test_polynomial_fit(self):
        i, x, Wx, y, Wy = np.loadtxt(f"{path}/data/Pearson_York.dat", unpack=True)
        data = np.array([x, y]).T
        cov = np.array([[1.0 / Wx, 0.0 * Wx], [0.0 * Wy, 1.0 / Wy]]).T

        class m(NonLinearModel):

            def __init__(self, data, cov, guessed_params, delta_params):
                self.data = data
                self.data_covariances = cov
                self.set_params(guessed_params)
                self.delta_params = delta_params
                self.mle_tolerances = np.array([1e-08] * len(data[:, 0]))

            def set_params(self, param_values):
                self.params = param_values

            def get_params(self):
                return parameter_dict(self.params)

            def function(self, x, flag):
                return np.array(
                    [
                        x[0],
                        parameter_dict(self.params)[0] * x[0] * x[0] * x[0]
                        + parameter_dict(self.params)[1] * x[0] * x[0]
                        + parameter_dict(self.params)[2] * x[0]
                        + parameter_dict(self.params)[3],
                    ]
                )

            def normal(self, x, flag):
                n = np.array(
                    [
                        3.0 * parameter_dict(self.params)[0] * x[0] * x[0]
                        + 2.0 * parameter_dict(self.params)[1] * x[0]
                        + 1.0 * parameter_dict(self.params)[2],
                        -1.0,
                    ]
                )
                return n / np.linalg.norm(n)

        guessed_params = np.array([-0.012, 0.161, -1.15, 6.142])
        delta_params = np.array([1e-05, 1e-05, 1e-05, 1e-05])
        fitted_curve = m(data, cov, guessed_params, delta_params)
        nonlinear_least_squares_fit(model=fitted_curve, param_tolerance=1e-05)
        self.assertArraysAlmostEqual([fitted_curve.WSS], [10.486904577])

    def test_line_fit_matches_closed_form_weighted_least_squares(self):
        """
        Fit y = a*x + b to data with negligible x-uncertainty, and check
        that popt, pcov and goodness_of_fit match the closed-form weighted
        least squares solution (the fit reduces to ordinary WLS when the
        x-coordinate of each datum is essentially exact).
        """

        class LineModel(NonLinearModel):

            def __init__(self, data, cov, guessed_params):
                self.data = data
                self.data_covariances = cov
                self.set_params(guessed_params)
                self.delta_params = np.array([1e-06, 1e-06])
                self.mle_tolerances = np.array([1e-10] * len(data))

            def get_params(self):
                return parameter_dict(self.params)

            def set_params(self, param_values):
                self.params = np.array(param_values, dtype=float)

            def function(self, x, flag=None):
                a, b = parameter_dict(self.params)
                return np.array([x[0], a * x[0] + b])

            def normal(self, x, flag=None):
                a, _ = parameter_dict(self.params)
                n = np.array([-a, 1.0])
                return n / np.linalg.norm(n)

        np.random.seed(0)
        n = 100
        true_a, true_b = (2.5, -1.3)
        x = np.linspace(0.0, 10.0, n)
        sigma_y = np.full(n, 0.3)
        y = true_a * x + true_b + np.random.normal(scale=sigma_y)
        data = np.column_stack([x, y])
        cov = np.zeros((n, 2, 2))
        for i in range(n):
            cov[i] = np.diag([0.0, sigma_y[i] ** 2])
        model = LineModel(data, cov, guessed_params=[0.5, 0.5])
        nonlinear_least_squares_fit(model, param_tolerance=1e-10)
        W = 1.0 / sigma_y**2
        X = np.column_stack([x, np.ones(n)])
        XtWX = X.T @ (W[:, None] * X)
        XtWy = X.T @ (W * y)
        beta_wls = np.linalg.solve(XtWX, XtWy)
        cov_wls = np.linalg.inv(XtWX) * model.goodness_of_fit
        self.assertArraysAlmostEqual(model.popt, beta_wls, tol=1e-06)
        self.assertArraysAlmostEqual(model.pcov.flatten(), cov_wls.flatten(), tol=1e-06)
        slope = model.popt[0]
        expected_noise_variance = (
            model.goodness_of_fit * sigma_y[0] ** 2 / (1.0 + slope**2)
        )
        self.assertFloatEqual(model.noise_variance, expected_noise_variance)

    def test_basic_unweighted_least_squares_exact_answer(self):
        """Test simple unweighted least squares with exact answer."""
        A = np.array([[1, 0], [1, 1]])
        b = np.array([1, 3])
        popt, pcov, res = weighted_constrained_least_squares(A, b)
        self.assertArraysAlmostEqual(popt, [1.0, 2.0])
        self.assertArraysAlmostEqual(pcov[0], [1.0, -1.0])
        self.assertArraysAlmostEqual(pcov[1], [-1.0, 2.0])
        self.assertAlmostEqual(res, 0.0)

    def test_basic_unweighted_least_squares(self):
        """Test simple unweighted least squares."""
        A = np.array([[1, 1], [1, 2], [1, 3]])
        b = np.array([1, 2, 2])
        popt, pcov, res = weighted_constrained_least_squares(A, b)
        self.assertArraysAlmostEqual(popt, [2.0 / 3.0, 1.0 / 2.0])
        self.assertArraysAlmostEqual(pcov[0], [7.0 / 3.0, -1.0])
        self.assertArraysAlmostEqual(pcov[1], [-1.0, 1.0 / 2.0])
        self.assertAlmostEqual(res, 1.0 / 6.0)

    def test_basic_unit_weighted_least_squares(self):
        """Test simple unit weighted least squares."""
        A = np.array([[1, 1], [1, 2], [1, 3]])
        b = np.array([1, 2, 2])
        Cov_b = np.diag([1, 1, 1])
        popt, pcov, res = weighted_constrained_least_squares(A, b, Cov_b)
        self.assertArraysAlmostEqual(popt, [2.0 / 3.0, 1.0 / 2.0])
        self.assertArraysAlmostEqual(pcov[0], [7.0 / 3.0, -1.0])
        self.assertArraysAlmostEqual(pcov[1], [-1.0, 1.0 / 2.0])
        self.assertAlmostEqual(res, 1.0 / 6.0)

    def test_unevenly_weighted_least_squares(self):
        """Test weighted least squares with non-identity covariance."""
        A = np.array([[1, 1], [1, 2], [1, 3]])
        b = np.array([1, 2, 2])
        Cov_b = np.diag([1, 1, 2])
        popt, pcov, res = weighted_constrained_least_squares(A, b, Cov_b=Cov_b)
        self.assertArraysAlmostEqual(popt, [4.0 / 7.0, 4.0 / 7.0], tol_zero=1e-12)
        self.assertArraysAlmostEqual(pcov[0], [19.0 / 7.0, -9.0 / 7.0])
        self.assertArraysAlmostEqual(pcov[1], [-9.0 / 7.0, 5.0 / 7.0])
        self.assertAlmostEqual(res, 1.0 / 7.0)

    def test_equality_constraints_least_squares(self):
        """Test equality constraint (force x1 + x2 = 1)."""
        A = np.array([[1, 2], [3, 4], [5, 6]])
        b = np.array([7, 8, 9])
        C = np.array([[1, 1]])
        c = np.array([0.5])
        popt, _, res = weighted_constrained_least_squares(
            A, b, equality_constraints=[C, c]
        )
        self.assertArraysAlmostEqual(popt, [-6.0, 6.5], tol_zero=1e-12)
        self.assertAlmostEqual(res, 0.0)

    def test_inequality_constraints_least_squares(self):
        """Test inequality constraint (x1 >= 2, x2 >= 0)."""
        A = np.array([[1, 1], [1, 2], [1, 3]])
        b = np.array([1, 2, 2])
        D = np.array([[-1, 0], [0, -1]])
        d = np.array([-2, 0])
        popt, pcov, res = weighted_constrained_least_squares(
            A, b, inequality_constraints=[D, d]
        )
        self.assertArraysAlmostEqual(popt, [2.0, 0.0], tol_zero=1e-12)
        self.assertArraysAlmostEqual(pcov[0], [7.0 / 3.0, -1.0])
        self.assertArraysAlmostEqual(pcov[1], [-1.0, 1.0 / 2.0])
        self.assertAlmostEqual(res, 1.0)

    def test_rank_deficient_least_squares(self):
        """Test rank-deficient A with allow_rank_deficient=False → expect error."""
        A = np.array([[1, 2], [2, 4], [3, 6]])
        b = np.array([1, 2, 3])
        with self.assertRaises(Exception):
            _, _, _ = weighted_constrained_least_squares(A, b)

    def test_rank_deficient_allowed_least_squares(self):
        """Test rank-deficient A with allow_rank_deficient=True."""
        A = np.array([[1, 2], [2, 4], [3, 6]])
        b = np.array([1, 2, 3])
        popt, pcov, res = weighted_constrained_least_squares(
            A, b, allow_rank_deficient=True
        )
        self.assertAlmostEqual(res, 0.0)
        self.assertArraysAlmostEqual(A @ popt, b)

    def test_fit_PTV_data(self):
        fo = burnman.minerals.HP_2011_ds62.fo()
        pressures = np.linspace(1000000000.0, 2000000000.0, 8)
        temperatures = np.ones_like(pressures) * parameter_dict(fo.params)["T_0"]
        PTV = np.empty((len(pressures), 3))
        for i in range(len(pressures)):
            fo.set_state(pressures[i], temperatures[i])
            PTV[i] = [pressures[i], temperatures[i], fo.molar_volume]
        params = ["V_0", "K_0", "Kprime_0"]
        fitted_eos = burnman.eos_fitting.fit_PTV_data(fo, params, PTV, verbose=False)
        zeros = np.zeros_like(fitted_eos.pcov[0])
        self.assertArraysAlmostEqual(fitted_eos.pcov[0], zeros)

    def test_fit_PTp_data(self):
        fo = burnman.minerals.HP_2011_ds62.fo()
        pressures = np.linspace(1000000000.0, 2000000000.0, 8)
        temperatures = np.ones_like(pressures) * parameter_dict(fo.params)["T_0"]
        PTG = np.empty((len(pressures), 3))
        PTS = np.empty((len(pressures), 3))
        PTH = np.empty((len(pressures), 3))
        PTa = np.empty((len(pressures), 3))
        for i in range(len(pressures)):
            fo.set_state(pressures[i], temperatures[i])
            PTG[i] = [pressures[i], temperatures[i], fo.molar_gibbs]
            PTS[i] = [pressures[i], temperatures[i], fo.molar_entropy]
            PTH[i] = [pressures[i], temperatures[i], fo.molar_enthalpy]
            PTa[i] = [pressures[i], temperatures[i], fo.thermal_expansivity]
        params = ["V_0", "K_0", "Kprime_0"]
        for flag, data in [("gibbs", PTG), ("S", PTS), ("H", PTH), ("alpha", PTa)]:
            flags = [flag] * len(pressures)
            fitted_eos = burnman.eos_fitting.fit_PTp_data(
                fo, params, flags, data, verbose=False
            )
            zeros = np.zeros_like(fitted_eos.pcov[0])
            self.assertArraysAlmostEqual(fitted_eos.pcov[0], zeros)

    def test_fit_PTV_data_w_priors(self):
        fo = burnman.minerals.HP_2011_ds62.fo()
        pressures = np.linspace(1000000000.0, 2000000000.0, 8)
        temperatures = np.ones_like(pressures) * parameter_dict(fo.params)["T_0"]
        PTV = np.empty((len(pressures), 3))
        for i in range(len(pressures)):
            fo.set_state(pressures[i], temperatures[i])
            PTV[i] = [pressures[i], temperatures[i], fo.molar_volume]
        params = ["V_0", "K_0", "Kprime_0"]
        priors = [
            parameter_dict(fo.params)["V_0"],
            parameter_dict(fo.params)["K_0"],
            parameter_dict(fo.params)["Kprime_0"],
        ]
        invcov = np.linalg.inv(
            np.diag(
                np.power(
                    np.array(
                        [parameter_dict(fo.params)["V_0"] / 10.0, 1000000000.0, 1.0]
                    ),
                    2.0,
                )
            )
        )
        fitted_eos = burnman.eos_fitting.fit_PTV_data(
            fo,
            params,
            PTV,
            param_priors=priors,
            param_prior_inv_cov_matrix=invcov,
            verbose=False,
        )
        zeros = np.zeros_like(fitted_eos.pcov[0])
        self.assertArraysAlmostEqual(fitted_eos.pcov[0], zeros)

    def test_fit_bounded_PTV_data(self):
        fo = burnman.minerals.HP_2011_ds62.fo()
        pressures = np.linspace(0.0, 10000000000.0, 8)
        temperatures = np.ones_like(pressures) * parameter_dict(fo.params)["T_0"]
        PTV = np.empty((len(pressures), 3))
        for i in range(len(pressures)):
            fo.set_state(pressures[i], temperatures[i])
            PTV[i] = [pressures[i], temperatures[i], fo.molar_volume]
        PTV[0, 2] *= 1.01
        PTV[-1, 2] *= 0.99
        params = ["V_0", "K_0", "Kprime_0"]
        bounds = np.array([[0.0, np.inf], [0.0, np.inf], [3.0, 4.0]])
        fitted_eos = burnman.eos_fitting.fit_PTV_data(
            fo, params, PTV, bounds=bounds, verbose=False
        )
        cp_bands = burnman.nonlinear_fitting.confidence_prediction_bands(
            model=fitted_eos,
            x_array=PTV,
            confidence_interval=0.95,
            f=attribute_function(fo, "V"),
            flag="V",
        )
        self.assertEqual(len(cp_bands[0]), len(PTV))
        self.assertEqual(len(cp_bands), 4)
        self.assertFloatEqual(3.0, fitted_eos.popt[2])
        s = pretty_string_values(
            fitted_eos.popt,
            fitted_eos.pcov,
            extra_decimal_places=1,
            combine_value_and_sigma=False,
        )
        self.assertEqual(len(s), 3)
        self.assertEqual(len(s[0]), 3)
        self.assertEqual(len(s[1]), 3)
        self.assertEqual(len(s[2]), 3)
        s = pretty_string_values(
            fitted_eos.popt,
            fitted_eos.pcov,
            extra_decimal_places=1,
            combine_value_and_sigma=True,
        )
        self.assertEqual(len(s), 3)
        self.assertEqual(len(s[0]), 3)
        self.assertEqual(len(s[1]), 3)
        self.assertEqual(len(s[2]), 3)

    def test_PTV_data_eos_numerical_failure(self):
        fo = burnman.minerals.HP_2011_ds62.fo()
        pressures = np.linspace(1000000000.0, 2000000000.0, 8)
        temperatures = np.ones_like(pressures) * parameter_dict(fo.params)["T_0"]
        fo.set_state(1000000000.0, parameter_dict(fo.params)["T_0"])
        V0 = fo.molar_volume
        fo.set_state(2000000000.0, parameter_dict(fo.params)["T_0"])
        V1 = fo.molar_volume
        volumes = V0 + (V1 - V0) * (pressures - 1000000000.0) / 1000000000.0
        PTV = np.array([pressures, temperatures, volumes]).T
        params = ["V_0", "K_0", "Kprime_0"]
        with self.assertRaises(Exception):
            _ = burnman.eos_fitting.fit_PTV_data(fo, params, PTV, verbose=False)

    def test_PTV_data_eos_exception(self):
        fo = burnman.minerals.SLB_2011.forsterite()
        pressures = np.linspace(1000000000.0, 100000000000.0, 8)
        temperatures = np.ones_like(pressures) * parameter_dict(fo.params)["T_0"]
        fo.set_state(1000000000.0, parameter_dict(fo.params)["T_0"])
        V0 = fo.molar_volume
        fo.set_state(2000000000.0, parameter_dict(fo.params)["T_0"])
        V1 = fo.molar_volume
        volumes = V0 + (V1 - V0) * (pressures - 1000000000.0) / 1000000000.0
        PTV = np.array([pressures, temperatures, volumes]).T
        params = ["V_0", "K_0", "Kprime_0"]
        with self.assertRaises(Exception):
            _ = burnman.eos_fitting.fit_PTV_data(fo, params, PTV, verbose=False)

    def test_fit_VTF_data(self):
        fo = burnman.minerals.SLB_2011.forsterite()
        pressures = np.linspace(1000000000.0, 2000000000.0, 8)
        temperatures = np.ones_like(pressures) * parameter_dict(fo.params)["T_0"]
        VTF = np.empty((len(pressures), 3))
        for i in range(len(pressures)):
            fo.set_state(pressures[i], temperatures[i])
            VTF[i] = [fo.molar_volume, temperatures[i], fo.molar_helmholtz]
        flags = ["helmholtz"] * len(temperatures)
        params = ["F_0", "V_0", "K_0", "Kprime_0"]
        fitted_eos = burnman.eos_fitting.fit_VTp_data(
            fo, params, flags, VTF, verbose=False
        )
        zeros = np.zeros_like(fitted_eos.pcov[0])
        self.assertArraysAlmostEqual(fitted_eos.pcov[0], zeros)

    def test_fit_VTP_data(self):
        fo = burnman.minerals.HP_2011_ds62.fo()
        pressures = np.linspace(1000000000.0, 2000000000.0, 8)
        temperatures = np.ones_like(pressures) * parameter_dict(fo.params)["T_0"]
        VTP = np.empty((len(pressures), 3))
        for i in range(len(pressures)):
            fo.set_state(pressures[i], temperatures[i])
            VTP[i] = [fo.molar_volume, temperatures[i], pressures[i]]
        params = ["V_0", "K_0", "Kprime_0"]
        fitted_eos = burnman.eos_fitting.fit_VTP_data(fo, params, VTP, verbose=False)
        zeros = np.zeros_like(fitted_eos.pcov[0])
        self.assertArraysAlmostEqual(fitted_eos.pcov[0], zeros)

    def test_fit_VTp_data(self):
        fo = burnman.minerals.HP_2011_ds62.fo()
        pressures = np.linspace(1000000000.0, 2000000000.0, 8)
        temperatures = np.ones_like(pressures) * parameter_dict(fo.params)["T_0"]
        VTS = np.empty((len(pressures), 3))
        VTH = np.empty((len(pressures), 3))
        for i in range(len(pressures)):
            fo.set_state(pressures[i], temperatures[i])
            VTS[i] = [fo.molar_volume, temperatures[i], fo.molar_entropy]
            VTH[i] = [fo.molar_volume, temperatures[i], fo.molar_enthalpy]
        params = ["V_0", "K_0", "Kprime_0"]
        for flag, data in [("S", VTS), ("H", VTH)]:
            flags = [flag] * len(pressures)
            fitted_eos = burnman.eos_fitting.fit_VTp_data(
                fo, params, flags, data, verbose=False
            )
            zeros = np.zeros_like(fitted_eos.pcov[0])
            self.assertArraysAlmostEqual(fitted_eos.pcov[0], zeros, tol_zero=1e-05)

    def test_fit_VTP_data_w_priors(self):
        fo = burnman.minerals.HP_2011_ds62.fo()
        pressures = np.linspace(1000000000.0, 2000000000.0, 8)
        temperatures = np.ones_like(pressures) * parameter_dict(fo.params)["T_0"]
        VTP = np.empty((len(pressures), 3))
        for i in range(len(pressures)):
            fo.set_state(pressures[i], temperatures[i])
            VTP[i] = [fo.molar_volume, temperatures[i], pressures[i]]
        params = ["V_0", "K_0", "Kprime_0"]
        priors = [
            parameter_dict(fo.params)["V_0"],
            parameter_dict(fo.params)["K_0"],
            parameter_dict(fo.params)["Kprime_0"],
        ]
        invcov = np.linalg.inv(
            np.diag(
                np.power(
                    np.array(
                        [parameter_dict(fo.params)["V_0"] / 10.0, 1000000000.0, 1.0]
                    ),
                    2.0,
                )
            )
        )
        fitted_eos = burnman.eos_fitting.fit_VTP_data(
            fo,
            params,
            VTP,
            param_priors=priors,
            param_prior_inv_cov_matrix=invcov,
            verbose=False,
        )
        zeros = np.zeros_like(fitted_eos.pcov[0])
        self.assertArraysAlmostEqual(fitted_eos.pcov[0], zeros)

    def test_fit_bounded_VTP_data(self):
        fo = burnman.minerals.HP_2011_ds62.fo()
        pressures = np.linspace(0.0, 10000000000.0, 8)
        temperatures = np.ones_like(pressures) * parameter_dict(fo.params)["T_0"]
        VTP = np.empty((len(pressures), 3))
        for i in range(len(pressures)):
            fo.set_state(pressures[i], temperatures[i])
            VTP[i] = [fo.molar_volume, temperatures[i], pressures[i]]
        VTP[0, 0] *= 1.01
        VTP[-1, 0] *= 0.99
        params = ["V_0", "K_0", "Kprime_0"]
        bounds = np.array([[0.0, np.inf], [0.0, np.inf], [3.0, 4.0]])
        fitted_eos = burnman.eos_fitting.fit_VTP_data(
            fo, params, VTP, bounds=bounds, verbose=False
        )
        cp_bands = burnman.nonlinear_fitting.confidence_prediction_bands(
            model=fitted_eos,
            x_array=VTP,
            confidence_interval=0.95,
            f=attribute_function(fo, "P", using_volume=True),
            flag="P",
        )
        self.assertEqual(len(cp_bands[0]), len(VTP))
        self.assertEqual(len(cp_bands), 4)
        self.assertFloatEqual(3.0, fitted_eos.popt[2])
        s = pretty_string_values(
            fitted_eos.popt,
            fitted_eos.pcov,
            extra_decimal_places=1,
            combine_value_and_sigma=False,
        )
        self.assertEqual(len(s), 3)
        self.assertEqual(len(s[0]), 3)
        self.assertEqual(len(s[1]), 3)
        self.assertEqual(len(s[2]), 3)
        s = pretty_string_values(
            fitted_eos.popt,
            fitted_eos.pcov,
            extra_decimal_places=1,
            combine_value_and_sigma=True,
        )
        self.assertEqual(len(s), 3)
        self.assertEqual(len(s[0]), 3)
        self.assertEqual(len(s[1]), 3)
        self.assertEqual(len(s[2]), 3)

    def test_VTP_data_eos_numerical_failure(self):
        fo = burnman.minerals.HP_2011_ds62.fo()
        pressures = np.linspace(1000000000.0, 2000000000.0, 8)
        temperatures = np.ones_like(pressures) * parameter_dict(fo.params)["T_0"]
        fo.set_state(1000000000.0, parameter_dict(fo.params)["T_0"])
        V0 = fo.molar_volume
        fo.set_state(2000000000.0, parameter_dict(fo.params)["T_0"])
        V1 = fo.molar_volume
        volumes = V0 + (V1 - V0) * (pressures - 1000000000.0) / 1000000000.0
        VTP = np.array([volumes, temperatures, pressures]).T
        params = ["V_0", "K_0", "Kprime_0"]
        with self.assertRaises(Exception):
            _ = burnman.eos_fitting.fit_VTP_data(fo, params, VTP, verbose=False)

    def test_VTP_data_eos_exception(self):
        fo = burnman.minerals.SLB_2011.forsterite()
        pressures = np.linspace(1000000000.0, 100000000000.0, 8)
        temperatures = np.ones_like(pressures) * parameter_dict(fo.params)["T_0"]
        fo.set_state(1000000000.0, parameter_dict(fo.params)["T_0"])
        V0 = fo.molar_volume
        fo.set_state(2000000000.0, parameter_dict(fo.params)["T_0"])
        V1 = fo.molar_volume
        volumes = V0 + (V1 - V0) * (pressures - 1000000000.0) / 1000000000.0
        VTP = np.array([volumes, temperatures, pressures]).T
        params = ["V_0", "K_0", "Kprime_0"]
        with self.assertRaises(Exception):
            _ = burnman.eos_fitting.fit_VTP_data(fo, params, VTP, verbose=False)

    def test_fit_composition_to_solution(self):
        gt = burnman.minerals.JH_2015.garnet()
        fitted_species = ["Fe", "Ca", "Mg", "Cr", "Al", "Si", "Fe3+"]
        species_amounts = np.array([1.1, 2.0, 0.0, 0, 1.9, 3.0, 0.1])
        species_covariances = np.eye(7) * 0.01 * 0.01
        species_conversions = {"Fe3+": {"Fef_B": 1.0}}
        np.random.seed(100)
        species_amounts = np.random.multivariate_normal(
            species_amounts, species_covariances
        )
        popt, _, _ = fit_composition_to_solution(
            gt,
            fitted_species,
            species_amounts,
            species_covariances,
            species_conversions,
        )
        gt.set_composition(popt)
        f = [0.004, 0.328, 0.619, 0.048, 0.0]
        f2 = np.round(gt.molar_fractions, 3)
        self.assertArraysAlmostEqual(f, f2)

    def test_composition_repeated_observations_preserve_zero_constraints(self):
        solution = DummyCompositionSolution(["MgO", "FeO"], ["[Mg]O", "[Fe]O"])
        for duplicate in ("Mg", "Mg_A"):
            for zero_first in (False, True):
                for normalize in (False, True):
                    with self.subTest(
                        duplicate=duplicate, zero_first=zero_first, normalize=normalize
                    ):
                        values = [0.0, 1.0, 2.0] if zero_first else [1.0, 0.0, 2.0]
                        amounts, _, residual = fit_composition_to_solution(
                            solution,
                            ["Mg", duplicate, "Fe"],
                            values,
                            None,
                            normalize=normalize,
                        )
                        self.assertArraysAlmostEqual(
                            amounts, [0.0, 1.0 if normalize else 2.0], tol_zero=1e-08
                        )
                        self.assertFloatEqual(residual, 0.5 if normalize else 1.0)

    def test_composition_zero_constraints_ignore_empty_conversion_rows(self):
        solution = DummyCompositionSolution(["MgO", "FeO"], ["[Mg]O", "[Fe]O"])
        amounts, _, residual = fit_composition_to_solution(
            solution,
            ["Mg", "no_species", "Mg_A", "Mg", "Fe"],
            [1.0, 0.0, 0.0, 0.0, 2.0],
            None,
            {"no_species": {"Mg": 0.0}},
            normalize=False,
        )
        self.assertArraysAlmostEqual(amounts, [0.0, 2.0], tol_zero=1e-08)
        self.assertFloatEqual(residual, 1.0)

    def test_composition_repeated_observation_zero_tolerance(self):
        solution = DummyCompositionSolution(["MgO", "FeO"], ["[Mg]O", "[Fe]O"])
        for small, expected_mg in ((1e-11, 0.0), (1e-09, (1.0 + 1e-09) / 2.0)):
            with self.subTest(observation=small):
                amounts, _, _ = fit_composition_to_solution(
                    solution,
                    ["Mg", "Mg", "Fe"],
                    [1.0, small, 2.0],
                    None,
                    normalize=False,
                )
                self.assertArraysAlmostEqual(
                    amounts, [expected_mg, 2.0], tol_zero=1e-08
                )

    def test_composition_conversion_coefficients(self):
        solution = DummyCompositionSolution(["MgO", "FeO"], ["[Mg]O", "[Fe]O"])
        conversions = {"twice_Mg": {"Mg": 2.0, "Fe_A": 0.0}, "half_Fe": {"Fe_A": 0.5}}
        for normalize in (False, True):
            with self.subTest(normalize=normalize):
                converted = fit_composition_to_solution(
                    solution,
                    ["twice_Mg", "half_Fe"],
                    [1.2, 0.7],
                    np.eye(2),
                    conversions,
                    normalize=normalize,
                )
                direct = fit_composition_to_solution(
                    solution,
                    ["Mg", "Fe_A"],
                    [0.6, 1.4],
                    np.diag([0.25, 4.0]),
                    normalize=normalize,
                )
                expected = [0.3, 0.7] if normalize else [0.6, 1.4]
                self.assertArraysAlmostEqual(converted[0], expected)
                self.assertArraysAlmostEqual(converted[0], direct[0])
                self.assertArraysAlmostEqual(
                    converted[1].flatten(), direct[1].flatten()
                )
                self.assertFloatEqual(converted[2], direct[2])

    def test_composition_conversion_equal_site_amounts(self):
        solution = DummyCompositionSolution(
            ["MgFeSiO4", "MgFeSiO4"], ["[Mg][Fe]SiO4", "[Fe][Mg]SiO4"]
        )
        amounts, covariance, residual = fit_composition_to_solution(
            solution,
            ["Mg_A", "Mg_B", "Mg_equal"],
            [0.8, 0.2, 0.0],
            np.eye(3),
            {"Mg_equal": {"Mg_A": 1.0, "Mg_B": -1.0}},
        )
        self.assertArraysAlmostEqual(amounts, [0.5, 0.5])
        self.assertArraysAlmostEqual(
            covariance.flatten(), np.array([1.0, -1.0, -1.0, 1.0]) / 6.0
        )
        self.assertFloatEqual(residual, 0.18)

    def test_composition_conversion_uses_earlier_definition(self):
        solution = DummyCompositionSolution(
            ["MgFeSiO4", "MgFeSiO4"], ["[Mg][Fe]SiO4", "[Fe][Mg]SiO4"]
        )
        conversions = {
            "twice_Mg": {"Mg_A": 2.0},
            "Mg_difference": {"twice_Mg": 1.5, "Mg_B": -0.25},
        }
        converted = fit_composition_to_solution(
            solution, ["Mg_difference", "Fe_A"], [0.725, 0.7], np.eye(2), conversions
        )
        direct = fit_composition_to_solution(
            solution,
            ["Mg_difference", "Fe_A"],
            [0.725, 0.7],
            np.eye(2),
            {"Mg_difference": {"Mg_A": 3.0, "Mg_B": -0.25}},
        )
        self.assertArraysAlmostEqual(converted[0], [0.3, 0.7])
        self.assertArraysAlmostEqual(converted[0], direct[0])
        self.assertArraysAlmostEqual(converted[1].flatten(), direct[1].flatten())
        self.assertFloatEqual(converted[2], direct[2])

    def test_fit_leaves_jacobian_and_residuals_consistent_with_popt(self):
        """
        After fitting, the Jacobian and weighted residuals should correspond to
        the final parameters in model.popt.
        """
        i, x, Wx, y, Wy = np.loadtxt(f"{path}/data/Pearson_York.dat", unpack=True)
        data = np.array([x, y]).T
        cov = np.array([[1.0 / Wx, 0.0 * Wx], [0.0 * Wy, 1.0 / Wy]]).T

        class m(NonLinearModel):

            def __init__(self, data, cov, guessed_params, delta_params):
                self.data = data
                self.data_covariances = cov
                self.set_params(guessed_params)
                self.delta_params = delta_params
                self.mle_tolerances = np.array([0.1] * len(data[:, 0]))

            def set_params(self, param_values):
                self.params = param_values

            def get_params(self):
                return parameter_dict(self.params)

            def function(self, x, flag):
                return np.array(
                    [
                        x[0],
                        parameter_dict(self.params)[0] * x[0]
                        + parameter_dict(self.params)[1],
                    ]
                )

            def normal(self, x, flag):
                n = np.array([parameter_dict(self.params)[0], -1.0])
                return n / np.linalg.norm(n)

        guessed_params = np.array([-0.5, 5.5])
        delta_params = np.array([0.001, 0.001])
        fitted_curve = m(data, cov, guessed_params, delta_params)
        nonlinear_least_squares_fit(model=fitted_curve, param_tolerance=1e-05)
        popt = np.copy(fitted_curve.popt)
        cached_jacobian = np.copy(fitted_curve.jacobian)
        cached_residuals = np.copy(fitted_curve.weighted_residuals)
        calculate_jacobian(fitted_curve)
        fresh_jacobian = np.copy(fitted_curve.jacobian)
        _, fresh_residuals, _ = find_mle(fitted_curve)
        self.assertArraysAlmostEqual(fitted_curve.get_params(), popt)
        self.assertArraysAlmostEqual(cached_residuals, fresh_residuals)
        self.assertArraysAlmostEqual(
            cached_jacobian.flatten(), fresh_jacobian.flatten()
        )
