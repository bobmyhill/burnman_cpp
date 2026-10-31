# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2012-2026 by the BurnMan Team.
"""Standalone port of upstream tests/test_einstein.py.

Reference: tests/reference/burnman.json.
Original numerical inputs, assertions and tolerances are retained.
"""

from original_test_support import BurnManTest
from burnman_cpp.eos import einstein


class Einstein(BurnManTest):

    def test_theta_derivatives(self):
        T = 300.0
        debye_T = 1100.0
        n = 2.0
        delta = 1e-06
        dF_dTheta_num = (
            einstein.helmholtz_energy(T, debye_T + delta, n)
            - einstein.helmholtz_energy(T, debye_T - delta, n)
        ) / (2 * delta)
        d2F_dTheta2_num = (
            einstein.dhelmholtz_dTheta(T, debye_T + delta, n)
            - einstein.dhelmholtz_dTheta(T, debye_T - delta, n)
        ) / (2 * delta)
        dS_dTheta_num = (
            einstein.entropy(T, debye_T + delta, n)
            - einstein.entropy(T, debye_T - delta, n)
        ) / (2 * delta)
        dF_dTheta_analytical = einstein.dhelmholtz_dTheta(T, debye_T, n)
        d2F_dTheta2_analytical = einstein.d2helmholtz_dTheta2(T, debye_T, n)
        dS_dTheta_analytical = einstein.dentropy_dTheta(T, debye_T, n)
        self.assertFloatEqual(dF_dTheta_num, dF_dTheta_analytical)
        self.assertFloatEqual(d2F_dTheta2_num, d2F_dTheta2_analytical)
        self.assertFloatEqual(dS_dTheta_num, dS_dTheta_analytical)
