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

"""Nonnegative phase amounts and the Bertka–Fei experimental compositions."""

from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.optimize import nnls

import burnman_cpp as bm
from burnman_cpp.optimize.composition_fitting import (
    fit_phase_proportions_to_bulk_composition as fit,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_FILE = ROOT / "examples/data/input_fitting/Bertka_Fei_1997_mars_mantle.dat"


def test_exact_mixture_and_unnormalized_phase_amounts():
    assert (
        bm.optimize.composition_fitting.fit_phase_proportions_to_bulk_composition is fit
    )
    phases = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
    bulk = np.array([0.25, 0.75, 1.0])
    amounts, covariance, residual = fit(phases, bulk)
    assert_allclose(amounts, [0.25, 0.75], rtol=1e-13)
    assert_allclose(phases @ amounts, bulk, rtol=1e-13)
    assert_allclose(covariance, [[2 / 3, -1 / 3], [-1 / 3, 2 / 3]], rtol=1e-13)
    assert residual < 1e-25
    scaled_amounts, scaled_covariance, scaled_residual = fit(phases, 4 * bulk)
    assert_allclose(scaled_amounts, [1, 3], rtol=1e-13)
    assert_allclose(scaled_covariance, covariance, rtol=1e-13)
    assert scaled_residual < 1e-25


def test_inexact_bulk_reaches_a_nonnegative_boundary():
    phases = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)
    bulk = np.array([-1, 2, 0], dtype=float)
    amounts, covariance, residual = fit(phases, bulk)
    assert_allclose(amounts, [0, 1], atol=1e-10)
    assert_allclose(covariance, [[2 / 3, -1 / 3], [-1 / 3, 2 / 3]], rtol=1e-13)
    assert_allclose(residual, 3.0, rtol=1e-12)
    zero, _, zero_residual = fit(phases, np.zeros(3))
    assert_allclose(zero, [0, 0], atol=0)
    assert zero_residual == 0


@pytest.mark.parametrize(
    "phases,bulk",
    [
        (np.empty((3, 0)), np.ones(3)),
        (np.eye(2), [1]),
        (np.eye(2), [1, np.nan]),
        ([[1, 0], [np.inf, 1]], [1, 2]),
        ([[1, 1], [2, 2]], [1, 2]),
    ],
)
def test_invalid_or_underdetermined_compositions_raise(phases, bulk):
    with pytest.raises(ValueError):
        fit(phases, bulk)


def test_all_mars_mantle_experiments_match_independent_nonnegative_fits():
    data = np.genfromtxt(DATA_FILE, names=True, dtype=None, encoding="utf8")
    compositions = np.array([data[name] for name in data.dtype.names[:6]])
    bulk = compositions[:, 0]
    for sample in np.unique(data["sample"]):
        if sample == "bulk_composition":
            continue
        phases = compositions[:, data["sample"] == sample]
        amounts, covariance, residual = fit(phases, bulk)
        expected, residual_norm = nnls(phases, bulk)
        assert_allclose(amounts, expected, atol=1e-7, rtol=1e-6, err_msg=sample)
        assert amounts.min() >= -1e-10, sample
        assert_allclose(
            residual, residual_norm**2, atol=1e-10, rtol=1e-9, err_msg=sample
        )
        inverse = np.linalg.pinv(phases)
        assert_allclose(
            covariance, inverse @ inverse.T, rtol=1e-8, atol=1e-9, err_msg=sample
        )


def test_optional_python_burnman_parity_and_data_provenance():
    burnman = pytest.importorskip("burnman")
    pytest.importorskip("cvxpy")
    from burnman.optimize.composition_fitting import (
        fit_phase_proportions_to_bulk_composition,
    )

    source = Path(burnman.__file__).parent / "data/input_fitting" / DATA_FILE.name
    assert DATA_FILE.read_bytes() == source.read_bytes()
    data = np.genfromtxt(DATA_FILE, names=True, dtype=None, encoding="utf8")
    compositions = np.array([data[name] for name in data.dtype.names[:6]])
    bulk = compositions[:, 0]
    for sample in np.unique(data["sample"]):
        if sample == "bulk_composition":
            continue
        phases = compositions[:, data["sample"] == sample]
        actual = fit(phases, bulk)
        expected = fit_phase_proportions_to_bulk_composition(phases, bulk)
        assert_allclose(actual[0], expected[0], rtol=1e-5, atol=1e-7, err_msg=sample)
        assert_allclose(actual[1], expected[1], rtol=1e-6, atol=1e-9, err_msg=sample)
        assert_allclose(actual[2], expected[2], rtol=1e-7, atol=1e-10, err_msg=sample)
