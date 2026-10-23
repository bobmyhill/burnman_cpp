"""Thermodynamic consistency of the illustrative least-squares liquid."""

import importlib
from pathlib import Path

import numpy as np
import pytest

import burnman_cpp as bm


@pytest.fixture
def example(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "examples"))
    return importlib.import_module("example_dekoker_2013")


@pytest.mark.parametrize(
    "pressure,temperature", [(0.0, 3000.0), (25e9, 3000.0), (136e9, 5000.0)]
)
def test_least_squares_gibbs_derivatives_and_partial_potentials(
    example, pressure, temperature
):
    model = example.LeastSquaresLiquid(use_seifertite=False)
    state = lambda p, t: model.state(float(p), float(t))[2]
    w = state(pressure, temperature)
    dp, dt = 1e6, 0.1
    numerical_s = -(
        state(pressure, temperature + dt)[:, 0]
        - state(pressure, temperature - dt)[:, 0]
    ) / (2 * dt)
    numerical_v = (
        state(pressure + dp, temperature)[:, 0]
        - state(pressure - dp, temperature)[:, 0]
    ) / (2 * dp)
    np.testing.assert_allclose(numerical_s, w[:, 2], rtol=1e-8, atol=1e-7)
    np.testing.assert_allclose(numerical_v, w[:, 3], rtol=1e-6, atol=1e-13)
    np.testing.assert_allclose(w[:, 0] + temperature * w[:, 2], w[:, 1], atol=1e-8)
    x, dx = 0.63, 1e-5
    mu = model.chemical_potentials(x, pressure, temperature)
    assert np.array([1 - x, x]) @ mu == pytest.approx(
        model.gibbs(x, pressure, temperature), abs=1e-8
    )
    dgdx = (
        model.gibbs(x + dx, pressure, temperature)
        - model.gibbs(x - dx, pressure, temperature)
    ) / (2 * dx)
    assert mu[1] - mu[0] == pytest.approx(dgdx, abs=0.003)


def test_example_keeps_original_liquids_and_solid_entropies(example):
    model = example.LeastSquaresLiquid()
    model.state(25e9, 2900.0)
    model.saturation_residuals(0.5, 25e9, 2900.0)
    for name, phase in zip(example.COMPOSITIONS, model.liquids):
        original = getattr(bm.minerals.DKS_2013_liquids, name + "_liquid")()
        original.set_state(25e9, 2900.0)
        assert phase.molar_entropy == pytest.approx(original.molar_entropy, abs=1e-10)
    for name, phase in zip(("periclase", "perovskite", "stishovite"), model.solids):
        assert (
            phase.params.S_0 == getattr(bm.minerals.DKS_2013_solids, name)().params.S_0
        )


@pytest.mark.parametrize(
    "pair,guess,expected",
    [
        ((0, 1), [0.41, 4950.0], [0.4150334055996865, 5010.028560226677]),
        ((1, 2), [0.71, 4550.0], [0.7153183565254315, 4549.632032141942]),
    ],
)
def test_final_example_retains_audited_least_squares_equilibria(
    example, pair, guess, expected
):
    # These are regression values for this illustrative approximation, not
    # claimed matches to the authors' continuous mixing or melting model.
    model = example.LeastSquaresLiquid()
    result, error = model.eutectic(136e9, pair, guess)
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-6)
    assert error < 1e-6
