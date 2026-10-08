"""Public constants retain the native values and thermodynamic units."""

import importlib

import numpy as np
import pytest

import burnman_cpp as bm


def test_constants_module_is_public_and_uses_native_values():
    from burnman_cpp import constants
    from burnman_cpp.constants import gas_constant

    assert importlib.import_module("burnman_cpp.constants") is constants
    assert "constants" in bm.__all__
    expected = dict(
        gas_constant=8.31446261815324,
        Avogadro=6.02214076e23,
        Boltzmann=1.380649e-23,
        G=6.67430e-11,
        Dirac=1.0545718176461565e-34,
        invcm=11.9627,
        logish_eps=1.0e-7,
    )
    assert set(constants.__all__) == set(expected)
    for name, value in expected.items():
        assert getattr(constants, name) == getattr(bm._core.constants, name) == value
    assert gas_constant == pytest.approx(
        constants.Avogadro * constants.Boltzmann, rel=2.0e-16
    )


def test_exported_gas_constant_matches_native_mixing_entropy(endmembers):
    fractions = np.array([0.3, 0.7])
    phase = bm.Solution(bm.IdealSolution(endmembers), fractions)
    phase.set_state(2.0e9, 1700.0)
    expected_entropy = -bm.constants.gas_constant * np.dot(fractions, np.log(fractions))
    assert phase.excess_entropy == pytest.approx(expected_entropy, rel=1.0e-14)
    assert phase.excess_gibbs == pytest.approx(-1700.0 * expected_entropy, rel=1.0e-14)
