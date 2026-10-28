# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Composite evaluation and the original manual-equilibration examples."""

from pathlib import Path
import runpy

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

import burnman_cpp as bm
from burnman_cpp.tools.chemistry import reaction_matrix_as_strings


@pytest.mark.parametrize("set_state", [False, True])
def test_evaluate_preserves_nested_material_and_phase_states(set_state):
    ol = bm.minerals.SLB_2011.mg_fe_olivine()
    ol.set_composition([0.9, 0.1])
    inner = bm.Assemblage([ol], [1.0])
    outer = bm.Assemblage([inner, bm.minerals.SLB_2011.periclase()], [0.8, 0.2])
    if set_state:
        outer.set_state(2.0e9, 1100.0)
    # A phase shared with another assemblage can have its own state.
    ol.set_state(3.0e9, 1300.0)
    density = ol.density
    pressures = np.array([[1.0e9, 5.0e9], [10.0e9, 20.0e9]])
    temperatures = np.full_like(pressures, 1500.0)
    values = outer.evaluate(["P", "T", "rho", "v_p"], pressures, temperatures)
    assert values.shape == (4, 2, 2)
    assert_array_equal(values[0], pressures)
    assert_array_equal(values[1], temperatures)
    assert (values[2:] > 0.0).all()
    assert outer.has_state() == set_state
    assert inner.has_state() == set_state
    if set_state:
        assert outer.pressure == inner.pressure == 2.0e9
        assert outer.temperature == inner.temperature == 1100.0
    assert ol.pressure == 3.0e9
    assert ol.temperature == 1300.0
    assert ol.density == density
    assert_array_equal(ol.molar_fractions, [0.9, 0.1])
    with pytest.raises(ValueError, match="Unknown scalar material property"):
        outer.evaluate(["rho", "unknown"], pressures, temperatures)
    assert outer.has_state() == set_state
    assert ol.density == density


def test_evaluate_scalar_empty_and_noncontiguous_arrays():
    mineral = bm.minerals.SLB_2011.forsterite()
    assert_array_equal(mineral.evaluate(["P", "T"], 1.0e9, 1000.0), [1.0e9, 1000.0])
    pressures = np.arange(12.0).reshape(3, 4).T * 1.0e9
    temperatures = np.full((3, 4), 1500.0).T
    assert_array_equal(mineral.evaluate(["P"], pressures, temperatures)[0], pressures)
    assert mineral.evaluate(["rho"], np.empty((0, 2)), np.empty((0, 2))).shape == (
        1,
        0,
        2,
    )
    assert mineral.evaluate([], [1.0e9], [1000.0]).shape == (0, 1)
    for pressures, temperatures in [
        ([1.0e9], [1000.0, 1500.0]),
        ([[1.0e9]], [1000.0]),
        ([np.nan], [1000.0]),
        ([1.0e9], [-1.0]),
        ([1.0e9], [np.inf]),
    ]:
        with pytest.raises(ValueError):
            mineral.evaluate(["rho"], pressures, temperatures)
    assert not mineral.has_state()


def test_reaction_basis_strings_use_current_endmember_names():
    ol = bm.minerals.SLB_2011.mg_fe_olivine()
    wad = bm.minerals.SLB_2011.mg_fe_wadsleyite()
    rock = bm.Assemblage([ol, wad], [0.5, 0.5])
    assert rock.reaction_basis_as_strings == reaction_matrix_as_strings(
        rock.reaction_basis, rock.endmember_names
    )
    assert len(rock.reaction_basis_as_strings) == rock.n_reactions == 2
    strings = rock.reaction_basis_as_strings
    strings.clear()
    assert len(rock.reaction_basis_as_strings) == 2
    independent = bm.Assemblage([bm.minerals.SLB_2011.forsterite()], [1.0])
    assert independent.reaction_basis_as_strings == []
    # The property also belongs to individual solid solutions.
    assert ol.reaction_basis_as_strings == []


def test_composite_example_equilibria_and_reference_landmarks(monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: plt.close("all"))
    example = Path(__file__).parents[2] / "examples/example_composite.py"
    result = runpy.run_path(str(example), run_name="__main__")
    # Landmarks printed by the original Python BurnMan example.
    assert_allclose(result["rho"][0], 4446.348, rtol=0.0, atol=0.001)
    assert_allclose(result["Vp"][0] / 1.0e3, 11.244, rtol=0.0, atol=0.001)
    assert_allclose(result["pt"][0] / 1.0e9, 0.431, rtol=0.0, atol=0.001)
    assert_allclose(result["pt"][1], 809.343, rtol=0.0, atol=0.001)
    assert_allclose(result["px"][1], 0.1648, rtol=0.0, atol=0.0001)
    assert len(result["pressures"]) == 21
    assert len(result["sol"]) == 20  # The last point is the pure-Mg limit.
    assert_allclose(result["x_fe_ols"][[0, -1]], [0.440, 0.0], atol=0.001, rtol=0.0)
    assert_allclose(result["x_fe_wads"][[0, -1]], [0.673, 0.0], atol=0.001, rtol=0.0)
    rock = result["rock"]
    assert_allclose(rock.phases[0].molar_fractions, [1.0, 0.0], atol=0.0, rtol=0.0)
    assert_allclose(rock.phases[1].molar_fractions, [1.0, 0.0], atol=0.0, rtol=0.0)
    assert_allclose(result["fo_mwd"].reaction_affinities, 0.0, atol=1.0e-6, rtol=0.0)
    for pressure, solution in zip(result["pressures"][:-1], result["sol"]):
        assert solution.success, solution.message
        bm.set_composition_and_state_from_parameters(rock, solution.x)
        assert_allclose(rock.pressure, pressure, rtol=0.0, atol=1.0)
        assert_allclose(rock.temperature, 1600.0, rtol=0.0, atol=1.0e-6)
        assert_allclose(rock.molar_fractions, [0.5, 0.5], atol=1.0e-8, rtol=0.0)
        assert_allclose(rock.reaction_affinities, 0.0, atol=1.0e-3, rtol=0.0)
        formula = rock.formula
        assert_allclose(
            [formula["Mg"] + formula["Fe"], formula["Si"], formula["O"]], [2, 1, 4]
        )
