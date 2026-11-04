# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Composite evaluation and reaction-basis formatting."""

import numpy as np
from numpy.testing import assert_array_equal
import pytest

import burnman_cpp as bm
from conftest import assert_material_amount_scaling
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


@pytest.mark.parametrize("nested", [False, True])
def test_intensive_and_extensive_composites(nested):
    child = bm.minerals.SLB_2011.forsterite()
    if nested:
        child = bm.Assemblage([child, bm.minerals.SLB_2011.fayalite()], [0.3, 0.7])
        child.n_moles = 3.0
    phase = bm.Assemblage([child, bm.minerals.SLB_2011.periclase()], [0.6, 0.4])
    phase.set_state(20e9, 1000.0)
    assert_material_amount_scaling(phase)
