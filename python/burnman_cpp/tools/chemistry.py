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

"""Reaction equilibria, component fugacities and shock temperatures (SI units)."""

from fractions import Fraction

import numpy as np

from .._core import chemistry as _chemistry
from .._core.chemistry import (
    dictionarize_formula,
    equilibrium_pressure,
    equilibrium_temperature,
    formula_mass,
    fugacity,
    hugoniot,
    invariant_point,
    reactions_from_stoichiometric_matrix,
    relative_fugacity,
)


def reaction_matrix_as_strings(reaction_matrix, compound_names):
    """Format signed reaction coefficients, with reactants on the left.

    Rows are reactions and columns correspond to ``compound_names``. Coefficients
    are displayed as integers or rational fractions where possible.
    """
    reactions = np.asarray(reaction_matrix, dtype=float)
    names = list(compound_names)
    if reactions.ndim != 2 or reactions.shape[1] != len(names):
        raise ValueError("Reaction columns must match the compound names.")
    if not np.isfinite(reactions).all():
        raise ValueError("Reaction coefficients must be finite.")
    strings = []
    for reaction in reactions:
        sides = [[], []]
        for coefficient, name in zip(reaction, names):
            if abs(coefficient) > 1.0e-10:
                amount = Fraction(float(abs(coefficient))).limit_denominator(1000000)
                sides[int(coefficient > 0.0)].append(f"{amount} {name}")
        strings.append(f"{' + '.join(sides[0])} = {' + '.join(sides[1])}")
    return strings


def reactions_from_formulae(formulae, compound_names, return_strings=True):
    """Enumerate minimal balanced reactions between fixed-composition compounds.

    Formulae may be element-amount dictionaries or strings such as ``Mg2SiO4``.
    Both reaction directions are included, with primitive integer coefficients.
    Return formatted strings by default; ``return_strings=False`` returns a
    NumPy array with one column per compound. Names are used only for strings.
    """
    formulae = [
        dictionarize_formula(formula) if isinstance(formula, str) else formula
        for formula in formulae
    ]
    reactions = _chemistry.reactions_from_formulae(formulae)
    if return_strings:
        return reaction_matrix_as_strings(reactions, compound_names)
    return reactions


__all__ = [
    "dictionarize_formula",
    "formula_mass",
    "fugacity",
    "relative_fugacity",
    "equilibrium_pressure",
    "equilibrium_temperature",
    "invariant_point",
    "hugoniot",
    "reactions_from_stoichiometric_matrix",
    "reactions_from_formulae",
    "reaction_matrix_as_strings",
]
