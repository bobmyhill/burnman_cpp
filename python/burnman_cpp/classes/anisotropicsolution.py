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

"""Native anisotropic solution mixing and composition derivatives."""

from copy import deepcopy

from .. import _core
from .anisotropicmineral import AnisotropicMineral


class AnisotropicSolution(_core.AnisotropicSolution):
    """Mix anisotropic endmembers with a scalar native solution model.

    psi_excess_function(lnV, Pth, fractions, params) returns three 6x6
    matrices (Psi excess and its lnV/Pth derivatives) and composition
    derivatives of Psi:I with shape (3,3,n_endmembers), at fixed V,T.
    Set composition before state when molar_fractions is omitted.
    """

    def __init__(
        self,
        name,
        solution_model,
        psi_excess_function,
        anisotropic_parameters,
        molar_fractions=None,
    ):
        self.anisotropic_params = anisotropic_parameters
        self.psi_excess_function = psi_excess_function
        if not callable(psi_excess_function):
            raise TypeError("psi_excess_function must be callable.")
        params = anisotropic_parameters
        _core.AnisotropicSolution.__init__(
            self,
            name,
            solution_model,
            lambda lnV, Pth, fractions: psi_excess_function(
                lnV, Pth, fractions, params
            ),
            molar_fractions,
        )

    def copy(self):
        fractions = self.molar_fractions if self.molar_fractions.size else None
        result = type(self)(
            self.name,
            self.solution_model.clone(),
            self.psi_excess_function,
            deepcopy(self.anisotropic_params),
            fractions,
        )
        result.T_0 = self.T_0
        if self.has_state():
            result.set_state(self.pressure, self.temperature)
        return result

    evaluate = AnisotropicMineral.evaluate


# The native class already inherits AnisotropicMineral. Reuse its Python
# aliases without adding a second native base through the adapter class.
for _name, _value in vars(AnisotropicMineral).items():
    if isinstance(_value, property):
        setattr(AnisotropicSolution, _name, _value)
