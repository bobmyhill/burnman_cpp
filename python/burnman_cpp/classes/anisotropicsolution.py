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


class RelaxedAnisotropicSolution(_core.RelaxedAnisotropicSolution):
    """Relax isochemical endmember exchanges and thermoelastic responses.

    Rows of relaxation_vectors describe rapid exchanges; rows of
    unrelaxed_vectors define independent composition coordinates. Set their
    fractions with set_composition(), then set P,T or molar V,T. relaxed=False
    keeps the supplied structural composition while synchronizing both objects.
    Tensor susceptibilities assume an interior structural minimum and T > 0.
    """

    def __init__(
        self, anisotropic_solution, relaxation_vectors=None, unrelaxed_vectors=None
    ):
        if relaxation_vectors is None and isinstance(
            anisotropic_solution, _core.RelaxedAnisotropicSolution
        ):
            _core.RelaxedAnisotropicSolution.__init__(self, anisotropic_solution)
        else:
            _core.RelaxedAnisotropicSolution.__init__(
                self, anisotropic_solution, relaxation_vectors, unrelaxed_vectors
            )
        self.anisotropic_params = getattr(
            anisotropic_solution, "anisotropic_params", None
        )
        self.psi_excess_function = getattr(
            anisotropic_solution, "psi_excess_function", None
        )

    def copy(self):
        return type(self)(self)

    evaluate = AnisotropicSolution.evaluate


for _name, _value in vars(AnisotropicSolution).items():
    if isinstance(_value, property):
        setattr(RelaxedAnisotropicSolution, _name, _value)
