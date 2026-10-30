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

"""Anisotropic mineral EOS with native tensor calculations and rotations."""

from copy import deepcopy

import numpy as np

from .. import _core
from .equilibratedcomposite import _ALIASES
from .._core import (
    convert_f_Pth_to_f_T_derivatives,
    deformation_gradient_alpha_and_compliance,
)


class AnisotropicMineral(_core.AnisotropicMineral):
    """Myhill (2022) EOS, following Python BurnMan's constructor.

    Cell parameters are molar lengths in metres and angles in degrees.
    Coefficients have shape (6,6,m,n), with m >= 2 and n >= 1. Alternatively,
    supply a parameter dictionary, psi_function(f, Pth, params) returning
    (Psi, dPsi/df, dPsi/dPth), and an explicit orthotropic boolean.
    Non-orthotropic models require frame_convention, a permutation of [0,1,2].
    """

    def __init__(
        self,
        isotropic_mineral,
        cell_parameters,
        anisotropic_parameters,
        frame_convention=None,
        psi_function=None,
        orthotropic=None,
    ):
        self._reference_cell_parameters = np.array(
            cell_parameters, dtype=float, copy=True
        )
        self._custom_psi_function = psi_function
        if psi_function is None:
            values = np.array(anisotropic_parameters, dtype=float, copy=True)
            super().__init__(
                isotropic_mineral, cell_parameters, values, frame_convention
            )
            self.anisotropic_params = {"c": values}
            self.psi_function = self.standard_psi_function
        else:
            if not isinstance(orthotropic, bool):
                raise ValueError(
                    "A custom Psi function requires an explicit orthotropic boolean."
                )
            params = anisotropic_parameters
            super().__init__(
                isotropic_mineral,
                cell_parameters,
                lambda f, Pth: psi_function(f, Pth, params),
                orthotropic,
                frame_convention,
            )
            self.anisotropic_params = params
            self.psi_function = psi_function

    @staticmethod
    def standard_psi_function(f, Pth, params):
        return _core._standard_anisotropic_psi(f, Pth, params["c"])

    @staticmethod
    def check_standard_parameters(anisotropic_parameters):
        return _core._check_anisotropic_parameters(anisotropic_parameters)

    def copy(self):
        parameters = deepcopy(self.anisotropic_params)
        result = type(self)(
            self.isotropic_mineral.copy(),
            self._reference_cell_parameters,
            parameters["c"] if self._custom_psi_function is None else parameters,
            self.frame_convention,
            self._custom_psi_function,
            self.orthotropic,
        )
        result.name = self.name
        if self.has_state():
            result.set_state(self.pressure, self.temperature)
        return result

    def evaluate(self, vars_list, pressures, temperatures, molar_fractions=None):
        """Evaluate scalar/tensor properties on an independent mineral copy."""
        pressures = np.asarray(pressures, dtype=float)
        temperatures = np.asarray(temperatures, dtype=float)
        if pressures.shape != temperatures.shape:
            raise ValueError("pressures and temperatures must have the same shape.")
        if (
            not np.all(np.isfinite(pressures))
            or not np.all(np.isfinite(temperatures))
            or np.any(temperatures < 0.0)
        ):
            raise ValueError("Require finite pressure and finite temperature >= 0.")
        names = list(vars_list)
        fractions = (
            None
            if molar_fractions is None
            else np.asarray(molar_fractions, dtype=float)
        )
        if fractions is not None and fractions.shape != pressures.shape + (
            getattr(self, "n_unrelaxed_vectors", self.n_endmembers),
        ):
            raise ValueError(
                "molar_fractions must match the sample shape and number of endmembers."
            )
        if not names or not pressures.size:
            return np.empty((len(names),) + pressures.shape)
        material = self.copy()
        if fractions is not None:
            material.set_composition(fractions.reshape(-1, fractions.shape[-1])[0])
        material.set_state(float(pressures.flat[0]), float(temperatures.flat[0]))
        output = [
            np.empty(pressures.shape + np.shape(getattr(material, name)))
            for name in names
        ]
        for index in np.ndindex(pressures.shape):
            if fractions is not None:
                material.set_composition(fractions[index])
            material.set_state(float(pressures[index]), float(temperatures[index]))
            for values, name in zip(output, names):
                values[index] = getattr(material, name)
        try:
            return np.array(output)
        except ValueError:
            return output


def _explicit_property(name):
    def get(self):
        raise NotImplementedError(
            f"{name} is ambiguous for an anisotropic mineral; query the explicit Reuss/Voigt or tensor property."
        )

    return property(get)


for _alias, _name in dict(
    _ALIASES,
    volume="molar_volume",
    entropy="molar_entropy",
    enthalpy="molar_enthalpy",
    helmholtz="molar_helmholtz",
    internal_energy="molar_internal_energy",
    heat_capacity_p="molar_heat_capacity_p",
    heat_capacity_v="molar_heat_capacity_v",
).items():
    if _alias in ("K_T", "K_S", "beta_T", "beta_S"):
        setattr(AnisotropicMineral, _alias, _explicit_property(_alias))
    elif _alias != _name:
        setattr(
            AnisotropicMineral,
            _alias,
            property(lambda self, name=_name: getattr(self, name)),
        )
