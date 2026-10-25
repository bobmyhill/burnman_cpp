# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Relaxed assemblages following Python BurnMan's RelaxedComposite interface."""

import numpy as np

from .. import _core
from .equilibratedcomposite import _ALIASES as _MOLAR_ALIASES

_ALIASES = dict(
    _MOLAR_ALIASES,
    V="volume",
    S="entropy",
    H="enthalpy",
    C_p="heat_capacity_p",
    C_v="heat_capacity_v",
    gibbs="gibbs",
    F="helmholtz",
    energy="internal_energy",
)


class RelaxedComposite(_core.RelaxedComposite):
    """Equilibrate an assemblage and relax selected reaction responses.

    ``RelaxedComposite(composite, relaxation_vectors)`` shares the phases of
    the supplied native Assemblage. Reaction vectors are rows, with columns
    in ``composite.endmember_names`` order, and must conserve each element.
    Use ``composite.reaction_basis`` for full relaxation.

    ``set_state(P, T, bulk_composition=None)`` equilibrates all phase fractions
    and solution compositions. The optional elemental composition defaults
    to ``unrelaxed.formula``. The chosen vectors control only the subsequent
    thermodynamic response, through the Gibbs Hessian pseudoinverse (rcond
    1e-22), as in Python BurnMan. ``dqdz_relaxed`` gives reaction-extent
    derivatives with respect to pressure and temperature.

    ``copy_state_from_unrelaxed()`` synchronizes state, phase fractions and
    mole amount after an external equilibrium solve. Relaxed derivatives
    require positive temperature and positive amounts of solution phases.
    V, S, H, C_p and C_v are extensive quantities; molar_* properties are per
    mole, following Python BurnMan.
    """

    def copy(self):
        """Return an independent relaxed composite with the same state."""
        return type(self)(self)

    def evaluate(self, vars_list, pressures, temperatures):
        """Evaluate properties over equal-shaped P,T arrays and restore P,T.

        Conventional aliases such as K_S, alpha and C_p are supported. Scalar
        properties return shape ``(len(vars_list), *pressures.shape)``;
        matrix properties append their own shape. Mixed property shapes
        return a list of arrays, following Python BurnMan.
        """
        pressures = np.asarray(pressures, dtype=float)
        temperatures = np.asarray(temperatures, dtype=float)
        if pressures.shape != temperatures.shape:
            raise ValueError("pressures and temperatures must have the same shape.")
        names = [_ALIASES.get(name, name) for name in vars_list]
        if pressures.size == 0 or not names:
            return np.empty((len(names),) + pressures.shape)
        previous = (self.pressure, self.temperature) if self.has_state() else None
        try:
            self.set_state(float(pressures.flat[0]), float(temperatures.flat[0]))
            output = [
                np.empty(pressures.shape + np.shape(getattr(self, name)))
                for name in names
            ]
            for index in np.ndindex(pressures.shape):
                self.set_state(float(pressures[index]), float(temperatures[index]))
                for values, name in zip(output, names):
                    values[index] = getattr(self, name)
        finally:
            if previous is None:
                self._clear_state()
            else:
                self.set_state(*previous)
        try:
            return np.array(output)
        except ValueError:
            return output

    @property
    def number_of_moles(self):
        return self.n_moles

    @number_of_moles.setter
    def number_of_moles(self, value):
        self.n_moles = value
        self.reset_cache()


def _property_alias(name):
    return property(lambda self: getattr(self, name))


def _extensive_property(name):
    return property(lambda self: getattr(self, name) * self.n_moles)


for _name, _molar in {
    "internal_energy": "molar_internal_energy",
    "helmholtz": "molar_helmholtz",
    "gibbs": "molar_gibbs",
    "enthalpy": "molar_enthalpy",
    "volume": "molar_volume",
    "entropy": "molar_entropy",
    "heat_capacity_p": "molar_heat_capacity_p",
    "heat_capacity_v": "molar_heat_capacity_v",
}.items():
    setattr(RelaxedComposite, _name, _extensive_property(_molar))

for _alias, _name in _ALIASES.items():
    if _alias != _name:
        setattr(RelaxedComposite, _alias, _property_alias(_name))

__all__ = ["RelaxedComposite"]
