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

"""A Composite interface with full phase discovery at each pressure/temperature."""

from collections.abc import Mapping
from copy import copy, deepcopy

import numpy as np

from .. import _core
from ..minerals import available_phase_names as _available_phase_names
from ..minerals import load_dataset

_ALIASES = {
    "P": "pressure",
    "T": "temperature",
    "rho": "density",
    "V": "molar_volume",
    "S": "molar_entropy",
    "H": "molar_enthalpy",
    "F": "molar_helmholtz",
    "gibbs": "molar_gibbs",
    "energy": "molar_internal_energy",
    "molar_gibbs_free_energy": "molar_gibbs",
    "molar_helmholtz_energy": "molar_helmholtz",
    "K_T": "isothermal_bulk_modulus_reuss",
    "K_S": "isentropic_bulk_modulus_reuss",
    "beta_T": "isothermal_compressibility_reuss",
    "beta_S": "isentropic_compressibility_reuss",
    "G": "shear_modulus",
    "v_p": "p_wave_velocity",
    "v_s": "shear_wave_velocity",
    "v_phi": "bulk_sound_velocity",
    "gr": "grueneisen_parameter",
    "alpha": "thermal_expansivity",
    "C_p": "molar_heat_capacity_p",
    "C_v": "molar_heat_capacity_v",
}


def _averaging_scheme(scheme):
    if isinstance(scheme, _core.AveragingType):
        return scheme
    names = {
        "Voigt": _core.AveragingType.Voigt,
        "Reuss": _core.AveragingType.Reuss,
        "VoigtReussHill": _core.AveragingType.VRH,
        "VRH": _core.AveragingType.VRH,
        "HashinShtrikmanLower": _core.AveragingType.HashinShtrikmanLower,
        "HashinShtrikmanUpper": _core.AveragingType.HashinShtrikmanUpper,
        "HashinShtrikmanAverage": _core.AveragingType.HashinShtrikman,
        "HashinShtrikman": _core.AveragingType.HashinShtrikman,
    }
    try:
        return names[scheme]
    except (KeyError, TypeError):
        raise ValueError(f"Unknown averaging scheme: {scheme!r}") from None


class EquilibratedComposite(_core.EquilibratedComposite):
    """A fixed-bulk composite that minimises Gibbs energy at each state.

    ``composition`` is a Composition or a dictionary of formula amounts.
    Dictionaries default to moles (elemental dictionaries also work); choose
    ``unit_type='mass'`` for masses in kg. Amounts retain their supplied scale.
    ``dataset`` is a thermodynamic dataset (such as SLB11), its name, a
    calibrated ModelSet class object, or a sequence of candidate Material class
    objects. ``phase_names`` optionally selects materials by their dataset names.
    ``EquilibratedComposite.available_phase_names(dataset)`` lists all available
    names, including pure endmembers and alternate names for materials.

    ``set_state(P, T)`` minimises Gibbs energy at the prescribed bulk composition,
    pressure and temperature. Competing phases may enter or leave, solutions
    change composition, and multiple copies of a solution may coexist.
    Candidate Material class objects are copied. The calculated phases,
    fractions, formula, n_moles and material properties are available after
    setting the state.

    By default, the previous equilibrium supplies initial estimates for the
    next state. Every result is checked against competing phases and for local
    stability. An unsuccessful estimate triggers a new phase search. Set
    ``warm_start=False`` to start every search independently.

    By default thermal derivatives include changes in equilibrium phase amounts
    and compositions. Set ``relaxed=False`` to hold the calculated phase
    fractions and compositions fixed when evaluating these derivatives.
    Undefined or unstable responses raise diagnostic exceptions.
    ``snapshot()`` returns an independent ordinary Assemblage at the current
    equilibrium. It does not re-equilibrate when its state changes.

    A failed set_state raises and preserves the last accepted state. Inspect
    ``last_equilibrium_attempt`` for its diagnostics and ``equilibrium_state``
    for the accepted phase amounts, compositions and mass-balance errors.
    """

    def __init__(
        self,
        composition,
        dataset,
        *,
        unit_type="molar",
        phase_names=None,
        settings=None,
        relaxed=True,
        averaging_scheme="VoigtReussHill",
        name="EquilibratedComposite",
        warm_start=True,
    ):
        if isinstance(composition, _core.Composition):
            bulk = copy(composition)
        elif isinstance(composition, Mapping):
            bulk = _core.Composition(dict(composition), unit_type)
        else:
            raise TypeError(
                "composition must be a Composition or a dictionary of formula amounts."
            )

        if (
            isinstance(dataset, str)
            or hasattr(dataset, "__name__")
            or hasattr(dataset, "phases")
        ):
            phases, metadata = load_dataset(dataset, phase_names)
        else:
            if phase_names is not None:
                raise ValueError(
                    "phase_names requires a thermodynamic dataset with named materials."
                )
            try:
                phases = list(dataset)
            except TypeError:
                raise TypeError(
                    "dataset must be a thermodynamic dataset, its name, a ModelSet class object or a sequence of Material class objects."
                ) from None
            if not phases or any(
                not isinstance(phase, _core.Material) for phase in phases
            ):
                raise ValueError(
                    "Supply a nonempty sequence of candidate Material class objects."
                )
            metadata = dict(
                dataset="custom", phase_names=[phase.name for phase in phases]
            )

        super().__init__(
            bulk.atomic_composition,
            phases,
            settings if settings is not None else _core.PhaseDiagramSettings(),
            relaxed,
            _averaging_scheme(averaging_scheme),
            warm_start,
        )
        self._composition = bulk
        self._dataset_metadata = metadata
        self.set_name(name)

    @staticmethod
    def available_phase_names(dataset):
        """Return all dataset names accepted by ``phase_names``, in sorted order.

        ``dataset`` is a thermodynamic dataset or its name. The returned names
        include pure endmembers and alternate names for materials. No bulk
        composition or pressure/temperature state is required.
        """
        return _available_phase_names(dataset)

    @property
    def composition(self):
        """An independent copy of the prescribed bulk composition."""
        return copy(self._composition)

    @property
    def atomic_composition(self):
        """Prescribed elemental mole amounts on their original scale."""
        return self._composition.atomic_composition

    @property
    def dataset_metadata(self):
        """Thermodynamic dataset information and the selected material names."""
        return deepcopy(self._dataset_metadata)

    @property
    def equilibrated(self):
        return self.equilibrium_state is not None

    @property
    def phase_volumes(self):
        """Phase volumes per current mole of composite (m³/mol)."""
        return self.molar_fractions * np.array(
            [phase.molar_volume for phase in self.phases]
        )

    @property
    def phase_masses(self):
        """Phase masses per current mole of composite (kg/mol)."""
        return self.molar_fractions * np.array(
            [phase.molar_mass for phase in self.phases]
        )

    def set_averaging_scheme(self, averaging_scheme):
        """Set the elastic averaging scheme, such as 'VoigtReussHill' or 'Reuss'."""
        super().set_averaging_scheme(_averaging_scheme(averaging_scheme))

    def evaluate(self, vars_list, pressures, temperatures):
        """Evaluate scalar material properties, re-equilibrating at every point.

        Pressure/temperature arrays must have the same shape. The result has
        shape ``(len(vars_list), *pressures.shape)``. Short property
        names such as rho, S, K_S and v_p are accepted. The original accepted
        state is restored afterwards, including when an evaluation fails.
        With ``warm_start=True``, successive points reuse the preceding
        equilibrium as an initial estimate while checking all competing phases.
        """
        names = [_ALIASES.get(name, name) for name in vars_list]
        available = set(_core.thermodynamic_property_names())
        unknown = [name for name in names if name not in available]
        if unknown:
            raise ValueError(f"Unknown scalar material properties: {unknown}")
        pressures = np.asarray(pressures, dtype=float)
        temperatures = np.asarray(temperatures, dtype=float)
        if pressures.shape != temperatures.shape:
            raise ValueError("pressures and temperatures must have the same shape.")
        values = np.empty((len(names),) + pressures.shape)
        if pressures.size == 0 or not names:
            return values
        original_state = (self.pressure, self.temperature) if self.has_state() else None
        try:
            for index in np.ndindex(pressures.shape):
                self.set_state(float(pressures[index]), float(temperatures[index]))
                for row, name in enumerate(names):
                    values[(row,) + index] = getattr(self, name)
        finally:
            if original_state is None:
                self._clear_state()
            else:
                self.set_state(*original_state)
        return values

    def __str__(self):
        if not self.has_state():
            return f"{self.name} (state not set)"
        phases = ", ".join(
            f"{phase.name}: {fraction:.6g}"
            for phase, fraction in zip(self.phases, self.molar_fractions)
        )
        return f"{self.name} at {self.pressure:g} Pa, {self.temperature:g} K\n{phases}"


def _property_alias(name):
    return property(lambda self: getattr(self, name))


for _alias, _name in _ALIASES.items():
    setattr(EquilibratedComposite, _alias, _property_alias(_name))

__all__ = ["EquilibratedComposite"]
