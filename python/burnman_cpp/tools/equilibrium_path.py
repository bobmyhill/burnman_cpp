# ------------------------------------------------------
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
#
# This file is part of BurnMan.
#
# Detailed license information governing the source code
# and contributions can be found in the LICENSE document.
#
# ------------------------------------------------------
"""Equilibrium mineral modes and properties along paths in P, T, S and V."""

from dataclasses import dataclass, field

import numpy as np

from .. import _core

_COORDINATES = dict(P="pressure", T="temperature", S="entropy", V="volume")


@dataclass
class EquilibriumPath:
    """Solved states and mineral amounts, with coexisting model copies summed.

    Amount arrays have shape (points, candidate models). Failed or excluded
    states have NaN rows. Original states retain each solution copy's amount,
    endmember proportions, active basis and equilibrium diagnostics.
    Optional property arrays are in ``values``; their failures are recorded in
    ``property_diagnostics`` without discarding successful equilibria.
    """

    states: tuple
    phase_names: tuple[str, ...]
    molar_amounts: np.ndarray
    atomic_amounts: np.ndarray
    values: dict = field(default_factory=dict)
    property_diagnostics: tuple = ()
    equilibrium_solves: int = 0

    @property
    def success(self):
        """True for accepted equilibria inside the allowed model domain."""
        return np.array([s.success and not s.outside_model_domain for s in self.states])

    @property
    def diagnostics(self):
        """One message per unsuccessful point, using zero-based array indices."""
        return tuple(
            f"Point {i}: {s.message}"
            for i, (s, ok) in enumerate(zip(self.states, self.success))
            if not ok
        )

    def coordinates(self, variable):
        """Read P (Pa), T (K), total S (J/K) or total V (m³) at every point."""
        name = _COORDINATES[variable.upper()]
        return np.array(
            [
                getattr(s, name) if ok else np.nan
                for s, ok in zip(self.states, self.success)
            ]
        )

    def fractions(self, basis="atomic"):
        """Return atomic or molar fractions, summing to one at solved points.

        Atomic fractions weight formula-unit amounts by atoms per formula unit;
        they are independent of arbitrary endmember formula-unit conventions.
        They differ from the molar fractions of minerals with different formula
        sizes. No interpolation or infilling is applied to missing states.
        """
        if basis not in ("atomic", "molar"):
            raise ValueError("basis must be 'atomic' or 'molar'.")
        amounts = self.atomic_amounts if basis == "atomic" else self.molar_amounts
        return amounts / amounts.sum(axis=1, keepdims=True)


def calculate_equilibrium_path(
    composition,
    phases,
    start,
    end,
    *,
    n_points=201,
    settings=None,
    pressure_range=(0.0, 150.0e9),
    temperature_range=(1.0, 6000.0),
    properties=(),
    relaxed=True,
    averaging_scheme=_core.AveragingType.VRH,
):
    """Calculate mineral modes along a straight path in any two P/T/S/V axes.

    ``start`` and ``end`` are dictionaries specifying the same two of P, T, S
    and V. Both coordinates may vary linearly between the endpoints. For example,
    ``{'P': 0., 'S': entropy}`` to ``{'P': 135.e9, 'S': entropy}`` defines an
    isentrope. Use Pa, K, total J/K and total m³; S/V use the supplied bulk's
    amount scale. To specify an isentrope by potential temperature, first find
    its surface entropy with ``stable_equilibrium(..., pressure=0.,
    temperature=1600.)``. Endpoints are included, and both coordinates are
    sampled uniformly in either direction.

    ``composition`` is a Composition or elemental mole dictionary. ``phases``
    supplies all candidate minerals/solutions, not a prescribed assemblage.
    One native workspace reuses the previous accepted equilibrium as its next
    starting guess, checking competing phases and local stability throughout.
    P/T search ranges guide inverse S/V solves; they do not specify a geotherm.
    ``settings.verbose=True`` prints periodic progress. Failed points retain
    messages and NaN modes; they are never joined by invented equilibria.
    Optional property evaluation uses the same arguments as
    :func:`sample_equilibrium_path`, with results in ``path.values``.
    """
    start = {str(k).upper(): float(v) for k, v in start.items()}
    end = {str(k).upper(): float(v) for k, v in end.items()}
    if (
        len(start) != 2
        or start.keys() != end.keys()
        or not start.keys() <= _COORDINATES.keys()
        or not np.isfinite([*start.values(), *end.values()]).all()
    ):
        raise ValueError(
            "Endpoints must specify the same two finite P, T, S or V coordinates."
        )
    if start == end:
        raise ValueError("The path endpoints must be different.")
    if (
        isinstance(n_points, bool)
        or not isinstance(n_points, (int, np.integer))
        or n_points < 2
    ):
        raise ValueError("n_points must be an integer >= 2.")
    variables = "".join(k for k in _COORDINATES if k in start)
    points = np.linspace(
        [start[k] for k in variables], [end[k] for k in variables], n_points
    )
    return sample_equilibrium_path(
        composition,
        phases,
        dict(zip(variables, points.T)),
        settings=settings,
        pressure_range=pressure_range,
        temperature_range=temperature_range,
        properties=properties,
        relaxed=relaxed,
        averaging_scheme=averaging_scheme,
    )


def sample_equilibrium_path(
    composition,
    phases,
    coordinates,
    *,
    settings=None,
    pressure_range=(0.0, 150.0e9),
    temperature_range=(1.0, 6000.0),
    properties=(),
    relaxed=True,
    averaging_scheme=_core.AveragingType.VRH,
):
    """Equilibrate an ordered, possibly curved path through two P/T/S/V axes.

    ``coordinates`` maps two variable names to equally sized, finite 1-D arrays
    in SI units, e.g. ``{'P': pressures, 'T': geotherm}``. Points need not be
    equally spaced. Neighbours supply warm starts as in the linear-path driver.
    Optional ``properties`` are evaluated directly on the accepted assemblage
    with no additional solves, retaining its exact active composition basis.
    ``relaxed=True`` includes mass-conserving chemical reactions in response
    properties such as expansivity; velocities use the selected elastic
    ``averaging_scheme`` (VRH by default). ``path.values`` contains arrays;
    equilibrium failures and property failures retain separate diagnostics.
    """
    coordinates = {str(k).upper(): np.asarray(v, float) for k, v in coordinates.items()}
    if len(coordinates) != 2 or not coordinates.keys() <= _COORDINATES.keys():
        raise ValueError("coordinates must supply two of P, T, S and V.")
    variables = "".join(k for k in _COORDINATES if k in coordinates)
    arrays = list(coordinates.values())
    if (
        any(a.ndim != 1 or not a.size or not np.isfinite(a).all() for a in arrays)
        or arrays[0].shape != arrays[1].shape
    ):
        raise ValueError("Path coordinates must be equally sized, finite 1-D arrays.")
    points = np.column_stack([coordinates[k] for k in variables])
    bulk = (
        composition.atomic_composition
        if isinstance(composition, _core.Composition)
        else composition
    )
    phases = list(phases)
    states, values, diagnostics, solves = _core._equilibrium_path(
        bulk,
        phases,
        points,
        variables,
        settings or _core.PhaseDiagramSettings(),
        pressure_range,
        temperature_range,
        [properties] if isinstance(properties, str) else list(properties),
        relaxed,
        averaging_scheme,
    )
    states = tuple(states)
    molar = np.full((len(states), len(phases)), np.nan)
    atomic = np.full_like(molar, np.nan)
    atoms = [
        (
            np.array([sum(f.values()) for f in p.endmember_formulae])
            if isinstance(p, _core.Solution)
            else np.array([sum(p.formula.values())])
        )
        for p in phases
    ]
    for row, state in enumerate(states):
        if not state.success or state.outside_model_domain:
            continue
        molar[row], atomic[row] = 0.0, 0.0
        for phase in state.phases:
            col = phase.candidate_index
            molar[row, col] += phase.amount
            atomic[row, col] += phase.amount * (
                np.asarray(phase.composition) @ atoms[col]
            )
    return EquilibriumPath(
        states,
        tuple(p.name for p in phases),
        molar,
        atomic,
        values,
        tuple(diagnostics),
        solves,
    )


def plot_mode_box(
    path,
    ax=None,
    *,
    variable="P",
    x=None,
    xlabel=None,
    basis="atomic",
    order=None,
    phase_aliases=None,
    colors=None,
):
    """Draw stacked equilibrium modes, leaving failed intervals blank.

    ``variable`` selects the native P/T/S/V abscissa, unless a same-length
    ``x`` array is supplied (e.g. depth or pressure in GPa). ``order`` lists
    candidate model names from bottom to top and must include every present
    model. Coexisting copies of a model share one band. Use ``phase_aliases``
    for display names and ``colors`` to map model names to fill colours.
    Return (figure, axes); shaded bands and lines use only calculated samples.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    if x is None:
        x = path.coordinates(variable)
    x = np.asarray(x, float)
    if x.shape != (len(path.states),):
        raise ValueError("x must contain one coordinate per path point.")
    fractions = path.fractions(basis)
    present = [
        name for j, name in enumerate(path.phase_names) if np.any(fractions[:, j] > 0)
    ]
    order = present if order is None else list(order)
    if len(set(order)) != len(order) or set(order) != set(present):
        raise ValueError("order must list every present candidate model exactly once.")
    lower = np.where(path.success, 0.0, np.nan)
    palette = plt.get_cmap("tab20")
    for index, name in enumerate(order):
        mode = fractions[:, path.phase_names.index(name)]
        upper = lower + mode
        ax.fill_between(
            x,
            lower,
            upper,
            color=(colors or {}).get(name, palette(index % 20)),
            label=(phase_aliases or {}).get(name, name),
            linewidth=0,
        )
        ax.plot(x, upper, color="black", lw=0.8)
        lower = upper
    ax.set(
        xlabel=xlabel
        or f"{_COORDINATES[variable.upper()].capitalize()} "
        f"({dict(P='Pa', T='K', S='J/K', V='m³')[variable.upper()]})",
        ylabel=f"{basis.capitalize()} fraction",
        ylim=(0, 1),
    )
    return ax.figure, ax
