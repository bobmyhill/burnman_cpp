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
"""Total thermodynamic potentials and their derivatives on natural coordinates.

Arrays have shape (len(y), len(x)). P is in Pa, T in K, S in total J/K and
V in total m³ on the supplied bulk scale. Potentials are total joules.
E denotes internal energy; U is accepted as a synonym.
"""

from dataclasses import dataclass, field
from pathlib import Path
import json
import tempfile

import numpy as np

from .properties import (
    _saved_property_grid_batches,
    evaluate_property_grid,
)

DERIVATIVES = ("d_dx", "d_dy", "d2_dx2", "d2_dxdy", "d2_dy2")
# Natural axes and coordinate curvature signs: +1 convex, -1 concave.
POTENTIALS = {
    "E": (("V", "S"), (1, 1), "molar_internal_energy"),
    "F": (("V", "T"), (1, -1), "molar_helmholtz"),
    "G": (("P", "T"), (-1, -1), "molar_gibbs"),
    "H": (("P", "S"), (-1, 1), "molar_enthalpy"),
}


def _natural_potential(variables):
    """Select the thermodynamic potential for either order of its natural axes."""
    return next(
        (
            symbol
            for symbol, (axes, _, _) in POTENTIALS.items()
            if set(variables) == set(axes)
        ),
        None,
    )


@dataclass
class PotentialGrid:
    """One potential, its gradient and Hessian, and original equilibrium metadata.

    x/y follow ``variables``: E(V,S), F(V,T), G(P,T) or H(P,S). Derivatives
    use these physical coordinates, not array indices. Missing observations
    are NaN; a missing derivative array can be None before infilling.
    Processing returns new potential/derivative arrays and retains metadata.
    """

    potential: str
    x: np.ndarray
    y: np.ndarray
    values: np.ndarray
    d_dx: np.ndarray | None = None
    d_dy: np.ndarray | None = None
    d2_dx2: np.ndarray | None = None
    d2_dxdy: np.ndarray | None = None
    d2_dy2: np.ndarray | None = None
    n_moles: np.ndarray | None = None
    field_numbers: np.ndarray | None = None
    equilibrium_success: np.ndarray | None = None
    diagnostics: list = field(default_factory=list)

    def __post_init__(self):
        self.potential = self.potential.upper().replace("U", "E")
        if self.potential not in POTENTIALS:
            raise ValueError("potential must be E, F, G or H.")
        self.x, self.y = np.asarray(self.x, float), np.asarray(self.y, float)
        if any(
            axis.ndim != 1
            or not len(axis)
            or not np.isfinite(axis).all()
            or (np.diff(axis) <= 0).any()
            for axis in (self.x, self.y)
        ):
            raise ValueError("Potential axes must be finite, nonempty and increasing.")
        self.values = np.asarray(self.values, float)
        if self.values.shape != (len(self.y), len(self.x)) or self.values.ndim != 2:
            raise ValueError("Potential values must have shape (len(y), len(x)).")
        for name in DERIVATIVES:
            value = getattr(self, name)
            if value is not None:
                value = np.asarray(value, float)
                if value.shape != self.values.shape:
                    raise ValueError(f"{name} must match the potential grid shape.")
                setattr(self, name, value)

    @property
    def variables(self):
        """Natural coordinate names, in x/y order."""
        return POTENTIALS[self.potential][0]

    @property
    def curvature(self):
        """Signs of the two coordinate curvatures; mixed curvature is unrestricted."""
        return POTENTIALS[self.potential][1]

    @property
    def gradient_bounds(self):
        """Bounds from V>=0, T>=0 and P>=0; entropy's reference is unrestricted."""
        return tuple(
            {
                "P": (0.0, np.inf),
                "S": (0.0, np.inf),
                "V": (-np.inf, 0.0),
                "T": (-np.inf, np.inf),
            }[v]
            for v in self.variables
        )

    def spacings(self):
        """Validate increasing uniform axes and return their physical increments."""
        steps = []
        for name, axis in zip(self.variables, (self.x, self.y)):
            if axis.ndim != 1 or len(axis) < 2 or not np.isfinite(axis).all():
                raise ValueError(f"The {name} axis needs at least two finite points.")
            delta = np.diff(axis)
            if (delta <= 0).any() or not np.allclose(
                delta, delta[0], rtol=1e-9, atol=0
            ):
                raise ValueError(
                    f"The {name} axis must be increasing and uniformly spaced."
                )
            steps.append(float(delta[0]))
        return tuple(steps)


def save_potential_grid(grids, path):
    """Save potential values, derivatives and source masks to a compressed NPZ.

    NaNs and infinities retain their original values. Optional derivative and
    metadata arrays are omitted when absent. Files contain no model objects or
    pickled data. Read with load_potential_grid(path), without candidate models,
    or inspect named arrays with numpy.load(path, allow_pickle=False).
    """
    destination = Path(path)
    if destination.suffix != ".npz":
        raise ValueError("Potential grids must be saved as .npz files.")
    payload = {
        name: getattr(grids, name)
        for name in (
            "x",
            "y",
            "values",
            *DERIVATIVES,
            "n_moles",
            "field_numbers",
            "equilibrium_success",
        )
        if getattr(grids, name) is not None
    }
    payload.update(
        format="burnman_cpp.potential_grid",
        schema_version=1,
        potential=grids.potential,
        diagnostics=json.dumps(grids.diagnostics, allow_nan=False),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **payload)
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return destination


def _load_potential_arrays(path, potential):
    """Read the data-only NPZ format and validate coordinates and array shapes."""
    with np.load(path, allow_pickle=False) as data:
        if (
            str(data["format"]) != "burnman_cpp.potential_grid"
            or int(data["schema_version"]) != 1
        ):
            raise ValueError("Unsupported saved potential-grid format.")
        names = (
            "potential",
            "x",
            "y",
            "values",
            *DERIVATIVES,
            "n_moles",
            "field_numbers",
            "equilibrium_success",
        )
        values = {name: data[name] for name in names if name in data}
        values["potential"] = str(values["potential"])
        values["diagnostics"] = json.loads(str(data["diagnostics"]))
    result = PotentialGrid(**values)
    if (
        potential is not None
        and potential.upper().replace("U", "E") != result.potential
    ):
        raise ValueError("Requested potential disagrees with the saved grid.")
    return result


def load_potential_grid(source, phases=None, *, potential=None):
    """Load potential arrays or evaluate saved equilibria without new solves.

    source is a saved property grid, its path, or a chunk manifest/directory.
    Supply the original candidate models in their original order. The default
    potential is G for PT, H for PS, F for TV and E for SV. TV/SV arrays are
    transposed into the natural V,T / V,S order. All response derivatives are
    relaxed, evaluated at the retained compositions and active site faces.
    Missing/undefined values remain NaN, with original equilibrium masks.
    A potential NPZ written by save_potential_grid instead loads its arrays
    directly and needs no models. Infill/smoothing are never automatic here.
    """
    if not hasattr(source, "solution_at") and Path(source).suffix == ".npz":
        return _load_potential_arrays(source, potential)
    if phases is None:
        raise ValueError("Supply candidate models when replaying saved equilibria.")
    x, y, batches = _saved_property_grid_batches(source)
    result = None
    for rows, cols, grid in batches:
        if result is None:
            symbol = potential or _natural_potential(grid.diagram_type)
            if symbol is None:
                raise ValueError(
                    "Potential processing requires a PT, PS, TV or SV grid."
                )
            result = PotentialGrid(symbol, x, y, np.full((len(y), len(x)), np.nan))
            transpose = tuple(grid.diagram_type) == result.variables[::-1]
            if tuple(grid.diagram_type) != result.variables and not transpose:
                raise ValueError(
                    "The saved grid must use the potential's natural coordinates."
                )
            if transpose:
                result.x, result.y, result.values = y, x, result.values.T.copy()
            for name in DERIVATIVES + ("n_moles",):
                setattr(result, name, np.full(result.values.shape, np.nan))
            result.field_numbers = np.zeros(result.values.shape, dtype="int32")
            result.equilibrium_success = np.zeros(result.values.shape, dtype=bool)
        expected = result.variables[::-1] if transpose else result.variables
        if tuple(grid.diagram_type) != expected:
            raise ValueError("Saved batches must use the same diagram coordinates.")
        if not (np.array_equal(grid.x, x[cols]) and np.array_equal(grid.y, y[rows])):
            raise ValueError("Saved batch coordinates disagree with the manifest.")
        symbol = result.potential
        names = [POTENTIALS[symbol][2], "molar_volume", "temperature"]
        names += {
            "G": [
                "molar_entropy",
                "molar_heat_capacity_p",
                "thermal_expansivity",
                "isothermal_compressibility_reuss",
            ],
            "H": [
                "molar_heat_capacity_p",
                "isentropic_thermal_gradient",
                "isentropic_compressibility_reuss",
            ],
            "F": [
                "pressure",
                "molar_entropy",
                "molar_heat_capacity_v",
                "thermal_expansivity",
                "isothermal_compressibility_reuss",
            ],
            "E": [
                "pressure",
                "molar_heat_capacity_v",
                "thermal_expansivity",
                "isothermal_compressibility_reuss",
                "isentropic_compressibility_reuss",
            ],
        }[symbol]
        replay = evaluate_property_grid(grid, phases, names, relaxed=True)
        moles = np.full(grid.equilibrium_success.shape, np.nan)
        for j, i in np.argwhere(grid.equilibrium_success):
            moles[j, i] = grid.solution_at(int(j), int(i)).n_moles
        v = replay.values
        volume, temperature = moles * v["molar_volume"], v["temperature"]
        with np.errstate(divide="ignore", invalid="ignore"):
            if symbol == "G":
                derivatives = (
                    volume,
                    -moles * v["molar_entropy"],
                    -volume * v["isothermal_compressibility_reuss"],
                    volume * v["thermal_expansivity"],
                    -moles * v["molar_heat_capacity_p"] / temperature,
                )
            elif symbol == "H":
                cp = moles * v["molar_heat_capacity_p"]
                derivatives = (
                    volume,
                    temperature,
                    -volume * v["isentropic_compressibility_reuss"],
                    v["isentropic_thermal_gradient"],
                    np.divide(
                        temperature,
                        cp,
                        out=np.full_like(cp, np.nan),
                        where=~np.isnan(cp) & (cp != 0),
                    ),
                )
            else:
                beta = v["isothermal_compressibility_reuss"]
                cv = moles * v["molar_heat_capacity_v"]
                derivatives = (
                    (
                        -v["pressure"],
                        -moles * v["molar_entropy"],
                        1.0 / (volume * beta),
                        -v["thermal_expansivity"] / beta,
                        -cv / temperature,
                    )
                    if symbol == "F"
                    else (
                        -v["pressure"],
                        temperature,
                        1.0 / (volume * v["isentropic_compressibility_reuss"]),
                        -v["thermal_expansivity"] * temperature / (beta * cv),
                        np.divide(
                            temperature,
                            cv,
                            out=np.full_like(cv, np.nan),
                            where=~np.isnan(cv) & (cv != 0),
                        ),
                    )
                )
        target = (cols, rows) if transpose else (rows, cols)
        arrays = (
            moles * v[POTENTIALS[symbol][2]],
            *derivatives,
            moles,
            grid.field_numbers,
            grid.equilibrium_success,
        )
        for name, values in zip(
            ("values", *DERIVATIVES, "n_moles", "field_numbers", "equilibrium_success"),
            arrays,
        ):
            getattr(result, name)[target] = values.T if transpose else values
        if replay.diagnostics:
            result.diagnostics.append(
                dict(
                    rows=(rows.start, rows.stop),
                    columns=(cols.start, cols.stop),
                    diagnostics=replay.diagnostics,
                )
            )
    if result is None:
        raise ValueError("The saved grid contains no batches.")
    return result
