# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Thermodynamic property grids on saved phase fields."""

import numpy as np

from .. import _core
from .phase_diagram import _phase_diagram_data


def phase_diagram_grid(
    result,
    phases,
    properties,
    *,
    resolution=(101, 101),
    x=None,
    y=None,
    relaxed=True,
    store_solutions=False,
):
    """Evaluate scalar Composite/Material properties over a completed diagram.

    ``result`` accepts a native result, saved dictionary or JSON path. Supply
    the original candidate models in their original order. ``properties`` is
    a name or sequence of names from ``thermodynamic_property_names()``.
    ``resolution=(nx, ny)`` makes an inclusive regular grid. Alternatively
    supply both increasing 1-D coordinate arrays ``x`` and ``y``. Coordinates
    use native diagram axis order and SI units: PT means x=P in Pa, y=T in K;
    S/V are totals on the saved bulk scale, X interpolates the saved endpoints.

    Return a ``PhaseDiagramPropertyGrid`` with ``x``, ``y``, ``values[name]``
    arrays of shape (ny, nx), one-based ``field_numbers``, and a ``success``
    mask (true when every requested property succeeded). Failed/unavailable
    values are NaN and have ``diagnostics``; excluded
    points have field number zero. Open, provisional and unidentified fields
    are skipped. Shared boundaries use the lowest Field number's one-sided
    value. Source data and candidate materials are unchanged.

    Each point equilibrates the saved assemblage; phase boundary discovery
    and global stability searches are not repeated. ``relaxed=False`` holds
    phase amounts and endmember compositions fixed during property response
    (the point itself is still equilibrated). ``relaxed=True`` includes all
    active mass-conserving reactions using the projected Gibbs Hessian,
    following Stixrude & Lithgow-Bertelloni, doi:10.1093/gji/ggab394, eqs 19–26.
    Relaxation changes alpha, Cp, compressibility and their derived properties;
    equilibrium densities and energies are the same in both modes. Singular
    reaction curvature yields NaN for relaxed response properties. Molar
    responses divide extensive derivatives by the current number of moles.
    Relaxed acoustic velocities use the relaxed adiabatic bulk modulus and
    the Composite's existing shear modulus; the paper specifies no shear
    relaxation law. Unrelaxed velocities retain Composite's seismic averaging.

    ``store_solutions=True`` retains every accepted equilibrium, including
    points where a requested property fails. Inspect ``equilibrium_success``
    and ``solution_at(row, col)`` (None for unsolved/excluded points). Save with
    ``save_property_grid`` and later use ``evaluate_property_grid`` to compute
    more properties or change relaxation mode without equilibrium solves.
    """
    data = _phase_diagram_data(result)
    if (x is None) != (y is None):
        raise ValueError("Supply both x and y, or neither.")
    if x is None:
        counts = np.asarray(resolution)
        if (
            counts.shape != (2,)
            or not np.issubdtype(counts.dtype, np.integer)
            or (counts < 2).any()
        ):
            raise ValueError("resolution must contain two integer counts >= 2.")
        native = _core.PhaseDiagramResult.from_dict(data)
        x, y = [
            np.linspace(*bounds, int(n))
            for bounds, n in zip(native.coordinate_ranges, counts)
        ]
    else:
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("x and y must be one-dimensional coordinate arrays.")
    if not isinstance(relaxed, (bool, np.bool_)):
        raise TypeError("relaxed must be a boolean.")
    if not isinstance(store_solutions, (bool, np.bool_)):
        raise TypeError("store_solutions must be a boolean.")
    names = [properties] if isinstance(properties, str) else list(properties)
    return _core._phase_diagram_grid(
        data, phases, names, x, y, bool(relaxed), bool(store_solutions)
    )


def save_property_grid(grid, path):
    """Save property arrays, masks, metadata and retained solutions; return Path.

    Use ``.json.gz`` for compressed JSON, or ``.json`` for plain JSON. Non-finite
    property values become null and are restored as NaN. Phase amounts, original
    compositions, exact active bases and their coefficients retain double
    precision. Files contain data only; model definitions are supplied again
    when evaluating additional properties. An existing file is replaced only
    after serialization and writing succeed.
    """
    import gzip
    import json
    from pathlib import Path
    import tempfile

    if not isinstance(grid, _core.PhaseDiagramPropertyGrid):
        raise TypeError("grid must be a PhaseDiagramPropertyGrid.")
    payload = json.dumps(grid.to_dict(), allow_nan=False, separators=(",", ":")).encode(
        "utf-8"
    )
    destination = Path(path)
    if destination.suffix == ".gz":
        payload = gzip.compress(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        temporary.replace(destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return destination


def load_property_grid(path):
    """Load a plain/compressed JSON property grid, including retained solutions.

    No model objects are deserialized and no equilibrium solves are performed.
    Supply the original candidate model definitions to evaluate more properties.
    """
    import gzip
    import json
    from pathlib import Path

    source = Path(path)
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError("A saved property grid must be a JSON object.")
    return _core.PhaseDiagramPropertyGrid.from_dict(data)


def evaluate_property_grid(grid, phases, properties, *, relaxed=None):
    """Evaluate properties at retained equilibria without repeating any solves.

    Accept a native grid, its saved dictionary, or a JSON/JSON.gz path. The grid
    must have been calculated with ``store_solutions=True``. Supply the same
    candidate model definitions in their original order. Preserve each point's
    physical P/T, amounts, compositions and exact active face; no interpolation,
    Gibbs minimization or equilibrium solving is performed. A previous property
    failure does not discard its equilibrium. Unsolved points remain masked.

    Return a new grid containing the requested properties and retained states;
    the input is unchanged. ``relaxed=None`` inherits the saved mode; True/False
    switches it. ``equilibrium_solves`` is zero for this evaluation. Source
    equilibrium diagnostics are retained; property diagnostics are recalculated.
    """
    from collections.abc import Mapping

    if isinstance(grid, Mapping):
        grid = _core.PhaseDiagramPropertyGrid.from_dict(dict(grid))
    elif not isinstance(grid, _core.PhaseDiagramPropertyGrid):
        grid = load_property_grid(grid)
    if relaxed is None:
        relaxed = grid.relaxed
    if not isinstance(relaxed, (bool, np.bool_)):
        raise TypeError("relaxed must be a boolean or None.")
    names = [properties] if isinstance(properties, str) else list(properties)
    return _core._evaluate_property_grid(grid, phases, names, bool(relaxed))
