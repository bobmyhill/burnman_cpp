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
    points have field number zero. Saved polygons propose assemblages, including
    provisional/open labels. Accept an equilibrated assigned assemblage without
    searching competing phases. Only if that assemblage fails, try fields across
    shared boundaries within one grid spacing in each coordinate. On nonuniform
    grids, this window reaches the immediately preceding/following coordinates;
    at an endpoint it extends by the nearest spacing. A singleton axis has zero
    width. Alternatives must pass equilibrium, curvature and competing-phase
    affinity checks.
    Model-domain exclusions remain masked. field_numbers records
    the accepted field, which can differ from the drawn polygon near a curved
    boundary. A point without a containing field remains unresolved.
    Source data and candidate materials are unchanged.

    Neighbouring energies and conjugate derivatives are checked against the
    convexity/concavity bounds of G(P,T), H(P,S), F(T,V) or U(S,V). These checks
    permit phase transitions and flat response modes; they do not apply the
    same curvature signs to non-natural P-V/T-S sections or changing bulk X.
    Only flagged points require independent phase discovery and comparison of
    competing roots, including different compositions of the same phase set.
    The saved diagram's boundaries are unchanged. Accepted neighbours in this
    row and the previous row supply phase
    amounts, compositions and P/T guesses before saved-state retries.
    A second pass visits failed or inconsistent equilibria in reverse row/column
    order, using successful neighbours to the right and in the following row.
    Consistent equilibria and their properties are preserved, including
    property failures.
    ``relaxed=False`` holds
    phase amounts and endmember compositions fixed during property response
    (the point itself is still equilibrated). ``relaxed=True`` includes all
    active mass-conserving reactions using the projected Gibbs Hessian,
    following Stixrude & Lithgow-Bertelloni, doi:10.1093/gji/ggab394, eqs 19–26.
    Relaxation changes alpha, Cp, compressibility and their derived properties;
    equilibrium densities and energies are the same in both modes. Singular
    reaction curvature yields NaN for undefined relaxed response properties.
    A diverging thermal expansivity with an unambiguous sign is returned as
    signed infinity; very soft positive curvature retains a finite response. Molar
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
    data, names, x, y = _grid_arguments(
        result, properties, resolution, x, y, relaxed, store_solutions
    )
    return _core._phase_diagram_grid(
        data, phases, names, x, y, bool(relaxed), bool(store_solutions)
    )


def _grid_arguments(result, properties, resolution, x, y, relaxed, store_solutions):
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
    return data, names, x, y


def phase_diagram_grid_chunks(
    result,
    phases,
    properties,
    *,
    resolution=(101, 101),
    x=None,
    y=None,
    relaxed=True,
    store_solutions=True,
    chunk_size=4096,
):
    """Yield ``(row_slice, column_slice, grid)`` with at most chunk_size points.

    Arguments and coordinates match ``phase_diagram_grid``. Prepare the saved
    diagram and field geometry once, then equilibrate bounded rectangular
    batches. Save/process each yielded grid and discard it before advancing;
    collecting the iterator defeats the memory limit. Retained solutions,
    property failures, holes and excluded fields have the usual grid semantics.
    Accepted neighbours are reused across batch boundaries with at most two
    rows of starting states retained. This also applies when store_solutions=False.
    Boundary proximity uses the full grid axes, independently of batch size.
    This iterator supplies the forward pass. After saving its batches, use
    ``retry_property_grid_chunks`` for the reverse pass across batch boundaries.
    Candidate models and the source are unchanged.
    """
    import operator

    if isinstance(chunk_size, (bool, np.bool_)):
        raise ValueError("chunk_size must be a positive integer.")
    try:
        chunk_size = operator.index(chunk_size)
    except TypeError as exc:
        raise ValueError("chunk_size must be a positive integer.") from exc
    if chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer.")
    data, names, x, y = _grid_arguments(
        result, properties, resolution, x, y, relaxed, store_solutions
    )
    # Validate the complete coordinate arrays, including joins between batches.
    for axis in (x, y):
        if not len(axis) or not np.isfinite(axis).all() or (np.diff(axis) <= 0).any():
            raise ValueError(
                "Grid coordinates must be finite, nonempty and increasing."
            )
    prepared = _core._PropertyGridder(data, phases, x, y)
    width = min(len(x), chunk_size)
    height = max(1, chunk_size // width)
    for j in range(0, len(y), height):
        rows = slice(j, min(j + height, len(y)))
        for i in range(0, len(x), width):
            cols = slice(i, min(i + width, len(x)))
            yield rows, cols, prepared.evaluate(
                names, x[cols], y[rows], bool(relaxed), bool(store_solutions)
            )


def load_property_grid_chunks(path):
    """Yield saved solution batches from a chunked-grid manifest or directory.

    Return ``(row_slice, column_slice, grid)`` one batch at a time. Each grid
    is usable with ``evaluate_property_grid`` for further properties without
    solving equilibria again. The manifest lists ordinary ``save_property_grid``
    JSON/JSON.gz files plus their half-open row/column bounds in the full grid.
    Files contain no model definitions. This iterator never loads the full grid.
    """
    source, _, records = _property_grid_chunks(path)
    for record in records:
        yield _load_grid_chunk(source.parent, record)


def retry_property_grid_chunks(result, phases, path):
    """Retry failed/inconsistent equilibria in saved batches, visiting backwards.

    Supply the original phase diagram and candidate models. ``path`` is a
    property-grid manifest or its directory; the manifest must retain the full
    ``x`` and ``y`` coordinate arrays. Yield ``(rows, columns, grid)`` in reverse
    batch order, replacing each attempted batch atomically on disk before
    yielding. Thermodynamically consistent states and properties remain unchanged. Remove
    diagnostics for recovered equilibria and retain any property errors.

    Only one solution batch and two rows of warm starts are retained. Accepted
    states prime those rows without solving again. Ordinary fallback fields
    still require a shared boundary within one grid spacing; thermodynamic
    violations trigger full phase discovery. Existing equilibrium_solves
    counts include both passes. Regenerate derived arrays/reports from the
    yielded grids so they reflect the recovered states.
    """
    source, manifest, records = _property_grid_chunks(path)
    if "x" not in manifest or "y" not in manifest:
        raise ValueError("Reverse grid retries require full x/y axes in the manifest.")
    prepared = _core._PropertyGridder(
        _phase_diagram_data(result), phases, manifest["x"], manifest["y"]
    )
    for record in sorted(
        records, key=lambda r: (r["rows"][0], r["columns"][0]), reverse=True
    ):
        rows, cols, grid = _load_grid_chunk(source.parent, record)
        attempted = np.any(~grid.equilibrium_success & (grid.field_numbers > 0))
        solves, diagnostics = grid.equilibrium_solves, tuple(grid.diagnostics)
        prepared.retry(grid)
        if attempted or grid.equilibrium_solves != solves or tuple(grid.diagnostics) != diagnostics:
            save_property_grid(grid, source.parent / record["file"])
        yield rows, cols, grid
        del grid


def _property_grid_chunks(path):
    """Read batch locations and metadata without loading any solution states."""
    import json
    from pathlib import Path

    source = Path(path)
    if source.is_dir():
        source /= "manifest.json"
    with source.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if (
        manifest.get("format") != "burnman_cpp.property_grid_chunks"
        or manifest.get("schema_version") != 1
    ):
        raise ValueError("Unsupported property-grid chunk manifest.")

    if "chunks" in manifest:
        records = manifest["chunks"]
    else:
        with (source.parent / manifest["chunks_file"]).open(
            encoding="utf-8"
        ) as journal:
            records = [json.loads(line) for line in journal if line.strip()]
    return source, manifest, records


def _load_grid_chunk(directory, record):
    """Load one batch and check its coordinate shape against the journal."""
    grid = load_property_grid(directory / record["file"])
    j, jend = record["rows"]
    i, iend = record["columns"]
    if j < 0 or i < 0 or jend - j != len(grid.y) or iend - i != len(grid.x):
        raise ValueError("Saved chunk shape disagrees with its manifest.")
    return slice(j, jend), slice(i, iend), grid


def save_property_grid(grid, path):
    """Save property arrays, masks, metadata and retained solutions; return Path.

    Use ``.json.gz`` for compressed JSON, or ``.json`` for plain JSON.
    Property NaNs become null; signed infinities use "+inf"/"-inf" strings.
    Loading restores their NumPy values. Phase amounts, original
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
    payload = grid.to_dict()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
        opener = gzip.open if destination.suffix == ".gz" else open
        with opener(temporary, "wt", encoding="utf-8") as stream:
            json.dump(payload, stream, allow_nan=False, separators=(",", ":"))
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
