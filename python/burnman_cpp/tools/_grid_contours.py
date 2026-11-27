# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Interpolate sampled fields to their boundaries and clip contour geometry."""

import numpy as np

_GEOMETRY_TOLERANCE = 1e-10  # Distances after rescaling the diagram axes.
_VALUE_TOLERANCE = 32 * np.finfo(float).eps


def _coordinates(records, axes):
    return np.asarray([[point[a] for a in axes] for point in records]).reshape(-1, 2)


def internal_boundary_samples(
    polygon,
    boundary,
    origin,
    span,
    value,
    *,
    grid_axes=(),
    axes=("pressure", "temperature"),
):
    """Sample both field-side limits along an internal discontinuity.

    The zero-width boundary is represented by slightly offset samples on each
    side. ``field_contours`` masks triangles that cross the centreline, keeping
    both limiting values without creating a wide invalid band around it.
    """
    from shapely import (
        distance as geometry_distance,
        points as geometry_points,
        line_locate_point,
    )
    from shapely.geometry import LineString, Polygon

    curves = boundary["limiting_curves"]
    if list(polygon.phases) != curves[0]["assemblage"]:
        return np.empty((0, 2)), np.empty(0), np.empty((0, 2))
    source = _coordinates(boundary["points"], axes)
    normalised = (source - origin) / span
    keep = np.r_[
        True, np.linalg.norm(np.diff(normalised, axis=0), axis=1) > _GEOMETRY_TOLERANCE
    ]
    normalised = normalised[keep]
    if len(normalised) < 2:
        return np.empty((0, 2)), np.empty(0), np.empty((0, 2))
    path = LineString(normalised)
    curve_distances = [
        line_locate_point(
            path, geometry_points((_coordinates(curve["points"], axes) - origin) / span)
        )
        for curve in curves
    ]
    distance = np.r_[
        0.0, np.cumsum(np.linalg.norm(np.diff(normalised, axis=0), axis=1))
    ]
    positions = [*distance, *curve_distances[0], *curve_distances[1]]
    for first, last, start, length in zip(
        normalised[:-1], normalised[1:], distance[:-1], np.diff(distance)
    ):
        for axis, nodes in enumerate(grid_axes):
            if last[axis] != first[axis]:
                fractions = ((nodes - origin[axis]) / span[axis] - first[axis]) / (
                    last[axis] - first[axis]
                )
                positions.extend(
                    start + length * fractions[(fractions > 0) & (fractions < 1)]
                )
    positions = np.unique(positions)
    positions = positions[np.r_[np.diff(positions) > _GEOMETRY_TOLERANCE, True]]
    centre = np.column_stack(
        [np.interp(positions, distance, normalised[:, i]) for i in range(2)]
    )
    tangent = np.gradient(centre, positions, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    right = np.column_stack((tangent[:, 1], -tangent[:, 0]))
    region = Polygon(
        (polygon.vertices - origin) / span,
        [(ring - origin) / span for ring in polygon.holes],
    )
    # Opposite limits need distinct mesh vertices. Keep their separation below
    # the distance to the field edge, including in fields thinner than a cell.
    clearance = geometry_distance(region.boundary, geometry_points(centre))
    roundoff = 8 * np.max(np.abs(np.spacing(origin + centre * span)) / span, axis=1)
    offset = np.minimum(
        clearance / 2, np.maximum(roundoff, np.sqrt(np.finfo(float).eps) * clearance)
    )
    offset[[0, -1]] = 0.0
    points, values, limits = [], [], []
    for side, sign in ((0, 1.0), (1, -1.0)):
        # Saved curves can have independent sampling and ordering. Match their
        # coordinates to the reaction geometry, never to array indices or IDs.
        states = curves[side]["points"]
        distances = curve_distances[side]
        side_values = np.asarray(
            [
                value(state, side, index) if state.get("success", False) else np.nan
                for index, state in enumerate(states)
            ]
        )
        order = np.argsort(distances, kind="stable")
        interpolated = np.full(len(positions), np.nan)
        if len(order):
            interpolated = np.interp(
                positions,
                distances[order],
                side_values[order],
                left=np.nan,
                right=np.nan,
            )
        limits.append(interpolated[[0, -1]])
        coordinates = origin + (centre + sign * offset[:, None] * right) * span
        for coordinate, scalar in zip(coordinates, interpolated):
            probe = np.clip(
                coordinate, origin + span * 1e-10, origin + span * (1 - 1e-10)
            )
            if polygon.contains_rectangle(probe, span * 1e-12):
                points.append(coordinate)
                values.append(scalar)
    junctions = ~np.isclose(limits[0], limits[1], rtol=_VALUE_TOLERANCE, atol=0.0)
    return (
        np.asarray(points).reshape(-1, 2),
        np.asarray(values),
        source[[0, -1]][junctions],
    )


def saved_scalar_samples(data, polygon, state_value, boundary_states, *, grid_axes=()):
    """Collect interior and field-side values, including internal replacements."""
    from collections import Counter
    from .phase_diagram import _section_axes

    _, axes = _section_axes(data)
    ranges = np.asarray([data[a + "_range"] for a in axes])
    origin, span = ranges[:, 0], np.diff(ranges).ravel()
    settings = data.get("settings", {})
    copies = settings.get("max_phase_instances", 3)
    expected = Counter(i // copies for i in polygon.phases)
    internal = [
        line
        for line in data.get("boundaries", [])
        if line.get("is_solution_replacement")
        and line.get("side_a") == line.get("side_b") == list(polygon.phases)
        and len(line.get("limiting_curves", [])) == 2
    ]
    if axes != ["pressure", "temperature"]:
        from shapely.geometry import LineString

        # Latent S/V limits can enclose finite area. Only coincident curves
        # describe an internal line in the selected calculation coordinates.
        internal = [
            line
            for line in internal
            if all(len(c["points"]) > 1 for c in line["limiting_curves"])
            and LineString(
                (_coordinates(line["limiting_curves"][0]["points"], axes) - origin)
                / span
            ).hausdorff_distance(
                LineString(
                    (_coordinates(line["limiting_curves"][1]["points"], axes) - origin)
                    / span
                )
            )
            <= _GEOMETRY_TOLERANCE
        ]
    # Junction IDs join independently solved endpoints, whose coordinates can
    # differ slightly. Include the node used by the polygon as well as each arm.
    junction_points = {
        line.get(key): []
        for line in internal
        for key in ("start_node", "end_node")
        if line.get(key, -1) >= 0
    }
    for line in data.get("boundaries", []):
        for index, name in ((0, "start_node"), (-1, "end_node")):
            if line.get(name) in junction_points:
                junction_points[line[name]].append(line["points"][index])
    for node in data.get("nodes", []):
        if node["id"] in junction_points:
            junction_points[node["id"]].append(node)
    endpoints = _coordinates(
        [p for group in junction_points.values() for p in group]
        + [p for line in internal for p in (line["points"][0], line["points"][-1])],
        axes,
    )
    points, values, barriers, invalid = [], [], [], []

    def append(position, state):
        scalar = float(state_value(state, polygon))
        if np.isfinite(scalar):
            points.append(position)
            values.append(scalar)

    for key, position, state in boundary_states(polygon):
        # The internal line retains both endpoint limits. A polygon alone
        # cannot select between those two roots at the shared junction.
        if (
            len(endpoints)
            and np.min(np.linalg.norm((endpoints - position) / span, axis=1))
            < _GEOMETRY_TOLERANCE
        ):
            continue
        append(position, state)
    boundary_count = len(points)
    interior = {}
    for state in data.get("samples", []):
        if not state.get("success") or state.get("outside_model_domain", False):
            continue
        minimum = settings.get("amount_tolerance", 1e-8) * sum(
            p["amount"] for p in state["phases"]
        )
        active = [p for p in state["phases"] if p["amount"] > minimum]
        if Counter(p["candidate_index"] for p in active) != expected:
            continue
        position = np.asarray([state[a] for a in axes])
        probe = np.clip(position, origin + span * 1e-10, origin + span * (1 - 1e-10))
        if polygon.contains_rectangle(probe, span * 1e-12):
            scalar = float(state_value(dict(state, phases=active), polygon))
            if np.isfinite(scalar):
                interior.setdefault(tuple(position), []).append(scalar)
    for position, scalars in interior.items():
        # Two coexistence limits can be saved at exactly the same coordinates.
        # Their explicit side curves supply the values; an interior sample
        # cannot choose between them, even if the drawn chord is slightly away.
        if not internal or np.allclose(
            scalars, scalars[0], rtol=_VALUE_TOLERANCE, atol=0
        ):
            points.append(position)
            values.append(scalars[0])
    for line in internal:
        extra_points, extra_values, junctions = internal_boundary_samples(
            polygon,
            line,
            origin,
            span,
            lambda state, side, index: state_value(state, polygon),
            grid_axes=grid_axes,
            axes=axes,
        )
        # Side limits take precedence over coincident interior observations.
        points = [*extra_points, *points]
        values = [*extra_values, *values]
        ends = _coordinates([line["points"][0], line["points"][-1]], axes)
        for position in junctions:
            name = "start_node" if np.array_equal(position, ends[0]) else "end_node"
            records = junction_points.get(line.get(name))
            invalid.extend(_coordinates(records, axes) if records else [position])
        barriers.append(_coordinates(line["points"], axes))
    return (
        np.asarray(points).reshape(-1, 2),
        np.asarray(values),
        barriers,
        np.asarray(invalid).reshape(-1, 2),
        boundary_count,
    )


def field_contours(points, values, polygon, invalid, levels, *, barriers=()):
    """Yield (level, coordinates) inside a field, including its boundary.

    Values at pre-existing boundary equilibria belong in points/values.
    Missing boundary vertices use a local affine estimate from this field's
    observations. Invalid grid nodes mask incident triangles. Rescaling by
    the field's own spans also supports fields much thinner than a grid cell.
    """
    from matplotlib.figure import Figure
    from matplotlib.tri import Triangulation
    from scipy.interpolate import LinearNDInterpolator
    from scipy.spatial import Delaunay, QhullError, cKDTree
    from shapely import (
        crosses,
        make_valid,
        union_all,
    )
    from shapely.geometry import LineString, Polygon

    rings = [polygon.vertices, *polygon.holes]
    vertices = np.vstack(rings)
    origin, scale = vertices.min(axis=0), np.ptp(vertices, axis=0)
    if (scale <= 0).any():
        return
    rings = [(ring - origin) / scale for ring in rings]
    region = Polygon(rings[0], rings[1:])
    if not region.is_valid:
        region = make_valid(region)
        if region.geom_type == "GeometryCollection":
            region = union_all(
                [
                    part
                    for part in region.geoms
                    if part.geom_type in ("Polygon", "MultiPolygon")
                ]
            )
    points = (np.asarray(points).reshape(-1, 2) - origin) / scale
    values = np.asarray(values)
    barriers = [
        (np.asarray(line, dtype=float).reshape(-1, 2) - origin) / scale
        for line in barriers
        if np.asarray(line).size >= 4
    ]
    finite = np.isfinite(values)
    invalid = np.vstack(
        [(np.asarray(invalid).reshape(-1, 2) - origin) / scale, points[~finite]]
    )
    points, values = points[finite], values[finite]
    # Boundary observations precede grid observations, so coincident vertices
    # retain the solved limiting composition rather than a nearby grid root.
    points, indices = np.unique(points, axis=0, return_index=True)
    values = values[indices]
    if len(points) < 3:
        return
    try:
        triangulation = Delaunay(points)
    except QhullError:
        return
    boundary = np.unique(np.vstack(rings), axis=0)
    estimates = LinearNDInterpolator(triangulation, values)(boundary)
    missing = np.flatnonzero(~np.isfinite(estimates))
    if len(missing):
        tree = cKDTree(points)
        for index in missing:
            _, near = tree.query(boundary[index], k=min(6, len(points)))
            design = np.column_stack(
                [np.ones(len(near)), points[near] - boundary[index]]
            )
            estimates[index] = np.linalg.lstsq(design, values[near], rcond=None)[0][0]
    coordinates = np.vstack([boundary, points, invalid])
    samples = np.r_[estimates, values, np.full(len(invalid), np.nan)]
    # Invalid observations take precedence at duplicate coordinates.
    coordinates, reverse_indices = np.unique(
        coordinates[::-1], axis=0, return_index=True
    )
    samples = samples[::-1][reverse_indices]
    try:
        triangles = Delaunay(coordinates).simplices
    except QhullError:
        return
    mask = ~np.isfinite(samples[triangles]).all(axis=1)
    if barriers:
        from shapely import polygons

        triangle_shapes = polygons(coordinates[triangles])
        for barrier in barriers:
            if len(barrier) >= 2:
                mask |= crosses(triangle_shapes, LineString(barrier))
    if mask.all():
        return
    mesh = Triangulation(*coordinates.T, triangles=triangles, mask=mask)
    used = samples[np.unique(triangles[~mask])]
    low, high = used.min(), used.max()
    selected = sorted({float(level) for level in levels if low <= level <= high})
    if not selected or low == high:
        return
    figure = Figure()
    try:
        contours = figure.subplots().tricontour(
            mesh, np.nan_to_num(samples), levels=selected
        )
        for level, segments in zip(contours.levels, contours.allsegs):
            for segment in segments:
                if len(segment) < 2:
                    continue
                clipped = LineString(segment).intersection(region)
                pieces = clipped.geoms if hasattr(clipped, "geoms") else [clipped]
                for piece in pieces:
                    if piece.geom_type == "LineString" and not piece.is_empty:
                        yield float(level), np.asarray(piece.coords) * scale + origin
    finally:
        figure.clear()
