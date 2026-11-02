# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.
"""Interpolate sampled fields to their boundaries and clip contour geometry."""

import numpy as np


def field_contours(points, values, polygon, invalid, levels):
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
    from shapely import make_valid, union_all
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
