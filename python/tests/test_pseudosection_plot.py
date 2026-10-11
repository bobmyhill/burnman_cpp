"""Analytic polygon geometry and phase counts, independent of thermodynamics."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import burnman_cpp as bm


def line(points, left, right):
    return dict(
        points=[dict(pressure=p, temperature=t) for p, t in points],
        side_b=left,
        side_a=right,
    )


def sample(p, t, phases):
    return dict(
        pressure=p,
        temperature=t,
        success=True,
        phases=[dict(id=i, amount=1.0) for i in phases],
    )


def diagram(lines, samples=()):
    return dict(
        pressure_range=(0.0, 1.0),
        temperature_range=(0.0, 1.0),
        boundaries=lines,
        samples=list(samples),
        fields=[],
        nodes=[],
    )


def square(low=0.2, high=0.8):
    return [(low, low), (high, low), (high, high), (low, high), (low, low)]


def check_joined_edges(geometry):
    def key(a, b):
        return tuple(sorted((tuple(a), tuple(b))))

    expected = Counter(key(*segment) for segment in geometry.boundary_segments)
    actual = Counter(
        key(a, b)
        for path in geometry.boundary_lines
        for a, b in zip(path[:-1], path[1:])
    )
    assert actual == expected


@pytest.mark.parametrize("endpoint", [False, True])
def test_plot_joins_matching_segments_only_between_endpoint_vertices(endpoint):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    points = [(0.0, 0.25), (0.3, 0.35), (0.7, 0.65), (1.0, 0.75)]
    data = diagram(
        [
            line(points[:2], [0, 1], [0, 2]),
            line(points[1:3], [0, 1], [0, 2]),
            line(points[2:][::-1], [0, 2], [0, 1]),
        ]
    )
    if endpoint:
        data["nodes"] = [dict(id=0, pressure=0.3, temperature=0.35)]
        data["boundaries"][0]["end_node"] = 0
        data["boundaries"][1]["start_node"] = 0
    fig, ax = bm.plot_pseudosection(
        data,
        pressure_unit="Pa",
        temperature_unit="K",
        swap_axes=True,
        label_assemblages=False,
        colorbar=False,
    )
    geometry = ax.pseudosection_geometry
    assert not geometry.diagnostics
    assert {tuple(p.phases) for p in geometry.polygons} == {(0, 1), (0, 2)}
    assert sum(p.area for p in geometry.polygons) == pytest.approx(1.0)
    check_joined_edges(geometry)
    outlines = next(c for c in ax.collections if isinstance(c, LineCollection))
    joined = [path for path in outlines.get_segments() if len(path) > 2]
    assert len(joined) == 1
    expected = np.asarray(points[1:] if endpoint else points)
    assert np.allclose(joined[0], expected) or np.allclose(joined[0], expected[::-1])
    plt.close(fig)


def test_dense_boundary_keeps_all_refined_points_in_rendering_and_svg():
    from io import BytesIO
    import re
    import xml.etree.ElementTree as ET
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    pressure = np.linspace(0.0, 1.0, 257)
    temperature = 0.5 + 0.15 * np.sin(2 * np.pi * pressure)
    points = np.column_stack((pressure, temperature))
    with mpl.rc_context({"path.simplify": True, "path.simplify_threshold": 1.0}):
        fig, ax = bm.plot_pseudosection(
            diagram([line(points, [0], [1])]),
            pressure_unit="Pa",
            temperature_unit="K",
            swap_axes=True,
            label_assemblages=False,
            colorbar=False,
        )
        try:
            check_joined_edges(ax.pseudosection_geometry)
            outlines = next(c for c in ax.collections if isinstance(c, LineCollection))
            path = max(outlines.get_paths(), key=lambda p: len(p.vertices))
            assert len(path.vertices) == len(points)
            fig.canvas.draw()
            rendered = np.vstack(
                [v for v, _ in path.iter_segments(transform=ax.transData, curves=False)]
            )
            np.testing.assert_allclose(rendered, ax.transData.transform(path.vertices))
            outlines.set_gid("phase-boundaries")
            output = BytesIO()
            fig.savefig(output, format="svg")
            ns = {"svg": "http://www.w3.org/2000/svg"}
            group = ET.fromstring(output.getvalue()).find(
                ".//svg:g[@id='phase-boundaries']", ns
            )
            counts = [
                len(re.findall(r"[ML]", p.attrib["d"]))
                for p in group.findall(".//svg:path", ns)
            ]
            assert sorted(counts) == sorted(
                len(p.vertices) for p in outlines.get_paths()
            )
        finally:
            plt.close(fig)


def test_joined_lines_keep_closed_loops_and_distinct_paths_between_same_vertices():
    points = square()
    data = diagram([line(points, [0, 1], [0])])
    loop = bm.pseudosection_field_polygons(data)
    check_joined_edges(loop)
    closed = [path for path in loop.boundary_lines if len(path) > 2]
    assert len(closed) == 1 and len(closed[0]) == 5
    np.testing.assert_array_equal(closed[0][0], closed[0][-1])
    assert sorted(p.area for p in loop.polygons) == pytest.approx([0.36, 0.64])
    # Explicit endpoints keep the two arcs separate, retaining the entire field.
    data["boundaries"] = [
        line(points[:3], [0, 1], [0]) | dict(start_node=0, end_node=1),
        line(points[2:], [0, 1], [0]) | dict(start_node=1, end_node=0),
    ]
    data["nodes"] = [
        dict(id=i, pressure=p, temperature=t)
        for i, (p, t) in enumerate(points[::2][:2])
    ]
    split = bm.pseudosection_field_polygons(data)
    check_joined_edges(split)
    assert len([path for path in split.boundary_lines if len(path) > 2]) == 2
    assert sorted(p.area for p in split.polygons) == pytest.approx([0.36, 0.64])


def test_joining_does_not_cross_a_change_of_assemblage_with_the_same_phase_count():
    points = [(0.0, 0.3), (0.4, 0.4), (0.6, 0.7), (1.0, 0.8)]
    data = diagram([line(points[:2], [0, 1], [0, 2]), line(points[1:], [0, 1], [0, 3])])
    geometry = bm.pseudosection_field_polygons(data)
    check_joined_edges(geometry)
    paths = [path for path in geometry.boundary_lines if len(path) > 2]
    assert len(paths) == 1 and len(paths[0]) == 3
    assert np.allclose(paths[0], points[1:]) or np.allclose(paths[0], points[:0:-1])


def test_joining_keeps_intersections_and_disconnected_boundaries_separate():
    vertical = [(0.5, 0.0), (0.5, 0.25), (0.5, 0.75), (0.5, 1.0)]
    horizontal = [(t, p) for p, t in vertical]
    crossing = bm.pseudosection_field_polygons(
        diagram([line(vertical, [0], [1]), line(horizontal, [0], [1])]),
        merge_fields=False,
    )
    check_joined_edges(crossing)
    paths = [path for path in crossing.boundary_lines if len(path) > 2]
    assert len(paths) == 2
    assert all(not any(np.allclose(p, [0.5, 0.5]) for p in path) for path in paths)
    disconnected = bm.pseudosection_field_polygons(
        diagram(
            [
                line([(0.2, 0.2), (0.3, 0.3), (0.4, 0.4)], [0], [1]),
                line([(0.6, 0.6), (0.7, 0.7), (0.8, 0.8)], [0], [1]),
            ]
        ),
        close_domain=False,
    )
    check_joined_edges(disconnected)
    assert len(disconnected.boundary_lines) == 2
    assert all(len(path) == 3 for path in disconnected.boundary_lines)


@pytest.mark.parametrize("closed", [False, True])
def test_closed_loops_preserve_boundaries_with_identical_phase_names(closed):
    points = square() if closed else [(0.5, 0.0), (0.5, 0.5), (0.5, 1.0)]
    boundary = line(points, [0, 1], [0, 1])
    for replacement in (False, True):
        boundary["is_solution_replacement"] = replacement
        geometry = bm.pseudosection_field_polygons(diagram([boundary]))
        assert not geometry.diagnostics
        assert len(geometry.polygons) == 2
        assert sorted(p.area for p in geometry.polygons) == pytest.approx(
            [0.36, 0.64] if closed else [0.5, 0.5])
        assert all(p.phases == [0, 1] for p in geometry.polygons)
        check_joined_edges(geometry)


@pytest.mark.parametrize("amount_tolerance", [1.0e-9, 1.0e-7, 1.0e-5])
@pytest.mark.parametrize("trace_fraction", [5.0e-8, 5.0e-6])
@pytest.mark.parametrize("total_amount", [1.0e-6, 1.0e6])
def test_field_counts_use_configured_relative_amount_tolerance(
    amount_tolerance, trace_fraction, total_amount
):
    interior = sample(0.5, 0.5, [0, 1])
    interior["phases"][0]["amount"] = total_amount * (1.0 - trace_fraction)
    interior["phases"][1]["amount"] = total_amount * trace_fraction
    data = diagram([], [interior])
    data["settings"] = dict(amount_tolerance=amount_tolerance)
    geometry = bm.pseudosection_field_polygons(data)
    assert not geometry.diagnostics
    assert len(geometry.polygons) == 1
    expected = [0, 1] if trace_fraction > amount_tolerance else [0]
    polygon = geometry.polygons[0]
    assert polygon.phases == expected
    assert polygon.n_phases == len(expected)
    assert polygon.area == pytest.approx(1.0)


def test_closed_loop_and_surrounding_field_preserve_hole():
    result = bm.pseudosection_field_polygons(
        diagram(
            [line(square(), [0, 1, 2], [0])],
            [sample(0.5, 0.5, [0, 1, 2]), sample(0.1, 0.1, [0])],
        )
    )
    assert not result.diagnostics
    assert len(result.polygons) == 2
    by_count = {p.n_phases: p for p in result.polygons}
    assert by_count[1].area == pytest.approx(0.64)
    assert by_count[3].area == pytest.approx(0.36)
    assert len(by_count[1].holes) == 1
    assert not by_count[3].holes
    for polygon in result.polygons:
        np.testing.assert_array_equal(polygon.vertices[0], polygon.vertices[-1])
        assert (polygon.label_position > 0.0).all() and (
            polygon.label_position < 1.0
        ).all()
        in_square = (polygon.label_position > 0.2).all() and (
            polygon.label_position < 0.8
        ).all()
        assert in_square == (polygon.n_phases == 3)
    enclosed = bm.pseudosection_field_polygons(
        diagram([line(square(), [0, 1, 2], [0])]), close_domain=False
    )
    assert len(enclosed.polygons) == 1 and enclosed.polygons[0].n_phases == 3


def test_tiny_face_area_is_stable_far_from_coordinate_origin():
    low, high = 0.9, 0.9 + 4.0e-8
    result = bm.pseudosection_field_polygons(
        diagram([line(square(low, high), [0, 1], [0])]), merge_fields=False
    )
    enclosed = next(p for p in result.polygons if p.n_phases == 2)
    assert enclosed.area == pytest.approx((high - low) ** 2, rel=1.0e-12, abs=0.0)
    assert sum(p.area for p in result.polygons) == pytest.approx(1.0, abs=1.0e-15)


def test_verified_interior_can_identify_face_thinner_than_snap_tolerance():
    triangle = [(0.1, 0.1), (0.9, 0.9), (0.5 + 2.0e-9, 0.5 - 2.0e-9), (0.1, 0.1)]
    interior = sample(0.5 + 2.0e-9 / 3.0, 0.5 - 2.0e-9 / 3.0, [0, 1, 2])
    data = diagram([line(triangle, [0], [0, 1, 2])], [interior])
    ordinary = bm.pseudosection_field_polygons(data, merge_fields=False)
    assert {p.n_phases for p in ordinary.polygons} == {1, 3}
    assert all(p.sample_index == -1 for p in ordinary.polygons)
    interior["is_field_verification"] = True
    verified = bm.pseudosection_field_polygons(data, merge_fields=False)
    assert not verified.diagnostics
    assert {p.n_phases for p in verified.polygons} == {1, 3}
    enclosed = next(p for p in verified.polygons if p.n_phases == 3)
    assert enclosed.phases == [0, 1, 2]
    assert enclosed.sample_index == 0


@pytest.mark.parametrize("axes", ["PS", "TV"])
@pytest.mark.parametrize("variance,expected_count", [(0, 2), (1, 3)])
def test_latent_invariant_loops_have_no_area_but_univariant_strips_remain(
    axes, variance, expected_count
):
    ranges = dict(P=(23.2e9, 23.9e9), T=(1400.0, 1700.0),
                  S=(300.0, 340.0), V=(3.3e-5, 3.7e-5))
    names = dict(P="pressure", T="temperature", S="entropy", V="volume")

    def point(x, y):
        values = dict(pressure=23.5e9, temperature=1532.0)
        for axis, value in zip(axes, (x, y)):
            low, high = ranges[axis]
            values[names[axis]] = low + value * (high - low)
        return values

    phases = [0, 1, 2, 3]
    data = dict(
        diagram_type=axes,
        pressure_range=ranges["P"], temperature_range=ranges["T"],
        entropy_range=ranges["S"], volume_range=ranges["V"],
        fields=[],
        nodes=[dict(id=i, kind="junction", gibbs_variance=variance,
                    pt_nullity=0, assemblage=phases, **point(0.5, y))
               for i, y in enumerate((0.0, 1.0))],
        boundaries=[dict(
            id=-1, start_node=0, end_node=1, assemblage=phases,
            side_b=[0, 1, 2], side_a=[0, 1, 3],
            points=[point(0.5, 0.0), point(0.5 + delta, 0.5), point(0.5, 1.0)],
        ) for delta in (-1.e-12, 1.e-12)],
        samples=[dict(success=True, **point(x, 0.5),
                      phases=[dict(id=i, amount=1.0) for i in ids])
                 for x, ids in [(0.25, [0, 1, 2]), (0.75, [0, 1, 3])]],
    )
    before = json.dumps(data, sort_keys=True)
    geometry = bm.pseudosection_field_polygons(
        data, tolerance=1.e-12, merge_fields=False
    )
    assert json.dumps(data, sort_keys=True) == before
    assert len(geometry.polygons) == expected_count
    assert sum(p.area for p in geometry.polygons) == pytest.approx(1.0)
    if variance == 0:
        assert not geometry.diagnostics
        assert {tuple(p.phases) for p in geometry.polygons} == {
            (0, 1, 2), (0, 1, 3)
        }
    else:
        assert any(0.0 < p.area < 1.e-11 for p in geometry.polygons)


def test_nested_and_disconnected_fields():
    data = diagram(
        [line(square(), [0, 1], [0]), line(square(0.4, 0.6), [0, 1, 2], [0, 1])]
    )
    result = bm.pseudosection_field_polygons(data)
    assert not result.diagnostics
    assert {p.n_phases: p.area for p in result.polygons} == pytest.approx(
        {1: 0.64, 2: 0.32, 3: 0.04}
    )
    first = [(p * 0.25, t * 0.25) for p, t in square()]
    second = [(p * 0.25 + 0.7, t * 0.25 + 0.7) for p, t in square()]
    result = bm.pseudosection_field_polygons(
        diagram([line(first, [0, 1], [0]), line(second, [0, 1], [0])])
    )
    assert sorted(p.n_phases for p in result.polygons) == [1, 2, 2]
    assert sum(p.area for p in result.polygons) == pytest.approx(1.0)
    assert len(next(p for p in result.polygons if p.n_phases == 1).holes) == 2


def test_crossing_curves_between_stable_vertices_form_one_field_loop():
    # Crossing coarse chords used to introduce several artificial sliver fields.
    first = [(0.2, 0.2), (0.7, 0.3), (0.3, 0.6), (0.8, 0.8)]
    second = [(0.8, 0.8), (0.2, 0.7), (0.7, 0.4), (0.2, 0.2)]
    data = diagram([
        line(first, [0, 1], [0]) | dict(start_node=0, end_node=1),
        line(second, [0, 1], [0]) | dict(start_node=1, end_node=0),
    ])
    data["nodes"] = [dict(id=i, pressure=p, temperature=t)
                     for i, (p, t) in enumerate([first[0], first[-1]])]
    before = json.dumps(data, sort_keys=True)
    geometry = bm.pseudosection_field_polygons(data)
    assert json.dumps(data, sort_keys=True) == before
    assert not geometry.diagnostics
    assert len(geometry.polygons) == 2
    assert {tuple(p.phases) for p in geometry.polygons} == {(0,), (0, 1)}
    assert all(not p.has_open_boundary for p in geometry.polygons)
    assert all(len(p.source_regions) == 1 for p in geometry.polygons)
    assert sum(p.area for p in geometry.polygons) == pytest.approx(1.0)
    assert len(geometry.boundary_lines) == 6  # two saved arcs and four frame sides
    np.testing.assert_array_equal(geometry.boundary_lines[0], first)
    np.testing.assert_array_equal(geometry.boundary_lines[1], second)
    check_joined_edges(geometry)


def test_explicit_junctions_close_adjacent_fields_without_polygon_union():
    data = diagram([
        line(square(), [0, 1], [0]),
        line([(0.5, 0.0), (0.5, 0.2)], [0], [0]) | dict(end_node=0),
        line([(0.5, 0.8), (0.5, 1.0)], [0], [0]) | dict(start_node=1),
    ])
    data["nodes"] = [dict(id=i, pressure=0.5, temperature=t)
                     for i, t in enumerate([0.2, 0.8])]
    for merge in (False, True):
        geometry = bm.pseudosection_field_polygons(data, merge_fields=merge)
        assert not geometry.diagnostics
        assert len(geometry.polygons) == 3
        assert sorted(p.area for p in geometry.polygons) == pytest.approx([0.32, 0.32, 0.36])
        assert all(len(p.source_regions) == 1 for p in geometry.polygons)
        assert sum(p.area for p in geometry.polygons) == pytest.approx(1.0)


def test_nested_closed_loops_are_retained_even_with_identical_labels():
    data = diagram([line(square(), [0], [0]), line(square(0.4, 0.6), [0, 1], [0])])
    geometry = bm.pseudosection_field_polygons(data)
    assert not geometry.diagnostics
    assert len(geometry.polygons) == 3
    assert sorted(p.area for p in geometry.polygons) == pytest.approx([0.04, 0.32, 0.64])
    assert sorted(len(p.holes) for p in geometry.polygons) == [0, 1, 1]
    assert all(len(p.source_regions) == 1 for p in geometry.polygons)
    assert sum(p.area for p in geometry.polygons) == pytest.approx(1.0)


def test_same_assemblage_loops_touching_at_a_stable_vertex_remain_separate():
    first = [(0.5, 0.5), (0.2, 0.5), (0.2, 0.2), (0.5, 0.2), (0.5, 0.5)]
    second = [(0.5, 0.5), (0.8, 0.5), (0.8, 0.8), (0.5, 0.8), (0.5, 0.5)]
    data = diagram([line(p, [0, 1], [0]) | dict(start_node=0, end_node=0)
                    for p in (first, second)])
    data["nodes"] = [dict(id=0, pressure=0.5, temperature=0.5)]
    geometry = bm.pseudosection_field_polygons(data)
    assert not geometry.diagnostics
    assert sorted(p.n_phases for p in geometry.polygons) == [1, 2, 2]
    assert sorted(p.area for p in geometry.polygons) == pytest.approx([0.09, 0.09, 0.82])
    assert all(len(p.source_regions) == 1 for p in geometry.polygons)


def test_hole_is_not_assigned_to_neighbour_touching_its_first_vertex():
    # The clockwise face walk starts at (.4,.4), also on the larger square.
    # That shared vertex alone must not turn the smaller field into its hole.
    first = [(0.4, 0.2), (0.4, 0.4), (0.2, 0.4), (0.2, 0.2), (0.4, 0.2)]
    data = diagram([line(first, [0, 1], [0]), line(square(0.4, 0.7), [0, 1, 2], [0])])
    result = bm.pseudosection_field_polygons(data)
    assert not result.diagnostics
    fields = {p.n_phases: p for p in result.polygons}
    assert {n: p.area for n, p in fields.items()} == pytest.approx(
        {1: 0.87, 2: 0.04, 3: 0.09}
    )
    assert len(fields[1].holes) == 2
    assert not fields[2].holes and not fields[3].holes


def test_grid_closed_loops_preserve_each_cell_and_assemblage():
    # Saved grid endpoints define cells even where neighbours share a label.
    grid = np.array(
        [
            [0, 0, 0, 1, 1],
            [0, 2, 0, 1, 2],
            [0, 0, 0, 2, 2],
            [1, 1, 1, 2, 1],
            [1, 2, 1, 1, 1],
        ]
    )
    n = len(grid)
    lines = []
    for i in range(1, n):
        for j in range(n):
            lines.append(
                line(
                    [(i / n, j / n), (i / n, (j + 1) / n)],
                    [int(grid[i - 1, j])],
                    [int(grid[i, j])],
                )
            )
    for j in range(1, n):
        for i in range(n):
            lines.append(
                line(
                    [(i / n, j / n), ((i + 1) / n, j / n)],
                    [int(grid[i, j])],
                    [int(grid[i, j - 1])],
                )
            )
    data = diagram(
        lines,
        [
            sample((i + 0.5) / n, (j + 0.5) / n, [int(grid[i, j])])
            for i in range(n)
            for j in range(n)
        ],
    )
    raw = bm.pseudosection_field_polygons(data, merge_fields=False)
    merged = bm.pseudosection_field_polygons(data)
    assert not raw.diagnostics and not merged.diagnostics
    assert len(raw.polygons) == n * n
    assert len(merged.polygons) == n * n
    for geometry in (raw, merged):
        assert sum(p.area for p in geometry.polygons) == pytest.approx(1.0)
        for p in geometry.polygons:
            cell = tuple(np.floor(p.label_position * n).astype(int))
            assert p.phases == [int(grid[cell])]
            assert p.area == pytest.approx(1.0 / (n * n))
            assert len(p.source_regions) == 1


def test_open_boundary_is_not_extended_to_close_a_field():
    data = diagram(
        [line([(0.5, 0.0), (0.5, 0.7)], [0], [0, 1])], [sample(0.1, 0.1, [0])]
    )
    result = bm.pseudosection_field_polygons(data, close_domain=False)
    assert not result.polygons
    result = bm.pseudosection_field_polygons(data)
    assert len(result.polygons) == 1 and result.polygons[0].phases == [0]
    assert result.polygons[0].has_open_boundary
    assert result.polygons[0].is_provisional
    assert result.diagnostics


def test_roundoff_in_repeated_curves_does_not_create_unfinished_fields():
    points = np.array(
        [
            (0.2, 0.0),
            (0.21, 0.025),
            (0.23, 0.05),
            (0.3, 0.1),
            (0.35, 0.2),
            (0.34, 0.3),
            (0.31, 0.4),
            (0.3, 1.0),
        ]
    )
    duplicate = points.copy()
    duplicate[1:-1] += (
        np.array([(3, -13), (9, 4), (-5, 6), (4, 3), (0.3, 5), (-7, -2)]) * 1.0e-15
    )
    geometry = bm.pseudosection_field_polygons(
        diagram([line(points, [0], [1]), line(duplicate[::-1], [1], [0])])
    )
    assert not geometry.diagnostics
    assert {tuple(p.phases) for p in geometry.polygons} == {(0,), (1,)}
    assert len(geometry.polygons) == 2
    assert sum(p.area for p in geometry.polygons) == pytest.approx(1.0)


@pytest.mark.parametrize("width,expected_count", [(1.0e-16, 1), (2.0e-9, 2)])
def test_roundoff_width_rings_are_removed_but_thin_fields_are_kept(
    width, expected_count
):
    triangle = [(0.15, 0.08), (0.16, 0.28), (0.155 + width, 0.18), (0.15, 0.08)]
    geometry = bm.pseudosection_field_polygons(
        diagram([line(triangle, [0], [0, 1])], [sample(0.5, 0.5, [0])]),
        tolerance=1.0e-12,
        merge_fields=False,
    )
    assert not geometry.diagnostics
    assert len(geometry.polygons) == expected_count
    assert {p.n_phases for p in geometry.polygons} == set(range(1, expected_count + 1))
    assert sum(p.area for p in geometry.polygons) == pytest.approx(1.0)


@pytest.mark.parametrize("width,expected_count", [(0.0, 1), (2.0e-9, 2)])
def test_intersection_roundoff_near_a_metasediment_junction(width, expected_count):
    # Three split chords near the sp/mt/ru event in the metasediment example.
    # Their apparent enclosed width is below the absolute coordinate roundoff
    # band. A wider triangle at the same location must remain a distinct field.
    triangle = [
        (0.003085696724597525, 0.16115956844058948),
        (0.0030856982462141706, 0.16115956405772522),
        (0.0030856976385097486 + width, 0.16115956580818627),
        (0.003085696724597525, 0.16115956844058948),
    ]
    geometry = bm.pseudosection_field_polygons(
        diagram([line(triangle, [0, 1], [0])], [sample(0.5, 0.5, [0])]),
        tolerance=1.0e-12,
        merge_fields=False,
    )
    assert not geometry.diagnostics
    assert len(geometry.polygons) == expected_count
    assert {p.n_phases for p in geometry.polygons} == set(range(1, expected_count + 1))
    assert sum(p.area for p in geometry.polygons) == pytest.approx(1.0)


def test_consistent_perimeter_takes_priority_over_a_near_chord_sample():
    data = diagram(
        [line([(0.5, 0.0), (0.5, 1.0)], [0], [1])],
        [sample(0.8, 0.5, [1]), sample(0.5 + 5.0e-9, 0.5, [1, 2])],
    )
    geometry = bm.pseudosection_field_polygons(data)
    assert not geometry.diagnostics
    assert {tuple(p.phases) for p in geometry.polygons} == {(0,), (1,)}
    assert all(not p.is_provisional for p in geometry.polygons)


@pytest.mark.parametrize("retains_equations", [False, True])
@pytest.mark.parametrize("trace_fraction", [-5.e-7, -5.e-10, 0.0, 5.e-8])
def test_subcutoff_samples_do_not_override_consistent_perimeter(
    retains_equations, trace_fraction
):
    interior = sample(0.5 + 5.e-9, 0.5, [0])
    if retains_equations:
        interior["phases"].append(dict(id=1, amount=trace_fraction))
    data = diagram(
        [line([(0.5, 0.0), (0.5, 1.0)], [0], [0, 1])],
        [sample(0.9, 0.5, [0, 1]), interior],
    )
    geometry = bm.pseudosection_field_polygons(data)
    assert not geometry.diagnostics
    assert {tuple(p.phases) for p in geometry.polygons} == {(0,), (0, 1)}


def test_near_edge_sample_does_not_change_perimeter_labels():
    data = diagram(
        [
            line([(0.3, 0.0), (0.3, 1.0)], [1], [0]),
            line([(0.5, 0.0), (0.5, 1.0)], [0], [1]),
        ],
        [
            sample(0.1, 0.5, [1]),
            sample(0.4, 0.5, [0]),
            sample(0.8, 0.5, [1]),
            sample(0.5 - 5.0e-9, 0.5, [1]),
        ],
    )
    geometry = bm.pseudosection_field_polygons(data, merge_fields=False)
    assert not geometry.diagnostics
    assert len(geometry.polygons) == 3
    assert [p.phases for p in sorted(geometry.polygons, key=lambda p: p.label_position[0])] == [[1], [0], [1]]


def test_closed_thin_field_survives_without_intersection_snapping():
    width = 2.e-9
    points = [(0.2, 0.2), (0.8 + width, 0.8), (0.8, 0.8), (0.2, 0.2)]
    geometry = bm.pseudosection_field_polygons(
        diagram([line(points, [0, 1], [0])]), tolerance=1.e-12)
    assert not geometry.diagnostics
    assert len(geometry.polygons) == 2
    field = next(p for p in geometry.polygons if p.n_phases == 2)
    assert field.area == pytest.approx(0.3 * width, rel=1.e-6)
    assert not field.has_open_boundary
    assert sum(p.area for p in geometry.polygons) == pytest.approx(1.0)


def test_saved_pyrolite_fragments_296_to_298_belong_to_one_vertex_loop():
    data = json.loads((Path(__file__).parent / "data" /
                       "pyrolite_stable_vertex_loop.json").read_text())
    geometry = bm.pseudosection_field_polygons(data)
    target = [27, 30, 33, 39, 42, 45, 60, 78]
    matches = []
    for point in data["fragment_label_positions"]:
        containing = [i for i, polygon in enumerate(geometry.polygons)
                      if polygon.contains_rectangle(point, [1.e-6, 1.e-12])]
        assert len(containing) == 1
        matches.extend(containing)
    assert len(set(matches)) == 1
    field = geometry.polygons[matches[0]]
    assert field.phases == target
    assert not field.has_open_boundary
    assert len(field.source_regions) == 1


@pytest.mark.parametrize("tip,open_count", [((0.1, 0.5), 1), ((0.3, 0.5), 2)])
def test_unfinished_branch_marks_its_field_provisional_and_keeps_both_labels(tip, open_count):
    data = diagram([line(square(), [0, 1], [0]), line([(0.2, 0.5), tip], [0], [0, 1]) | dict(start_node=0)])
    data["nodes"] = [dict(id=0, pressure=0.2, temperature=0.5)]
    result = bm.pseudosection_field_polygons(data)
    assert sorted(p.n_phases for p in result.polygons) == [1, 2]
    provisional = [p for p in result.polygons if p.is_provisional]
    assert len(provisional) == 1
    assert provisional[0].n_phases == open_count
    assert provisional[0].has_open_boundary
    assert len(result.diagnostics) == 1


def test_domain_edges_close_fields_and_count_solution_instances():
    # Different IDs of the same candidate count as distinct coexisting phases.
    data = diagram(
        [line([(0.5, 0.0), (0.5, 1.0)], [0], [0, 1])],
        [sample(0.25, 0.5, [0]), sample(0.75, 0.5, [0, 1])],
    )
    result = bm.pseudosection_field_polygons(data)
    assert not result.diagnostics
    assert sorted((p.n_phases, p.area) for p in result.polygons) == [(1, 0.5), (2, 0.5)]
    data["samples"][0]["phases"].append(dict(id=2, amount=0.0))
    assert sorted(
        p.n_phases for p in bm.pseudosection_field_polygons(data).polygons
    ) == [1, 2]


def test_invalid_geometry_is_rejected():
    with pytest.raises(ValueError, match="tolerance"):
        bm.pseudosection_field_polygons(diagram([]), tolerance=0.0)
    with pytest.raises(ValueError, match="finite"):
        bm.pseudosection_field_polygons(
            diagram([line([(0.0, 0.0), (np.nan, 1.0)], [0], [1])])
        )


def test_conflicting_perimeter_without_samples_remains_unidentified():
    points = square()
    data = diagram([line(points[:3], [0, 1], [3]), line(points[2:], [0, 2], [3])])
    result = bm.pseudosection_field_polygons(data)
    inner = next(p for p in result.polygons if p.area == pytest.approx(0.36))
    assert not inner.phases and inner.n_phases == 0
    assert inner.is_provisional
    assert len(result.diagnostics) == 1
    assert "conflicting perimeter assemblages" in result.diagnostics[0]


def test_native_equilibrium_result_and_saved_json_range_names():
    from burnman_cpp.minerals import HP11

    settings = bm.PseudosectionSettings()
    settings.pressure_seeds = settings.temperature_seeds = 3
    result = bm.pseudosection(
        {"Al": 2.0, "Si": 1.0, "O": 5.0},
        [HP11.andalusite(), HP11.ky(), HP11.sill()],
        (1e5, 1e9),
        (500.0, 1200.0),
        settings,
    )
    polygons = bm.pseudosection_field_polygons(result)
    assert not polygons.diagnostics
    assert len(polygons.polygons) == 3
    assert all(p.n_phases == 1 for p in polygons.polygons)
    assert sum(p.area for p in polygons.polygons) == pytest.approx(1.0)
    data = diagram([line(square(), [0, 1], [0])])
    data["calculation_pressure_range_Pa"] = data.pop("pressure_range")
    data["temperature_range_K"] = data.pop("temperature_range")
    assert sorted(
        p.n_phases for p in bm.pseudosection_field_polygons(data).polygons
    ) == [1, 2]


def test_plot_has_discrete_phase_count_colours_and_hole(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = diagram([line(square(), [0, 1, 2], [0])])
    fig, ax = bm.plot_pseudosection(
        data, pressure_unit="Pa", temperature_unit="K", fill_alpha=1.0
    )
    assert sorted(ax.collections[0].get_array()) == [1.0, 3.0]
    assert ax.collections[0].norm.boundaries.tolist() == [0.5, 1.5, 2.5, 3.5]
    assert fig.axes[1].get_ylabel() == "Number of phases"
    # Rendered centre must use the inner field's colour, not an outer patch
    # painted across its hole. Check pixels well away from line antialiasing.
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    for point, count in [((0.5, 0.5), 3), ((0.1, 0.1), 1)]:
        x, y = ax.transData.transform(point).astype(int)
        expected = ax.collections[0].cmap(ax.collections[0].norm(count))[:3]
        np.testing.assert_allclose(
            rgba[rgba.shape[0] - 1 - y, x, :3] / 255.0, expected, atol=0.01
        )
    fig.savefig(tmp_path / "filled.svg")
    plt.close(fig)


@pytest.mark.parametrize("named", [False, True])
def test_assemblage_colours_preserve_solution_multiplicity_and_polygon_holes(named):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    data = diagram([line(square(), [0, 1, 2], [2])])
    data["phase_names"] = ["g", "g #2", "q"]
    colours = (
        {("q", "g", "g"): "red", ("q",): "blue"}
        if named
        else {
            (2, 1, 0): "red",
            (2,): "blue",
        }
    )
    fig, ax = bm.plot_pseudosection(
        data,
        pressure_unit="Pa",
        temperature_unit="K",
        fill_alpha=1.0,
        assemblage_colors=colours,
        label_assemblages=False,
    )
    # An assemblage palette must not leave a misleading phase-count colourbar.
    assert len(fig.axes) == 1
    assert ax.collections[0].get_array() is None
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    for point, colour in [((0.5, 0.5), "red"), ((0.1, 0.1), "blue")]:
        x, y = ax.transData.transform(point).astype(int)
        np.testing.assert_allclose(
            rgba[rgba.shape[0] - 1 - y, x, :3] / 255.0, to_rgba(colour)[:3], atol=0.01
        )
    plt.close(fig)


def test_plot_labels_assemblages_and_solution_multiplicity():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = diagram([line(square(), [0, 1, 2], [3, 4, 5])])
    data["phase_names"] = ["g", "pl", "q", "hb", "hb #2", "q"]
    fig, ax = bm.plot_pseudosection(
        data, pressure_unit="Pa", temperature_unit="K", phase_aliases={"g": "gt"}
    )
    assert {text.get_text() for text in ax.texts} == {"gt pl q", "2hb q"}
    fig.canvas.draw()
    plt.close(fig)
    fig, ax = bm.plot_pseudosection(
        data, label_assemblages=False, pressure_unit="Pa", temperature_unit="K"
    )
    assert not ax.texts
    plt.close(fig)


@pytest.mark.parametrize("merge_fields", [True, False])
def test_plot_labels_each_explicit_closed_loop(merge_fields):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    data = diagram(
        [
            line([(0.0, 0.5), (1.0, 0.5)], [0, 1], [0, 1]),
            line([(0.5, 0.0), (0.5, 1.0)], [0, 1], [0, 1]),
        ]
    )
    data["nodes"] = [dict(id=0, pressure=0.5, temperature=0.5)]
    data["phase_names"] = ["gt", "q"]
    fig, ax = bm.plot_pseudosection(
        data,
        pressure_unit="Pa",
        temperature_unit="K",
        merge_fields=merge_fields,
        show_unresolved=False,
        label_key_path=None,
        label_fontsize=10.0,
    )
    assert len(ax.pseudosection_geometry.polygons) == 4
    assert len(ax.texts) == 4
    assert {t.get_text() for t in ax.texts} == {"gt q"}
    assert {t.get_fontsize() for t in ax.texts} == {10.0}
    outlines = next(c for c in ax.collections if isinstance(c, LineCollection))
    internal = [
        s
        for s in outlines.get_segments()
        if not np.any(np.all(s == 0.0, axis=0) | np.all(s == 1.0, axis=0))
    ]
    assert internal
    plt.close(fig)


def test_label_rectangles_respect_holes_and_concave_edges():
    result = bm.pseudosection_field_polygons(diagram([line(square(), [0, 1], [0])]))
    outer = next(p for p in result.polygons if p.n_phases == 1)
    assert outer.contains_rectangle([0.1, 0.5], [0.04, 0.3])
    # Every corner is in the outer field, but the rectangle encloses its hole.
    assert not outer.contains_rectangle([0.5, 0.5], [0.45, 0.45])
    inner = next(p for p in result.polygons if p.n_phases == 2)
    assert inner.contains_rectangle([0.5, 0.5], [0.1, 0.2])
    assert not inner.contains_rectangle([0.5, 0.5], [0.4, 0.1])
    points = [
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.6, 1.0),
        (0.6, 0.3),
        (0.4, 0.3),
        (0.4, 1.0),
        (0.0, 1.0),
        (0.0, 0.0),
    ]
    concave = bm.pseudosection_field_polygons(
        diagram([line(points, [0], [1])]), close_domain=False
    ).polygons[0]
    # All corners fit, while the top edge crosses the indentation.
    assert not concave.contains_rectangle([0.5, 0.5], [0.4, 0.4])
    with pytest.raises(ValueError, match="half-sizes"):
        inner.contains_rectangle([0.5, 0.5], [0.0, 1.0])


def test_uniform_font_and_numbered_assemblage_document(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    narrow = [(0.499, 0.1), (0.501, 0.1), (0.501, 0.9), (0.499, 0.9), (0.499, 0.1)]
    data = diagram([line(narrow, list(range(12)), [12])])
    data["phase_names"] = [f"phase{i}" for i in range(12)] + ["background"]
    data["phase_names"][0] = "phase|0"
    key_path = tmp_path / "field_names.md"
    fig, ax = bm.plot_pseudosection(
        data,
        pressure_unit="Pa",
        temperature_unit="K",
        label_fontsize=11.0,
        label_key_path=key_path,
    )
    assert {text.get_fontsize() for text in ax.texts} == {11.0}
    assert ax.pseudosection_label_key
    assert " ".join(data["phase_names"][:12]) in ax.pseudosection_label_key.values()
    assert all(
        str(number) in {text.get_text() for text in ax.texts}
        for number in ax.pseudosection_label_key
    )
    assert key_path.is_file()
    key = key_path.read_text()
    for number, name in ax.pseudosection_label_key.items():
        escaped_name = name.replace("|", r"\|")
        assert f"| {number} | {escaped_name} |" in key
    # The number does not fit in the narrow region either: retain the chosen
    # size and show its connection to the correct field.
    assert any(getattr(text, "arrow_patch", None) is not None for text in ax.texts)
    fig.canvas.draw()
    boxes = [text.get_window_extent() for text in ax.texts]
    assert all(not a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i + 1 :])
    plt.close(fig)
    suppressed = tmp_path / "suppressed.md"
    fig, ax = bm.plot_pseudosection(
        data,
        label_assemblages=False,
        label_key_path=suppressed,
        pressure_unit="Pa",
        temperature_unit="K",
    )
    assert not ax.texts and not suppressed.exists()
    plt.close(fig)
    with pytest.raises(ValueError, match="label_fontsize"):
        bm.plot_pseudosection(data, label_fontsize=0.0)


def test_eos_domain_mask_is_subdivided_and_never_gets_an_assemblage_colour():
    data = diagram([], [sample(0.8, 0.5, [0, 1])])
    data["excluded_regions"] = [[(0.0, 0.4), (0.2, 1.0), (0.0, 1.0), (0.0, 0.4)]]
    result = bm.pseudosection_field_polygons(data)
    assert not result.diagnostics
    assert len(result.polygons) == 2
    masked = next(p for p in result.polygons if p.outside_model_domain)
    field = next(p for p in result.polygons if not p.outside_model_domain)
    assert masked.n_phases == 0 and not masked.phases
    assert masked.area == pytest.approx(0.06)
    assert field.n_phases == 2 and field.area == pytest.approx(0.94)
    assert sum(p.area for p in result.polygons) == pytest.approx(1.0)


def test_frame_roundoff_does_not_create_zero_kelvin_slivers():
    data = diagram(
        [line([(0.0, 2.0e-15), (0.4, 2.0e-15), (1.0, 0.0)], [0, 1], [0])],
        [sample(0.5, 0.5, [0, 1])],
    )
    geometry = bm.pseudosection_field_polygons(data)
    assert len(geometry.polygons) == 1
    assert geometry.polygons[0].area == pytest.approx(1.0)
    assert geometry.polygons[0].n_phases == 2


def test_mixed_perimeter_uses_saved_interior_evidence_and_keeps_internal_edges():
    points = square()
    data = diagram(
        [line(points[:3], [0, 1], [3]), line(points[2:], [0, 2], [3])],
        [sample(0.5, 0.5, [0, 2])],
    )
    geometry = bm.pseudosection_field_polygons(data)
    inner = next(p for p in geometry.polygons if p.area == pytest.approx(0.36))
    assert inner.phases == [0, 2]
    assert inner.is_provisional and not inner.has_open_boundary
    assert "conflicting perimeter assemblages" in geometry.diagnostics[0]
    assert len(geometry.boundary_segments) == 8


def test_conflicting_perimeter_is_not_decided_by_length_or_point_count():
    # Neither a longer arc nor more stored points establishes the assemblage.
    short = [(0.2, y) for y in np.linspace(0.8, 0.2, 101)]
    data = diagram([
        line(square()[:4], [0, 1], [3]),
        line(short, [0, 2], [3]),
    ])
    geometry = bm.pseudosection_field_polygons(data)
    inner = next(p for p in geometry.polygons if p.area == pytest.approx(0.36))
    assert not inner.phases and inner.n_phases == 0
    assert inner.is_provisional


@pytest.mark.parametrize("case", ["at1", "mix1g"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("native", [False, True])
def test_loop_constraints_recover_coexisting_solution_instances(case, reverse, native):
    # Actual loops whose finite neighbour probes supplied incorrect labels.
    # Independent equilibria confirmed two spinels / two clinopyroxenes.
    fixture = json.loads(
        (Path(__file__).parent / "data/green_provisional_loops.json").read_text()
    )[case]
    data = fixture["data"]
    if reverse:
        for boundary in data["boundaries"]:
            boundary["points"].reverse()
            boundary["side_a"], boundary["side_b"] = boundary["side_b"], boundary["side_a"]
            boundary["start_node"], boundary["end_node"] = boundary["end_node"], boundary["start_node"]
    geometry = bm.pseudosection_field_polygons(
        bm.PseudosectionResult.from_dict(data) if native else data)
    matches = [p for p in geometry.polygons if p.contains_rectangle(
        fixture["position"], [1.0e-4, 1.0e-8])]
    assert len(matches) == 1
    field = matches[0]
    assert field.phases == fixture["expected"]
    assert field.n_phases == len(fixture["expected"])
    assert not field.is_provisional and not field.has_open_boundary


def test_conflicting_hole_does_not_override_agreed_exterior():
    data = diagram([
        line(square(), [0, 1], [0]),
        line(square(0.4, 0.6), [2], [0, 2]),
    ], [sample(0.3, 0.3, [0, 1])])
    geometry = bm.pseudosection_field_polygons(data, merge_fields=False)
    surrounding = next(p for p in geometry.polygons if p.holes and p.area < 0.5)
    assert surrounding.phases == [0, 1]
    assert surrounding.is_provisional
    assert len(surrounding.holes) == 1


def test_no_interior_or_perimeter_label_uses_nearest_saved_sample_provisionally():
    data = diagram(
        [line([(0.5, 0.0), (0.5, 1.0)], [], [])],
        [sample(0.2, 0.5, [0]), sample(0.3, 0.5, [1, 2])],
    )
    geometry = bm.pseudosection_field_polygons(data)
    right = next(p for p in geometry.polygons if p.label_position[0] > 0.5)
    assert right.phases == [1, 2]
    assert right.sample_index == -1
    assert right.is_provisional
    assert len(geometry.polygons) == 2
    assert all(p.is_provisional for p in geometry.polygons)


def test_no_assemblage_evidence_stays_unknown():
    geometry = bm.pseudosection_field_polygons(diagram([]))
    assert len(geometry.polygons) == 1
    assert geometry.polygons[0].n_phases == 0
    assert not geometry.polygons[0].phases
    assert geometry.polygons[0].is_provisional
    assert "no assemblage evidence" in geometry.diagnostics[0]
