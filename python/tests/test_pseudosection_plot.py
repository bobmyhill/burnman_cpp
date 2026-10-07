"""Analytic polygon geometry and phase counts, independent of thermodynamics."""

import json
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
    data = diagram([line(triangle, [0], [0, 1])], [interior])
    ordinary = bm.pseudosection_field_polygons(data, merge_fields=False)
    assert {p.n_phases for p in ordinary.polygons} == {1, 2}
    interior["is_field_verification"] = True
    verified = bm.pseudosection_field_polygons(data, merge_fields=False)
    assert not verified.diagnostics
    assert {p.n_phases for p in verified.polygons} == {1, 3}
    enclosed = next(p for p in verified.polygons if p.n_phases == 3)
    assert enclosed.phases == [0, 1, 2]
    assert enclosed.sample_index == 0


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


def test_crossings_and_duplicate_lines_do_not_make_false_faces():
    lines = [
        line([(0.0, 0.5), (1.0, 0.5)], [0, 1], [0, 1]),
        line([(0.5, 0.0), (0.5, 1.0)], [0, 1], [0, 1]),
    ]
    data = diagram(lines + lines)
    raw = bm.pseudosection_field_polygons(data, merge_fields=False)
    assert len(raw.polygons) == 4
    assert all(p.n_phases == 2 and p.area == pytest.approx(0.25) for p in raw.polygons)
    result = bm.pseudosection_field_polygons(data)
    assert not result.diagnostics
    assert len(result.polygons) == 1
    assert result.polygons[0].n_phases == 2
    assert result.polygons[0].area == pytest.approx(1.0)
    assert result.polygons[0].source_regions == [0, 1, 2, 3]
    assert len(result.boundary_segments) < len(raw.boundary_segments)
    assert all(
        np.any(np.all(s == 0.0, axis=0) | np.all(s == 1.0, axis=0))
        for s in result.boundary_segments
    )


def test_merge_adjacent_regions_preserves_holes_and_different_assemblages():
    # A redundant divider splits the annulus around a genuine phase field.
    data = diagram(
        [
            line(square(), [0, 1], [0]),
            line([(0.5, 0.0), (0.5, 0.2)], [0], [0]),
            line([(0.5, 0.8), (0.5, 1.0)], [0], [0]),
        ]
    )
    raw = bm.pseudosection_field_polygons(data, merge_fields=False)
    assert len(raw.polygons) == 3
    merged = bm.pseudosection_field_polygons(data)
    assert not merged.diagnostics
    assert len(merged.polygons) == 2
    outer = next(p for p in merged.polygons if p.phases == [0])
    assert outer.area == pytest.approx(0.64)
    assert len(outer.holes) == 1 and len(outer.source_regions) == 2
    assert not outer.contains_rectangle([0.5, 0.5], [0.45, 0.45])
    assert sum(p.area for p in merged.polygons) == pytest.approx(
        sum(p.area for p in raw.polygons)
    )
    # Equal counts must not erase phase-in/phase-out boundaries.
    swapped = bm.pseudosection_field_polygons(
        diagram([line([(0.5, 0.0), (0.5, 1.0)], [0, 1], [0, 2])])
    )
    assert len(swapped.polygons) == 2
    assert {tuple(p.phases) for p in swapped.polygons} == {(0, 1), (0, 2)}


def test_merge_nested_identical_fields_removes_redundant_hole():
    data = diagram([line(square(), [0], [0]), line(square(0.4, 0.6), [0, 1], [0])])
    raw = bm.pseudosection_field_polygons(data, merge_fields=False)
    merged = bm.pseudosection_field_polygons(data)
    assert len(raw.polygons) == 3 and len(merged.polygons) == 2
    outer = next(p for p in merged.polygons if p.phases == [0])
    assert outer.area == pytest.approx(0.96)
    assert len(outer.holes) == 1 and len(outer.source_regions) == 2
    assert outer.contains_rectangle([0.3, 0.5], [0.03, 0.3])
    assert not outer.contains_rectangle([0.5, 0.5], [0.4, 0.4])


def test_same_assemblage_regions_touching_only_at_a_point_remain_separate():
    first = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5), (0.0, 0.0)]
    second = [(0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (0.5, 1.0), (0.5, 0.5)]
    merged = bm.pseudosection_field_polygons(
        diagram([line(first, [0, 1], [0]), line(second, [0, 1], [0])])
    )
    assert sorted(p.n_phases for p in merged.polygons) == [1, 1, 2, 2]
    assert all(len(p.source_regions) == 1 for p in merged.polygons)


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


def test_grid_unions_match_connected_assemblages_and_preserve_each_area():
    # Independent cell adjacency gives the expected connected regions,
    # including concave unions, enclosed islands and repeated assemblages.
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
    cells = {
        i: tuple(np.floor(p.label_position * n).astype(int))
        for i, p in enumerate(raw.polygons)
    }
    remaining = set(cells.values())
    expected = set()
    while remaining:
        start = min(remaining)
        region, queue = {start}, [start]
        remaining.remove(start)
        while queue:
            i, j = queue.pop()
            for neighbour in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                if neighbour in remaining and grid[neighbour] == grid[start]:
                    remaining.remove(neighbour)
                    region.add(neighbour)
                    queue.append(neighbour)
        expected.add((int(grid[start]), frozenset(region)))
    actual = set()
    for p in merged.polygons:
        region = frozenset(cells[i] for i in p.source_regions)
        actual.add((p.phases[0], region))
        assert p.area == pytest.approx(len(region) / (n * n))
    assert actual == expected
    assert sum(len(p.holes) for p in merged.polygons) >= 1


def test_open_boundary_is_not_extended_to_close_a_field():
    data = diagram(
        [line([(0.5, 0.0), (0.5, 0.7)], [0], [0, 1])], [sample(0.1, 0.1, [0])]
    )
    result = bm.pseudosection_field_polygons(data, close_domain=False)
    assert not result.polygons
    result = bm.pseudosection_field_polygons(data)
    assert len(result.polygons) == 1 and result.polygons[0].n_phases == 0
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


def test_snapping_intersections_preserves_closed_thin_fields():
    curves = json.loads(
        (Path(__file__).parent / "data" / "pyrolite_narrow_curves.json").read_text()
    )
    geometry = bm.pseudosection_field_polygons(
        diagram([line(points, [0], [0]) for points in curves]), merge_fields=False
    )
    assert not geometry.diagnostics
    assert all(p.n_phases == 1 and not p.has_open_boundary for p in geometry.polygons)
    assert min(p.area for p in geometry.polygons) < 1.0e-9
    assert sum(p.area for p in geometry.polygons) == pytest.approx(1.0)


@pytest.mark.parametrize("tip,filled_count", [((0.1, 0.5), 2), ((0.3, 0.5), 1)])
def test_unfinished_branch_only_invalidates_the_field_containing_it(tip, filled_count):
    data = diagram([line(square(), [0, 1], [0]), line([(0.2, 0.5), tip], [0], [0, 1])])
    result = bm.pseudosection_field_polygons(data)
    assert sorted(p.n_phases for p in result.polygons) == [0, filled_count]
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


def test_known_count_does_not_invent_an_ambiguous_assemblage_label():
    points = square()
    data = diagram([line(points[:3], [0, 1], [3]), line(points[2:], [0, 2], [3])])
    result = bm.pseudosection_field_polygons(data)
    inner = next(p for p in result.polygons if p.n_phases == 2)
    assert not inner.phases
    assert len(result.diagnostics) == 1
    assert "known phase count" in result.diagnostics[0]


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


@pytest.mark.parametrize("merge_fields,field_count", [(True, 1), (False, 4)])
def test_plot_merged_fields_uses_one_label_and_omits_internal_lines(
    merge_fields, field_count
):
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
    assert len(ax.pseudosection_geometry.polygons) == field_count
    assert len(ax.texts) == field_count
    assert {t.get_text() for t in ax.texts} == {"gt q"}
    assert {t.get_fontsize() for t in ax.texts} == {10.0}
    outlines = next(c for c in ax.collections if isinstance(c, LineCollection))
    internal = [
        s
        for s in outlines.get_segments()
        if not np.any(np.all(s == 0.0, axis=0) | np.all(s == 1.0, axis=0))
    ]
    assert bool(internal) == (not merge_fields)
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
