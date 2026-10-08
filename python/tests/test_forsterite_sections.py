"""Independent thermodynamic sections preserve Mg2SiO4 coexistence fields."""

from pathlib import Path
import runpy

import numpy as np
import pytest

import burnman_cpp as bm


@pytest.fixture(scope="module")
def example():
    return runpy.run_path(
        str(
            Path(__file__).resolve().parents[2]
            / "examples"
            / "example_forsterite_thermodynamic_sections.py"
        )
    )


@pytest.fixture(scope="module")
def sections(example):
    return example["calculate"](quick=True)


@pytest.fixture(scope="module")
def contours(example, sections):
    saved = {name: section.to_dict() for name, section in sections.items()}
    result = example["calculate_contours"](saved)
    assert saved == {name: section.to_dict() for name, section in sections.items()}
    return result


@pytest.mark.parametrize("diagram", ["PT", "PS", "TV", "SV"])
def test_forsterite_contours_satisfy_targets_and_additive_properties(
    example, contours, diagram
):
    minerals = example["candidate_phases"]()
    for name, records in contours[diagram].items():
        for record in records:
            result = record["contours"]
            assert result["resolved"], result["diagnostics"]
            assert not result["diagnostics"]
            for line in result["lines"]:
                for point in line["points"]:
                    totals = dict(S=0.0, V=0.0)
                    for phase in point["phases"]:
                        mineral = minerals[phase["candidate_index"]]
                        mineral.set_state(point["pressure"], point["temperature"])
                        totals["S"] += phase["amount"] * mineral.molar_entropy
                        totals["V"] += phase["amount"] * mineral.molar_volume
                    assert point["mass_balance_error"] < 1.0e-9
                    assert point["entropy"] == pytest.approx(totals["S"], abs=1.0e-7)
                    assert point["volume"] == pytest.approx(totals["V"], abs=1.0e-15)
                    value = dict(P=point["pressure"], T=point["temperature"], **totals)[
                        name
                    ]
                    tolerance = dict(P=1.0, T=1.0e-6, S=1.0e-6, V=1.0e-15)[name]
                    assert value == pytest.approx(
                        record["value"], abs=tolerance, rel=0.0
                    )
        assert any(record["contours"]["lines"] for record in records)
    if diagram != "PT":
        assert any(
            len(line["phases"]) == 3
            for records in contours[diagram].values()
            for record in records
            for line in record["contours"]["lines"]
        )


@pytest.mark.parametrize("normalized", [False, True])
def test_forsterite_contour_labels_use_display_units_and_one_fontsize(
    sections, contours, example, tmp_path, normalized
):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    fig, axes = example["plot"](
        sections,
        tmp_path,
        label_fontsize=6.0,
        contours=contours,
        volume_unit="kg/m3" if normalized else "cm3",
        entropy_unit="kB/atom" if normalized else "J/K",
    )
    expected_s = [
        f"{record['value'] / (example['ENTROPY_SCALE'] if normalized else 1.0):.3g}"
        for record in contours["PT"]["S"]
    ]
    expected_v = (
        "4200" if normalized else f"{example['BULK_MASS'] / 4200.0 * 1.0e6:.3g}"
    )
    for ax in axes.flat:
        assert all(text.get_fontsize() == 6.0 for text in ax.texts)
        assert all(text.get_fontsize() == 6.0 for text in ax.get_legend().get_texts())
    entropy_labels = [
        t.get_text()
        for t in axes[0, 0].texts
        if to_rgba(t.get_color()) == to_rgba("red")
    ]
    assert entropy_labels and set(entropy_labels) <= set(expected_s)
    assert expected_v in [t.get_text() for t in axes[0, 0].texts]
    assert "23.4" in [t.get_text() for t in axes[1, 1].texts]
    assert "1500" in [t.get_text() for t in axes[0, 1].texts]
    plt.close(fig)


@pytest.mark.parametrize("diagram,count", [("PT", 3), ("PS", 6), ("TV", 6), ("SV", 7)])
def test_forsterite_sections_close_all_fields(sections, diagram, count):
    result = sections[diagram]
    assert result.diagram_type == diagram
    assert result.resolved, result.diagnostics
    assert all(s.success for s in result.samples)
    assert all(
        line.start_node >= 0 and line.end_node >= 0 for line in result.boundaries
    )
    geometry = bm.pseudosection_field_polygons(result)
    assert not geometry.diagnostics
    assert len(geometry.polygons) == count
    assert all(p.n_phases > 0 and not p.has_open_boundary for p in geometry.polygons)
    assert sum(p.area for p in geometry.polygons) == pytest.approx(1.0, abs=1.0e-8)


def test_four_phase_sv_field_is_the_latent_entropy_volume_triangle(sections, example):
    result = sections["SV"]
    polygon = next(
        p for p in bm.pseudosection_field_polygons(result).polygons if p.n_phases == 4
    )
    invariant = next(n for n in sections["PT"].nodes if n.kind == "junction")
    minerals = example["candidate_phases"]()
    for mineral in minerals:
        mineral.set_state(invariant.pressure, invariant.temperature)
    per, bdg, aki, rw = minerals
    vertices = np.array(
        [
            [rw.molar_entropy, rw.molar_volume],
            [
                per.molar_entropy + aki.molar_entropy,
                per.molar_volume + aki.molar_volume,
            ],
            [
                per.molar_entropy + bdg.molar_entropy,
                per.molar_volume + bdg.molar_volume,
            ],
        ]
    )
    ranges = np.array(result.coordinate_ranges)
    normalized = vertices / (ranges[:, 1] - ranges[:, 0])
    a, b = normalized[1:] - normalized[0]
    expected_area = abs(a[0] * b[1] - a[1] * b[0]) * 0.5
    assert polygon.area == pytest.approx(expected_area, rel=1.0e-6)
    verified = [
        s for s in result.samples if s.is_field_verification and len(s.phases) == 4
    ]
    assert verified
    for sample in verified:
        assert sample.pressure == pytest.approx(invariant.pressure, abs=0.3)
        assert sample.temperature == pytest.approx(invariant.temperature, abs=1.0e-6)
        assert {p.id for p in sample.phases} == {0, 1, 2, 3}


@pytest.mark.parametrize("normalized", [False, True])
def test_plot_units_are_optional_and_preserve_native_sv_geometry(
    sections, example, normalized
):
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection

    result = sections["SV"]
    before = result.to_dict()
    options = dict(entropy_unit="kB/atom", volume_unit="kg/m3") if normalized else {}
    fig, ax = bm.plot_pseudosection(
        result,
        swap_axes=True,
        colorbar=False,
        label_fontsize=6.0,
        label_key_path=None,
        **options,
    )
    entropy_scale = example["ENTROPY_SCALE"] if normalized else 1.0
    expected_y = (3600.0, 4500.0) if normalized else result.volume_range
    assert ax.get_xlim() == pytest.approx(
        np.array(result.entropy_range) / entropy_scale
    )
    assert ax.get_ylim() == pytest.approx(expected_y)
    assert ax.get_xlabel().startswith("Entropy")
    assert ax.get_ylabel() == ("Density (kg/m³)" if normalized else "Volume (m³)")
    collection = next(c for c in ax.collections if isinstance(c, PatchCollection))
    polygon = ax.pseudosection_geometry.polygons[0]
    native = np.array(polygon.vertices)
    expected = np.column_stack(
        [
            native[:, 0] / entropy_scale,
            example["BULK_MASS"] / native[:, 1] if normalized else native[:, 1],
        ]
    )
    np.testing.assert_allclose(collection.get_paths()[0].vertices, expected)
    assert all(text.get_fontsize() == 6.0 for text in ax.texts)
    assert result.to_dict() == before
    plt.close(fig)
