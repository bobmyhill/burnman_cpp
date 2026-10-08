"""Reusable contour batches, phase selection and display options."""

import json

import numpy as np
import pytest

import burnman_cpp as bm
from conftest import oxide_params


@pytest.fixture
def section():
    phase = bm.Mineral(oxide_params())
    settings = bm.PseudosectionSettings()
    settings.pressure_seeds = settings.temperature_seeds = 2
    settings.max_phase_instances = 1
    result = bm.pseudosection(
        {"Mg": 1.0, "O": 1.0}, [phase], (3.0e9, 5.0e9), (800.0, 1200.0), settings
    )
    return result, phase


def test_contour_batches_preserve_levels_native_targets_and_source(section, capsys):
    from burnman_cpp.tools import pseudosection_contour_levels

    result, phase = section
    saved = result.to_dict()
    before = json.dumps(saved, sort_keys=True)
    levels = [4.5e9, 3.5e9]
    records = pseudosection_contour_levels(
        saved, iter([phase]), iter(levels), bm.PressureConstraint, verbose=True
    )
    assert [r["value"] for r in records] == levels
    assert json.loads(json.dumps(records, allow_nan=False)) == records
    assert json.dumps(saved, sort_keys=True) == before
    for record in records:
        assert record["contours"]["resolved"]
        assert record["contours"]["equilibrium_solves"] > 0
        for line in record["contours"]["lines"]:
            for point in line["points"]:
                assert point["pressure"] == pytest.approx(record["value"], abs=1.0)
    assert capsys.readouterr().out.count("native solves") == len(levels)


def test_batches_accept_field_factories_and_retain_failed_level_diagnostics(section):
    result, phase = section
    settings = bm.PseudosectionContourSettings()
    settings.max_trace_steps = 1
    saved = result.to_dict()
    del saved["composition_start"]
    records = bm.pseudosection_contour_levels(
        saved,
        [phase],
        [1000.0],
        lambda value: lambda *_: bm.TemperatureConstraint(value),
        settings=settings,
        composition=result.composition_start,
    )
    assert not records[0]["contours"]["resolved"]
    assert records[0]["contours"]["diagnostics"]


def test_batch_rejects_nonfinite_levels_before_starting_solves(section):
    result, phase = section
    calls = []
    with pytest.raises(ValueError, match="finite"):
        bm.pseudosection_contour_levels(
            result, [phase], [4.0e9, np.nan], lambda value: calls.append(value)
        )
    assert not calls


@pytest.mark.parametrize("phase_name,instance", [("g", 2), ("g #2", None)])
def test_phase_factory_selects_saved_instance_in_the_field_layout(
    two_phase_assemblage, phase_name, instance
):
    a = two_phase_assemblage
    bulk = {"Mg": 0.5, "Fe": 0.5, "O": 1.0}
    saved = dict(phase_names=["g", "g #2", "matrix"])
    parameters = bm.get_equilibration_parameters(a, bulk)
    sites = a.get_phase(0).site_names
    ambiguous = bm.PhaseCompositionConstraint.for_diagram(
        saved, "g", sites, [1, 0], [1, 1], 0.7
    )
    with pytest.raises(ValueError, match="select an instance"):
        ambiguous(a, parameters, [1, 0])
    factory = bm.PhaseCompositionConstraint.for_diagram(
        saved, phase_name, sites, [1, 0], [1, 1], 0.7, instance=instance
    )
    # Saved branch #2 is first in this assemblage; IDs are not phase indices.
    constraint = factory(a, parameters, [1, 0])
    assert isinstance(constraint, bm.PhaseCompositionConstraint)
    result = bm.equilibrate(
        bulk, a, [bm.PressureConstraint(2.0e9), constraint], tol=1.0e-9
    )
    assert result.sol_array.item().success, result.sol_array.item().message
    expected_t = 10000.0 / (bm.constants.gas_constant * np.log(0.7 / 0.3))
    assert a.temperature == pytest.approx(expected_t, abs=1.0e-5)
    np.testing.assert_allclose(a.get_phase(0).molar_fractions, [0.7, 0.3], atol=1.0e-9)
    np.testing.assert_allclose(a.get_phase(1).molar_fractions, [0.3, 0.7], atol=1.0e-9)
    assert factory(a, parameters, [2, 0]) is None


def test_phase_factory_skips_absent_phases_and_active_pure_faces(section):
    _, phase = section
    a = bm.Assemblage([phase], [1.0])
    parameters = bm.get_equilibration_parameters(a, {"Mg": 1.0, "O": 1.0})
    factory = bm.PhaseCompositionConstraint.for_diagram(
        dict(phase_names=["g", "matrix"]), "g", ["Mg_A", "Fe_A"], [1, 0], [1, 1], 0.5
    )
    assert factory(a, parameters, [0]) is None
    assert factory(a, parameters, [1]) is None


@pytest.mark.parametrize(
    "variable,value,expected",
    [("P", 4.0e9, "40"), ("T", 1000, "726.85"), (None, 0.7, "0.7")],
)
def test_contour_labels_inherit_units_and_allow_explicit_display_options(
    section, variable, value, expected
):
    import matplotlib.pyplot as plt

    result, phase = section
    contours = bm.pseudosection_contours(
        result, [phase], bm.TemperatureConstraint(1000)
    )
    fig, ax = bm.plot_pseudosection(result, label_key_path=None)
    bm.plot_pseudosection_contours(
        result,
        contours,
        ax,
        variable=variable,
        value=value,
        label_fontsize=8.0,
        label_placement="single",
        pressure_unit="kbar",
    )
    assert len([t for t in ax.texts if t.get_text() == expected]) == 1
    assert ax.collections[-1].get_label() == expected
    plt.close(fig)


@pytest.mark.parametrize(
    "placement,expected_y", [("single", 0.2), ("coexistence", 0.8)]
)
def test_label_placement_can_prefer_coexistence_over_the_longest_segment(
    placement, expected_y
):
    import matplotlib.pyplot as plt

    saved = dict(diagram_type="PT", pressure_range=(0, 1), temperature_range=(0, 1))
    contours = dict(
        diagram_type="PT",
        lines=[
            dict(
                phases=[0],
                points=[
                    dict(temperature=0.05, pressure=0.2),
                    dict(temperature=0.95, pressure=0.2),
                ],
            ),
            dict(
                phases=[0, 1, 2],
                points=[
                    dict(temperature=0.2, pressure=0.8),
                    dict(temperature=0.8, pressure=0.8),
                ],
            ),
        ],
    )
    fig, ax = plt.subplots()
    ax.set(xlim=(0, 1), ylim=(0, 1))
    bm.plot_pseudosection_contours(
        saved,
        contours,
        ax,
        value=0.5,
        label_format=".2f",
        label_placement=placement,
        pressure_unit="Pa",
        temperature_unit="K",
    )
    assert len(ax.texts) == 1
    assert ax.texts[0].get_text() == "0.50"
    assert ax.texts[0].get_position()[1] == pytest.approx(expected_y)
    plt.close(fig)
