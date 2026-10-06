"""Native constraint continuation on saved fields; no phase diagram rebuild."""

import json
from pathlib import Path
import runpy
import numpy as np
import pytest
import burnman_cpp as bm
from conftest import oxide_params, make_model
from test_pseudosection import crossing_phases


@pytest.fixture
def simple_diagram():
    phase = bm.Mineral(oxide_params())
    options = bm.PseudosectionSettings()
    options.pressure_seeds = options.temperature_seeds = 2
    options.max_phase_instances = 1
    result = bm.pseudosection(
        {"Mg": 1.0, "O": 1.0},
        [phase],
        (3.0e9, 5.0e9),
        (800.0, 1200.0),
        options,
    )
    return result, phase


@pytest.mark.parametrize(
    "kind", ["pressure", "temperature", "volume", "entropy", "linear"]
)
def test_scalar_constraints_trace_saved_json_and_retain_source(simple_diagram, kind):
    result, phase = simple_diagram
    phase.set_state(4.0e9, 1000.0)
    constraints = dict(
        pressure=bm.PressureConstraint(4.0e9),
        temperature=bm.TemperatureConstraint(1000.0),
        volume=bm.VolumeConstraint(phase.molar_volume),
        entropy=bm.EntropyConstraint(phase.molar_entropy),
        linear=bm.LinearXConstraint([1.0, 2.0e6, 0.0], 6.0e9),
    )
    before = result.to_dict()
    saved = json.loads(json.dumps(before))
    contours = bm.pseudosection_contours(saved, [phase], constraints[kind])
    assert result.to_dict() == before
    assert saved == before
    assert contours.resolved, contours.diagnostics
    assert len(contours.lines) == 1
    line = contours.lines[0]
    assert len(line.points) > 10
    assert line.termination == "field boundary; field boundary"
    for p in line.points:
        assert p.mass_balance_error < 1.0e-9
        phase.set_state(p.pressure, p.temperature)
        if kind == "pressure":
            assert p.pressure == pytest.approx(4.0e9, abs=0.3)
        elif kind == "temperature":
            assert p.temperature == pytest.approx(1000.0, abs=1.0e-6)
        elif kind == "volume":
            assert phase.molar_volume == pytest.approx(
                line.points[0].volume, abs=1.0e-15
            )
        elif kind == "entropy":
            assert phase.molar_entropy == pytest.approx(
                line.points[0].entropy, abs=1.0e-7
            )
        else:
            assert p.pressure + 2.0e6 * p.temperature == pytest.approx(6.0e9, abs=3.0)
    serialized = json.loads(json.dumps(contours.to_dict(), allow_nan=False))
    assert serialized["lines"][0]["points"][0]["phases"][0]["candidate_index"] == 0


def test_ellipse_finds_and_closes_an_interior_loop(simple_diagram):
    result, phase = simple_diagram
    contour = bm.pseudosection_contours(
        result,
        [phase],
        bm.PTEllipseConstraint([4.0e9, 1000.0], [5.0e8, 100.0]),
    )
    assert contour.resolved, contour.diagnostics
    assert len(contour.lines) == 1
    line = contour.lines[0]
    assert line.closed
    np.testing.assert_allclose(
        [line.points[0].pressure, line.points[0].temperature],
        [line.points[-1].pressure, line.points[-1].temperature],
    )
    for p in line.points:
        assert np.hypot(
            (p.pressure - 4.0e9) / 5.0e8, (p.temperature - 1000.0) / 100.0
        ) == pytest.approx(1.0, abs=1.0e-8)


def test_contours_stop_at_saved_phase_fields_and_factories_can_skip():
    options = bm.PseudosectionSettings()
    options.pressure_seeds = options.temperature_seeds = 3
    options.max_phase_instances = 1
    phases = crossing_phases()
    result = bm.pseudosection(
        {"Mg": 1.0, "Fe": 1.0, "O": 2.0},
        phases,
        (0.0, 2.0e9),
        (600.0, 1400.0),
        options,
    )
    calls = []

    def factory(a, parameters, ids):
        calls.append(ids)
        assert parameters.n_parameters == bm.get_parameter_vector(a).size
        return bm.PressureConstraint(0.5e9) if 2 in ids else None

    contour = bm.pseudosection_contours(result, phases, factory)
    assert contour.resolved, contour.diagnostics
    assert len(calls) == 4
    assert len(contour.lines) == 1
    temperatures = [p.temperature for p in contour.lines[0].points]
    assert min(temperatures) == pytest.approx(600.0, abs=1.0e-5)
    assert max(temperatures) == pytest.approx(1000.0, abs=1.0e-5)
    assert contour.lines[0].phases == [0, 2]


def test_constraint_and_source_validation(simple_diagram):
    result, phase = simple_diagram
    with pytest.raises(TypeError, match="constraint must"):
        bm.pseudosection_contours(result, [phase], 4.0)
    with pytest.raises(TypeError, match="factory must return"):
        bm.pseudosection_contours(result, [phase], lambda *_: 4.0)
    with pytest.raises(ValueError, match="candidate names"):
        bm.pseudosection_contours(result, [], bm.PressureConstraint(4.0e9))
    with pytest.raises(ValueError, match="match the saved bulk"):
        bm.pseudosection_contours(
            result,
            [phase],
            bm.PressureConstraint(4.0e9),
            composition={"Mg": 2.0, "O": 2.0},
        )
    legacy = result.to_dict()
    del legacy["composition_start"]
    with pytest.raises(ValueError, match="composition_start"):
        bm.pseudosection_contours(legacy, [phase], bm.PressureConstraint(4.0e9))
    assert bm.pseudosection_contours(
        legacy,
        [phase],
        bm.PressureConstraint(4.0e9),
        composition=result.composition_start,
    ).resolved


def test_phase_composition_constraint_uses_the_field_parameter_layout():
    # Two ideal solutions partition analytically, Mg/(Mg+Fe) in the Mg-rich
    # phase = 1/(1+exp(-10000/RT)), independently of P.
    phases = [
        bm.Solution(make_model(shifts=(0.0, 10000.0)), [0.8, 0.2], name="g"),
        bm.Solution(make_model(shifts=(10000.0, 0.0)), [0.2, 0.8], name="matrix"),
    ]
    options = bm.PseudosectionSettings()
    options.max_phase_instances = 1
    options.pressure_seeds = options.temperature_seeds = 2
    diagram = bm.pseudosection(
        {"Mg": 0.5, "Fe": 0.5, "O": 1.0},
        phases,
        (1.0e9, 2.0e9),
        (1000.0, 2000.0),
        options,
    )
    target = 1.0 / (1.0 + np.exp(-10000.0 / (8.31446261815324 * 1500.0)))

    def factory(a, parameters, ids):
        i = ids.index(0)
        return bm.PhaseCompositionConstraint(
            i,
            a.get_phase(i).site_names,
            [1.0, 0.0],
            [1.0, 1.0],
            target,
            a,
            parameters,
        )

    contours = bm.pseudosection_contours(diagram, phases, factory)
    assert contours.resolved, contours.diagnostics
    assert len(contours.lines) == 1
    for p in contours.lines[0].points:
        assert p.temperature == pytest.approx(1500.0, abs=1.0e-5)
        garnet = next(s for s in p.phases if s.candidate_index == 0)
        assert garnet.composition[0] / garnet.composition.sum() == pytest.approx(
            target, abs=1.0e-9
        )
        assert p.mass_balance_error < 1.0e-8


def test_phase_fraction_constraint_on_latent_volume_field():
    first = bm.Mineral(oxide_params())
    second = bm.CombinedMineral([first], [1.0], [5000.0, 0.0, -1.0e-6], name="dense")
    first.set_state(5.0e9, 1000.0)
    v = first.molar_volume
    options = bm.PseudosectionSettings()
    options.volume_seeds = options.composition_seeds = 4
    options.max_phase_instances = 1
    options.step = 0.1
    diagram = bm.pseudosection(
        {"Mg": 1.0, "O": 1.0},
        [first, second],
        diagram="VX",
        volume_range=(v - 1.2e-6, v + 0.2e-6),
        temperature=1000.0,
        composition_end={"Mg": 1.01, "O": 1.01},
        settings=options,
    )

    def factory(a, parameters, ids):
        return (
            bm.PhaseFractionConstraint(ids.index(1), 0.3, parameters)
            if len(ids) == 2
            else None
        )

    contours = bm.pseudosection_contours(diagram, [first, second], factory)
    assert contours.resolved, contours.diagnostics
    assert len(contours.lines) == 1
    for p in contours.lines[0].points:
        amounts = {s.candidate_index: s.amount for s in p.phases}
        assert amounts[1] / sum(amounts.values()) == pytest.approx(0.3, abs=1.0e-8)
        assert p.pressure == pytest.approx(5.0e9, abs=0.3)
        assert p.temperature == pytest.approx(1000.0, abs=1.0e-6)
        assert p.volume == pytest.approx(
            (1.0 + 0.01 * p.composition_coordinate) * (v - 0.3e-6), abs=1.0e-14
        )


@pytest.mark.parametrize(
    "diagram", ["PT", "PS", "PV", "TS", "TV", "SV", "PX", "TX", "SX", "VX"]
)
def test_contours_use_all_shared_section_coordinates(diagram):
    phase = bm.Mineral(oxide_params())
    phase.set_state(4.0e9, 1000.0)
    values = dict(P=4.0e9, T=1000.0, S=phase.molar_entropy, V=phase.molar_volume)
    widths = dict(P=1.0e8, T=1.0, S=0.05, V=1.0e-9)
    names = dict(
        P="pressure", T="temperature", S="entropy", V="volume", X="composition"
    )
    options = bm.PseudosectionSettings()
    options.max_phase_instances = 1
    options.pressure_seeds = options.temperature_seeds = options.entropy_seeds = (
        options.volume_seeds
    ) = options.composition_seeds = 3
    kwargs = {
        names[c] + "_range": (values[c] - widths[c], values[c] + widths[c])
        for c in diagram
        if c != "X"
    }
    if "X" in diagram:
        kwargs["composition_end"] = {"Mg": 1.001, "O": 1.001}
        kwargs["temperature" if diagram[0] != "T" else "pressure"] = (
            1000.0 if diagram[0] != "T" else 4.0e9
        )
    if "P" not in diagram and "pressure" not in kwargs:
        kwargs["pressure_range"] = (0.0, 8.0e9)
    if "T" not in diagram and "temperature" not in kwargs:
        kwargs["temperature_range"] = (500.0, 1500.0)
    source = bm.pseudosection(
        {"Mg": 1.0, "O": 1.0},
        [phase],
        diagram=diagram,
        settings=options,
        **kwargs,
    )
    constraint_type = dict(
        P=bm.PressureConstraint,
        T=bm.TemperatureConstraint,
        S=bm.EntropyConstraint,
        V=bm.VolumeConstraint,
    )
    first = diagram[0]
    contour = bm.pseudosection_contours(
        json.loads(json.dumps(source.to_dict())),
        [phase],
        constraint_type[first](values[first]),
    )
    assert contour.resolved, contour.diagnostics
    assert len(contour.lines) == 1
    for p in contour.lines[0].points:
        assert getattr(p, names[first]) == pytest.approx(values[first], rel=1.0e-9)
        phase.set_state(p.pressure, p.temperature)
        amount = 1.0 + 0.001 * p.composition_coordinate
        assert p.volume == pytest.approx(amount * phase.molar_volume, abs=1.0e-15)
        assert p.entropy == pytest.approx(amount * phase.molar_entropy, abs=1.0e-8)


def test_linear_free_composition_constraint_keeps_its_original_reference():
    phase = bm.Mineral(oxide_params())
    options = bm.PseudosectionSettings()
    options.max_phase_instances = 1
    options.pressure_seeds = options.composition_seeds = 3
    result = bm.pseudosection(
        {"Mg": 1.0, "O": 1.0},
        [phase],
        diagram="PX",
        pressure_range=(3.0e9, 5.0e9),
        temperature=1000.0,
        composition_end={"Mg": 1.1, "O": 1.1},
        settings=options,
    )

    def factory(a, parameters, ids):
        initial_x = (a.n_moles - 1.0) / 0.1
        row = np.zeros(parameters.n_parameters)
        row[-1] = 1.0
        return bm.LinearXConstraint(row, 0.4 - initial_x)

    contours = bm.pseudosection_contours(result, [phase], factory)
    assert contours.resolved, contours.diagnostics
    assert len(contours.lines) == 1
    for p in contours.lines[0].points:
        assert p.composition_coordinate == pytest.approx(0.4, abs=1.0e-9)
        assert p.phases[0].amount == pytest.approx(1.04, abs=1.0e-9)


@pytest.mark.parametrize(
    "units", [dict(), dict(volume_unit="kg/m3", entropy_unit="kB/atom", swap_axes=True)]
)
def test_separate_contour_plotter_inherits_units_and_accepts_json(units):
    import matplotlib.pyplot as plt

    phase = bm.Mineral(oxide_params())
    phase.set_state(4.0e9, 1000.0)
    v, s = phase.molar_volume, phase.molar_entropy
    options = bm.PseudosectionSettings()
    options.max_phase_instances = 1
    options.entropy_seeds = options.volume_seeds = 2
    result = bm.pseudosection(
        {"Mg": 1.0, "O": 1.0},
        [phase],
        diagram="SV",
        entropy_range=(s - 0.05, s + 0.05),
        volume_range=(v - 1.0e-9, v + 1.0e-9),
        pressure_range=(0.0, 8.0e9),
        temperature_range=(500.0, 1500.0),
        settings=options,
    )
    contours = bm.pseudosection_contours(result, [phase], bm.VolumeConstraint(v))
    fig, ax = bm.plot_pseudosection(result, label_key_path=None, **units)
    collections = len(ax.collections)
    bm.plot_pseudosection_contours(
        result,
        json.loads(json.dumps(contours.to_dict())),
        ax,
        label="constant V",
        label_fontsize=9.0,
    )
    assert len(ax.collections) == collections + 1
    points = ax.collections[-1].get_paths()[0].vertices
    if units:
        expected_density = (
            sum(
                bm.Composition(
                    result.composition_start, "molar"
                ).mass_composition.values()
            )
            / v
        )
        np.testing.assert_allclose(points[:, 1], expected_density, rtol=1.0e-9)
        assert all(
            text.get_fontsize() == 9.0
            for text in ax.texts
            if text.get_text() == "constant V"
        )
    else:
        np.testing.assert_allclose(points[:, 0], v, rtol=1.0e-9)
    with pytest.raises(ValueError, match="must match"):
        bm.plot_pseudosection_contours(result, contours, ax, temperature_unit="K")
    plt.close(fig)


def test_metasediment_example_uses_volume_and_ferrous_garnet_constraints(monkeypatch):
    examples = Path(__file__).resolve().parents[2] / "examples"
    monkeypatch.syspath_prepend(str(examples))
    example = runpy.run_path(str(examples / "example_metasediment_contours.py"))
    diagram = json.loads(
        (Path(__file__).parent / "data/metasediment_contour_field.json").read_text()
    )
    # An interior verification state gives a contour crossing the field;
    # a corner extremum can touch the field at just one point.
    seed = next(p for p in diagram["samples"] if p["is_field_verification"])
    g = next(p for p in seed["phases"] if p["candidate_index"] == 6)
    composition = g["composition"]
    ratio = (composition[0] + composition[3]) / (
        composition[0] + composition[1] + composition[3]
    )
    work = []
    amounts = []
    for p in seed["phases"]:
        phase = example["candidate_phases"]()[p["candidate_index"]]
        if isinstance(phase, bm.Solution):
            phase.set_composition(p["composition"])
        work.append(phase)
        amounts.append(p["amount"])
    a = bm.Assemblage(work, amounts)
    a.n_moles = sum(amounts)
    a.set_state(seed["pressure"], seed["temperature"])
    expected_density = 0.1 / (a.n_moles * a.molar_volume)
    results = example["calculate"](
        diagram, [expected_density], [ratio], seed_grid=3, step=0.1
    )
    density = results["density"][0]["contours"]
    garnet = results["garnet"][0]["contours"]
    assert density["resolved"], density["diagnostics"]
    assert garnet["resolved"], garnet["diagnostics"]
    assert garnet["lines"]
    assert density["lines"]
    for line in garnet["lines"]:
        for point in line["points"]:
            g = next(p for p in point["phases"] if p["candidate_index"] == 6)
            c = np.asarray(g["composition"])
            mg, fe2, fe3 = 3.0 * (c[0] + c[3]), 3.0 * c[1], 2.0 * c[3]
            assert mg / (mg + fe2) == pytest.approx(ratio, abs=1.0e-7)
            assert mg / (mg + fe2 + fe3) != pytest.approx(ratio, abs=1.0e-3)
            assert point["mass_balance_error"] < 1.0e-8
    for line in density["lines"]:
        for point in line["points"]:
            assert 0.1 / point["volume"] == pytest.approx(expected_density, rel=1.0e-8)


def test_contours_are_split_at_holes_in_the_saved_field(simple_diagram):
    source, phase = simple_diagram
    saved = source.to_dict()
    saved["excluded_regions"] = [
        [
            [3.8e9, 900.0],
            [4.2e9, 900.0],
            [4.2e9, 1100.0],
            [3.8e9, 1100.0],
            [3.8e9, 900.0],
        ]
    ]
    # The original verification point lies in the new hole. Add an interior
    # state in the remaining field so its assemblage can be classified.
    phase.set_state(3.4e9, 1000.0)
    saved["samples"].append(
        dict(
            saved["samples"][0],
            pressure=3.4e9,
            temperature=1000.0,
            entropy=phase.molar_entropy,
            volume=phase.molar_volume,
            gibbs=phase.molar_gibbs,
            is_field_verification=True,
        )
    )
    contours = bm.pseudosection_contours(
        saved, [phase], bm.TemperatureConstraint(1000.0)
    )
    assert contours.resolved, contours.diagnostics
    assert len(contours.lines) == 2
    endpoints = sorted(
        p.pressure for line in contours.lines for p in [line.points[0], line.points[-1]]
    )
    np.testing.assert_allclose(endpoints, [3.0e9, 3.8e9, 4.2e9, 5.0e9], atol=1.0)
    for line in contours.lines:
        for p in line.points:
            assert p.pressure <= 3.8e9 + 1.0 or p.pressure >= 4.2e9 - 1.0
