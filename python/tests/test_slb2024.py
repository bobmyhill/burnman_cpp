"""SLB2024 thermodynamics, redox and EOS-domain regressions."""

import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest

import burnman_cpp as bm
from burnman_cpp.minerals import SLB_2024 as SLB

SOLUTIONS = [
    "c2c_pyroxene",
    "calcium_ferrite_structured_phase",
    "clinopyroxene",
    "garnet",
    "ilmenite",
    "ferropericlase",
    "new_aluminous_phase",
    "olivine",
    "orthopyroxene",
    "plagioclase",
    "post_perovskite",
    "bridgmanite",
    "ringwoodite",
    "mg_fe_aluminous_spinel",
    "wadsleyite",
]


@pytest.fixture(scope="module")
def example():
    return runpy.run_path(
        str(
            Path(__file__).resolve().parents[2]
            / "examples/example_pyrolite_pseudosection.py"
        )
    )


def test_pyrolite_bulk_has_exact_initial_ferric_ratio(example):
    oxides = example["PYROLITE_OXIDES"]
    assert sum(oxides.values()) == pytest.approx(100.0)
    composition = example["PYROLITE_COMPOSITION"]
    assert isinstance(composition, bm.Composition)
    assert sum(composition.mass_composition.values()) == pytest.approx(0.1)
    fe = composition.atomic_composition["Fe"]
    ferric = 2.0 * composition.molar_composition["Fe2O3"]
    assert ferric / fe == pytest.approx(0.03, abs=1.0e-15)
    reduced = example["pyrolite_composition"](0.0)
    reference = reduced.atomic_composition
    bulk = composition.atomic_composition
    for element in ["Na", "Ca", "Mg", "Fe", "Al", "Si", "Cr"]:
        assert bulk[element] / bulk["Si"] == pytest.approx(
            reference[element] / reference["Si"]
        )
    assert bulk["O"] / bulk["Si"] > reference["O"] / reference["Si"]
    with pytest.raises(ValueError):
        example["pyrolite_composition"](1.1)


def test_pyrolite_example_resume_accepts_roundoff_but_rejects_changed_bulk(
    example, tmp_path, monkeypatch
):
    previous = json.loads(
        (Path(__file__).parent / "data" / "pyrolite_model_limit.json").read_text()
    )["previous"]
    oxides = example["PYROLITE_OXIDES"].copy()
    oxides["FeO"] = float(np.nextafter(oxides["FeO"], np.inf))
    previous["oxide_wt_percent"] = oxides
    source = tmp_path / "pyrolite.json"
    source.write_text(json.dumps(previous))
    sentinel = object()

    def continue_saved(bulk, phases, data, opts):
        assert bulk == example["PYROLITE_COMPOSITION"].atomic_composition
        assert data["oxide_wt_percent"] == oxides
        assert phases and opts.verbose
        return sentinel

    monkeypatch.setattr(bm, "refine_pseudosection", continue_saved)
    assert example["refine"](source, verbose=True) is sentinel
    oxides["FeO"] *= 1.01
    source.write_text(json.dumps(previous))
    with pytest.raises(ValueError, match="different bulk composition"):
        example["refine"](source)


@pytest.mark.parametrize(
    "pressure,temperature",
    [
        (0.0, 0.0),
        (50.0e9, 0.0),
        (150.0e9, 0.0),
        (0.0, 1600.0),
        (30.0e9, 2000.0),
        (100.0e9, 3000.0),
        (150.0e9, 4000.0),
    ],
)
def test_native_pyrolite_equilibria_conserve_bulk_and_verify_stability(
    example, pressure, temperature
):
    state = bm.stable_equilibrium(
        example["PYROLITE_COMPOSITION"].atomic_composition,
        example["candidate_phases"](),
        pressure,
        temperature,
    )
    assert state.success, state.message
    assert not state.outside_model_domain
    assert state.mass_balance_error < 1.0e-8
    assert state.equilibrium_error < 0.02
    assert state.minimum_affinity > -0.2
    assert all(p.amount > 0.0 for p in state.phases)
    # Ferric production is permitted to consume ferrous iron and generate
    # metal in an oxygen-closed bulk; the initial ferric ratio is not buffered.
    if pressure == 30.0e9:
        assert any(p.name.startswith("Fe-") for p in state.phases)
        bg = next(p for p in state.phases if p.name == "bg")
        assert bg.composition[3:6].sum() > 0.0


def test_infeasible_hot_low_pressure_bulk_is_an_explicit_model_limit(example):
    state = bm.stable_equilibrium(
        example["PYROLITE_COMPOSITION"].atomic_composition,
        example["candidate_phases"](),
        0.0,
        4000.0,
        example["settings"](),
    )
    assert not state.success
    assert state.outside_model_domain
    assert state.excluded_phases
    assert any("Forsterite" in reason for reason in state.excluded_phases)


def test_required_orthopyroxene_masks_its_spinodal_even_when_bulk_remains_feasible(
    example,
):
    bulk = example["PYROLITE_COMPOSITION"].atomic_composition
    opts = example["settings"]()
    opts.required_eos_phases = []
    admissible = bm.stable_equilibrium(
        bulk, example["candidate_phases"](), 7.0e9, 3880.0, opts
    )
    assert admissible.success, admissible.message
    opts.required_eos_phases = ["opx"]
    required = bm.stable_equilibrium(
        bulk, example["candidate_phases"](), 7.0e9, 3880.0, opts
    )
    assert not required.success and required.outside_model_domain
    assert "Mg-Tschermak" in required.message


def test_eos_exclusion_can_be_disabled_for_strict_candidate_validation(example):
    opts = example["settings"]()
    opts.exclude_invalid_eos = False
    opts.required_eos_phases = []
    state = bm.stable_equilibrium(
        example["PYROLITE_COMPOSITION"].atomic_composition,
        example["candidate_phases"](),
        100.0e9,
        3000.0,
        opts,
    )
    assert not state.success
    assert "stable EOS branch" in state.message


@pytest.mark.parametrize("case_index", range(3))
def test_trace_spin_activation_continues_without_adding_an_identical_phase(
    example, case_index
):
    """Real CF/metal/ppv edges that used to stop at a trace spin component."""
    cases = json.loads(
        (Path(__file__).parent / "data" / "pyrolite_endpoints.json").read_text()
    )["cases"]
    case = cases[case_index]
    previous = case["previous"]
    opts = bm.PseudosectionSettings()
    opts.max_lines = 1
    opts.max_recovery_passes = 1
    result = bm.refine_pseudosection(
        example["PYROLITE_COMPOSITION"].atomic_composition,
        example["candidate_phases"](),
        previous,
        opts,
    )
    line = result.boundaries[0]
    start = case["end"] == "start"
    assert line.termination.split("; ")[0 if start else 1] in [
        "junction",
        "domain edge",
    ]
    original = previous["boundaries"][0]["points"][0 if start else -1]
    endpoint = line.points[0 if start else -1]
    distance = np.linalg.norm(
        [
            (endpoint.pressure - original["pressure"]) / 150.0e9,
            (endpoint.temperature - original["temperature"]) / 4000.0,
        ]
    )
    assert distance > 0.04
    added = line.points[: len(line.points) - 2] if start else line.points[2:]
    assert added
    candidate_count = len(example["candidate_phases"]())
    for point in added:
        assert point.mass_balance_error < 1.0e-8
        assert point.minimum_affinity >= -0.2
        assert point.residual < 0.02
        for candidate in range(candidate_count):
            copies = [
                p.composition for p in point.phases if p.candidate_index == candidate
            ]
            for i, one in enumerate(copies):
                for two in copies[i + 1 :]:
                    assert np.linalg.norm(one - two) > 1.0e-4


@pytest.mark.parametrize(
    "case_index,termination",
    [(0, "domain edge"), (1, "solution critical point"), (2, "junction")],
)
def test_cold_frame_critical_and_close_redox_endpoints(
    example, case_index, termination
):
    cases = json.loads(
        (Path(__file__).parent / "data" / "pyrolite_cold_endpoints.json").read_text()
    )["cases"]
    opts = bm.PseudosectionSettings()
    opts.max_lines = 1
    opts.max_recovery_passes = 1
    previous = cases[case_index]["previous"]
    result = bm.refine_pseudosection(
        example["PYROLITE_COMPOSITION"].atomic_composition,
        example["candidate_phases"](),
        previous,
        opts,
    )
    line = result.boundaries[0]
    point = line.points[-1]
    assert line.termination.split("; ")[1] == termination
    assert line.end_node >= 0
    assert point.mass_balance_error < 1.0e-8
    assert point.minimum_affinity >= -0.2
    if case_index == 0:
        # Classification preserves the already accepted thermodynamic point.
        assert point.pressure == previous["boundaries"][0]["points"][-1]["pressure"]
        assert (
            point.temperature == previous["boundaries"][0]["points"][-1]["temperature"]
        )
    if case_index == 1:
        assert result.nodes[line.end_node].kind == "critical_point"
        assert result.nodes[line.end_node].critical_mode.size > 0
    if case_index == 2:
        node = result.nodes[line.end_node]
        assert node.kind == "junction" and len(node.zero_phases) == 2
        assert (
            point.temperature > previous["boundaries"][0]["points"][-1]["temperature"]
        )
        for zero in node.zero_phases:
            assert abs(next(p.amount for p in point.phases if p.id == zero)) < 1.0e-9


def test_resumed_model_limit_remains_attached_to_recomputed_eos_envelope(example):
    case = json.loads(
        (Path(__file__).parent / "data" / "pyrolite_model_limit.json").read_text()
    )
    previous = case["previous"]
    opts = example["settings"]()
    opts.max_lines = 1
    opts.max_recovery_passes = 0
    result = bm.refine_pseudosection(
        example["PYROLITE_COMPOSITION"].atomic_composition,
        example["candidate_phases"](),
        previous,
        opts,
    )
    point = result.boundaries[0].points[0 if case["start"] else -1]
    saved = previous["boundaries"][0]["points"][0 if case["start"] else -1]
    assert (
        point.pressure == saved["pressure"]
        and point.temperature == saved["temperature"]
    )
    location = np.array([point.pressure, point.temperature])
    assert any(
        np.min(np.linalg.norm((ring - location) / [150.0e9, 4000.0], axis=1)) < 1.0e-12
        for ring in result.excluded_regions
    )
    # The common serializer keeps the envelope, original compositions and
    # required-model policy; resuming JSON needs no example-specific settings.
    saved = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    restored = bm.PseudosectionResult.from_dict(saved)
    assert restored.to_dict() == saved
    resumed = bm.refine_pseudosection(
        example["PYROLITE_COMPOSITION"].atomic_composition,
        example["candidate_phases"](),
        saved,
    )
    assert resumed.settings.required_eos_phases == ["opx"]
    assert any(
        np.min(np.linalg.norm((ring - location) / [150.0e9, 4000.0], axis=1)) < 1.0e-12
        for ring in resumed.excluded_regions
    )


def test_legacy_pyrolite_json_retains_eos_policy_and_failure_reasons(example):
    case = json.loads(
        (Path(__file__).parent / "data" / "pyrolite_model_limit.json").read_text()
    )
    saved = case["previous"]
    saved.update(
        excludes_invalid_eos=True,
        active_solution_faces=True,
        required_eos_phases=["opx"],
        resolved=False,
        diagnostics=["saved diagnostic"],
    )
    state = bm.stable_equilibrium(
        example["PYROLITE_COMPOSITION"].atomic_composition,
        example["candidate_phases"](),
        0.0,
        4000.0,
    )
    saved["samples"] = [
        dict(
            pressure=state.pressure,
            temperature=state.temperature,
            success=state.success,
            outside_model_domain=state.outside_model_domain,
            message=state.message,
            excluded_phases=state.excluded_phases,
            gibbs=state.gibbs,
            mass_balance_error=state.mass_balance_error,
            minimum_affinity=state.minimum_affinity,
            equilibrium_error=state.equilibrium_error,
            phases=[],
        )
    ]
    restored = bm.PseudosectionResult.from_dict(saved)
    assert restored.settings.required_eos_phases == ["opx"]
    assert not restored.resolved and restored.diagnostics == ["saved diagnostic"]
    assert restored.samples[0].outside_model_domain
    assert restored.samples[0].excluded_phases == state.excluded_phases
    assert restored.samples[0].message == state.message


def test_nearby_redox_junctions_keep_distinct_phase_rules_on_resume(example):
    previous = json.loads(
        (Path(__file__).parent / "data" / "pyrolite_nearby_junctions.json").read_text()
    )
    opts = example["settings"]()
    opts.max_lines = 4
    opts.max_recovery_passes = 0
    result = bm.refine_pseudosection(
        example["PYROLITE_COMPOSITION"].atomic_composition,
        example["candidate_phases"](),
        previous,
        opts,
    )
    ids = []
    for old, line in zip(previous["boundaries"], result.boundaries):
        start = old["start_node"] == 0
        index = line.start_node if start else line.end_node
        ids.append(index)
        node = result.nodes[index]
        point = line.points[0 if start else -1]
        assert (
            np.linalg.norm(
                (
                    np.array([node.pressure, node.temperature])
                    - [point.pressure, point.temperature]
                )
                / [150.0e9, 4000.0]
            )
            < 1.0e-8
        )
    # Two iron-polymorph representations of one event still share a node;
    # the two chemically different nearby events have their own nodes.
    assert ids[0] == ids[1]
    assert len(set(ids)) == 3


@pytest.mark.parametrize(
    "factory", [SLB.fea, SLB.feg, SLB.fee, SLB.mag, SLB.smag, SLB.hmag]
)
def test_zero_temperature_magnetic_and_electronic_limits_are_finite(factory):
    phase = factory()
    phase.set_state(30.0e9, 0.0)
    values = [
        phase.molar_gibbs,
        phase.molar_entropy,
        phase.molar_volume,
        phase.molar_heat_capacity_p,
    ]
    assert np.isfinite(values).all()
    assert phase.molar_heat_capacity_p == pytest.approx(0.0, abs=1.0e-12)
    phase.set_state(30.0e9, 1.0e-5)
    assert phase.molar_gibbs == pytest.approx(values[0], abs=0.01)


def test_spinodal_bracket_recovers_the_stable_root_and_rejects_the_other():
    # A 4000 K forsterite root exists just above its thermal pressure minimum.
    phase = SLB.fo()
    phase.set_state(3.0e9, 4000.0)
    assert phase.isothermal_bulk_modulus_reuss > 0.0
    volume = phase.molar_volume
    phase.set_state(3.01e9, 4000.0)
    assert phase.molar_volume < volume
    phase.set_state(0.0, 4000.0)
    with pytest.raises(ValueError, match="limiting pressure"):
        _ = phase.molar_volume


def test_zero_site_multiplicity_survives_polytope_basis_transformation():
    original = SLB.ferropericlase()
    basis = np.eye(original.n_endmembers)[[0, 4]]
    reduced = bm.transform_solution_to_new_basis(original, basis, [0.75, 0.25])
    original.set_composition([0.75, 0.0, 0.0, 0.0, 0.25])
    for phase in [original, reduced]:
        phase.set_state(30.0e9, 1600.0)
    assert reduced.molar_gibbs == pytest.approx(original.molar_gibbs, abs=1.0e-7)


@pytest.mark.parametrize(
    "name",
    SOLUTIONS + ["fea", "feg", "fee", "st", "qtz", "neph", "mag", "smag", "hmag"],
)
def test_sLB2024_catalogue_matches_python_reference(name):
    reference = pytest.importorskip("burnman").minerals.SLB_2024
    native = getattr(SLB, name)()
    pure = getattr(reference, name)()
    if isinstance(native, bm.Solution):
        fractions = np.arange(1.0, native.n_endmembers + 1)
        fractions /= fractions.sum()
        native.set_composition(fractions)
        pure.set_composition(fractions)
    for pressure, temperature in [(20.0e9, 1200.0), (80.0e9, 3000.0)]:
        native.set_state(pressure, temperature)
        pure.set_state(pressure, temperature)
        properties = [
            "molar_gibbs",
            "molar_entropy",
            "molar_volume",
            "molar_heat_capacity_p",
            "shear_modulus",
        ]
        if isinstance(native, bm.Solution):
            properties += ["partial_gibbs", "gibbs_hessian"]
        for prop in properties:
            np.testing.assert_allclose(
                getattr(native, prop),
                getattr(pure, prop),
                rtol=2.0e-10,
                atol=2.0e-5,
                err_msg=f"{name}, {pressure}, {temperature}: {prop}",
            )
