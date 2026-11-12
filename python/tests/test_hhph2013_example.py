# SPDX-License-Identifier: GPL-3.0-or-later
# SPDX-FileCopyrightText: Copyright (C) 2026 by the BurnMan Team.

"""Published activities, calibration offsets and closed-bulk equilibrium."""

from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pytest

import burnman_cpp as bm
from burnman_cpp.minerals import HHPH_2013 as hp
from burnman_cpp.minerals import model_sets
from burnman_cpp.tools.phase_diagram_helpers import load_example


@pytest.fixture(scope="module")
def example():
    return load_example(
        str(
            Path(__file__).resolve().parents[2]
            / "examples/example_hhph2013_phase_diagram.py"
        )
    )


def test_high_pressure_ideal_activities_follow_published_exponents(example):
    phases = {p.name: p for p in example.PHASES}
    # HHPH2013 Appendix: effective multiplicities differ from atom counts.
    mg, fe, ca, na = 11 / 18, 1 / 6, 1 / 6, 1 / 18
    al, mgb, feb, si = 7 / 12, 1 / 12, 1 / 12, 1 / 4
    garnet = [
        mg**3 * al**2,
        fe**3 * al**2,
        ca**3 * al**2,
        4 * mg**3 * mgb * si,
        4 * mg**3 * feb * si,
        27 * mg**2 * na * al * si,
    ]
    mg, fe, ca, na = 1 / 3, 1 / 3, 1 / 6, 1 / 6
    al, mgb, feb, si = 5 / 12, 1 / 6, 1 / 12, 1 / 3
    cf = [
        mg * al,
        ca * al,
        2 * mg * np.sqrt(mgb * si),
        2 * fe * np.sqrt(feb * si),
        2 * fe * np.sqrt(mgb * si),
        2 * na * np.sqrt(al * si),
    ]
    na3, ca3, mg3, fe3 = 1 / 7, 1 / 7, 2 / 7, 3 / 7
    mg2, fe2 = 5 / 7, 2 / 7
    al1, si1, mg1, fe1 = 17 / 42, 13 / 42, 3 / 14, 1 / 14
    normalizer = 1 / ((5 / 6) ** 2.5 * (1 / 6) ** 0.5)
    nal = [
        normalizer * na3 * mg2**2 * al1**2.5 * si1**0.5,
        ca3 * mg2**2 * al1**3,
        mg3 * mg2**2 * al1**3,
        8 * mg3 * mg2**2 * (mg1 * si1) ** 1.5,
        8 * fe3 * fe2**2 * (fe1 * si1) ** 1.5,
        8 * fe3 * mg2**2 * (mg1 * si1) ** 1.5,
        8 * fe3 * fe2**2 * (mg1 * si1) ** 1.5,
    ]
    for name, expected in [
        ("g", garnet),
        ("cf", cf),
        ("nal", nal),
        ("aki", [1 / 3, np.sqrt(2) / 3, np.sqrt(2) / 3]),
    ]:
        model = bm.IdealSolution(phases[name].solution_model.endmembers)
        actual = model.compute_activities(
            20.0e9, 1500.0, np.full(len(expected), 1 / len(expected))
        )
        assert_allclose(actual, expected, atol=1.0e-14, rtol=1.0e-12)


@pytest.mark.parametrize("pressure,temperature", [(15e9, 1000.0), (30e9, 2000.0)])
def test_dqf_and_ordered_endmembers_preserve_gibbs_and_volume(
    example, pressure, temperature
):
    phases = {p.name: p for p in model_sets.hhph2013().phases}
    # Independently evaluate the archived DQF expressions in SI.
    cases = [
        ("mpv", 1, [("fpv", 1)], -9500, 0),
        ("mpv", 2, [("cpv", 1)], 60000, 0),
        ("cpv", 0, [("mpv", 1)], 35000, 0),
        ("nal", 5, [("fsnal", 1 / 6), ("msnal", 5 / 6)], 2000, 0),
        ("nal", 6, [("fsnal", 0.5), ("msnal", 0.5)], 6500, 0),
        ("cor", 2, [("mcor", 1), ("fak", 1), ("mak", -1)], -15000, 0),
        ("cpx", 1, [("fs", 1)], 3800 - 3 * temperature + 3e-7 * pressure, 3e-7),
        ("cpx", 4, [("en", 1)], 3500 - 2 * temperature + 4.8e-7 * pressure, 4.8e-7),
        ("hpx", 3, [("di", 1)], -100 + 0.211 * temperature + 5e-8 * pressure, 5e-8),
        ("hpx", 4, [("mgts", 1), ("hen", 1), ("en", -1)], -1000, 0),
    ]
    for phase, index, recipe, delta_g, delta_v in cases:
        actual = phases[phase].solution_model.endmembers[index][0]
        actual.set_state(pressure, temperature)
        g, v = 0.0, 0.0
        for name, coefficient in recipe:
            base = getattr(hp, name)()
            base.set_state(pressure, temperature)
            g += coefficient * base.molar_gibbs
            v += coefficient * base.molar_volume
        assert actual.molar_gibbs - g == pytest.approx(delta_g, abs=1.0e-8)
        assert actual.molar_volume - v == pytest.approx(delta_v, abs=1.0e-17)


@pytest.mark.parametrize(
    "pressure,temperature", [(3e9, 1473.15), (16.5e9, 1873.15), (30e9, 2273.15)]
)
def test_requested_bulk_equilibria_conserve_each_element(
    example, pressure, temperature
):
    bulk = example.BULK_COMPOSITION.atomic_composition
    result = bm.stable_equilibrium(bulk, example.PHASES, pressure, temperature)
    assert result.success, result.message
    assert result.minimum_affinity >= -0.2
    actual = dict.fromkeys(bulk, 0.0)
    for state in result.phases:
        phase = example.PHASES[state.candidate_index]
        if isinstance(phase, bm.Solution):
            endmembers = [m for m, _ in phase.solution_model.endmembers]
        else:
            endmembers = [phase]
        for fraction, endmember in zip(state.composition, endmembers):
            for element, count in endmember.formula.items():
                actual[element] += state.amount * fraction * count
    for element in bulk:
        assert actual[element] == pytest.approx(bulk[element], rel=1.0e-8)
    assert example.FIGURE_1_OXIDE_MOLES == dict(
        SiO2=39.4, Al2O3=2.0, CaO=3.3, MgO=49.5, FeO=5.2, Na2O=0.26
    )
    assert example.MAX_PYROXENE_CA == pytest.approx(0.2)
    assert example.REFINEMENT_RESOLUTION == (201, 201)
    assert example.PRESSURE_RANGE == (3e9, 30e9)
    assert example.TEMPERATURE_RANGE == (1473.15, 2273.15)
    for phase in result.phases:
        if phase.name in ("opx", "hpx"):
            assert phase.composition[3] <= 0.2 + 1e-9


@pytest.mark.parametrize("pressure", [9.0, 20.0, 24.0, 30.0])
def test_ca_limit_preserves_mg_rich_branch_and_does_not_pin_compositions(
    example, pressure
):
    from benchmark_hhph2013 import benchmark_phases

    bulk = bm.Composition(
        dict(SiO2=39.4, Al2O3=2.0, CaO=3.3, MgO=49.5, FeO=5.2, Na2O=0.26), "molar"
    )
    bulk.renormalize("mass", "total", 0.1)
    temperature = 1357.0 + 11.6 * pressure + 273.15
    states = []
    for cap in [0.1, 0.5]:
        phases = benchmark_phases(max_pyroxene_ca=cap)
        assert {"opx", "hpx"} <= {p.name for p in phases}
        state = bm.stable_equilibrium(
            bulk.atomic_composition, phases, pressure * 1e9, temperature
        )
        assert state.success, state.message
        states.append(state)
        for phase in state.phases:
            if phase.name in ("opx", "hpx"):
                assert phase.composition[3] < 0.1 - 1e-6
    assert states[0].gibbs == pytest.approx(states[1].gibbs, abs=1e-8)
    assert [p.name for p in states[0].phases] == [p.name for p in states[1].phases]
    if pressure == 9.0:
        assert "hpx" in {p.name for p in states[0].phases}
    else:
        assert "hpx" not in {p.name for p in states[0].phases}


def test_archive_includes_ds62_cats_ordering_and_fscf_reference(example):
    from burnman_cpp.minerals import HP_2011_ds62

    phases = {p.name: p for p in model_sets.hhph2013().phases}
    cats = phases["cpx"].solution_model.endmembers[2][0]
    ds62 = HP_2011_ds62.cats()
    for mineral in [cats, ds62]:
        mineral.set_state(15e9, 1800.0)
    assert cats.molar_gibbs == pytest.approx(ds62.molar_gibbs, abs=1e-8)
    fscf = phases["cf"].solution_model.endmembers[3][0]
    original = hp.fscf()
    for mineral in [fscf, original]:
        mineral.set_state(25e9, 2000.0)
    assert fscf.molar_gibbs == pytest.approx(original.molar_gibbs, abs=1e-8)
