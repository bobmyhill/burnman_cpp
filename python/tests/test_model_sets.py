"""Matched HPx families and published 2022 Temkin melt activities."""

from pathlib import Path
import runpy

import numpy as np
import pytest
import burnman_cpp as bm

from burnman_cpp.minerals import (
    model_sets,
    IG18,
    HGP18,
    MP14,
    MB16,
)


@pytest.mark.parametrize("family", ["metapelite", "metabasite"])
def test_model_sets_have_independent_phases_and_ps94_water(family):
    first, second = getattr(model_sets, family)(), getattr(model_sets, family)()
    names = [p.name for p in first.phases]
    assert len(set(names)) == len(names)
    assert first.dataset == "HP11"
    assert "PS94" in first.to_dict()["water"]
    water = first.phases[names.index("H2O")]
    # Published and-bi-mu-q-H2O-2.5-600.txt, also checked by the native
    # tests/include/thermocalc_reference.hpp oracle (2.5 kbar, 600 C).
    water.set_state(2.5e8, 873.15)
    assert water.molar_gibbs == pytest.approx(-367307.85, abs=10.0)
    assert water.molar_enthalpy == pytest.approx(-240824.10, abs=10.0)
    first.phases[0].set_name("changed")
    assert second.phases[0].name == names[0]


def test_igneous_collection_has_full_melt_and_records_its_scope():
    models = model_sets.igneous()
    assert "G25" in models.version
    assert models.dataset == "HPx_ds636"
    assert "G25" in models.notes and "aqueous fluid" in models.notes
    assert models.to_dict()["notes"] == models.notes
    phases = {p.name: p for p in models.phases}
    assert phases.keys() >= {
        "ol",
        "cpx",
        "opx",
        "g",
        "sp",
        "fsp",
        "hb",
        "bi",
        "ms",
        "ep",
        "cd",
        "ilm",
        "melt",
        "fluid",
    }
    assert phases["melt"].n_endmembers == 12
    assert "Cr" in phases["melt"].elements
    assert sum(p.name == "fsp" for p in models.phases) == 1


@pytest.mark.parametrize("pressure,temperature", [(2.0e8, 1000.0), (2.0e9, 1400.0)])
def test_igneous_olivine_retains_its_original_ordered_endmember(pressure, temperature):
    ol = IG18.ol()
    ol.set_composition([0.0, 0.0, 0.0, 1.0])
    ol.set_state(pressure, temperature)
    assert {k: v for k, v in ol.formula.items() if v} == {
        "Mg": 1.0,
        "Fe": 1.0,
        "Si": 1.0,
        "O": 4.0,
    }
    # The original Python declaration was the mean of fa and fo, with no
    # correction. Its reused variable name must not select a pyroxene species.
    fa, fo = HGP18.fa(), HGP18.fo()
    for p in (fa, fo):
        p.set_state(pressure, temperature)
    for prop in ("molar_gibbs", "molar_entropy", "molar_volume"):
        assert getattr(ol, prop) == pytest.approx(
            0.5 * (getattr(fa, prop) + getattr(fo, prop)), rel=8.0e-8
        )
    assert IG18.cfm_cpx().formula["Si"] == 2.0  # The pyroxene species stays available.


@pytest.mark.parametrize("temperature", [1173.15, 1673.15])
def test_igneous_equilibrium_handles_chromium_free_bulk(temperature):
    example = runpy.run_path(
        str(Path(__file__).parents[2] / "examples/example_basalt_pseudosection.py")
    )
    bulk = example["BASALT_COMPOSITION"].atomic_composition
    assert bulk.get("Cr", 0.0) == 0.0
    # Projecting this chemical face produced a -9.5e-17 starting fraction;
    # NLopt's strict box bounds rejected the otherwise feasible initial state.
    state = bm.stable_equilibrium(bulk, model_sets.igneous().phases, 1.0e9, temperature)
    assert state.success, state.message
    assert state.mass_balance_error < 1.0e-8
    assert state.minimum_affinity >= -0.2
    assert state.equilibrium_error <= 0.02


def test_metabasite_keeps_clinopyroxene_calibrations_separate():
    dio, aug = model_sets.metabasite("dio"), model_sets.metabasite("aug")
    assert dio.name != aug.name
    assert sum(p.name == "cpx" for p in dio.phases) == 1
    assert sum(p.name == "cpx" for p in aug.phases) == 1
    assert dio.phases[1].endmember_names != aug.phases[1].endmember_names
    with pytest.raises(ValueError, match="one model"):
        model_sets.metabasite("both")


@pytest.mark.parametrize("factory", [MP14.liq, MB16.L])
def test_melt_gibbs_derivatives_with_variable_site_multiplicities(factory):
    s = factory()
    p = np.arange(1.0, s.n_endmembers + 1)
    p /= p.sum()
    s.set_composition(p)
    s.set_state(8.0e8, 1073.15)
    mu, hessian = s.partial_gibbs.copy(), s.gibbs_hessian.copy()
    step = 1.0e-5
    for i in range(1, len(p)):
        high, low = p.copy(), p.copy()
        high[i] += step
        high[0] -= step
        low[i] -= step
        low[0] += step
        s.set_composition(high)
        g_high, mu_high = s.molar_gibbs, s.partial_gibbs.copy()
        s.set_composition(low)
        assert (g_high - s.molar_gibbs) / (2 * step) == pytest.approx(
            mu[i] - mu[0], abs=0.02
        )
        np.testing.assert_allclose(
            (mu_high - s.partial_gibbs) / (2 * step),
            hessian[:, i] - hessian[:, 0],
            rtol=1.0e-6,
            atol=0.03,
        )


@pytest.mark.parametrize(
    "calibration,module,expected",
    [
        ("G25", "IG25", {"fluid", "hb", "bi", "ms", "ep", "cd"}),
        ("W24", "IG24", {"nph", "kals", "lct", "mel"}),
    ],
)
def test_igneous_calibrations_are_complete_and_independent(
    calibration, module, expected
):
    from burnman_cpp import minerals

    first, second = model_sets.igneous(calibration), model_sets.igneous(calibration)
    names = [p.name for p in first.phases]
    assert len(names) == len(set(names))
    assert expected <= set(names)
    assert first.dataset == "HPx_ds636"
    assert first.phases[7].n_endmembers == (12 if calibration == "G25" else 14)
    assert sum(isinstance(p, bm.Solution) for p in first.phases) == (
        14 if calibration == "G25" else 12
    )
    first.phases[0].set_name("changed")
    assert second.phases[0].name == "ol"
    assert getattr(minerals, module) is not None
    if calibration == "W24":
        assert all(p.formula.get("H", 0.0) == 0.0 for p in first.phases)
    else:
        fluid = next(p for p in first.phases if p.name == "fluid")
        assert fluid.n_endmembers == 11
        assert "Si" in fluid.elements and "H" in fluid.elements
    with pytest.raises(ValueError, match="G25.*W24"):
        model_sets.igneous("H18")
